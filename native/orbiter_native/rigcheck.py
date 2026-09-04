"""What the rig can actually do right now — measured on the spot.

Every number a scan depends on is knowable before scanning, and each one has
a failure that looks exactly like every other from the panel: no points. The
counts on the SCAN panel read the same for a calibration whose pieces
describe different cameras, for a laser that is off, and for a subject out of
reach. This asks each question separately and prints the answer.

    orbiter-rigcheck

It talks to camserver and to the Orbiter server, takes a snapshot from each
eye, and returns non-zero when it finds something that stops a scan — so it
also works as a check after a calibration sweep rather than only as a thing
to read.

The measurements, in the order they gate a point:

  cameras   exposure mode, flicker setting, the frame rate each eye is
            really running at. Auto exposure hunts, and a camera that has
            dropped to 20 fps is holding the shutter open for 50 ms.
  pairing   how far apart in the capture clock the two eyes land. They
            free-run; the offset walks, and a scan pairs within 20 ms.
  calibration  what the server holds, and how much data each piece rests on.
  stripe    where the sheet says the stripe can be against where it is.
  veto      how far the right eye's stripe sits from where the left eye's
            candidates project into its frame. This is the one that says
            whether the calibration can scan at all.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from typing import Any

import cv2
import httpx
import numpy as np

from .config import parse as parse_config
from .laser import find_stripe_pixels
from .laserplane import from_config as plane_from_config
from .scan import ScanParams, _on_plane, _reach_mm, _veto_offset, stripe_rows
from .stereo import StereoRig, result_from_config

#: Frame headers carry the capture instant; it is the only clock that times
#: both cameras, so it is the only one a pair can be judged on.
STAMP = re.compile(rb"X-Capture-Monotonic:\s*([0-9.]+)", re.I)

OK, WARN, BAD = "ok", "warn", "bad"


def nearest_gaps(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """For each left capture instant, the distance to the nearest right one.

    What the pairing actually gets to choose from: with two free-running
    cameras the offset between the eyes sits wherever it lands and walks
    slowly, so this is a distribution, not a number.
    """
    if not len(left) or not len(right):
        return np.empty(0)
    right = np.sort(right)
    idx = np.searchsorted(right, left)
    out = np.full(len(left), np.inf)
    for k in (-1, 0):
        j = np.clip(idx + k, 0, len(right) - 1)
        out = np.minimum(out, np.abs(left - right[j]))
    return out


def _stamps(host: str, cam: str, seconds: float, into: list[float]) -> None:
    end = time.monotonic() + seconds
    try:
        with httpx.stream("GET", f"{host}/stream/{cam}?sync=1", timeout=seconds + 15) as r:
            tail = b""
            for chunk in r.iter_bytes(65536):
                for m in STAMP.finditer(tail + chunk):
                    into.append(float(m.group(1)))
                tail = chunk[-64:]
                if time.monotonic() > end:
                    break
    except httpx.HTTPError as exc:                       # a camera can be busy
        into.append(float("nan"))
        print(f"    {cam}: stream unreadable — {exc}")


def _snapshot(host: str, cam: str) -> np.ndarray | None:
    r = httpx.get(f"{host}/snapshot/{cam}", timeout=10)
    r.raise_for_status()
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def _controls(host: str, cam: str) -> dict[str, Any]:
    r = httpx.get(f"{host}/api/controls/{cam}", timeout=8).json()
    return {c["slug"]: c for c in r.get("controls", [])}


#: A line's mark: nothing to see, worth knowing, in the way of a scan.
MARK = {OK: "  ", WARN: " !", BAD: " X"}


def say(state: str, line: str) -> None:
    print(f"{MARK[state]} {line}")


def check_cameras(host: str, cams: dict[str, str]) -> list[str]:
    """Exposure mode and flicker setting, read back from the driver."""
    print("cameras")
    bad = []
    for side, cam in cams.items():
        try:
            c = _controls(host, cam)
        except httpx.HTTPError as exc:
            say(BAD, f"{side:5s} {cam}: controls unreadable — {exc}")
            bad.append(f"{cam} unreachable")
            continue
        auto = c.get("auto_exposure", {}).get("value")
        line = c.get("power_line_frequency", {}).get("value")
        # 1 is Manual Mode. Anything else is the camera choosing its own
        # exposure, which on this rig meant 50 ms and a smeared stripe.
        say(OK if auto == 1 else WARN,
            f"{side:5s} {cam}: auto_exposure={auto} "
            f"({'manual' if auto == 1 else 'the camera is choosing'})")
        if auto != 1:
            bad.append(f"{cam} auto exposure")
        say(OK if line == 1 else WARN,
            f"{side:5s} {cam}: power_line_frequency={line} "
            f"({'50 Hz' if line == 1 else 'not 50 Hz — mains banding'})")
    return bad


def check_pairing(host: str, cams: dict[str, str], seconds: float,
                  window_s: float) -> list[str]:
    """Frame rate per eye, and how often a pair falls inside the window."""
    print(f"\npairing (watching {seconds:.0f} s)")
    got: dict[str, list[float]] = {s: [] for s in cams}
    threads = [threading.Thread(target=_stamps, args=(host, cam, seconds, got[side]))
               for side, cam in cams.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps = {s: np.array([v for v in got[s] if np.isfinite(v)]) for s in got}
    bad = []
    for side, cam in cams.items():
        fps = len(stamps[side]) / seconds
        say(OK if fps >= 25 else WARN, f"{side:5s} {cam}: {fps:.1f} fps")
        if fps < 25:
            bad.append(f"{cam} at {fps:.0f} fps")
    gaps = nearest_gaps(stamps["left"], stamps["right"])
    if not len(gaps):
        say(BAD, "no frames from one of the eyes")
        return bad + ["no frames"]
    share = float((gaps <= window_s).mean())
    say(OK if share >= 0.5 else WARN,
        f"nearest partner: median {np.median(gaps) * 1000:.1f} ms, "
        f"p90 {np.percentile(gaps, 90) * 1000:.1f} ms")
    say(OK if share >= 0.5 else WARN,
        f"inside the scan's {window_s * 1000:.0f} ms window: {100 * share:.0f}% of frames")
    if share < 0.5:
        bad.append(f"only {100 * share:.0f}% of frames can pair")
    return bad


def check_calibration(rig_cfg: dict) -> list[str]:
    """What the server holds, and how much data each piece rests on."""
    print("\ncalibration on the server")
    bad = []
    for side in ("left", "right"):
        k = (rig_cfg.get(side) or {}).get("intrinsics")
        if not k:
            say(BAD, f"K {side:5s}: none")
            bad.append(f"no {side} intrinsics")
            continue
        sf = k.get("sigma_f")
        rel = None if sf is None else float(sf) / float(k["fx"])
        say(OK if rel is not None and rel < 0.005 else WARN,
            f"K {side:5s}: {k.get('views')} views, rms {k.get('rms_px', float('nan')):.2f} px, "
            f"focal ±{'?' if rel is None else f'{100 * rel:.2f}%'}")
    ex = rig_cfg.get("extrinsics") or {}
    if ex:
        say(OK, f"pair   : {ex.get('views')} views, "
                f"rms {ex.get('rms_px', float('nan')):.2f} px, "
                f"baseline {ex.get('baseline_mm', float('nan')):.1f} mm")
    else:
        say(BAD, "pair   : none")
        bad.append("no extrinsics")
    lp = rig_cfg.get("laser_plane") or {}
    if lp:
        # A sheet from a handful of frames all at one distance fits them and
        # says nothing about anywhere else.
        say(OK if lp.get("frames", 0) >= 6 else WARN,
            f"sheet  : {lp.get('frames')} frames, {lp.get('points')} points, "
            f"d {lp.get('d', float('nan')):.2f} mm, "
            f"rms {lp.get('rms_mm', float('nan')):.2f} mm")
    else:
        say(BAD, "sheet  : none")
        bad.append("no laser plane")
    for side in ("left", "right"):
        ro = (rig_cfg.get(side) or {}).get("readout")
        if ro:
            ms = ro["seconds"] * 1000.0
            # Longer than a frame interval is not a readout; it is a fit that
            # found the wrong slope.
            say(OK if 1.0 < ms < 34.0 else WARN,
                f"readout {side:5s}: {ms:.1f} ms ± {ro['sigma_s'] * 1000:.1f}")
    return bad


def check_stripe(cfg, rig_cfg: dict, host: str, cams: dict[str, str],
                 params: ScanParams) -> list[str]:
    """Where the sheet says the stripe can be, where it is, and whether the
    other eye agrees — the question the veto asks on every pixel."""
    print("\nstripe and veto")
    bad: list[str] = []
    frames = {}
    for side, cam in cams.items():
        img = _snapshot(host, cam)
        if img is None:
            say(BAD, f"{side}: no frame from {cam}")
            return ["no frame"]
        frames[side] = img
    wh = (frames["left"].shape[1], frames["left"].shape[0])
    right_wh = (frames["right"].shape[1], frames["right"].shape[0])
    kl = cfg.left.intrinsics_for(wh) if cfg.left else None
    kr = cfg.right.intrinsics_for(right_wh) if cfg.right else None
    geom = result_from_config(cfg.extrinsics_raw, wh)
    plane = plane_from_config(cfg.laser_plane_raw, wh)
    if kl is None or kr is None or geom is None or plane is None:
        say(BAD, "the calibration does not resolve at this frame size — "
                 "intrinsics, pair and sheet are all needed")
        return ["calibration does not resolve"]
    rig = StereoRig(kl, kr, geom)

    bands = {s: stripe_rows(plane, rig, params.range_mm, size, s)
             for s, size in (("left", wh), ("right", right_wh))}
    pix = {s: find_stripe_pixels(frames[s], rows=bands[s]) for s in ("left", "right")}
    for side in ("left", "right"):
        p, band = pix[side], bands[side]
        say(OK if p.ok else BAD,
            f"{side:5s}: rows {band[0]}..{band[1]} of {frames[side].shape[0]}, "
            f"{p.count} lit px" + (f" — {p.reason}" if p.reason else ""))
        if not p.ok:
            bad.append(f"no stripe in the {side} eye")
    if bad:
        return bad

    px = np.stack([pix["left"].x, pix["left"].y], axis=1).astype(np.float64)
    cand = _on_plane(kl, plane, px)
    ok = np.isfinite(cand).all(axis=1)
    reach = _reach_mm(cand[ok], rig, "left")
    lo, hi = params.range_mm
    inside = float(((reach >= lo) & (reach <= hi)).mean()) if len(reach) else 0.0
    say(OK if inside > 0.2 else WARN,
        f"reach : median {np.median(reach):.0f} mm, {100 * inside:.0f}% within "
        f"{lo:.0f}-{hi:.0f} mm")

    off = _veto_offset(pix["right"], rig.project_right(cand[ok]))
    if not np.isfinite(off):
        say(BAD, "veto  : the eyes share no scanline — nothing to compare")
        return ["the eyes share no scanline"]
    good = abs(off) <= params.confirm_px
    say(OK if good else BAD,
        f"veto  : the eyes disagree by {off:+.1f} px "
        f"(confirm_px is {params.confirm_px}) — "
        f"{'a scan will confirm' if good else 'nothing will confirm'}")
    if not good:
        bad.append(f"veto residual {off:+.1f} px")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="orbiter-rigcheck", description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://localhost:8000",
                    help="Orbiter server holding the calibration")
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="how long to watch both streams for the pairing check")
    ap.add_argument("--quick", action="store_true",
                    help="skip the timed pairing check")
    a = ap.parse_args(argv)

    try:
        raw = httpx.get(f"{a.server}/config", timeout=8).json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"cannot read {a.server}/config — {exc}")
        return 2
    cfg = parse_config(raw)
    rig_cfg = raw.get("stereo_rig") or {}
    host = cfg.camserver.rstrip("/")
    if not host:
        print("the server names no camserver")
        return 2
    cams = {side: getattr(cfg, side).camera_id
            for side in ("left", "right") if getattr(cfg, side)}
    if len(cams) != 2:
        print("the server names fewer than two cameras")
        return 2
    print(f"camserver {host} · left {cams['left']} · right {cams['right']}\n")

    params = ScanParams()
    trouble = check_cameras(host, cams)
    if not a.quick:
        from .scanworker import PAIR_WINDOW_S
        trouble += check_pairing(host, cams, a.seconds, PAIR_WINDOW_S)
    trouble += check_calibration(rig_cfg)
    try:
        trouble += check_stripe(cfg, rig_cfg, host, cams, params)
    except httpx.HTTPError as exc:
        print(f"  X stripe check failed — {exc}")
        trouble.append("stripe check failed")

    print()
    if trouble:
        print("in the way of a scan: " + "; ".join(trouble))
        return 1
    print("nothing in the way of a scan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
