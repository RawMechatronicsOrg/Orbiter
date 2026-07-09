"""Motion-plate foreground for a turntable shoot — temporal background subtraction.

The rig's defining property: **AZ rotates the platform, the camera is
azimuthally static.** So the frames of one elevation ring (the az-sweep at a
fixed EL) are shot by a *physically stationary* camera — the background is
pixel-identical across the ring and only the assembly (object + board + disc)
moves. That makes "what changed between frames" a direct, physical answer to
"what is the object", with none of the guessing the SAM2 probe-grid does:

    ring frames ─▶ per-pixel temporal median  ─▶ plate  (the empty scene,
                                                          reconstructed with no
                                                          empty-table shot)
    frame, plate ─▶ robust colour residual ─▶ threshold ─▶ motion mask

Why per-pixel median is the empty scene: at any background pixel the object
projects there for only a short arc of azimuths, so across N≈24–37 frames the
pixel shows background in the large majority of them — the median lands on
background. Only a narrow on-axis core (where a tall object always sits) is
contaminated, and that core is foreground anyway (hole-filled downstream).

This is pure NumPy/OpenCV — no torch, no GPU, no dependence on pose accuracy.
The poses are only needed later, for the deterministic disc/board stamp and
the working-volume clamp (``geom_prompt.py``). Run standalone to eyeball the
plate + overlays on a real scan::

    python motion_plate.py --images <scan>/photos --out /tmp/motion_check
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

# Elevation tag in a frame name, e.g. ``003_az030_el+12.jpg`` -> ring "+12".
# AZ varies within a ring (camera static); EL changes between rings (camera
# moves), so EL is the grouping key.
_RING_RE = re.compile(r"_el([+-]?\d+)", re.IGNORECASE)

_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Defaults (every one is also a CLI flag / tool tunable).
#
# Plate/motion are computed at this long-side resolution then the mask is
# upsampled to full res — a 12 MP stack of 30+ frames is needless GB and the
# masks don't need the pixels (the SAM2 path already works at 1024).
_LONG_SIDE = 1280
# Foreground threshold = max(floor, k · noise_sigma). noise_sigma is a robust
# per-ring estimate of the background residual (sensor + JPEG); k sets how many
# sigmas above the noise floor counts as "moved". The absolute floor (0–255
# colour-distance units) keeps a near-noiseless ring from flagging speckle.
_RESIDUAL_K = 6.0
_RESIDUAL_FLOOR = 14.0
# Morphology: clean speckle (open) then close small gaps (close), radius in px
# at the working resolution.
_OPEN_PX = 2
_CLOSE_PX = 4


def parse_ring_key(name: str) -> str | None:
    """Elevation ring key from a frame name (``"+12"``), or None if untagged."""
    m = _RING_RE.search(Path(name).name)
    return m.group(1) if m else None


def discover_images(root: Path) -> list[tuple[Path, Path]]:
    """Sorted ``[(relpath, abspath)]`` of images under ``root`` (recursive)."""
    out: list[tuple[Path, Path]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXT:
            out.append((p.relative_to(root), p))
    return out


def group_rings(
    frames: list[tuple[Path, Path]],
) -> "OrderedDict[str, list[int]]":
    """Map ring key -> indices into ``frames`` (insertion order preserved).

    Untagged frames (no ``_elNN`` in the name) collapse into a single
    ``"_all"`` ring — the median is still meaningful if the whole shoot is one
    static-camera sweep, just coarser. A caller with priors can instead group
    by clustered camera elevation; the name tag is the cheap default that
    works on every Orbiter scan today.
    """
    rings: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, (rel, _ab) in enumerate(frames):
        key = parse_ring_key(rel.as_posix()) or "_all"
        rings.setdefault(key, []).append(i)
    return rings


def load_small(
    path: Path, long_side: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Read a frame and downscale so its long side is ``long_side``.

    Returns ``(small_bgr, scale, (H, W))`` where ``scale`` maps full-res ->
    small and ``(H, W)`` is the ORIGINAL size (to upsample the mask back).
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"unreadable image: {path}")
    h, w = bgr.shape[:2]
    s = min(1.0, long_side / float(max(h, w)))
    if s < 1.0:
        small = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = bgr
    return small, s, (h, w)


def build_plate(smalls: list[np.ndarray]) -> np.ndarray:
    """Per-pixel temporal median of a ring's (downscaled) frames -> the empty
    scene. ``smalls`` must share a shape.

    Median (not mean) is what rejects the moving assembly: a mean would smear
    a ghost of the object across the plate, a median lands cleanly on the
    background value seen in most frames.
    """
    stack = np.stack(smalls, axis=0)            # (N, H, W, 3) uint8
    return np.median(stack, axis=0).astype(np.uint8)


def build_plate_robust(
    smalls: list[np.ndarray],
    motion_masks: list[np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    """Rebuild the plate ignoring per-frame motion pixels — kills the on-axis
    "ghost" a plain median bakes in.

    A tall, near-axis object projects onto the central column in a LARGE
    fraction of azimuths, so the plain median there lands on the object, not
    the background. Given a first-pass motion mask per frame, re-take the
    per-pixel median over ONLY the frames where that pixel was background. A
    pixel that is motion in *every* frame (the object's permanent on-axis core)
    has no background sample — it falls back to the plain-median plate, which
    is fine: that core is foreground and gets hole-filled downstream anyway.
    """
    import warnings

    stack = np.stack(smalls, axis=0).astype(np.float32)     # (N, H, W, 3)
    mstack = np.stack([m.astype(bool) for m in motion_masks], axis=0)  # (N,H,W)
    stack[mstack] = np.nan                                   # ignore moving px
    # A pixel masked in EVERY frame (the object's permanent on-axis core) has
    # no background sample -> nanmedian returns NaN with an "All-NaN slice"
    # RuntimeWarning. Expected; we patch those pixels from the fallback below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plate = np.nanmedian(stack, axis=0)                 # (H, W, 3)
    holes = np.isnan(plate)
    if holes.any():
        plate[holes] = fallback.astype(np.float32)[holes]
    return np.clip(plate, 0, 255).astype(np.uint8)


def colour_residual(small: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean colour distance ``‖frame − plate‖`` (float32 HxW).

    Colour distance (not grey absdiff) so a coloured object on a neutral
    table separates cleanly even at equal luma.
    """
    d = small.astype(np.float32) - plate.astype(np.float32)
    return np.sqrt(np.einsum("hwc,hwc->hw", d, d))


def estimate_noise(residuals: list[np.ndarray]) -> float:
    """Robust background-noise sigma for a ring (colour-distance units).

    The background dominates every frame, so the MEDIAN residual over a frame
    sits in the background-noise regime; the ring median of those is a stable
    sigma that ignores the moving object (a few % of pixels) entirely.
    1.4826 rescales a median-abs deviation to a Gaussian sigma.
    """
    per_frame = [float(np.median(r)) for r in residuals]
    return 1.4826 * float(np.median(per_frame)) if per_frame else 0.0


def motion_mask_small(
    residual: np.ndarray,
    sigma: float,
    *,
    k: float = _RESIDUAL_K,
    floor: float = _RESIDUAL_FLOOR,
    open_px: int = _OPEN_PX,
    close_px: int = _CLOSE_PX,
) -> np.ndarray:
    """Threshold a colour residual into a cleaned boolean motion mask.

    Threshold = ``max(floor, k·sigma)``: ``k·sigma`` tracks the actual noise
    of this ring, ``floor`` guards a near-noiseless ring from flagging JPEG
    speckle. Open removes salt, close bridges thin texture-poor gaps in the
    object body.
    """
    thr = max(floor, k * sigma)
    m = (residual >= thr).astype(np.uint8)
    if open_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    if close_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return m > 0


# ── standalone validation ─────────────────────────────────────────────────────

def _overlay(small_bgr: np.ndarray, mask_small: np.ndarray) -> np.ndarray:
    """Foreground in colour, suppressed region dimmed + red, green outline."""
    out = small_bgr.copy()
    bg = ~mask_small
    out[bg] = (0.35 * out[bg] + np.array([0.0, 0.0, 90.0])).astype(np.uint8)
    cont, _ = cv2.findContours(mask_small.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cont, -1, (0, 255, 0), 2)
    return out


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate the motion plate on a scan.")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--long-side", type=int, default=_LONG_SIDE)
    p.add_argument("--k", type=float, default=_RESIDUAL_K)
    p.add_argument("--floor", type=float, default=_RESIDUAL_FLOOR)
    p.add_argument("--samples-per-ring", type=int, default=3,
                   help="How many per-ring frame overlays to dump.")
    p.add_argument("--max-rings", type=int, default=0,
                   help="Process only the first N rings (0 = all).")
    args = p.parse_args(argv)

    frames = discover_images(args.images.resolve())
    if not frames:
        print(f"error: no images under {args.images}", file=sys.stderr)
        return 1
    rings = group_rings(frames)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(frames)} frames in {len(rings)} ring(s): "
          + ", ".join(f"{k}({len(v)})" for k, v in rings.items()))

    ring_items = list(rings.items())
    if args.max_rings > 0:
        ring_items = ring_items[:args.max_rings]

    for key, idxs in ring_items:
        smalls, scale = [], 1.0
        for i in idxs:
            small, scale, _hw = load_small(frames[i][1], args.long_side)
            smalls.append(small)
        plate = build_plate(smalls)
        resid = [colour_residual(s, plate) for s in smalls]
        sigma = estimate_noise(resid)
        thr = max(args.floor, args.k * sigma)

        safe = key.replace("+", "p").replace("-", "m")
        cv2.imwrite(str(args.out / f"plate_{safe}.jpg"), plate,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        covs = []
        step = max(1, len(idxs) // max(1, args.samples_per_ring))
        for j, i in enumerate(idxs):
            mask = motion_mask_small(resid[j], sigma, k=args.k, floor=args.floor)
            covs.append(float(mask.mean()))
            if j % step == 0:
                name = frames[i][0].stem
                cv2.imwrite(str(args.out / f"ov_{safe}_{name}.jpg"),
                            _overlay(smalls[j], mask),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
        cov = np.asarray(covs)
        print(f"  ring {key}: {len(idxs)} frames  sigma={sigma:.2f}  "
              f"thr={thr:.1f}  coverage median={np.median(cov) * 100:.1f}% "
              f"[{cov.min() * 100:.1f}–{cov.max() * 100:.1f}%]")

    print(f"\nwrote plates + overlays to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
