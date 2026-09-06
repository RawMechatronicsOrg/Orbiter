"""Hold each camera's exposure where the laser's stripe is bright but not
clipped, and put it back when the camera forgets.

Two facts about this rig decide the design. The cameras are cheap UVC
webcams behind camserver, which reopens a device on every USB hiccup, and
a reopen resets the V4L2 controls — so an exposure set once is not an
exposure kept; something has to notice and set it again. And the scan's
depth comes from the left eye's stripe centroid: with the core clipped at
255 the score profile is flat and the centroid wanders by a pixel, which is
more than a millimetre at the working distance. The board, meanwhile, wants
light. So the exposure is steered by the stripe alone — the brightest
thing in the frame by far — to a peak in the low 200s, from the red values
the detector already gathers under the stripe (`laser.exposure_of`), and
no lower than that: everything else in the frame keeps as much light as
the stripe allows.

`ExposureKeeper` runs its own thread: HTTP to camserver must never sit on
the GUI thread. The window feeds it each eye's stripe peak from the results
it paints and reads a snapshot back for the toolbar; the guide reads the
same snapshot to say "STRIPE SATURATED".
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

#: Where the stripe's red peak should sit (`laser.exposure_of`): high, for
#: the score's dynamic range, with the top of the profile still there.
PEAK_LOW = 200.0
PEAK_HIGH = 240.0
#: Above this share of stripe pixels at the ceiling the stripe counts as
#: clipped whatever the percentile says.
CLIPPED_MAX = 0.05
#: "Exposure Time, Absolute" is in 100 µs steps on these cameras: 10 is a
#: millisecond, 330 a whole frame at 30 fps — longer only lowers the rate.
EXPOSURE_MIN = 10
EXPOSURE_MAX = 330
#: A clipped stripe hides how far over it is, so the step down is the big
#: one; the step up is gentle, into a band that is easy to overshoot.
STEP_DOWN = 0.7
STEP_UP = 1.2
#: After a write the sensor applies the new time a frame or two later and
#: the peak takes a moment to be observed at it: no second step before this.
SETTLE_S = 1.5
#: How often the camera is asked what it actually holds.
VERIFY_S = 5.0
#: An observation older than this says nothing about the exposure now.
FRESH_S = 2.0
#: What a reopen forgets besides the time itself: manual exposure mode and
#: the mains flicker filter (50 Hz here).
HELD = {"auto_exposure": 1, "power_line_frequency": 1}
EXPOSURE = "exposure_time_absolute"


class CameraControls:
    """camserver's V4L2 knobs: `GET/POST /api/controls/{cam}`, and the
    per-camera status that counts device reopens."""

    def __init__(self, host: str, timeout_s: float = 3.0) -> None:
        self.host = host.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def read(self, cam: str) -> dict[str, int | None]:
        """Slug → value for every control the camera reports."""
        r = self._client.get(f"{self.host}/api/controls/{cam}")
        r.raise_for_status()
        data = r.json()
        items = data.get("controls", data) if isinstance(data, dict) else data
        return {str(c.get("slug") or c.get("name")): c.get("value") for c in items}

    def write(self, cam: str, values: dict[str, int]) -> dict[str, Any]:
        """Write controls; returns camserver's verdict — `applied` per slug
        (requested / sent / readback / honoured) and `rejected` per slug
        with the reason, `"*"` when the camera as a whole refused."""
        r = self._client.post(f"{self.host}/api/controls/{cam}",
                              json={"controls": {k: int(v) for k, v in values.items()}})
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code >= 400 and not isinstance(data, dict):
            r.raise_for_status()
        return data if isinstance(data, dict) else {}

    def reopens(self, cam: str) -> int | None:
        """How many times camserver has reopened the device, or None when
        the status does not say."""
        r = self._client.get(f"{self.host}/api/cameras/{cam}")
        r.raise_for_status()
        worker = r.json().get("worker") or {}
        n = worker.get("device_reopens")
        return None if n is None else int(n)


@dataclass
class EyeExposure:
    """One eye's exposure as the keeper sees it."""

    cam: str | None = None
    #: The exposure time held, in the camera's units; None until known.
    target: int | None = None
    #: Recent (when, peak, clipped) from the stripe, newest last.
    peaks: deque = field(default_factory=lambda: deque(maxlen=8))
    applied_at: float = -1e9
    verified_at: float = -1e9
    reopens: int | None = None
    #: The last read-back of the exposure time, for the toolbar.
    seen: int | None = None
    #: The camera refuses exposure writes outright (firmware); steering is
    #: pointless and the operator is told so.
    rejected: str | None = None
    #: The camera takes the write but never reads the time back (0 or
    #: nothing): a mismatch on read is then no reason to write again.
    blind: bool = False
    note: str = ""

    def recent(self, now: float) -> tuple[float, float] | None:
        """Median peak and clipped share over the fresh observations."""
        fresh = [(p, c) for t, p, c in self.peaks if now - t <= FRESH_S and p == p]
        if len(fresh) < 3:
            return None
        peaks = sorted(p for p, _ in fresh)
        clipped = sorted(c for _, c in fresh)
        return peaks[len(peaks) // 2], clipped[len(clipped) // 2]

    @property
    def saturated(self) -> bool:
        """Whether the newest fresh observations show a clipped stripe."""
        r = self.recent(time.monotonic())
        return r is not None and (r[0] >= PEAK_HIGH + 8 or r[1] > CLIPPED_MAX)


def steer(target: int, peak: float, clipped: float) -> int:
    """The next exposure time for a stripe at `peak` with `clipped` of its
    pixels at the ceiling, holding `target` now. Pure, for the tests."""
    if peak >= PEAK_HIGH + 8 or clipped > CLIPPED_MAX:
        return max(EXPOSURE_MIN, int(target * STEP_DOWN))
    if peak < PEAK_LOW:
        return min(EXPOSURE_MAX, int(target * STEP_UP) + 1)
    return target


class ExposureKeeper:
    """Keeps both eyes' exposures, on a thread of its own.

    `auto` steers the held time by the stripe; off, the held time is what
    the operator set. Either way the keeper re-applies it, with `HELD`,
    when the camera reports a reopen or reads back something else.
    """

    def __init__(self, path: Path | None = None,
                 controls: CameraControls | None = None) -> None:
        self.path = path
        self._controls = controls
        self._lock = threading.Lock()
        self.eyes: dict[str, EyeExposure] = {"left": EyeExposure(), "right": EyeExposure()}
        self.auto = True
        self.laser_on = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    # ── configuration (GUI thread) ────────────────────────────────────────

    def configure(self, host: str | None, cams: dict[str, str | None]) -> None:
        """The camserver to talk to and which camera each eye is."""
        with self._lock:
            if host and (self._controls is None or self._controls.host != host.rstrip("/")):
                if self._controls is not None:
                    self._controls.close()
                self._controls = CameraControls(host)
            for side, cam in cams.items():
                eye = self.eyes[side]
                if eye.cam != cam:
                    eye.cam, eye.reopens, eye.seen = cam, None, None
                    eye.rejected, eye.blind = None, False
                    eye.verified_at = -1e9

    def set_auto(self, on: bool) -> None:
        with self._lock:
            self.auto = bool(on)
        self._save()

    def set_target(self, side: str, value: int) -> None:
        """The operator's own exposure for an eye; applied at the next tick."""
        with self._lock:
            eye = self.eyes[side]
            eye.target = int(min(max(value, EXPOSURE_MIN), 2000))
            eye.applied_at = -1e9
            eye.verified_at = -1e9      # force a write on the next verify
        self._save()

    def set_laser(self, on: bool) -> None:
        with self._lock:
            self.laser_on = bool(on)

    def observe(self, side: str, peak: float, clipped: float) -> None:
        """One frame's stripe brightness for an eye (NaN without a stripe)."""
        with self._lock:
            self.eyes[side].peaks.append((time.monotonic(), peak, clipped))

    def snapshot(self) -> dict[str, EyeExposure]:
        """A copy of each eye's state for the toolbar and the guide."""
        with self._lock:
            return {side: EyeExposure(cam=e.cam, target=e.target, peaks=deque(e.peaks, maxlen=8),
                                      applied_at=e.applied_at, verified_at=e.verified_at,
                                      reopens=e.reopens, seen=e.seen, rejected=e.rejected,
                                      blind=e.blind, note=e.note)
                    for side, e in self.eyes.items()}

    def saturated(self) -> set[str]:
        """Eyes whose stripe is clipped right now."""
        with self._lock:
            return {side for side, e in self.eyes.items() if e.saturated}

    # ── the thread ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="exposure", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._controls is not None:
                self._controls.close()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self.tick(time.monotonic())
            except Exception:                                  # noqa: BLE001
                log.exception("exposure keeper tick")

    def tick(self, now: float) -> None:
        """One pass over both eyes: verify what the camera holds, steer by
        the stripe, write when something changed. Split out for the tests."""
        with self._lock:
            controls, auto, laser = self._controls, self.auto, self.laser_on
            eyes = {side: e for side, e in self.eyes.items() if e.cam}
        if controls is None:
            return
        for side, eye in eyes.items():
            write = False
            if now - eye.verified_at >= VERIFY_S:
                write = self._verify(controls, eye, now)
            if (auto and laser and eye.rejected is None and now - eye.applied_at >= SETTLE_S
                    and eye.target is not None):
                with self._lock:
                    recent = eye.recent(now)
                if recent is not None:
                    new = steer(eye.target, *recent)
                    if new != eye.target:
                        with self._lock:
                            eye.target = new
                            eye.note = (f"stripe peak {recent[0]:.0f}, "
                                        f"{100 * recent[1]:.0f}% clipped → exposure {new}")
                        write = True
                        self._save()
            if write:
                self._apply(controls, eye, now)

    def _verify(self, controls: CameraControls, eye: EyeExposure, now: float) -> bool:
        """Read the camera; True when what it holds must be written again."""
        try:
            held = controls.read(eye.cam)
            reopens = controls.reopens(eye.cam)
        except (httpx.HTTPError, ValueError) as exc:
            with self._lock:
                eye.note = f"camserver: {_reason(exc)}"
                eye.verified_at = now
            return False
        seen = held.get(EXPOSURE)
        with self._lock:
            eye.verified_at = now
            eye.seen = None if seen is None else int(seen)
            reopened = eye.reopens is not None and reopens is not None and reopens != eye.reopens
            eye.reopens = reopens
            if eye.target is None:
                # First sight of this camera: adopt what it holds if that is
                # a real time, else the control's usual default.
                eye.target = int(seen) if seen and int(seen) >= EXPOSURE_MIN else 100
            stale = (any(held.get(k) != v for k, v in HELD.items())
                     or (eye.seen != eye.target and not eye.blind))
            if reopened:
                eye.note = f"camera reopened ({reopens}): exposure set again"
            elif stale:
                eye.note = f"camera holds {eye.seen}, wanted {eye.target}: set again"
            return reopened or stale

    def _apply(self, controls: CameraControls, eye: EyeExposure, now: float) -> None:
        with self._lock:
            values = {**HELD, EXPOSURE: eye.target}
            cam = eye.cam
        try:
            verdict = controls.write(cam, values)
        except (httpx.HTTPError, ValueError) as exc:
            with self._lock:
                eye.note = f"write failed: {_reason(exc)}"
                eye.applied_at = now
            return
        rejected = verdict.get("rejected") or {}
        applied = verdict.get("applied") or {}
        with self._lock:
            eye.applied_at = now
            if "*" in rejected:
                eye.note = f"camera: {rejected['*']}"
                return
            if EXPOSURE in rejected:
                eye.rejected = str(rejected[EXPOSURE])
                eye.note = f"camera refuses exposure writes: {eye.rejected}"
                return
            eye.rejected = None
            about = applied.get(EXPOSURE)
            back = about.get("readback") if isinstance(about, dict) else None
            if isinstance(about, dict) and about.get("honoured") is False:
                eye.rejected = "not honoured"
                eye.note = "camera took the exposure write and did nothing with it"
                return
            # A camera that never reads the time back is trusted on its word.
            eye.blind = isinstance(about, dict) and (back is None or int(back) < EXPOSURE_MIN)
            eye.seen = eye.target if eye.blind or back is None else int(back)

    # ── persistence ───────────────────────────────────────────────────────


    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.auto = bool(data.get("auto", True))
        for side, eye in self.eyes.items():
            t = data.get(side)
            if isinstance(t, int) and t >= EXPOSURE_MIN:
                eye.target = t

    def _save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {"auto": self.auto, **{s: e.target for s, e in self.eyes.items()}}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            log.warning("could not save %s", self.path)


def _reason(exc: Exception) -> str:
    """camserver's own words when it has any — `{"rejected": {"*": "camera
    is not running"}}` says more than `HTTPStatusError`."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            rejected = data.get("rejected") or {}
            if rejected:
                return str(next(iter(rejected.values())))
            if data.get("detail"):
                return str(data["detail"])
        except ValueError:
            pass
        return f"HTTP {resp.status_code}"
    return exc.__class__.__name__
