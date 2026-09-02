"""Scanning off the GUI thread: pair the eyes' results, triangulate, accumulate.

Why a thread of its own. The first version ran `scan_frame` in the GUI thread,
in the slot that received each eye's result. Measured on this rig at 1080p,
that slot was doing 32 ms of triangulation per pair, 8.6 ms of frame
conversion per eye at 50 results a second, and a min/max over the whole cloud
per pair (42 ms at a million points) — more than a second of work per second.
Qt's queued signals do not drop, so the backlog grew without bound and the
window fell further behind the cameras the longer it ran. Detection already
runs off the GUI thread; this puts scanning there too, and leaves the GUI
thread nothing but painting the newest frame and reading a few counters.

Pairing. Each eye's detector skips frames on its own, so "the newest left and
the newest right" are rarely from the same instant. Both eyes' recent results
are kept and paired by camserver's capture clock — the one clock that timed
both sensors — oldest first, each result used once. On this rig that finds a
partner for about twice as many frames as newest-against-newest did.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import RigConfig
from .laser import LaserPoints
from .laserplane import LaserPlane, from_config as plane_from_config
from .scan import CloudOverlay, PointCloud, ScanFrame, ScanParams, scan_frame
from .stereo import StereoRig, result_from_config
from .worker import EyeResult, Latest

log = logging.getLogger("orbiter_native.scanworker")

#: Two frames count as simultaneous within this much of camserver's capture
#: clock. Wider than calibration's 4 ms: the subject turns slowly by hand, and
#: 10 ms found partners for 26 of 58 left frames on this rig against 11 at 4 ms.
PAIR_WINDOW_S = 0.010

#: Results kept per eye while waiting for a partner. At 30 fps this is half a
#: second — far more than the two streams ever drift apart.
_HISTORY = 16

#: Cloud points handed to the eyes for drawing, at most. Projecting them costs
#: about 1 ms per 30k, per eye, per frame.
OVERLAY_MAX = 40000


@dataclass
class ScanInput:
    """What scanning needs from one eye's result — without its 6 MB frame."""

    capture_mono: float
    stripe: LaserPoints | None
    board_R: np.ndarray | None
    board_t: np.ndarray | None
    wh: tuple[int, int]


@dataclass
class ScanStatus:
    """For the panel: the cloud so far, and what the last pair did."""

    n_points: int
    bounds: tuple[np.ndarray, np.ndarray] | None
    pairs: int
    frame: ScanFrame | None = None
    #: A blocking condition — no calibration, no board, laser off.
    note: str | None = None


class ScanWorker:
    """Owns the scan thread, the cloud, and the overlay snapshot."""

    def __init__(self) -> None:
        # Guards the history, the configuration and the cloud.
        self._lock = threading.Lock()
        self._hist: dict[str, deque[ScanInput]] = {
            "left": deque(maxlen=_HISTORY), "right": deque(maxlen=_HISTORY)}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._cfg: RigConfig | None = None
        self._params = ScanParams()
        self._pairs = 0
        # Projection geometry, rebuilt only when the calibration or the frame
        # size changes. Touched by the scan thread alone.
        self._geom_key = None
        self._geom: tuple[StereoRig | None, LaserPlane | None] = (None, None)

        self.cloud = PointCloud()
        self.overlay = CloudOverlay()
        #: Newest counters; the GUI takes them on its own clock.
        self.status = Latest()

    # ── configuration (GUI thread) ────────────────────────────────────────

    def set_config(self, cfg: RigConfig) -> None:
        with self._lock:
            self._cfg = cfg

    def set_params(self, params: ScanParams) -> None:
        with self._lock:
            self._params = params

    def set_active(self, on: bool) -> None:
        with self._lock:
            self._active = on
            if not on:
                for q in self._hist.values():
                    q.clear()

    def clear(self) -> None:
        with self._lock:
            self.cloud.clear()
            self._pairs = 0
            snap = self.cloud.decimated(OVERLAY_MAX)
        self.overlay.publish(snap)
        self._publish(None, None)

    def export(self, path: str) -> int:
        with self._lock:
            return self.cloud.write_ply(path)

    # ── input (detector threads) ──────────────────────────────────────────

    def offer(self, res: EyeResult) -> None:
        """Take one eye's result. Cheap: it runs on the detector thread."""
        if res.capture_mono is None:
            return
        board = res.board
        item = ScanInput(res.capture_mono, res.stripe,
                         None if board is None else board.R,
                         None if board is None else board.t, res.wh)
        with self._lock:
            if not self._active:
                return
            self._hist[res.side].append(item)
        self._wake.set()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="scan", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── scan thread ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.25):
                continue
            self._wake.clear()
            while not self._stop.is_set():
                pair = self._take_pair()
                if pair is None:
                    break
                try:
                    self._process(*pair)
                except Exception:                              # noqa: BLE001
                    # One odd frame must not end scanning for the session.
                    log.exception("scan frame raised; continuing")

    def _take_pair(self):
        """The oldest left result that has a partner, with that partner.

        Both are removed, along with everything older on either side — those
        could only have paired with results already gone. A left without a
        partner is kept waiting while one could still arrive, that is while the
        newest right is not yet later than the left plus the window; once it
        is, that left has been passed over and is skipped.
        """
        with self._lock:
            left, right = self._hist["left"], self._hist["right"]
            for i, a in enumerate(left):
                best = None
                for j, b in enumerate(right):
                    gap = abs(a.capture_mono - b.capture_mono)
                    if gap <= PAIR_WINDOW_S and (best is None or gap < best[0]):
                        best = (gap, j)
                if best is not None:
                    b = right[best[1]]
                    for _ in range(i + 1):
                        left.popleft()
                    for _ in range(best[1] + 1):
                        right.popleft()
                    return a, b, self._cfg, self._params
                if not right or right[-1].capture_mono <= a.capture_mono + PAIR_WINDOW_S:
                    return None          # its partner may still be on the way
            return None

    def _process(self, a: ScanInput, b: ScanInput, cfg: RigConfig | None,
                 params: ScanParams) -> None:
        rig, plane = self._geometry(cfg, a.wh)
        if rig is None:
            self._publish(None, "no pair calibration for this frame size — "
                                "solve intrinsics and the stereo pair first")
            return
        if a.stripe is None or b.stripe is None:
            self._publish(None, "laser detector is off")
            return
        if a.board_R is None or a.board_t is None:
            self._publish(None, "board not visible — it is what defines the scan volume")
            return
        frame = scan_frame(rig, a.stripe, b.stripe, a.board_R, a.board_t, params, plane)
        snap = None
        with self._lock:
            self._pairs += 1
            if frame.n_kept:
                self.cloud.add(frame.points_board)
                snap = self.cloud.decimated(OVERLAY_MAX)
        if snap is not None:
            self.overlay.publish(snap)
        self._publish(frame, None)

    def _geometry(self, cfg: RigConfig | None, wh: tuple[int, int]):
        """Projection geometry for this calibration and frame size, cached.

        Keyed by the calibration's content rather than the config object: the
        poll hands over a fresh object every two seconds with the same numbers
        in it, and an object id can be reused after the old one is freed.
        """
        if cfg is None:
            return None, None
        key = (wh,
               cfg.left.intrinsics_raw if cfg.left else None,
               cfg.right.intrinsics_raw if cfg.right else None,
               cfg.extrinsics_raw, cfg.laser_plane_raw)
        if key != self._geom_key:
            rig = None
            if cfg.left is not None and cfg.right is not None:
                kl = cfg.left.intrinsics_for(wh)
                kr = cfg.right.intrinsics_for(wh)
                geom = result_from_config(cfg.extrinsics_raw, wh)
                if kl is not None and kr is not None and geom is not None:
                    rig = StereoRig(kl, kr, geom)
            self._geom_key = key
            self._geom = (rig, plane_from_config(cfg.laser_plane_raw, wh))
        return self._geom

    def _publish(self, frame: ScanFrame | None, note: str | None) -> None:
        with self._lock:
            st = ScanStatus(len(self.cloud), self.cloud.bounds(), self._pairs,
                            frame, note)
        self.status.put(st)
