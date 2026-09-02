"""Per-eye capture and detection threads.

Two threads per camera, not one, and the split is the whole point.

Reading and detecting in a single loop means the socket is only drained while
detection is idle. Detection costs ~13 ms per 1280x720 frame on this machine
against a ~33 ms frame interval, so it fits — until it does not: a busier
scene, a second camera contending, a bigger resolution, and the reader stops
keeping up. The TCP buffer then fills with frames nobody has read, and the
"live" view silently becomes a recording running further and further behind.
That failure is invisible: the frame rate looks fine and every frame is late.

So the reader thread does nothing but drain the socket and decode, always at
line rate, keeping only the newest frame. The detector thread takes whatever
is newest whenever it is free. Frames that arrive while detection is busy are
skipped, never queued — the overlay may update at 20 Hz while the view runs at
30, and that is the honest tradeoff, with latency bounded either way.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import httpx
import numpy as np
from PySide6.QtCore import QObject, Signal

from .config import Eye, RigConfig
from .cvcore import BoardSpec
from .detect import BoardDetector, BoardHit, draw_board
from .intrinsics import ViewDescriptor, describe
from .laser import (
    LaserLine,
    LaserParams,
    LaserPoints,
    board_mask,
    find_laser_line,
    find_laser_points,
)
from .laser import draw as draw_laser
from .orient import Orientation, apply as orient_apply, map_points
from .scan import CloudOverlay
from .source import Frame, MjpegReader

log = logging.getLogger("orbiter_native.worker")

#: Reconnect backoff after the stream drops. Small enough to recover promptly
#: when camserver restarts, capped so an absent host is not hammered.
_RETRY_INITIAL_S = 0.5
_RETRY_MAX_S = 5.0

#: Rolling window for the displayed rates, in samples.
_WINDOW = 60


@dataclass
class EyeStats:
    """Rolling numbers for one eye's overlay."""

    recv_fps: float = 0.0
    detect_fps: float = 0.0
    detect_ms: float = 0.0
    laser_ms: float = 0.0
    corners: int = 0
    coverage: float = 0.0
    #: Laser fit, when the frame produced a usable one.
    laser_inliers: int = 0
    laser_points: int = 0
    laser_rms_px: float = float("nan")
    laser_angle_deg: float = float("nan")
    laser_reason: str | None = None
    #: Stripe points found in scan mode.
    stripe_points: int = 0
    #: Server-side age of the last frame at send time, from `X-Age-Ms`.
    server_age_ms: float | None = None
    frames: int = 0
    error: str | None = None


class Latest:
    """A one-slot mailbox: writers overwrite, readers take.

    Deliberately not a Queue — a queue would buffer, and a buffered frame is a
    stale frame nobody wants by the time it is read.
    """

    def __init__(self) -> None:
        self._item = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def put(self, item) -> None:
        with self._lock:
            self._item = item
        self._ready.set()

    def take(self, timeout: float = 0.5):
        """Newest item, or None if none arrived within `timeout`."""
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            item, self._item = self._item, None
            self._ready.clear()
        return item

    def clear(self) -> None:
        with self._lock:
            self._item = None
        self._ready.clear()


@dataclass
class EyeResult:
    """One detection pass, ready for the GUI thread."""

    side: str
    #: Oriented BGR frame with overlays already drawn.
    bgr: np.ndarray
    stats: EyeStats
    board: BoardHit | None = None
    laser: LaserLine | None = None
    #: Shape-free stripe points — what scanning consumes. Empty unless the
    #: worker is in scan mode.
    stripe: LaserPoints | None = None
    #: Frame size, needed to interpret corner coordinates and to check that a
    #: stored calibration was solved at this resolution.
    wh: tuple[int, int] = (0, 0)
    #: Where/how the board sits, for the calibration capture's diversity test.
    descriptor: ViewDescriptor | None = None
    #: camserver's capture instant. Both cameras are timed by the SAME clock on
    #: that machine, so this is what pairs a left frame with its right one —
    #: our own arrival times are not comparable between two sockets.
    capture_mono: float | None = None


class EyeWorker(QObject):
    """Owns the reader and detector threads for one eye."""

    #: Rare, so a queued Qt signal suits it. Per-frame results do NOT go
    #: through a signal: Qt's queue never drops, so a GUI thread that falls
    #: behind backlogs frames without bound. They go to sinks — `add_sink` —
    #: which is where the GUI's one-slot mailbox and the scan worker's pairing
    #: history plug in.
    status = Signal(str, object)   # side, error-or-None

    def __init__(self, side: str) -> None:
        super().__init__()
        self.side = side
        self._latest = Latest()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._reader: MjpegReader | None = None

        # Guarded by `_cfg_lock`: swapped wholesale on a config change rather
        # than mutated field by field, so a detector pass never sees half of
        # one configuration and half of another.
        self._cfg_lock = threading.Lock()
        self._url: str | None = None
        self._orientation = Orientation()
        self._board_spec: BoardSpec | None = None
        self._eye = None
        self._laser_on = False
        self._scan_mode = False
        self._laser_params = LaserParams()
        self._overlay: CloudOverlay | None = None
        self._sinks: list[Callable[[EyeResult], None]] = []

        self._recv_times: deque[float] = deque(maxlen=_WINDOW)
        self._det_times: deque[float] = deque(maxlen=_WINDOW)
        self._stats = EyeStats()

    # ── configuration ─────────────────────────────────────────────────────

    def apply_config(self, cfg: RigConfig, eye: Eye | None) -> bool:
        """Adopt a new config. Returns True if the stream must be restarted.

        Only a changed URL forces a restart; orientation, board and laser
        settings are picked up by the next frame without dropping the stream.
        """
        url = cfg.stream_url(eye)
        with self._cfg_lock:
            restart = url != self._url
            self._url = url
            self._orientation = eye.orientation if eye else Orientation()
            self._board_spec = cfg.board
            self._eye = eye
        return restart

    def set_laser(self, enabled: bool, params: LaserParams | None = None) -> None:
        with self._cfg_lock:
            self._laser_on = enabled
            if params is not None:
                self._laser_params = params

    def set_scan_mode(self, on: bool) -> None:
        """Switch the stripe detector between its two genuinely different jobs.

        Calibration wants a straight line fitted to the stripe ON the board:
        the model holds because the board is flat, and the fit denoises it and
        proves it. Scanning wants the stripe's actual shape over the WHOLE
        frame: on a subject it is a broken curve, that shape is the
        measurement, and it mostly falls outside the board's outline anyway.
        Measured on a real frame from this rig, the line fit kept 126 of 649
        stripe points — the 523 it dropped were the object.
        """
        with self._cfg_lock:
            self._scan_mode = on

    def set_overlay(self, overlay: CloudOverlay | None) -> None:
        """The scanned cloud to draw over this eye's frames while scanning."""
        with self._cfg_lock:
            self._overlay = overlay

    def add_sink(self, sink: Callable[[EyeResult], None]) -> None:
        """Called from the detector thread with every result. Must be quick and
        must not touch widgets — the GUI reads its mailbox on its own clock."""
        self._sinks.append(sink)

    def _snapshot(self):
        with self._cfg_lock:
            return (self._url, self._orientation, self._board_spec,
                    self._eye, self._laser_on, self._laser_params,
                    self._scan_mode, self._overlay)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for target, name in ((self._read_loop, "read"), (self._detect_loop, "detect")):
            t = threading.Thread(target=target, name=f"{self.side}-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        if self._reader:
            self._reader.stop()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def restart_stream(self) -> None:
        """Drop the current connection; the reader loop reconnects to the new URL."""
        if self._reader:
            self._reader.stop()
        self._latest.clear()

    # ── reader thread ─────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        backoff = _RETRY_INITIAL_S
        while not self._stop.is_set():
            url = self._snapshot()[0]
            if not url:
                self._set_error("no camera assigned to this eye")
                time.sleep(0.5)
                continue
            self._reader = MjpegReader(url)
            try:
                for frame in self._reader.frames():
                    if self._stop.is_set():
                        return
                    if self._snapshot()[0] != url:
                        break            # config changed under us; reconnect
                    self._recv_times.append(frame.recv_mono)
                    self._latest.put(frame)
                    self._set_error(None)
                    backoff = _RETRY_INITIAL_S
            except httpx.HTTPError as exc:
                self._set_error(f"{exc.__class__.__name__}: {exc}")
            except Exception as exc:                     # noqa: BLE001
                log.exception("%s reader crashed", self.side)
                self._set_error(f"{exc.__class__.__name__}: {exc}")
            if self._stop.is_set():
                return
            time.sleep(backoff)
            backoff = min(backoff * 2, _RETRY_MAX_S)

    def _set_error(self, err: str | None) -> None:
        if err != self._stats.error:
            self._stats.error = err
            self.status.emit(self.side, err)

    # ── detector thread ───────────────────────────────────────────────────

    def _detect_loop(self) -> None:
        detector = BoardDetector()
        while not self._stop.is_set():
            frame: Frame | None = self._latest.take(timeout=0.25)
            if frame is None:
                continue
            try:
                self._detect_one(detector, frame)
            except Exception as exc:                     # noqa: BLE001
                # One bad frame, or one unforeseen shape, must not silently
                # end this eye for the rest of the session — the window would
                # keep showing the last image with no hint that detection had
                # stopped.
                log.exception("%s detector raised; continuing", self.side)
                self._set_error(f"detector: {exc.__class__.__name__}: {exc}")

    def _detect_one(self, detector: BoardDetector, frame: Frame) -> None:
        (_, orientation, spec, eye, laser_on, laser_params,
         scan_mode, overlay) = self._snapshot()
        detector.set_spec(spec)

        h, w = frame.gray.shape
        # Resolved against THIS frame's size: intrinsics solved at another
        # resolution are refused rather than silently misapplied.
        intrinsics = eye.intrinsics_for((w, h)) if eye else None
        board = detector.detect(frame.gray, intrinsics) if detector.ready else BoardHit()
        descriptor = (describe(board.corners, board.ids, detector.board, (w, h))
                      if board.corners is not None else None)

        # The stripe detector has two genuinely different jobs.
        #
        # CALIBRATION: the stripe falls on the flat board, so it IS a straight
        # line; the fit denoises it and proves it, and the search is confined
        # to the board because points off it are not on the plane being solved.
        #
        # SCANNING: the stripe falls on the subject, where it is a broken curve
        # whose shape is the measurement, and most of it lands outside the
        # board's outline. Measured on a real frame from this rig, the line fit
        # kept 126 of 649 points — the 523 it dropped were the object. So no
        # fit and no mask; the 3D volume does the rejecting, in millimetres.
        laser = LaserLine()
        points = LaserPoints()
        hull = None
        if laser_on and scan_mode:
            points = find_laser_points(frame.bgr, None, laser_params)
        elif laser_on:
            mask = board_mask(board.corners, frame.gray.shape)
            if mask is None:
                laser = LaserLine(reason="board not detected")
            else:
                laser = find_laser_line(frame.bgr, mask, laser_params)
                hull = cv2.convexHull(board.corners.reshape(-1, 2).astype(np.int32))

        # Detect on ORIGINAL pixels, then orient for display, then map the
        # detections into the oriented frame. Detecting on the oriented
        # image instead would report coordinates in a frame that depends on
        # a UI setting — useless to any calibration consumer.
        bgr = np.ascontiguousarray(orient_apply(frame.bgr, orientation))
        draw_board(bgr, _orient_board(board, orientation, w, h))
        if laser_on and scan_mode:
            # Single pixels, not a polyline: the stripe breaks wherever the
            # subject does, and a segment drawn across a break would show a
            # surface that was never measured.
            _draw_points(bgr, points.points if orientation.is_identity
                         else _map_all(points.points, orientation, w, h))
            if overlay is not None and board.R is not None and intrinsics is not None:
                _draw_cloud(bgr, overlay.points(), board.R, board.t, intrinsics,
                            orientation, w, h)
        elif laser_on:
            draw_laser(bgr, _orient_laser(laser, orientation, w, h),
                       _orient_hull(hull, orientation, w, h))

        self._det_times.append(time.monotonic())
        self._publish(frame, board, laser, bgr, descriptor, points)

    def _publish(self, frame, board, laser, bgr, descriptor, points) -> None:
        s = self._stats
        s.frames += 1
        s.recv_fps = _rate(self._recv_times)
        s.detect_fps = _rate(self._det_times)
        s.detect_ms = board.ms
        s.laser_ms = laser.ms
        s.corners = board.count
        s.coverage = board.coverage(*frame.gray.shape[::-1])
        s.laser_points = int(len(laser.points))
        s.laser_inliers = laser.n_inliers
        s.laser_rms_px = laser.rms_px
        s.laser_angle_deg = laser.angle_deg
        s.laser_reason = points.reason if points.count or points.reason != "no data"             else laser.reason
        s.stripe_points = points.count
        s.server_age_ms = frame.server_age_ms
        h, w = frame.gray.shape
        res = EyeResult(
            self.side, bgr, _copy_stats(s), board, laser, stripe=points,
            wh=(w, h), descriptor=descriptor, capture_mono=frame.capture_mono,
        )
        for sink in self._sinks:
            sink(res)


# ── helpers ───────────────────────────────────────────────────────────────


def _draw_points(bgr: np.ndarray, pts: np.ndarray) -> None:
    """Draw the raw stripe centroids — one pixel each, no line joining them.

    Not a polyline: the stripe breaks wherever the subject does, and drawing a
    segment across a break would show a surface that was never measured.
    """
    if not len(pts):
        return
    p = np.rint(pts).astype(np.int32)
    ok = ((p[:, 0] >= 0) & (p[:, 0] < bgr.shape[1])
          & (p[:, 1] >= 0) & (p[:, 1] < bgr.shape[0]))
    bgr[p[ok, 1], p[ok, 0]] = (80, 235, 80)


def _draw_cloud(bgr: np.ndarray, pts_board: np.ndarray, R: np.ndarray,
                t: np.ndarray, k, o: Orientation, w: int, h: int) -> None:
    """Project the scanned cloud into this eye and draw it, 2×2 px per point.

    Board-frame points through THIS eye's own board pose: each eye places the
    cloud from what it sees, so the two overlays disagreeing is itself a sign
    that the board poses do. Two pixels because the 1080p frame is shown at
    about a third of its size and a single pixel would be lost to the scaling.
    """
    if not len(pts_board):
        return
    cam = pts_board @ np.asarray(R, float).T + np.asarray(t, float).reshape(1, 3)
    ahead = cam[:, 2] > 1.0
    if not ahead.any():
        return
    px, _ = cv2.projectPoints(cam[ahead], np.zeros(3), np.zeros(3), k.K, k.D)
    px = px.reshape(-1, 2)
    if not o.is_identity:
        px = map_points(px, w, h, o)
    p = np.rint(px).astype(np.int32)
    hh, ww = bgr.shape[:2]
    ok = ((p[:, 0] >= 0) & (p[:, 0] < ww - 1)
          & (p[:, 1] >= 0) & (p[:, 1] < hh - 1))
    p = p[ok]
    for dx in (0, 1):
        for dy in (0, 1):
            bgr[p[:, 1] + dy, p[:, 0] + dx] = (0, 150, 255)


def _rate(times: "deque[float]") -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0


def _copy_stats(s: EyeStats) -> EyeStats:
    """Stats cross a thread boundary; hand the GUI a snapshot, not the live object."""
    return EyeStats(**vars(s))


def _map_all(pts: np.ndarray, o: Orientation, w: int, h: int) -> np.ndarray:
    """Map an (N, 2) array of (x, y) into oriented coordinates."""
    return map_points(pts, w, h, o).astype(np.float32)


def _orient_board(hit: BoardHit, o: Orientation, w: int, h: int) -> BoardHit:
    """Copy of `hit` with corners mapped into oriented coordinates, for drawing."""
    if hit.corners is None or o.is_identity:
        return hit
    mapped = _map_all(hit.corners.reshape(-1, 2), o, w, h).reshape(-1, 1, 2)
    return BoardHit(corners=mapped, ids=hit.ids, R=hit.R, t=hit.t, ms=hit.ms)


def _orient_laser(line: LaserLine, o: Orientation, w: int, h: int) -> LaserLine:
    """Copy of `line` in oriented coordinates, for drawing only.

    The line handed to any calibration consumer stays in original coordinates —
    this is purely so the overlay lands on the pixels it describes.
    """
    if o.is_identity or line.points.size == 0:
        return line
    pts = _map_all(line.points, o, w, h)
    out = LaserLine(points=pts, inliers=line.inliers, rms_px=line.rms_px,
                    ms=line.ms, reason=line.reason)
    if line.point is not None and line.direction is not None:
        # Map two points on the line and rebuild it, rather than trying to
        # rotate a direction vector through a transform that also mirrors.
        a = line.point - line.direction * 100.0
        b = line.point + line.direction * 100.0
        ma, mb = _map_all(np.stack([a, b]), o, w, h)
        d = mb - ma
        n = float(np.hypot(d[0], d[1]))
        if n > 1e-6:
            out.point, out.direction = ma, d / n
    return out


def _orient_hull(hull: np.ndarray | None, o: Orientation, w: int, h: int):
    if hull is None or o.is_identity:
        return hull
    return _map_all(hull.reshape(-1, 2), o, w, h).astype(np.int32).reshape(-1, 1, 2)
