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

from . import gpu
from .config import Eye, RigConfig
from .cvcore import BoardSpec
from .detect import BoardDetector, BoardHit
from .intrinsics import ViewDescriptor, describe
from .laser import (
    LaserLine,
    LaserParams,
    StripePixels,
    board_mask,
    find_laser_line,
    find_stripe_pixels,
)
from .orient import Orientation
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
    #: The corners were followed from the previous frame, not found by a
    #: full ChArUco pass — see `detect.TrackParams`.
    tracked: bool = False
    #: Laser fit, when the frame produced a usable one.
    laser_inliers: int = 0
    laser_points: int = 0
    laser_rms_px: float = float("nan")
    laser_angle_deg: float = float("nan")
    laser_reason: str | None = None
    #: Stripe points found in scan mode.
    stripe_points: int = 0
    #: Decoded and scored on the GPU (`gpu.py`) rather than by OpenCV.
    gpu: bool = False
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
    #: The frame for the view — unoriented, nothing drawn on it, shared with
    #: the reader rather than copied, and possibly half size on the GPU path
    #: (`wh` is the true size). The view orients it and draws the geometry
    #: below over it; nothing may write into it.
    bgr: np.ndarray
    stats: EyeStats
    orientation: Orientation = Orientation()
    #: This eye's intrinsics at this frame size, when it has them: what
    #: projects the cloud into the view. None until the pair is calibrated.
    intrinsics: object | None = None
    #: Hull of the detected corners the calibration-mode laser search was
    #: confined to, (N, 1, 2) int, or None.
    hull: np.ndarray | None = None
    board: BoardHit | None = None
    laser: LaserLine | None = None
    #: Every stripe pixel with its score — what scanning consumes. Empty
    #: unless the worker is in scan mode.
    stripe: StripePixels | None = None
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

    def __init__(self, side: str, gpu: bool = False) -> None:
        super().__init__()
        self.side = side
        self._gpu = gpu
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
        #: Rows the sheet can appear in while scanning (`scan.stripe_rows`),
        #: or None for the whole frame. Pushed by the window from the scan
        #: worker, which owns the geometry it follows from.
        self._stripe_rows: tuple[int, int] | None = None
        #: Whether the scan can use a stripe from this eye right now: the
        #: scan worker answers for the right eye (it knows whether the left
        #: has had a pose lately). None means always.
        self._scan_gate: Callable[[], bool] | None = None
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
        self._push_full_bgr()

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
        self._push_full_bgr()

    def _needs_full_bgr(self) -> bool:
        """Only the calibration-mode line fit reads the full colour frame on
        the CPU; scanning scores the stripe on the GPU frame."""
        with self._cfg_lock:
            return self._laser_on and not self._scan_mode

    def _push_full_bgr(self) -> None:
        reader = self._reader
        if reader is not None:
            reader.full_bgr = self._needs_full_bgr()

    def set_stripe_rows(self, rows: tuple[int, int] | None) -> None:
        with self._cfg_lock:
            self._stripe_rows = rows

    def set_scan_gate(self, gate: Callable[[], bool] | None) -> None:
        with self._cfg_lock:
            self._scan_gate = gate

    def add_sink(self, sink: Callable[[EyeResult], None]) -> None:
        """Called from the detector thread with every result. Must be quick and
        must not touch widgets — the GUI reads its mailbox on its own clock."""
        self._sinks.append(sink)

    def _snapshot(self):
        with self._cfg_lock:
            return (self._url, self._orientation, self._board_spec,
                    self._eye, self._laser_on, self._laser_params,
                    self._scan_mode, self._stripe_rows, self._scan_gate)

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
            self._reader = MjpegReader(url, gpu=self._gpu)
            self._reader.full_bgr = self._needs_full_bgr()
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
         scan_mode, stripe_rows, scan_gate) = self._snapshot()
        detector.set_spec(spec)

        h, w = frame.gray.shape
        # Resolved against THIS frame's size: intrinsics solved at another
        # resolution are refused rather than silently misapplied.
        intrinsics = eye.intrinsics_for((w, h)) if eye else None
        # While scanning, the right eye's board pose is only ever drawn — and
        # the window draws it from the left's through the extrinsics — so
        # the right eye skips ChArUco and spends its frame on the stripe.
        if detector.ready and board_wanted(self.side, scan_mode):
            board = detector.detect(frame.gray, intrinsics)
        else:
            detector.forget()
            board = BoardHit()
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
        pixels = StripePixels()
        hull = None
        if laser_on and scan_mode:
            if not stripe_wanted(self.side, board.R is not None,
                                 True if scan_gate is None else scan_gate()):
                pixels = StripePixels(wh=(w, h), reason="no board pose — nothing to place")
            else:
                # The whole-frame score is the expensive one; on a GPU frame
                # it runs where the pixels already are.
                pixels = (gpu.stripe_pixels(frame.rgb_gpu, laser_params, stripe_rows)
                          if frame.rgb_gpu is not None
                          else find_stripe_pixels(frame.bgr, laser_params, stripe_rows))
        elif laser_on:
            mask = board_mask(board.corners, frame.gray.shape)
            if mask is None:
                laser = LaserLine(reason="board not detected")
            elif frame.bgr is None:
                # The reader is switching to full frames; the next one has it.
                laser = LaserLine(reason="waiting for a full-size frame")
            else:
                laser = find_laser_line(frame.bgr, mask, laser_params)
                hull = cv2.convexHull(board.corners.reshape(-1, 2).astype(np.int32))

        # Everything above was found on ORIGINAL pixels and is published in
        # original coordinates. Orienting is the view's job (`glview`): it
        # draws the frame through the orientation and maps the geometry the
        # same way, so no oriented copy of the frame is ever made, and a
        # calibration consumer never sees coordinates that depend on a UI
        # setting.
        self._det_times.append(time.monotonic())
        self._publish(frame, board, laser, hull, descriptor, pixels,
                      orientation, intrinsics)

    def _publish(self, frame, board, laser, hull, descriptor, pixels,
                 orientation, intrinsics) -> None:
        s = self._stats
        s.frames += 1
        s.recv_fps = _rate(self._recv_times)
        s.detect_fps = _rate(self._det_times)
        s.detect_ms = board.ms
        s.laser_ms = laser.ms
        s.corners = board.count
        s.coverage = board.coverage(*frame.gray.shape[::-1])
        s.tracked = board.tracked
        s.laser_points = int(len(laser.points))
        s.laser_inliers = laser.n_inliers
        s.laser_rms_px = laser.rms_px
        s.laser_angle_deg = laser.angle_deg
        s.laser_reason = (pixels.reason if pixels.count or pixels.reason != "no data"
                          else laser.reason)
        s.stripe_points = pixels.count
        s.gpu = frame.rgb_gpu is not None
        s.server_age_ms = frame.server_age_ms
        h, w = frame.gray.shape
        res = EyeResult(
            self.side, frame.display, _copy_stats(s), orientation=orientation,
            intrinsics=intrinsics, hull=hull, board=board, laser=laser,
            stripe=pixels, wh=(w, h), descriptor=descriptor,
            capture_mono=frame.capture_mono,
        )
        for sink in self._sinks:
            sink(res)


# ── helpers ───────────────────────────────────────────────────────────────


def board_wanted(side: str, scan_mode: bool) -> bool:
    """Whether this eye runs ChArUco on the frame. While scanning, the right
    eye's board pose is only drawn, and the window draws it from the left's
    through the extrinsics — so the right eye spends its frame on the stripe."""
    return not (scan_mode and side == "right")


def stripe_wanted(side: str, has_pose: bool, left_recent: bool) -> bool:
    """Whether the stripe is worth scoring on this frame while scanning.

    The scan places points through the LEFT eye's board pose, so a left
    frame without one has nothing to place and the right eye's stripe is
    only wanted while the left has had a pose lately — the right eye does
    not detect the board while scanning, so it asks the scan worker.
    """
    return has_pose if side == "left" else left_recent


def _rate(times: "deque[float]") -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0


def _copy_stats(s: EyeStats) -> EyeStats:
    """Stats cross a thread boundary; hand the GUI a snapshot, not the live object."""
    return EyeStats(**vars(s))


