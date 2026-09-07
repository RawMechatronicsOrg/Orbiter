"""Main window: the two eyes side by side, plus what drives them.

The server owns configuration; this window polls `GET /config` and pushes what
changed down into the workers. Change an eye's orientation in the web Stereo
tab, press Apply there, and this window follows within a poll interval without
a restart — one place to set the run's baseline, two places that honour it.

The GUI thread only paints. Detection and scanning run in their own threads
and leave their newest result in a one-slot mailbox; a 30 Hz timer here takes
whatever is newest and shows it. Nothing queues: when the workers outrun the
painter, frames are skipped, never backlogged. The previous design delivered
every result through a queued Qt signal and did the scan maths in the slot —
at 1080p that was more than a second of GUI-thread work per second, and the
window fell further behind the cameras the longer it ran.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QMainWindow,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import httpx

from . import recon
from .calibpanel import CalibrationPanel
from .cloudview import CloudPanel
from .config import ConfigClient, Eye, RigConfig
from .cvcore import BoardSpec
from .exposure import EXPOSURE_MIN, ExposureKeeper
from .guide import Guide
from .guidepanel import GuideBanner
from .laser import LaserParams
from .panel import EyePanel
from .photos import (
    BoardSnapshot,
    EyeSnapshot,
    Extrinsics,
    PhotoSession,
    PhotoWriter,
    RigSnapshot,
)
from .posesmooth import median_pose
from .scan import ScanVolume
from .scanpanel import ScanPanel
from .scanworker import POSE_RECENT_S, ScanWorker
from .screens import adapter_of_window, same_gpu
from .stereo import compose_left_pose, compose_right_pose, result_from_config
from .worker import EyeWorker, Latest

log = logging.getLogger("orbiter_native.app")

#: How often to re-read the server's config. A human editing settings in the
#: web tab is the only thing that changes it, so seconds are plenty.
_CONFIG_POLL_MS = 2000

#: How often the window looks for newer results. Faster than the cameras is
#: pointless; slower would show frames late.
_PAINT_MS = 33
#: The guide's prompt is re-read this often: quick enough that "hold still"
#: follows the hand, slow enough that the big type does not flicker.
_GUIDE_MS = 250
#: How often the recon thread's lines are drained into the panel's one status
#: line. Faster than a person reads them; slower than any step takes.
_RECON_MS = 250
#: Recon ticks between two walks of the session directory while a run is
#: going. At `_RECON_MS` that is two and a half seconds — often enough that
#: an operator watching the disk fill sees it move, rare enough that a walk
#: over a dense run's thousands of files is not what the GUI thread does.
_SIZE_TICKS = 10
#: How long `closeEvent` waits for the recon thread once it has been asked to
#: stop. It stops at its child's next line of output, and every step it did
#: finish is in `recon-state.json`, so a run that outlasts this is resumed.
_RECON_JOIN_S = 3.0


def _f(value: Any, default: float = float("nan")) -> float:
    """A number out of the server's JSON, or `default`. Missing and malformed
    are the same thing here: a session records what it knows."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _aruco_dict_name(dict_id: int) -> str:
    """OpenCV's `DICT_*` constant as its own name.

    The manifest is read by people and by a converter, and neither wants to
    look `DICT_5X5_100` up from `5`. The server stores only the int, so the
    name is resolved here, against the OpenCV this rig actually runs.
    """
    names = sorted(name for name in dir(cv2.aruco)
                   if name.startswith("DICT_") and getattr(cv2.aruco, name) == dict_id)
    return names[0] if names else str(dict_id)


def _board_snapshot(spec: BoardSpec | None) -> BoardSnapshot | None:
    if spec is None:
        return None
    return BoardSnapshot(squares_x=int(spec.squares_x), squares_y=int(spec.squares_y),
                         square_mm=float(spec.square_length_mm),
                         marker_mm=float(spec.marker_length_mm),
                         dictionary=_aruco_dict_name(int(spec.aruco_dict_id)))


def _eye_snapshot(eye: Eye | None, live_wh: tuple[int, int] | None) -> EyeSnapshot | None:
    """One eye as the session records it, or None when no camera is assigned.

    The size is the intrinsics' own: a camera matrix is only valid at the
    resolution it was solved at, and that is the resolution the photographs
    have to be at for anybody to use them. The live frame size is the
    fallback, so a session opened before the first solve still says which
    camera took the pictures.
    """
    if eye is None or not eye.camera_id:
        return None
    k = eye.intrinsics_raw or {}
    try:
        solved_wh = (int(k["width"]), int(k["height"]))
    except (KeyError, TypeError, ValueError):
        solved_wh = live_wh or (0, 0)
    return EyeSnapshot(
        camera_id=eye.camera_id,
        wh=solved_wh,
        fx=_f(k.get("fx"), 0.0), fy=_f(k.get("fy"), 0.0),
        cx=_f(k.get("cx"), 0.0), cy=_f(k.get("cy"), 0.0),
        dist=tuple(_f(d, 0.0) for d in (k.get("dist") or ())),
        rms_px=_f(k.get("rms_px")),
    )


def _extrinsics_snapshot(raw: dict[str, Any] | None) -> Extrinsics | None:
    """The pair's geometry as stored, or None when it has not been solved."""
    if not isinstance(raw, dict):
        return None
    try:
        R = np.asarray(raw["R"], float).reshape(3, 3)
        t_mm = np.asarray(raw["T"], float).ravel()
    except (KeyError, TypeError, ValueError):
        return None
    return Extrinsics(R=R, t_mm=t_mm, rms_px=_f(raw.get("rms_px")))


def rig_snapshot(cfg: RigConfig, volume: ScanVolume,
                 live_wh: dict[str, tuple[int, int] | None]) -> RigSnapshot:
    """The rig frozen at the instant a session opens: both eyes, the pair, the
    board, and the volume being scanned inside.

    Mapped here rather than in `photos.py` because this window is the only
    thing that holds a `RigConfig`; the session module sits underneath the
    config client in the import graph and takes the snapshot ready-made.
    """
    return RigSnapshot(
        board=_board_snapshot(cfg.board),
        volume=volume,
        left=_eye_snapshot(cfg.left, live_wh.get("left")),
        right=_eye_snapshot(cfg.right, live_wh.get("right")),
        extrinsics=_extrinsics_snapshot(cfg.extrinsics_raw),
    )


class MainWindow(QMainWindow):
    def __init__(self, server: str, gpu: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Orbiter — native CV workbench")
        self.resize(1500, 780)

        self._client = ConfigClient(server)
        self._config: RigConfig | None = None
        #: Each eye's latest own board pose (R, t, when, frame size), for
        #: drawing the other eye's cloud when that eye has none of its own.
        self._poses: dict[str, tuple | None] = {"left": None, "right": None}
        self._extrinsics_raw: dict | None = None
        self._extrinsics_at: tuple = (None, None)
        self._extrinsics = None

        self.scanner = ScanWorker()
        #: Newest result per eye, waiting for the paint timer.
        self._inbox = {"left": Latest(), "right": Latest()}
        self.panels = {"left": EyePanel("left"), "right": EyePanel("right")}
        self.workers: dict[str, EyeWorker] = {}
        self._rows_pushed: dict[str, tuple[int, int] | None] = {}
        for side, panel in self.panels.items():
            w = EyeWorker(side, gpu=gpu)
            # Errors are rare, so a queued signal is fine for them; results
            # are not, so they go to mailboxes the timer drains.
            w.status.connect(self._on_status, Qt.ConnectionType.QueuedConnection)
            w.add_sink(self._inbox[side].put)
            w.add_sink(self.scanner.offer)
            if side == "right":
                w.set_scan_gate(self.scanner.pose_recent)
            panel.set_overlay(self.scanner.overlay)
            self.workers[side] = w

        self.calib = CalibrationPanel()
        self.calib.save_requested.connect(self._save_intrinsics)
        # Calibrating wants the stripe detected; the checkbox pushes it down.
        self.calib.laser_requested.connect(
            lambda on: self._laser.setChecked(True) if on else None)

        self.scan = ScanPanel(self.scanner)
        self.scan.active_changed.connect(self._on_scan_toggled)
        self.scan.cloud_cleared.connect(self._new_session)
        self.scan.photos_toggled.connect(self._on_photos_toggled)
        self.scan.photo_pass_toggled.connect(self._on_photo_pass_toggled)
        self.scan.reconstruct_requested.connect(self._reconstruct)
        self.scan.abort_requested.connect(self._abort_recon)
        self.scan.open_folder_requested.connect(self._open_session_folder)
        self.cloud = CloudPanel()

        #: The photographs: where they go, and the thread that writes them.
        #: Opened lazily — on "Clear cloud", or at the first scan or photo
        #: after launch — because a session is a directory on the operator's
        #: disk and an app that merely started has taken no photographs.
        self._session: PhotoSession | None = None
        self._writer: PhotoWriter | None = None
        #: The reconstruction: the thread running the chain, the Event that
        #: aborts it, and the lines it has produced since the timer last took
        #: them. The list is appended to on the recon thread and swapped out
        #: on the GUI thread, under `_recon_lock` and nothing else.
        self._recon_thread: threading.Thread | None = None
        self._recon_cancel = threading.Event()
        self._recon_lock = threading.Lock()
        self._recon_lines: list[str] = []
        self._recon_step = ""
        self._recon_last = ""
        #: Ticks the recon timer has fired, so the size walk can be one in
        #: `_SIZE_TICKS` of them rather than one per tick.
        self._recon_ticks = 0

        # The guide: which stage, what to do, where to bring the board.
        self.guide = Guide()
        self.banner = GuideBanner()
        self.banner.back_requested.connect(self._guide_back)
        self.banner.next_requested.connect(self._guide_next)
        self.banner.toggled.connect(self._guide_toggled)
        self.calib.cleared.connect(self._guide_restart)
        self.calib.rig_moved.connect(self._guide_restart)
        #: Each eye's newest frame size, for the guide's target rectangle.
        self._wh: dict[str, tuple[int, int] | None] = {"left": None, "right": None}
        #: Each eye's stream error, for the guide: None while frames arrive.
        self._offline: dict[str, str | None] = {"left": None, "right": None}
        #: Each eye's last own poses, for drawing the cloud through their
        #: median rather than through whatever this frame's corners gave:
        #: the scan places points through a steadied pose (`posesmooth`),
        #: and the overlay must not jump where the points do not.
        self._pose_hist: dict[str, deque] = {"left": deque(maxlen=7), "right": deque(maxlen=7)}
        #: The scan's last frame while scanning, for the guide's check.
        self._scan_frame = None

        # Each eye's exposure: steered by the stripe's brightness and set
        # again whenever camserver reopens a device and forgets it.
        self.exposure = ExposureKeeper(Path.home() / ".orbiter-native" / "exposure.json")
        self.exposure.start()
        #: The toolbar's spin boxes are being set from the keeper, not by hand.
        self._exposure_syncing = False

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.panels["left"])
        split.addWidget(self.panels["right"])
        side = QSplitter(Qt.Orientation.Vertical)
        side.addWidget(self.calib)
        side.addWidget(self.scan)
        side.addWidget(self.cloud)
        side.setStretchFactor(2, 1)
        split.addWidget(side)
        split.setSizes([560, 560, 380])

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(4, 4, 4, 0)
        outer.setSpacing(4)
        outer.addWidget(self.banner)
        outer.addWidget(split, 1)
        self.setCentralWidget(root)

        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self._server_label = QLabel(f"server {server}")
        self._server_label.setStyleSheet("color:#8b9aac; font-family:Consolas;")
        self.statusBar().addPermanentWidget(self._server_label)
        # Which GPU draws the window against which GPU drives its monitor:
        # known once the first GL context exists, checked again whenever the
        # window changes screens (screens.py has the why).
        self._gl_renderer: str | None = None
        self._screen_watched = False
        self._gpu_label = QLabel()
        self._gpu_label.setStyleSheet("color:#fbbf24; font-family:Consolas;")
        self._gpu_label.hide()
        self.statusBar().addPermanentWidget(self._gpu_label)
        self.panels["left"].view.gl_ready.connect(self._on_gl_ready)

        self.scanner.start()
        for w in self.workers.values():
            w.start()

        self._paint_timer = QTimer(self)
        self._paint_timer.timeout.connect(self._paint)
        self._paint_timer.start(_PAINT_MS)

        self._guide_timer = QTimer(self)
        self._guide_timer.timeout.connect(self._guide_tick)
        self._guide_timer.start(_GUIDE_MS)

        self._exposure_timer = QTimer(self)
        self._exposure_timer.timeout.connect(self._exposure_tick)
        self._exposure_timer.start(1000)

        # Runs only while a reconstruction does: there is nothing to drain
        # otherwise, and an hour-long run is the only thing that produces lines.
        self._recon_timer = QTimer(self)
        self._recon_timer.timeout.connect(self._recon_tick)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_config)
        self._timer.start(_CONFIG_POLL_MS)
        self._poll_config()

    # ── toolbar ───────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("controls")
        bar.setMovable(False)

        # Off by default — it costs a few ms per frame and only means anything
        # while the laser is actually on.
        self._laser = QCheckBox("laser line")
        self._laser.setToolTip(
            "Find the red stripe where it crosses the ChArUco board and fit a "
            "straight line to it. Restricted to the board: points on the bench "
            "are not on the board plane and would poison the calibration."
        )
        self._laser.toggled.connect(self._apply_laser)
        bar.addWidget(self._laser)

        bar.addWidget(QLabel("  redness "))
        self._threshold = QSpinBox()
        self._threshold.setRange(1, 255)
        self._threshold.setValue(LaserParams().redness_min)
        self._threshold.setToolTip(
            "Minimum r - max(g, b) for a pixel to count as stripe. Redness, not "
            "brightness: on this board the white squares are as bright as the "
            "laser but not remotely as red."
        )
        self._threshold.valueChanged.connect(lambda _v: self._apply_laser())
        bar.addWidget(self._threshold)

        bar.addSeparator()
        self._exp_auto = QCheckBox("auto exposure")
        self._exp_auto.setToolTip(
            "Steer each camera's exposure time by the laser stripe: its red peak is "
            "kept in the low 200s, bright but not clipped — a clipped stripe has a "
            "flat profile and its centroid wanders by a pixel, more than a millimetre "
            "of depth here. Off, the times beside are yours. Either way they are set "
            "again when camserver reopens a camera and forgets them."
        )
        self._exp_auto.setChecked(self.exposure.auto)
        self._exp_auto.toggled.connect(self._on_exposure_auto)
        bar.addWidget(self._exp_auto)
        self._exp_spin: dict[str, QSpinBox] = {}
        for side in ("left", "right"):
            bar.addWidget(QLabel(f"  {side[0].upper()} "))
            spin = QSpinBox()
            spin.setRange(EXPOSURE_MIN, 2000)
            spin.setSingleStep(10)
            spin.setSpecialValueText("—")            # the minimum stands for "not known yet"
            spin.setToolTip(f"Exposure time of the {side} camera, in 0.1 ms steps "
                            "(100 = 10 ms; 330 is a whole frame at 30 fps).")
            spin.setEnabled(not self.exposure.auto)
            spin.valueChanged.connect(lambda v, s=side: self._on_exposure_spin(s, v))
            bar.addWidget(spin)
            self._exp_spin[side] = spin
        self._exp_label = QLabel("")
        self._exp_label.setStyleSheet("color:#8b9aac; font-family:Consolas;")
        bar.addWidget(self._exp_label)

        bar.addSeparator()
        reload_act = QAction("Reload config", self)
        reload_act.triggered.connect(self._poll_config)
        bar.addAction(reload_act)

    def _apply_laser(self) -> None:
        params = LaserParams(redness_min=self._threshold.value())
        on = self._laser.isChecked()
        for w in self.workers.values():
            w.set_laser(on, params)
        self.calib.set_laser_active(on)
        self.exposure.set_laser(on)

    # ── config ────────────────────────────────────────────────────────────

    def _poll_config(self) -> None:
        cfg, err = self._client.fetch()
        if err or cfg is None:
            self.statusBar().showMessage(f"server unreachable — {err}")
            return

        self._config = cfg
        self._extrinsics_raw = cfg.extrinsics_raw
        board = "no board configured"
        if cfg.board:
            board = (f"board {cfg.board.squares_x}x{cfg.board.squares_y} · "
                     f"{cfg.board.square_length_mm:g}mm sq")
        self.statusBar().showMessage(
            f"camserver {cfg.camserver or '—'} · baseline {cfg.baseline_mm:g} mm "
            f"(nominal) · {board}"
        )

        self.calib.set_config(cfg)
        self.scanner.set_config(cfg)
        self.exposure.configure(cfg.camserver, {
            side: getattr(getattr(cfg, side, None), "camera_id", None)
            for side in ("left", "right")})
        for side, worker in self.workers.items():
            eye = getattr(cfg, side)
            self.panels[side].set_eye(eye)
            if worker.apply_config(cfg, eye):
                worker.restart_stream()

    def _save_intrinsics(self, per_side: dict) -> None:
        """Store a solve on the server, through the same command the web tab uses.

        The server is the owner of this state; writing it anywhere else would
        give the rig two sources of truth for its own calibration. Sent over
        HTTP rather than the WS command channel because this app is a read-only
        client of the model otherwise and does not hold a socket for it.
        """
        # The pair geometry is a rig-level field, not an eye's; the panel
        # flags it with a reserved key so one save covers both solves.
        extr = per_side.pop("_extrinsics", None)
        plane = per_side.pop("_laser_plane", None)
        # Per-eye fields as the panel keyed them: intrinsics, readout.
        args: dict = {side: dict(fields) for side, fields in per_side.items()}
        if plane is not None:
            args["laser_plane"] = plane
        if extr is not None:
            args["extrinsics"] = extr
            # The measured baseline supersedes whatever nominal value was typed
            # into the web tab — it comes from the same solve as the geometry.
            if "baseline_mm" in extr:
                args["baseline_mm"] = extr["baseline_mm"]
        try:
            r = httpx.post(f"{self._client.server}/command/set_stereo_rig",
                           json=args, timeout=5.0)
            if r.status_code == 400:
                # The handler's own message: both eyes on one camera, a
                # malformed calibration. The reason is the useful part.
                self.statusBar().showMessage(
                    f"server refused: {r.json().get('detail')}")
                return
            r.raise_for_status()
        except httpx.HTTPError as exc:
            self.statusBar().showMessage(f"could not save intrinsics — {exc}")
            return
        self.statusBar().showMessage("intrinsics saved to the server")
        self._poll_config()

    # ── worker output ─────────────────────────────────────────────────────

    def _on_scan_toggled(self, on: bool) -> None:
        """Scanning without the laser detector finds nothing, so turn it on."""
        if on and not self._laser.isChecked():
            self._laser.setChecked(True)      # this also pushes it to the workers
        if on:
            # A scan is one of the two things that opens a session: the cloud
            # and the photographs of one run belong in one directory.
            self._open_session()
        for w in self.workers.values():
            w.set_scan_mode(on)
        for panel in self.panels.values():
            panel.set_scanning(on)
        if not on:
            self._scan_frame = None

    def _extrinsics_for(self, wh: tuple[int, int] | None):
        """The pair geometry at the left eye's live frame size, or None.

        Resolved against the size, the way everything else here is: a
        calibration solved at another resolution is refused rather than
        silently misapplied. Drawn through an unchecked one, the right eye's
        overlay would agree with the left's while the scan itself refused the
        same numbers — and the two overlays disagreeing is the diagnostic.
        """
        if wh is None or self._extrinsics_raw is None:
            return None
        if (self._extrinsics_raw, wh) != self._extrinsics_at:
            self._extrinsics_at = (self._extrinsics_raw, wh)
            self._extrinsics = result_from_config(self._extrinsics_raw, wh)
        return self._extrinsics

    def _paint(self) -> None:
        """Show whatever is newest. Anything older was skipped, not queued."""
        fresh = {side: box.take(0.0) for side, box in self._inbox.items()}
        now = time.monotonic()
        for side, res in fresh.items():
            if res is not None:
                self._wh[side] = res.wh
                self.exposure.observe(side, res.stats.stripe_peak, res.stats.stripe_clipped)
            if res is not None and res.board is not None and res.board.R is not None:
                self._poses[side] = (res.board.R, res.board.t, now, res.wh)
        for side, held in self._poses.items():
            if held is not None and now - held[2] > POSE_RECENT_S:
                # A pose an eye has not had for a while is not one to draw the
                # other eye's cloud through: the two overlays disagreeing is
                # a diagnostic, and a stale pose would fake agreement.
                self._poses[side] = None
        for side, res in fresh.items():
            if res is None:
                continue
            pose = None
            if res.board is not None and res.board.R is not None:
                hist = self._pose_hist[side]
                hist.append((res.board.R, res.board.t))
                if len(hist) >= 3:
                    pose = median_pose(list(hist), len(hist) - 1)
            else:
                # The board is out of this eye's sight: a pose from before
                # it left is no neighbour of the one it comes back with.
                self._pose_hist[side].clear()
                # No pose of its own this frame: for drawing, the other eye's
                # recent one carried across through the pair.
                other = self._poses["right" if side == "left" else "left"]
                geom = self._extrinsics_for(other[3]) if other is not None else None
                if geom is not None:
                    carry = compose_right_pose if side == "right" else compose_left_pose
                    pose = carry(other[0], other[1], geom)
            self.panels[side].on_result(res, pose)
            self.calib.on_result(res)
        status = self.scanner.status.take(0.0)
        if status is not None:
            self.scan.on_status(status)
            if status.frame is not None:
                self._scan_frame = status.frame
        # The same decimated snapshot the eyes draw; the view uploads it
        # only when the scan thread published a new one.
        self.cloud.set_live_points(self.scanner.overlay.points(), len(self.scanner.cloud),
                                   self.scanner.overlay.colors())
        rows = dict(self.scanner.stripe_rows)
        if rows != self._rows_pushed:
            for side, worker in self.workers.items():
                worker.set_stripe_rows(rows.get(side))
            self._rows_pushed = rows

    def _on_status(self, side: str, error: object) -> None:
        self._offline[side] = error if isinstance(error, str) else None
        self.panels[side].on_status(self._offline[side])

    # ── photographs ───────────────────────────────────────────────────────

    def _open_session(self) -> PhotoSession | None:
        """The session photographs go into, opened on demand.

        Refused without a config, with a message that says why: the rig
        snapshot is what makes a session reconstructable, and one taken before
        the server answered would carry no intrinsics, no pair and no board —
        eighty photographs nobody can use, discovered an hour later.
        """
        if self._session is not None:
            return self._session
        cfg = self._config
        if cfg is None:
            # The panel, not the status bar: `_poll_config` rewrites that
            # every two seconds, and this is a refusal the operator has to
            # read before understanding why the checkbox came back up.
            self.scan.set_notice(
                "no rig config yet — the server has not answered, so a session "
                "would carry no intrinsics, no pair and no board. Photographs "
                "stay off until it does.")
            return None
        try:
            session = PhotoSession(
                rig=rig_snapshot(cfg, self.scan.params().volume, self._wh))
            writer = PhotoWriter(session)
            writer.start()
        except Exception as exc:                              # noqa: BLE001
            # A read-only sessions root, a disk with nothing left on it, a
            # thread the OS refused: raising out of a Qt slot would take the
            # toggle with it and leave the window running with no session and
            # no word about why.
            log.exception("the photo session could not be opened")
            self.scan.set_notice(f"could not open a photo session — {exc}")
            return None
        self.scanner.set_session(session, writer)
        self._session, self._writer = session, writer
        self.scan.set_session(session.session_id)
        self.scan.set_notice("")
        log.info("photo session %s in %s", session.session_id, session.path)
        self.statusBar().showMessage(f"photo session {session.session_id} — {session.path}")
        return session

    def _close_session(self) -> None:
        """Stop writing into the current session and forget it; the files stay.

        The writer is drained rather than dropped: what is queued was earned
        by the operator standing at the bench, and it is a fraction of a
        second's work to finish it.
        """
        writer, self._writer = self._writer, None
        self._session = None
        self.scanner.set_session(None, None)
        self.scan.set_session("")
        if writer is not None:
            writer.stop()

    def _new_session(self) -> None:
        """The cloud was cleared, so the run is over: the next photographs
        belong to the next session, not to the cloud that has just gone."""
        self._close_session()
        self._open_session()

    def _on_photos_toggled(self, on: bool) -> None:
        """Three calls, not one — and missing any of them is silent.

        The eye workers retain each frame's JPEG bytes and measure how sharp
        it is; the scan worker builds candidates, asks the policy and enqueues
        what it accepts. Without the first there are no bytes to write;
        without the second nothing is ever decided. Either way the session
        comes out empty, and nobody finds out until the pass is over.
        """
        if on and self._open_session() is None:
            # Refused: put the switch back where it was, which comes through
            # here again as False and disarms all three.
            self.scan.photos.setChecked(False)
            return
        for w in self.workers.values():
            w.set_photo_capture(on)
        self.scanner.arm_photos(on)

    def _on_photo_pass_toggled(self, on: bool) -> None:
        """Photo-pass mode is the scan worker's alone: the eyes go on doing
        exactly what "take photos" asked of them, and only the pairing changes."""
        self.scanner.set_photo_pass(on)

    # ── the reconstruction ────────────────────────────────────────────────

    def _reconstruct(self, mode: str) -> None:
        """Export the cloud the photographs were taken alongside, then run the
        chain on a thread of its own.

        `laser.ply` is written first, and from here: it comes out of the live
        cloud, which only this process holds, and `write_sparse` — the chain's
        very first step — reads it. The run itself never touches a widget; it
        appends lines to a list a timer drains.
        """
        if self._recon_thread is not None:
            return
        session = self._session
        if session is None:
            self.scan.set_notice(
                "no session yet — take some photographs before reconstructing")
            return
        try:
            n = self.scanner.export(str(session.path / "laser.ply"))
        except Exception as exc:                              # noqa: BLE001
            # Not just `OSError`: `export` walks the cloud and writes a PLY,
            # and anything it raises out of this slot would skip `closeEvent`
            # on the way out. The panel keeps the sentence; the status bar
            # would lose it at the next config poll.
            log.exception("laser.ply could not be written")
            self.scan.set_notice(f"could not write laser.ply — {exc}")
            return
        self.scan.set_notice("")
        self._recon_cancel = threading.Event()
        self._recon_step, self._recon_last = "", f"laser.ply: {n} points"
        with self._recon_lock:
            self._recon_lines = []
        cancel, path = self._recon_cancel, session.path

        def work() -> None:
            try:
                recon.run(path, mode, on_line=self._recon_line, cancel=cancel)
            except Exception as exc:                              # noqa: BLE001
                # A refusal, a failed step and an abort all carry a full
                # sentence; the log file has the rest. What must not happen is
                # a thread that dies with the panel still saying "running".
                log.exception("the reconstruction raised")
                self._recon_line(f"{exc.__class__.__name__}: {exc}")

        # Started first, assigned second: `start` can raise, and a
        # `_recon_thread` holding a thread that never ran would wedge the
        # double-start guard above for the rest of the session.
        thread = threading.Thread(target=work, name="recon", daemon=True)
        thread.start()
        self._recon_thread = thread
        self.scan.set_reconstructing(True)
        self.scan.set_recon_status(self._recon_step, self._recon_last)
        self._recon_timer.start(_RECON_MS)

    def _recon_line(self, line: str) -> None:
        """The recon thread's one reach into this window: append, and return.

        Accumulate-and-swap rather than `worker.Latest`, which overwrites —
        right for a status object, wrong for a line stream, where it would
        drop nearly every line between two ticks.
        """
        with self._recon_lock:
            self._recon_lines.append(line)

    def _recon_tick(self) -> None:
        """Show what the run has said, how much of the disk it has taken, and
        — once it has ended — hand the button back.

        The order matters at the end. The lines are drained, the thread is
        asked whether it is still running, and if it is not they are drained
        once more: this is the tick that stops the timer, so a line written
        between the first drain and the question has nobody left to take it.
        """
        self._recon_ticks += 1
        self._drain_recon_lines()
        thread = self._recon_thread
        if thread is None:
            return
        if thread.is_alive():
            # A dense run fills the disk for an hour with the cameras idle:
            # nothing pairs, so no `ScanStatus` is published and the size
            # label would sit at what the run started with. The walk is not
            # free, which is why it is one tick in `_SIZE_TICKS` and not this
            # one.
            if self._recon_ticks % _SIZE_TICKS == 0:
                self._show_session_size()
            return
        self._recon_thread = None
        # Once more, now the thread has ended: whatever it wrote between the
        # swap above and its last breath is still in the list, and the timer
        # that would have taken it is about to stop.
        self._drain_recon_lines()
        self._recon_timer.stop()
        self.scan.set_reconstructing(False)
        self._show_session_size()

    def _drain_recon_lines(self) -> None:
        """Take every line the recon thread has written since the last call,
        and show the newest of them beside the step it belongs to."""
        with self._recon_lock:
            lines, self._recon_lines = self._recon_lines, []
        for line in lines:
            name, _, rest = line.partition(": ")
            if rest == "running" and name in recon.CANONICAL_STEPS:
                self._recon_step = name
        if lines:
            self._recon_last = lines[-1]
            self.scan.set_recon_status(self._recon_step, self._recon_last)

    def _show_session_size(self) -> None:
        """Measure the session again and put the number on the panel.

        A reconstruction writes gigabytes through paths the writer's byte
        counter never sees, so the counter has to be replaced by a walk rather
        than added to. Called from a Qt timer, so what the walk can raise —
        a file that vanished between `rglob` and `stat`, a directory that went
        away — is caught here and not out of the slot.
        """
        session = self._session
        if session is None:
            return
        try:
            self.scan.set_session_bytes(session.rewalk())
        except Exception as exc:                              # noqa: BLE001
            log.exception("the session could not be measured")
            self.scan.set_notice(f"could not measure {session.path} — {exc}")

    def _abort_recon(self) -> None:
        """Abort is the cancel Event, and nothing else: the chain stops at its
        child's next line of output and terminates it. Every step that did
        finish stays recorded, so the next run resumes rather than starts
        over."""
        self._recon_cancel.set()
        self.scan.set_recon_status(self._recon_step, "aborting")

    def _open_session_folder(self) -> None:
        session = self._session
        if session is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(session.path)))

    # ── the guide ─────────────────────────────────────────────────────────

    def _guide_tick(self) -> None:
        if not self.banner.enabled.isChecked():
            return
        frames = {side: (wh[0], wh[1], self.panels[side].orientation)
                  for side, wh in self._wh.items() if wh is not None}
        f = self._scan_frame if self.scan.scanning else None
        prompt = self.guide.update(
            self.calib.flow, frames, laser_on=self._laser.isChecked(),
            scanning=self.scan.scanning,
            veto_px=None if f is None else float(f.veto_px),
            kept=0 if f is None else int(f.n_kept),
            auto=self.calib.auto.isChecked(),
            offline={side for side, err in self._offline.items() if err},
            saturated=self.exposure.saturated(),
            exposure_auto=self._exp_auto.isChecked())
        # The guide reads the flow; this is where its stage reaches it.
        self.calib.flow.solo = prompt.solo
        self.banner.set_prompt(prompt)
        for side, panel in self.panels.items():
            panel.view.set_target(prompt.target if prompt.eye == side else None)
            panel.view.set_highlight(prompt.eye in (side, "both"))

    # ── exposure ──────────────────────────────────────────────────────────

    def _on_exposure_auto(self, on: bool) -> None:
        self.exposure.set_auto(on)
        for spin in self._exp_spin.values():
            spin.setEnabled(not on)

    def _on_exposure_spin(self, side: str, value: int) -> None:
        if not self._exposure_syncing and not self._exp_auto.isChecked():
            self.exposure.set_target(side, value)

    def _exposure_tick(self) -> None:
        """The toolbar follows the keeper: the times it holds, the stripe's
        peak per eye, and the last thing it had to do."""
        snap = self.exposure.snapshot()
        self._exposure_syncing = True
        try:
            for side, eye in snap.items():
                spin = self._exp_spin[side]
                want = spin.minimum() if eye.target is None else eye.target
                if spin.value() != want and (eye.target is not None or self._exp_auto.isChecked()):
                    spin.setValue(want)
        finally:
            self._exposure_syncing = False
        now = time.monotonic()
        peaks = []
        for side, eye in snap.items():
            r = eye.recent(now)
            peaks.append(f"{side[0].upper()} {r[0]:.0f}" + ("!" if eye.saturated else "")
                         if r is not None else f"{side[0].upper()} —")
        note = max((e.note for e in snap.values() if e.note), key=len, default="")
        self._exp_label.setText("  peak " + " ".join(peaks) + (f"  · {note}" if note else ""))

    def _guide_back(self) -> None:
        self.guide.back()
        self._guide_tick()

    def _guide_next(self) -> None:
        self.guide.next()
        self._guide_tick()

    def _guide_restart(self) -> None:
        self.guide.restart()
        self._guide_tick()

    def _guide_toggled(self, on: bool) -> None:
        if on:
            self._guide_tick()
            return
        # Off: the eyes pair as usual, and nothing of the guide stays drawn.
        self.calib.flow.solo = None
        for panel in self.panels.values():
            panel.view.set_target(None)
            panel.view.set_highlight(False)

    # ── which GPU draws this window ─────────────────────────────────────

    def _on_gl_ready(self, renderer: str) -> None:
        self._gl_renderer = renderer
        handle = self.windowHandle()
        if handle is not None and not self._screen_watched:
            handle.screenChanged.connect(lambda _screen: self._check_gpu_topology())
            self._screen_watched = True
        self._check_gpu_topology()

    def _check_gpu_topology(self) -> None:
        """Warn when the GPU drawing the window is not the GPU driving its
        monitor. The desktop then copies every frame between the two - on
        the lab PC 40% of a GPU at 30 frames/s, and a stalled desktop with
        the window maximised. The remedy is on the desktop (which monitor is
        primary, which GPU drives it), so this says so rather than guessing
        at a rate cap that only moves the cliff."""
        renderer = self._gl_renderer
        adapter = adapter_of_window(int(self.winId())) if renderer else None
        if not renderer or adapter is None or same_gpu(renderer, adapter):
            self._gpu_label.hide()
            return
        text = (f"OpenGL draws on {renderer}, this monitor is on {adapter}: every "
                f"frame is copied between the two GPUs. Move the window to a monitor "
                f"on {renderer}, or make this monitor the primary display.")
        if self._gpu_label.text() != "⚠ " + text:
            log.warning(text)
        self._gpu_label.setText("⚠ " + text)
        self._gpu_label.show()

    # ── shutdown ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the threads before the window goes away.

        Without this the reader threads keep a socket open and Qt tears down
        widgets underneath the timer still firing.

        The reconstruction is asked to stop and waited on briefly rather than
        abandoned: it holds a child process, and a container left running past
        the window that started it is one nobody will think to kill. What it
        finished is in `recon-state.json`, so the next run resumes.
        """
        self._paint_timer.stop()
        self._timer.stop()
        self._recon_timer.stop()
        self._recon_cancel.set()
        thread, self._recon_thread = self._recon_thread, None
        if thread is not None:
            thread.join(_RECON_JOIN_S)
            if thread.is_alive():
                log.warning("the reconstruction did not stop within %.0f s; "
                            "its state file records what finished", _RECON_JOIN_S)
        self._close_session()
        for w in self.workers.values():
            w.stop()
        self.scanner.stop()
        self.exposure.stop()
        self._client.close()
        super().closeEvent(event)
