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
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

import httpx

from .calibpanel import CalibrationPanel
from .cloudview import CloudPanel
from .config import ConfigClient, RigConfig
from .laser import LaserParams
from .panel import EyePanel
from .scanpanel import ScanPanel
from .scanworker import LEFT_POSE_RECENT_S, ScanWorker
from .stereo import compose_right_pose, result_from_config
from .worker import EyeWorker, Latest

log = logging.getLogger("orbiter_native.app")

#: How often to re-read the server's config. A human editing settings in the
#: web tab is the only thing that changes it, so seconds are plenty.
_CONFIG_POLL_MS = 2000

#: How often the window looks for newer results. Faster than the cameras is
#: pointless; slower would show frames late.
_PAINT_MS = 33


class MainWindow(QMainWindow):
    def __init__(self, server: str, gpu: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Orbiter — native CV workbench")
        self.resize(1500, 780)

        self._client = ConfigClient(server)
        self._config: RigConfig | None = None
        self._left_pose: tuple | None = None
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
                w.set_scan_gate(self.scanner.left_pose_recent)
            panel.set_overlay(self.scanner.overlay)
            self.workers[side] = w

        self.calib = CalibrationPanel()
        self.calib.save_requested.connect(self._save_intrinsics)
        # Calibrating wants the stripe detected; the checkbox pushes it down.
        self.calib.laser_requested.connect(
            lambda on: self._laser.setChecked(True) if on else None)

        self.scan = ScanPanel(self.scanner)
        self.scan.active_changed.connect(self._on_scan_toggled)
        self.cloud = CloudPanel()

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
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)
        self.setCentralWidget(root)

        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self._server_label = QLabel(f"server {server}")
        self._server_label.setStyleSheet("color:#8b9aac; font-family:Consolas;")
        self.statusBar().addPermanentWidget(self._server_label)

        self.scanner.start()
        for w in self.workers.values():
            w.start()

        self._paint_timer = QTimer(self)
        self._paint_timer.timeout.connect(self._paint)
        self._paint_timer.start(_PAINT_MS)

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
        reload_act = QAction("Reload config", self)
        reload_act.triggered.connect(self._poll_config)
        bar.addAction(reload_act)

    def _apply_laser(self) -> None:
        params = LaserParams(redness_min=self._threshold.value())
        on = self._laser.isChecked()
        for w in self.workers.values():
            w.set_laser(on, params)
        self.calib.set_laser_active(on)

    # ── config ────────────────────────────────────────────────────────────

    def _poll_config(self) -> None:
        cfg, err = self._client.fetch()
        if err or cfg is None:
            self.statusBar().showMessage(f"server unreachable — {err}")
            return

        self._config = cfg
        self._extrinsics = result_from_config(cfg.extrinsics_raw, None)
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
        for w in self.workers.values():
            w.set_scan_mode(on)
        for panel in self.panels.values():
            panel.set_scanning(on)

    def _paint(self) -> None:
        """Show whatever is newest. Anything older was skipped, not queued."""
        fresh = {side: box.take(0.0) for side, box in self._inbox.items()}
        left = fresh["left"]
        now = time.monotonic()
        if left is not None and left.board is not None and left.board.R is not None:
            self._left_pose = (left.board.R, left.board.t, now)
        elif self._left_pose is not None and now - self._left_pose[2] > LEFT_POSE_RECENT_S:
            # A pose the left eye has not had for a while is not one to draw
            # the right eye's cloud through: the two overlays disagreeing is
            # a diagnostic, and a stale pose would fake agreement.
            self._left_pose = None
        for side, res in fresh.items():
            if res is None:
                continue
            pose = None
            if (side == "right" and (res.board is None or res.board.R is None)
                    and self._left_pose is not None and self._extrinsics is not None):
                # The right eye skipped ChArUco while scanning: its board
                # pose for drawing follows from the left's.
                pose = compose_right_pose(self._left_pose[0], self._left_pose[1], self._extrinsics)
            self.panels[side].on_result(res, pose)
            self.calib.on_result(res)
        status = self.scanner.status.take(0.0)
        if status is not None:
            self.scan.on_status(status)
        # The same decimated snapshot the eyes draw; the view uploads it
        # only when the scan thread published a new one.
        self.cloud.set_live_points(self.scanner.overlay.points(), len(self.scanner.cloud))
        rows = dict(self.scanner.stripe_rows)
        if rows != self._rows_pushed:
            for side, worker in self.workers.items():
                worker.set_stripe_rows(rows.get(side))
            self._rows_pushed = rows

    def _on_status(self, side: str, error: object) -> None:
        self.panels[side].on_status(error if isinstance(error, str) else None)

    # ── shutdown ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the threads before the window goes away.

        Without this the reader threads keep a socket open and Qt tears down
        widgets underneath the timer still firing.
        """
        self._paint_timer.stop()
        self._timer.stop()
        for w in self.workers.values():
            w.stop()
        self.scanner.stop()
        self._client.close()
        super().closeEvent(event)
