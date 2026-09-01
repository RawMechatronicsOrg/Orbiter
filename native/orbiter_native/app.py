"""Main window: the two eyes side by side, plus what drives them.

The server owns configuration; this window polls `GET /config` and pushes what
changed down into the workers. Change an eye's orientation in the web Stereo
tab, press Apply there, and this window follows within a poll interval without
a restart — one place to set the run's baseline, two places that honour it.
"""

from __future__ import annotations

import logging

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
from .config import ConfigClient, RigConfig
from .laser import LaserParams
from .panel import EyePanel
from .worker import EyeResult, EyeWorker

log = logging.getLogger("orbiter_native.app")

#: How often to re-read the server's config. A human editing settings in the
#: web tab is the only thing that changes it, so seconds are plenty.
_CONFIG_POLL_MS = 2000


class MainWindow(QMainWindow):
    def __init__(self, server: str) -> None:
        super().__init__()
        self.setWindowTitle("Orbiter — native CV workbench")
        self.resize(1500, 780)

        self._client = ConfigClient(server)
        self._config: RigConfig | None = None

        self.panels = {"left": EyePanel("left"), "right": EyePanel("right")}
        self.workers: dict[str, EyeWorker] = {}
        for side, panel in self.panels.items():
            w = EyeWorker(side)
            # Qt queues these across the thread boundary, so the slots run on
            # the GUI thread and may touch widgets safely.
            w.result.connect(self._on_result, Qt.ConnectionType.QueuedConnection)
            w.status.connect(self._on_status, Qt.ConnectionType.QueuedConnection)
            self.workers[side] = w

        self.calib = CalibrationPanel()
        self.calib.save_requested.connect(self._save_intrinsics)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.panels["left"])
        split.addWidget(self.panels["right"])
        split.addWidget(self.calib)
        split.setSizes([620, 620, 260])

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

        for w in self.workers.values():
            w.start()

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
        for w in self.workers.values():
            w.set_laser(self._laser.isChecked(), params)

    # ── config ────────────────────────────────────────────────────────────

    def _poll_config(self) -> None:
        cfg, err = self._client.fetch()
        if err or cfg is None:
            self.statusBar().showMessage(f"server unreachable — {err}")
            return

        self._config = cfg
        board = "no board configured"
        if cfg.board:
            board = (f"board {cfg.board.squares_x}x{cfg.board.squares_y} · "
                     f"{cfg.board.square_length_mm:g}mm sq")
        self.statusBar().showMessage(
            f"camserver {cfg.camserver or '—'} · baseline {cfg.baseline_mm:g} mm "
            f"(nominal) · {board}"
        )

        self.calib.set_board_spec(cfg.board)
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
        args = {side: {"intrinsics": k} for side, k in per_side.items()}
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

    # ── worker signals ────────────────────────────────────────────────────

    def _on_result(self, res: EyeResult) -> None:
        self.panels[res.side].on_result(res)
        self.calib.on_result(res)

    def _on_status(self, side: str, error: object) -> None:
        self.panels[side].on_status(error if isinstance(error, str) else None)

    # ── shutdown ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the threads before the window goes away.

        Without this the reader threads keep a socket open and Qt tears down
        widgets underneath the queued signals still in flight.
        """
        self._timer.stop()
        for w in self.workers.values():
            w.stop()
        self._client.close()
        super().closeEvent(event)
