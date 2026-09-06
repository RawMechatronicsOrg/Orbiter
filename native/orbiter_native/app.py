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
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
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

from .calibpanel import CalibrationPanel
from .cloudview import CloudPanel
from .config import ConfigClient, RigConfig
from .exposure import EXPOSURE_MIN, ExposureKeeper
from .guide import Guide
from .guidepanel import GuideBanner
from .laser import LaserParams
from .panel import EyePanel
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
        self.cloud = CloudPanel()

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
            if res.board is None or res.board.R is None:
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
        """
        self._paint_timer.stop()
        self._timer.stop()
        for w in self.workers.values():
            w.stop()
        self.scanner.stop()
        self.exposure.stop()
        self._client.close()
        super().closeEvent(event)
