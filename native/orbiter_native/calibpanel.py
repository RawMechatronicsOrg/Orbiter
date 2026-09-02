"""Calibration panel: one switch, and the board does the rest.

The operator moves the board; `calibflow.CalibrationFlow` decides what each
frame is good for — a still view, a pair, a stripe across the board, a brisk
twist for the readout — and solves in the background as the sets grow. This
panel shows the sets, the solves and the one line that matters: what to do
with the board next. Improvements go to the server on their own.

The guidance is the point. `calibrateCamera` returns a confident answer from a
set of head-on views and that answer can be 12% wrong in focal length with a
reprojection RMS indistinguishable from a good solve — measured, see
`intrinsics.MIN_TILT_SPREAD`. Nothing in the result reveals it. So the tilt
spread and the frame coverage are shown live, while the operator can still act
on them, and the advice line names the weakest link.

Solves run on a thread of their own and hand their outcome back through a
one-slot mailbox the panel's timer drains, the way frames reach the GUI:
nothing per frame goes through a queued signal, and the window never waits on
`calibrateCamera`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .calibflow import CalibrationFlow, Job, Outcome
from .config import RigConfig
from .cvcore import build_board
from .intrinsics import MIN_TILT_SPREAD
from .laserplane import from_config as plane_from_config
from .worker import EyeResult, Latest

log = logging.getLogger("orbiter_native.calibpanel")

#: Captured views live here between runs. A board sweep costs the operator
#: minutes, and losing it to a restart also loses the SIMULTANEOUS views that
#: stereoCalibrate needs — a session that solved intrinsics and then restarted
#: could not solve the pair at all without sweeping again.
SAMPLES_PATH = Path.home() / ".orbiter-native" / "calib-views.npz"

#: How often the panel drains the solve mailbox and asks whether a cycle is due.
_TICK_MS = 400


def _ema(prev: float | None, value: float, alpha: float = 0.2) -> float:
    """A slow average of a live figure, so the big number does not jitter."""
    return value if prev is None else prev + alpha * (value - prev)


class CoverageMap(QWidget):
    """Which parts of the frame the board has visited, as a grid of cells.

    Distortion is only measurable where the board actually went, so the empty
    cells are an instruction, not a decoration.
    """

    def __init__(self, grid: int = 6, parent=None) -> None:
        super().__init__(parent)
        self.grid = grid
        self._cells = np.zeros((grid, grid), bool)
        self.setFixedSize(96, 96)

    def set_cells(self, cells: np.ndarray) -> None:
        self._cells = cells
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        w = self.width() / self.grid
        h = self.height() / self.grid
        for gy in range(self.grid):
            for gx in range(self.grid):
                hit = bool(self._cells[gy, gx])
                p.fillRect(int(gx * w) + 1, int(gy * h) + 1,
                           int(w) - 2, int(h) - 2,
                           QColor(70, 200, 120) if hit else QColor(38, 44, 52))
        p.end()


class CalibrationPanel(QFrame):
    """The switch, the sets, the scoreboard, the advice."""

    #: Emitted with what to store: `{side: {"intrinsics": ..., "readout": ...}}`
    #: with only the solved keys, plus `_extrinsics` / `_laser_plane`.
    save_requested = Signal(dict)
    #: Calibrating wants the stripe detected; the window owns that switch.
    laser_requested = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")   # see the stylesheet in __main__
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.flow = CalibrationFlow()
        self._samples_path = SAMPLES_PATH
        self._outcome = Latest()
        self._thread: threading.Thread | None = None
        self._activity = ""
        self._solving_since: float | None = None
        # Live inputs to the error budget, smoothed: how far the board is
        # from the left eye, and how noisy the stripe's centroid is across
        # the stripe, from the calibration-mode line fits.
        self._z_mm: float | None = None
        self._stripe_px: float | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        title = QLabel("CALIBRATION")
        title.setStyleSheet(
            "color:#7cc4ff; font-weight:600; letter-spacing:2px; font-size:12px;")
        root.addWidget(title)

        # The one number: what a scanned point is expected to be off by,
        # with this calibration, at the distance the board is now.
        self.error_big = QLabel("—")
        self.error_big.setStyleSheet(
            "color:#8b9aac; font-family:Consolas; font-size:30px; font-weight:600;")
        self.error_big.setToolTip(
            "Expected one-sigma error of a scanned point, in mm, from this "
            "calibration: the stripe centroid's noise through the laser sheet, "
            "the sheet's own residual, the focal length's uncertainty over the "
            "scan volume, and the rolling shutter while it is unmeasured. Added "
            "in quadrature. Keep calibrating and watch it fall."
        )
        root.addWidget(self.error_big)
        self.error_terms = QLabel("expected point error — needs the left camera "
                                  "matrix and the laser sheet")
        self.error_terms.setWordWrap(True)
        self.error_terms.setStyleSheet(
            "color:#8b9aac; font-family:Consolas; font-size:11px;")
        root.addWidget(self.error_terms)

        self.auto = QCheckBox("calibrate continuously")
        self.auto.setToolTip(
            "Move the board about. A still board showing something new becomes a "
            "view (both eyes at once: a pair); the stripe straight across a still "
            "board feeds the laser plane; a brisk twist feeds the readout time. "
            "Everything is solved again in the background as the sets grow, from "
            "the raw corners through the best intrinsics so far."
        )
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._on_auto)
        root.addWidget(self.auto)

        self.autosave = QCheckBox("save improvements to the server")
        self.autosave.setToolTip(
            "A solve replaces what the server holds only when it has more data "
            "at a residual no worse than a little, or a lower residual."
        )
        self.autosave.setChecked(True)
        root.addWidget(self.autosave)

        row = QHBoxLayout()
        self.btn_capture = QPushButton("Capture")
        self.btn_capture.setToolTip("Take whatever both eyes see now, still or not.")
        self.btn_capture.clicked.connect(self._capture)
        self.btn_solve = QPushButton("Solve now")
        self.btn_solve.clicked.connect(self._solve_now)
        self.btn_save = QPushButton("Save now")
        self.btn_save.setToolTip("Send every current solve to the server, better or not.")
        self.btn_save.clicked.connect(self._save_all)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear)
        for b in (self.btn_capture, self.btn_solve, self.btn_save, self.btn_clear):
            row.addWidget(b)
        root.addLayout(row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        self._stat_labels: dict[str, QLabel] = {}
        for r, key in enumerate(("views", "paired", "tilt", "novelty",
                                 "laser pts", "motion")):
            name = QLabel(key)
            name.setStyleSheet("color:#8b9aac; font-size:11px;")
            val = QLabel("—")
            val.setStyleSheet("color:#dde5ee; font-family:Consolas; font-size:11px;")
            grid.addWidget(name, r, 0)
            grid.addWidget(val, r, 1)
            self._stat_labels[key] = val
        root.addLayout(grid)

        cov = QHBoxLayout()
        self.coverage = {"left": CoverageMap(), "right": CoverageMap()}
        for side in ("left", "right"):
            box = QVBoxLayout()
            lab = QLabel(side)
            lab.setStyleSheet("color:#8b9aac; font-size:10px;")
            lab.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            box.addWidget(lab)
            box.addWidget(self.coverage[side])
            cov.addLayout(box)
        root.addLayout(cov)

        self.scoreboard = QLabel("—")
        self.scoreboard.setStyleSheet(
            "color:#dde5ee; font-family:Consolas; font-size:11px;")
        root.addWidget(self.scoreboard)

        self.advice = QLabel("waiting for the board")
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet(
            "color:#ffd166; font-family:Consolas; font-size:11px;")
        root.addWidget(self.advice)

        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setStyleSheet(
            "color:#8b9aac; font-family:Consolas; font-size:11px;")
        root.addWidget(self.report)
        root.addStretch(1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_TICK_MS)

    # ── config ────────────────────────────────────────────────────────────

    def set_laser_active(self, on: bool) -> None:
        self.flow.laser_active = on

    def set_config(self, cfg: RigConfig) -> None:
        """Adopt the server's board and whatever it already holds: intrinsics
        to solve the rest through, and the strength of each stored figure, so
        a challenger has to beat it."""
        flow = self.flow
        board = build_board(cfg.board) if cfg.board is not None else None
        if flow.set_board(cfg.board, board):
            self.report.setText("board spec changed — captured sets discarded")
            self._persist()
        elif board is not None and not len(flow.samples) and not flow.results:
            # First board spec of the session: pick up whatever the last run
            # captured, so a restart does not cost another sweep.
            n = flow.samples.load(self._samples_path, board)
            if n:
                self.report.setText(f"loaded {n} views from the previous run "
                                    f"({len(flow.samples.paired())} paired)")
        for side in ("left", "right"):
            eye = getattr(cfg, side)
            if eye is None:
                continue
            flow.set_known_intrinsics(side, eye.intrinsics_for(None), eye.intrinsics_raw)
            r = eye.readout_raw or {}
            if isinstance(r, dict) and "seconds" in r:
                flow.set_stored(f"readout:{side}", int(r.get("views", 0) or 0),
                                float(r.get("sigma_s", float("nan")) or float("nan")))
        x = cfg.extrinsics_raw or {}
        if "R" in x:
            flow.set_stored("stereo", int(x.get("views", 0) or 0),
                            float(x.get("rms_px", float("nan")) or float("nan")))
        p = cfg.laser_plane_raw or {}
        if "n" in p:
            flow.set_stored("plane", int(p.get("frames", 0) or 0),
                            float(p.get("rms_mm", float("nan")) or float("nan")))
            if flow.stored_plane is None:
                flow.stored_plane = plane_from_config(p, None)
        self._refresh()

    # ── live feed ─────────────────────────────────────────────────────────

    def on_result(self, res: EyeResult) -> None:
        if res.side == "left":
            board = res.board
            if board is not None and board.t is not None and float(board.t[2]) > 0:
                self._z_mm = _ema(self._z_mm, float(board.t[2]))
            if res.laser is not None and res.laser.ok and np.isfinite(res.laser.rms_px):
                self._stripe_px = _ema(self._stripe_px, float(res.laser.rms_px))
        note = self.flow.offer(res, auto=self.auto.isChecked())
        if note:
            self._activity = note
            if "view" in note:
                self._persist()
        self._refresh_live(res)

    def _refresh_live(self, res: EyeResult) -> None:
        flow = self.flow
        nov = flow.samples.novelty(res.side, res.descriptor)
        self._stat_labels["novelty"].setText(
            "—" if not np.isfinite(nov) else f"{nov:.3f}")
        if not flow.laser_active:
            self._stat_labels["laser pts"].setText("detector off")
        else:
            sk = flow.plane.skipped
            self._stat_labels["laser pts"].setText(
                f"{len(flow.plane)} / {flow.plane.frames}f"
                + (f"  (moving {sk['moving']})" if sk["moving"] else ""))
        self._stat_labels["motion"].setText(
            f"L {flow.motion.count('left')}  R {flow.motion.count('right')} frames")
        self._refresh()

    def _refresh(self) -> None:
        flow = self.flow
        self._stat_labels["views"].setText(str(len(flow.samples)))
        self._stat_labels["paired"].setText(str(len(flow.samples.paired())))
        # Per eye, named. The two cameras see the board from different angles,
        # so a tilt that is rich for one can be near flat for the other.
        tl, tr = (flow.samples.tilt_spread(s) for s in ("left", "right"))
        short = [n for n, v in (("L", tl), ("R", tr)) if v < MIN_TILT_SPREAD]
        self._stat_labels["tilt"].setText(
            f"L {tl:.1f}  R {tr:.1f}  / {MIN_TILT_SPREAD:.0f}"
            + ("  ✓" if not short else f"  tilt more for {'+'.join(short)}"))
        for side in ("left", "right"):
            self.coverage[side].set_cells(flow.samples.coverage(side))
        self._refresh_error()
        self.scoreboard.setText("\n".join(flow.scoreboard()))
        self.advice.setText(flow.advice() if flow.board is not None
                            else "no board spec from the server")
        status = self._activity
        if self._solving_since is not None:
            status = f"solving… {time.monotonic() - self._solving_since:.0f} s" \
                     + (f"   ({self._activity})" if self._activity else "")
        self.report.setText(status)

    def _refresh_error(self) -> None:
        b = self.flow.expected_error(self._z_mm, self._stripe_px)
        if b is None:
            self.error_big.setText("—")
            self.error_big.setStyleSheet(
                "color:#8b9aac; font-family:Consolas; font-size:30px; font-weight:600;")
            self.error_terms.setText("expected point error — needs the left camera "
                                     "matrix and the laser sheet")
            return
        total = b.total_mm
        colour = "#5fd38d" if total <= 0.5 else "#ffd166" if total <= 1.5 else "#ff6b6b"
        self.error_big.setText(f"≈ {total:.2f} mm")
        self.error_big.setStyleSheet(
            f"color:{colour}; font-family:Consolas; font-size:30px; font-weight:600;")
        lines = [f"expected point error @ {b.z_mm:.0f} mm, stripe {b.stripe_px:.2f} px",
                 f"stripe {b.stripe_mm:.2f} · sheet {b.sheet_mm:.2f} · "
                 f"scale {b.scale_mm:.2f} · shutter {b.shutter_mm:.2f} mm"]
        if b.assumed:
            lines.append("assumed: " + ", ".join(b.assumed))
        self.error_terms.setText("\n".join(lines))

    # ── the cycle ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.monotonic()
        out = self._outcome.take(0.0)
        if out is not None:
            self._finish(out, now)
        if self.auto.isChecked() and self.flow.due(now):
            self._start_cycle(now)

    def _start_cycle(self, now: float) -> None:
        job = self.flow.snapshot(now)
        self._solving_since = now

        def work(job: Job = job) -> None:
            try:
                self._outcome.put(self.flow.run(job))
            except Exception as exc:                          # noqa: BLE001
                log.exception("calibration cycle raised")
                self._outcome.put(Outcome(reasons={"cycle": f"{exc.__class__.__name__}: {exc}"}))

        self._thread = threading.Thread(target=work, name="calib-solve", daemon=True)
        self._thread.start()
        self._refresh()

    def _finish(self, out: Outcome, now: float) -> None:
        self._solving_since = None
        payload = self.flow.finish(out, now)
        if "cycle" in out.reasons:
            self._activity = f"solve failed: {out.reasons['cycle']}"
        else:
            solved = ", ".join(k for k in out.results)
            self._activity = (f"cycle {out.seconds:.1f} s: {solved or 'nothing solved'}"
                              + (" — saving" if payload and self.autosave.isChecked() else ""))
        if payload and self.autosave.isChecked():
            self.save_requested.emit(payload)
        self._refresh()

    # ── actions ───────────────────────────────────────────────────────────

    def _on_auto(self, on: bool) -> None:
        if on:
            self.laser_requested.emit(True)
        self._refresh()

    def _capture(self) -> None:
        n = self.flow.capture()
        self._activity = f"captured view {n}" if n else "nothing to capture — no board in view"
        if n:
            self._persist()
        self._refresh()

    def _solve_now(self) -> None:
        if self.flow.running:
            self._activity = "a cycle is already running"
        elif self.flow.board is None:
            self._activity = "no board spec from the server"
        else:
            self.flow.request()
            self._start_cycle(time.monotonic())
        self._refresh()

    def _save_all(self) -> None:
        payload = self.flow.payload_current()
        if not payload:
            self._activity = "nothing solved yet"
        else:
            self.save_requested.emit(payload)
            self._activity = "sent every current solve to the server"
        self._refresh()

    def _clear(self) -> None:
        self.flow.clear()
        self._persist()
        self._activity = "cleared"
        self._refresh()

    def _persist(self) -> None:
        """Keep the captured views on disk. A failure here must not stop capture."""
        try:
            self.flow.samples.save(self._samples_path)
        except OSError as exc:
            log.warning("could not persist captured views: %s", exc)
