"""Calibration capture: collect ChArUco views, solve per-eye intrinsics, store.

The operator moves the board; this panel decides which of those views are worth
keeping, says out loud when the set is still degenerate, and refuses to solve
from one that is.

The guidance is the point. `calibrateCamera` returns a confident answer from a
set of head-on views and that answer can be 12% wrong in focal length with a
reprojection RMS indistinguishable from a good solve — measured, see
`intrinsics.MIN_TILT_SPREAD`. Nothing in the result reveals it. So the tilt
spread and the frame coverage are shown live, while the operator can still act
on them, rather than reported afterwards when the only remedy is to start over.

Views are captured in PAIRS, matched on camserver's capture clock. Both cameras
are timed by that same clock on that machine, so it is what says two frames are
simultaneous — our own arrival times came through two different sockets and are
not comparable. Intrinsics do not need the pairing; `stereoCalibrate` does, and
it would be a waste of the operator's time to sweep the board twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, Signal
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

from .cvcore import build_board
from .intrinsics import (
    MIN_TILT_SPREAD,
    MIN_VIEWS,
    EyeView,
    PairSample,
    SampleSet,
    SolveResult,
)
from .intrinsics import solve as solve_intrinsics
from .stereo import StereoResult
from .stereo import calibrate as solve_stereo
from .worker import EyeResult

log = logging.getLogger("orbiter_native.calibpanel")

#: Two frames count as simultaneous within this much of camserver's clock. The
#: pair is held to about 0.05 ms by camserver itself, so this is generous; it
#: only has to be tighter than the ~33 ms frame interval.
_PAIR_WINDOW_S = 0.010

#: The board must be this still before a view is captured, measured as median
#: corner movement between consecutive detections. Motion blur rounds corners
#: off and quietly biases the solve.
_STILL_PX = 1.0


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


@dataclass
class _Pending:
    """The newest result from one eye, waiting for its partner."""

    result: EyeResult
    prev_corners: np.ndarray | None = None


class CalibrationPanel(QFrame):
    """Capture controls, live diversity guidance, and the solve."""

    #: Emitted with `{side: intrinsics-dict}` when the operator saves a solve.
    save_requested = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.samples = SampleSet()
        self._board = None
        self._board_spec = None
        self._pending: dict[str, _Pending] = {}
        self._last_corners: dict[str, np.ndarray] = {}
        self._results: dict[str, SolveResult] = {}
        self._stereo: StereoResult | None = None
        #: Intrinsics currently known for each eye — from a fresh solve here,
        #: or from the server when a previous run already stored them.
        self._known_k: dict[str, object] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        title = QLabel("CALIBRATION")
        title.setStyleSheet(
            "color:#7cc4ff; font-weight:600; letter-spacing:2px; font-size:12px;")
        root.addWidget(title)

        self.auto = QCheckBox("auto-capture new views")
        self.auto.setToolTip(
            "Capture whenever the board is still and shows the solve something "
            "it has not seen — a new place in the frame, a new distance or a "
            "new tilt. Near-duplicates are skipped: twenty of the same view are "
            "worth less than six different ones."
        )
        self.auto.setChecked(True)
        root.addWidget(self.auto)

        row = QHBoxLayout()
        self.btn_capture = QPushButton("Capture")
        self.btn_capture.clicked.connect(lambda: self._capture(force=True))
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear)
        row.addWidget(self.btn_capture)
        row.addWidget(self.btn_clear)
        root.addLayout(row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        self._stat_labels: dict[str, QLabel] = {}
        for r, key in enumerate(("views", "paired", "tilt", "novelty")):
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

        self.btn_solve = QPushButton("Solve intrinsics")
        self.btn_solve.clicked.connect(self._solve)
        root.addWidget(self.btn_solve)

        self.btn_stereo = QPushButton("Solve stereo pair")
        self.btn_stereo.setToolTip(
            "Needs intrinsics for both eyes and views where BOTH saw the board "
            "at the same instant. Gives the real baseline and the geometry "
            "triangulation runs on."
        )
        self.btn_stereo.clicked.connect(self._solve_stereo)
        root.addWidget(self.btn_stereo)

        self.btn_save = QPushButton("Save to server")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        root.addWidget(self.btn_save)

        self.report = QLabel("move the board around, tilting it between shots")
        self.report.setWordWrap(True)
        self.report.setStyleSheet(
            "color:#8b9aac; font-family:Consolas; font-size:11px;")
        root.addWidget(self.report)
        root.addStretch(1)

    # ── config ────────────────────────────────────────────────────────────

    def set_intrinsics(self, side: str, k) -> None:
        """Adopt intrinsics the server already holds, so the stereo solve does
        not require re-running the per-eye one in this session."""
        if k is not None:
            self._known_k.setdefault(side, k)

    def set_board_spec(self, spec) -> None:
        """Adopt the board spec from the server. Captured views are dropped on
        a change: they were measured against a different board and mixing the
        two would solve for a geometry that never existed."""
        if spec == self._board_spec:
            return
        self._board_spec = spec
        self._board = build_board(spec) if spec is not None else None
        if self.samples:
            self._clear()
            self.report.setText("board spec changed — captured views discarded")

    # ── live feed ─────────────────────────────────────────────────────────

    def on_result(self, res: EyeResult) -> None:
        """Take one eye's newest result and, when both eyes line up, capture."""
        if res.descriptor is not None or res.board is not None:
            self._pending[res.side] = _Pending(res)
        self._refresh_live(res)
        if self.auto.isChecked():
            self._capture(force=False)

    def _still(self, res: EyeResult) -> bool:
        """True when the board has barely moved since the previous detection."""
        if res.board is None or res.board.corners is None:
            return False
        cur = res.board.corners.reshape(-1, 2)
        prev = self._last_corners.get(res.side)
        self._last_corners[res.side] = cur
        if prev is None or len(prev) != len(cur):
            return False
        return float(np.median(np.linalg.norm(cur - prev, axis=1))) <= _STILL_PX

    def _refresh_live(self, res: EyeResult) -> None:
        left = self._pending.get("left")
        right = self._pending.get("right")
        nov = max(
            self.samples.novelty("left", left.result.descriptor if left else None),
            self.samples.novelty("right", right.result.descriptor if right else None),
        )
        self._stat_labels["novelty"].setText(
            "—" if not np.isfinite(nov) else f"{nov:.3f}")
        self._stat_labels["views"].setText(str(len(self.samples)))
        self._stat_labels["paired"].setText(str(len(self.samples.paired())))
        spreads = [self.samples.tilt_spread(s) for s in ("left", "right")]
        worst = min(spreads)
        self._stat_labels["tilt"].setText(
            f"{worst:.1f} / {MIN_TILT_SPREAD:.0f}"
            + ("  ✓" if worst >= MIN_TILT_SPREAD else "  tilt more"))
        for side in ("left", "right"):
            self.coverage[side].set_cells(self.samples.coverage(side))

    # ── capture ───────────────────────────────────────────────────────────

    def _capture(self, force: bool) -> None:
        left = self._pending.get("left")
        right = self._pending.get("right")
        if self._board is None or (left is None and right is None):
            return

        # Simultaneity is judged on camserver's clock, which times both
        # cameras. Frames from two sockets have unrelated arrival times.
        lm = left.result.capture_mono if left else None
        rm = right.result.capture_mono if right else None
        if lm is not None and rm is not None and abs(lm - rm) > _PAIR_WINDOW_S:
            return

        views: dict[str, EyeView | None] = {"left": None, "right": None}
        for side, pend in (("left", left), ("right", right)):
            if pend is None:
                continue
            r = pend.result
            if r.board is None or r.board.corners is None or r.descriptor is None:
                continue
            if not force and not self._still(r):
                return
            views[side] = EyeView(r.board.corners, r.board.ids, r.wh, r.descriptor)

        if views["left"] is None and views["right"] is None:
            return
        if not force and not self.samples.is_new(
            views["left"].descriptor if views["left"] else None,
            views["right"].descriptor if views["right"] else None,
        ):
            return

        self.samples.add(PairSample(left=views["left"], right=views["right"]))
        self._pending.clear()
        self.report.setText(f"captured view {len(self.samples)}")

    def _clear(self) -> None:
        self.samples.clear()
        self._results.clear()
        self.btn_save.setEnabled(False)
        self.report.setText("cleared")

    # ── solve ─────────────────────────────────────────────────────────────

    def _solve(self) -> None:
        if self._board is None:
            self.report.setText("no board spec from the server")
            return
        self._results.clear()
        lines: list[str] = []
        for side in ("left", "right"):
            views = self.samples.views(side)
            if len(views) < MIN_VIEWS:
                lines.append(f"{side}: {len(views)} views, need {MIN_VIEWS}")
                continue
            res, why = solve_intrinsics(views, self._board,
                                        tilt_spread=self.samples.tilt_spread(side))
            if res is None:
                lines.append(f"{side}: {why}")
                continue
            self._results[side] = res
            self._known_k[side] = res.intrinsics
            i = res.intrinsics
            lines.append(
                f"{side}: fx {i.fx:.1f} fy {i.fy:.1f} cx {i.cx:.1f} cy {i.cy:.1f}\n"
                f"  rms {res.rms_px:.3f} px over {res.n_views} views "
                f"@ {res.wh[0]}x{res.wh[1]}\n"
                f"  k1 {i.dist[0]:+.4f} k2 {i.dist[1]:+.4f}"
            )
        self.report.setText("\n".join(lines) or "nothing to solve")
        self.btn_save.setEnabled(bool(self._results))

    def _solve_stereo(self) -> None:
        """Solve the pair geometry from the simultaneous views."""
        if self._board is None:
            self.report.setText("no board spec from the server")
            return
        missing = [s for s in ("left", "right") if s not in self._known_k]
        if missing:
            self.report.setText(
                f"stereo needs intrinsics for {', '.join(missing)} — solve those "
                "first, or load a calibration the server already has")
            return
        pairs = self.samples.paired()
        if not pairs:
            self.report.setText(
                "no views where both eyes saw the board at the same instant")
            return
        wh = pairs[0].left.wh
        res, why = solve_stereo(pairs, self._board,
                                self._known_k["left"], self._known_k["right"], wh)
        if res is None:
            self._stereo = None
            self.report.setText(f"stereo: {why}")
            return
        self._stereo = res
        self.report.setText(
            f"stereo: baseline {res.baseline_mm:.1f} mm  rms {res.rms_px:.3f} px\n"
            f"  over {res.n_views} simultaneous views @ {res.wh[0]}x{res.wh[1]}"
        )
        self.btn_save.setEnabled(True)

    def _save(self) -> None:
        if not self._results and self._stereo is None:
            return
        payload: dict = {side: res.as_config() for side, res in self._results.items()}
        if self._stereo is not None:
            payload["_extrinsics"] = self._stereo.as_config()
        self.save_requested.emit(payload)
        self.report.setText(
            self.report.text() + "\n\nsent to server — the panels will show "
            "board pose once it comes back")
