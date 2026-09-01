"""Scan controls: run the triangulation, show what it kept and what it dropped.

The rejection counters are the useful part of this panel. A scan that produces
nothing looks identical whether the board is out of view, the laser is off, the
subject is outside the box or the two eyes are looking at different stripes —
and each of those calls for a different fix. So they are counted separately and
shown while scanning, not summarised afterwards.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .scan import PointCloud, ScanFrame, ScanParams, ScanVolume

log = logging.getLogger("orbiter_native.scanpanel")


class ScanPanel(QFrame):
    """Toggle scanning, watch the counters, export the cloud."""

    #: Emitted when scanning is switched on or off, so the window can make sure
    #: the laser detector is running — scanning without it finds nothing.
    active_changed = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")   # see the stylesheet in __main__
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.cloud = PointCloud()
        self._last: ScanFrame | None = None
        self._frames = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        title = QLabel("SCAN")
        title.setStyleSheet(
            "color:#7cc4ff; font-weight:600; letter-spacing:2px; font-size:12px;")
        root.addWidget(title)

        self.active = QCheckBox("scanning")
        self.active.setToolTip(
            "Triangulate the laser stripe from both eyes and keep the points "
            "inside the box above the board. Needs the pair calibration, a "
            "visible board and the laser detector switched on."
        )
        self.active.toggled.connect(self.active_changed.emit)
        root.addWidget(self.active)

        box = QGridLayout()
        box.setHorizontalSpacing(8)
        box.addWidget(QLabel("box height"), 0, 0)
        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(10.0, 2000.0)
        self.height_mm.setValue(ScanVolume().height_mm)
        self.height_mm.setSuffix(" mm")
        box.addWidget(self.height_mm, 0, 1)
        box.addWidget(QLabel("half width"), 1, 0)
        self.half_mm = QDoubleSpinBox()
        self.half_mm.setRange(10.0, 2000.0)
        self.half_mm.setValue(ScanVolume().half_width_mm)
        self.half_mm.setSuffix(" mm")
        box.addWidget(self.half_mm, 1, 1)
        for w in (self.height_mm, self.half_mm):
            w.setToolTip(
                "The volume above the board that scanning keeps, in the BOARD's "
                "own frame — so it stays put as the board moves, and the bench, "
                "your hands and the far wall fall outside it without needing to "
                "be recognised."
            )
        root.addLayout(box)

        row = QHBoxLayout()
        self.btn_clear = QPushButton("Clear cloud")
        self.btn_clear.clicked.connect(self._clear)
        self.btn_export = QPushButton("Export PLY")
        self.btn_export.clicked.connect(self._export)
        row.addWidget(self.btn_clear)
        row.addWidget(self.btn_export)
        root.addLayout(row)

        self.stats = QLabel("idle")
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet(
            "color:#8b9aac; font-family:Consolas; font-size:11px;")
        self.stats.setAlignment(Qt.AlignmentFlag.AlignTop)
        root.addWidget(self.stats)
        root.addStretch(1)

    # ── configuration ─────────────────────────────────────────────────────

    def params(self) -> ScanParams:
        return ScanParams(volume=ScanVolume(height_mm=self.height_mm.value(),
                                            half_width_mm=self.half_mm.value()))

    @property
    def scanning(self) -> bool:
        return self.active.isChecked()

    # ── live ──────────────────────────────────────────────────────────────

    def on_frame(self, frame: ScanFrame) -> None:
        """Accumulate one frame pair's contribution and refresh the counters."""
        self._last = frame
        self._frames += 1
        if frame.n_kept:
            self.cloud.add(frame.points_board)
        self._refresh()

    def _refresh(self) -> None:
        f = self._last
        lines = [f"cloud   {len(self.cloud)} points"]
        b = self.cloud.bounds()
        if b is not None:
            lo, hi = b
            lines.append(f"extent  x {lo[0]:+.0f}..{hi[0]:+.0f}  "
                         f"y {lo[1]:+.0f}..{hi[1]:+.0f}  z {lo[2]:+.0f}..{hi[2]:+.0f} mm")
        if f is None:
            self.stats.setText("\n".join(lines))
            return
        if f.reason:
            lines.append(f"frame   — {f.reason}")
        else:
            lines.append(f"frame   {f.n_kept}/{f.n_candidates} kept")
            lines.append(f"no match {f.n_rejected_nomatch} · "
                         f"ambiguous {f.n_rejected_ambiguous}")
            lines.append(f"shallow  {f.n_rejected_geometry} · "
                         f"outside box {f.n_rejected_volume}")
            if f.n_kept:
                lines.append(f"reproj  med {np.median(f.reproj_px):.2f} px")
        self.stats.setText("\n".join(lines))

    def note(self, message: str) -> None:
        """Show a blocking condition — no calibration, no board, laser off."""
        self.stats.setText(f"cloud   {len(self.cloud)} points\n{message}")

    # ── actions ───────────────────────────────────────────────────────────

    def _clear(self) -> None:
        self.cloud.clear()
        self._last = None
        self._frames = 0
        self._refresh()

    def _export(self) -> None:
        if not len(self.cloud):
            self.stats.setText("nothing to export yet")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export point cloud", "scan.ply", "PLY (*.ply)")
        if not path:
            return
        try:
            n = self.cloud.write_ply(path)
        except OSError as exc:
            self.stats.setText(f"could not write {path}: {exc}")
            return
        self.stats.setText(f"wrote {n} points to {path}")
