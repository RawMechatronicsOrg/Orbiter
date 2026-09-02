"""Scan controls: switch scanning on, watch what it keeps and drops, export.

The rejection counters are the useful part of this panel. A scan that produces
nothing looks identical whether the board is out of view, the laser is off, the
subject is outside the box or the two eyes are looking at different stripes —
and each of those calls for a different fix. So they are counted separately and
shown while scanning, not summarised afterwards.

The panel owns no data. The cloud lives in `ScanWorker`, on its own thread;
this widget pushes settings down and shows the status that comes back up.
"""

from __future__ import annotations

import logging

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

from .scan import ScanParams, ScanVolume
from .scanworker import ScanStatus, ScanWorker

log = logging.getLogger("orbiter_native.scanpanel")


class ScanPanel(QFrame):
    """Toggle scanning, watch the counters, export the cloud."""

    #: Emitted when scanning is switched on or off, so the window can make sure
    #: the laser detector is running — scanning without it finds nothing.
    active_changed = Signal(bool)

    def __init__(self, scanner: ScanWorker, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")   # see the stylesheet in __main__
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._scanner = scanner
        self._status: ScanStatus | None = None

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
            "visible board and the laser detector switched on. The cloud is "
            "drawn over both eyes in orange."
        )
        self.active.toggled.connect(self._toggle)
        root.addWidget(self.active)

        box = QGridLayout()
        box.setHorizontalSpacing(8)
        box.addWidget(QLabel("box height"), 0, 0)
        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(10.0, 2000.0)
        self.height_mm.setValue(ScanVolume().height_mm)
        self.height_mm.setSuffix(" mm")
        box.addWidget(self.height_mm, 0, 1)
        box.addWidget(QLabel("radius"), 1, 0)
        self.radius_mm = QDoubleSpinBox()
        self.radius_mm.setRange(10.0, 2000.0)
        self.radius_mm.setValue(ScanVolume().radius_mm)
        self.radius_mm.setSuffix(" mm")
        box.addWidget(self.radius_mm, 1, 1)
        box.addWidget(QLabel("floor"), 2, 0)
        self.floor_mm = QDoubleSpinBox()
        self.floor_mm.setRange(0.0, 100.0)
        self.floor_mm.setValue(ScanVolume().floor_mm)
        self.floor_mm.setSuffix(" mm")
        self.floor_mm.setToolTip(
            "Points closer to the board than this are the board's own surface. "
            "Plane-based points carry about 1 mm of noise at half a metre."
        )
        box.addWidget(self.floor_mm, 2, 1)
        for w in (self.height_mm, self.radius_mm):
            w.setToolTip(
                "A cylinder standing on the board — the volume scanning keeps, "
                "in the BOARD's own frame, so it stays put as the board moves. "
                "The bench, your hands and the wall behind fall outside it "
                "without needing to be recognised. The board is a disc, so the "
                "radius is naturally its own: 144 mm on this rig."
            )
        for w in (self.height_mm, self.radius_mm, self.floor_mm):
            w.valueChanged.connect(self._push_params)
        root.addLayout(box)

        row = QHBoxLayout()
        self.btn_clear = QPushButton("Clear cloud")
        self.btn_clear.clicked.connect(self._scanner.clear)
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
                                            radius_mm=self.radius_mm.value(),
                                            floor_mm=self.floor_mm.value()))

    def _push_params(self, _value=None) -> None:
        self._scanner.set_params(self.params())

    def _toggle(self, on: bool) -> None:
        self._push_params()
        self._scanner.set_active(on)
        self.active_changed.emit(on)

    @property
    def scanning(self) -> bool:
        return self.active.isChecked()

    # ── live ──────────────────────────────────────────────────────────────

    def on_status(self, status: ScanStatus) -> None:
        self._status = status
        self._refresh()

    def _refresh(self) -> None:
        st = self._status
        if st is None:
            self.stats.setText("idle")
            return
        lines = [f"cloud   {st.n_points} points · {st.pairs} pairs"]
        if st.bounds is not None:
            lo, hi = st.bounds
            lines.append(f"extent  x {lo[0]:+.0f}..{hi[0]:+.0f}  "
                         f"y {lo[1]:+.0f}..{hi[1]:+.0f}  z {lo[2]:+.0f}..{hi[2]:+.0f} mm")
        f = st.frame
        if st.note:
            lines.append(f"frame   — {st.note}")
        elif f is not None and f.reason:
            lines.append(f"frame   — {f.reason}")
        elif f is not None:
            lines.append(f"frame   {f.n_kept}/{f.n_scanlines} scanlines kept")
            lines.append(f"pixels  {f.n_confirmed}/{f.n_pixels} confirmed by the right eye")
            lines.append(f"unconfirmed {f.n_rejected_unconfirmed} · "
                         f"outside {f.n_rejected_volume}")
        self.stats.setText("\n".join(lines))

    # ── actions ───────────────────────────────────────────────────────────

    def _export(self) -> None:
        if self._status is None or not self._status.n_points:
            self.stats.setText("nothing to export yet")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export point cloud", "scan.ply", "PLY (*.ply)")
        if not path:
            return
        try:
            n = self._scanner.export(path)
        except OSError as exc:
            self.stats.setText(f"could not write {path}: {exc}")
            return
        self.stats.setText(f"wrote {n} points to {path}")
