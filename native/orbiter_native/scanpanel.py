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
        box.addWidget(QLabel("reach from"), 3, 0)
        self.reach_lo = QDoubleSpinBox()
        self.reach_lo.setRange(0.0, 5000.0)
        self.reach_lo.setValue(ScanParams().range_mm[0])
        self.reach_lo.setSuffix(" mm")
        box.addWidget(self.reach_lo, 3, 1)
        box.addWidget(QLabel("reach to"), 4, 0)
        self.reach_hi = QDoubleSpinBox()
        self.reach_hi.setRange(0.0, 5000.0)
        self.reach_hi.setValue(ScanParams().range_mm[1])
        self.reach_hi.setSuffix(" mm")
        box.addWidget(self.reach_hi, 4, 1)
        for w in (self.reach_lo, self.reach_hi):
            w.setToolTip(
                "The scanner's working range: a point must lie this far from the "
                "line through the two camera centres. Nearer is the rig itself or "
                "a hand, farther is the wall; neither has to be recognised. Holds "
                "without a board pose, unlike the cylinder."
            )
        for w in (self.height_mm, self.radius_mm, self.floor_mm,
                  self.reach_lo, self.reach_hi):
            w.valueChanged.connect(self._push_params)
        root.addLayout(box)

        clean_row = QHBoxLayout()
        self.clean = QCheckBox("clean, merge to")
        self.clean.setToolTip(
            "Show and export the confident cloud rather than every voxel: a voxel "
            "with fewer than three neighbours within 2 mm is a glint or a hand; "
            "a voxel seen once where its neighbours were seen three times or "
            "more is a flicker the later passes never confirmed. What survives "
            "is merged on cells of the size beside, each voxel weighted by its "
            "precision — a point seen from close outweighs one seen from far."
        )
        self.clean.setChecked(ScanParams().clean)
        self.clean.toggled.connect(self._push_params)
        clean_row.addWidget(self.clean)
        self.merge_mm = QDoubleSpinBox()
        self.merge_mm.setRange(0.5, 5.0)
        self.merge_mm.setSingleStep(0.5)
        self.merge_mm.setValue(ScanParams().clean_merge_mm)
        self.merge_mm.setSuffix(" mm")
        self.merge_mm.setToolTip("Cell size of the confident cloud. Larger is quieter and "
                                 "loses detail; the point noise is about 1 mm at 400 mm.")
        self.merge_mm.valueChanged.connect(self._push_params)
        clean_row.addWidget(self.merge_mm)
        clean_row.addStretch(1)
        root.addLayout(clean_row)

        refine_row = QHBoxLayout()
        self.refine = QCheckBox("stereo refine, pair ≤")
        self.refine.setToolTip(
            "Fuse each point's sheet depth with the depth the right eye's own stripe "
            "centroid implies: the pair's baseline is longer than the laser's offset, "
            "so the right eye reads depth finer — as long as the pair is calibrated. "
            "Off above the pair residual beside: a poor pair would bend the surface "
            "rather than sharpen it."
        )
        self.refine.setChecked(ScanParams().stereo_refine)
        self.refine.toggled.connect(self._push_params)
        refine_row.addWidget(self.refine)
        self.refine_rms = QDoubleSpinBox()
        self.refine_rms.setRange(0.2, 5.0)
        self.refine_rms.setSingleStep(0.1)
        self.refine_rms.setValue(ScanParams().stereo_refine_max_rms_px)
        self.refine_rms.setSuffix(" px")
        self.refine_rms.setToolTip("The pair's reprojection residual above which the right "
                                   "eye is not trusted with depth.")
        self.refine_rms.valueChanged.connect(self._push_params)
        refine_row.addWidget(self.refine_rms)
        refine_row.addStretch(1)
        root.addLayout(refine_row)

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
        lo, hi = sorted((self.reach_lo.value(), self.reach_hi.value()))
        return ScanParams(range_mm=(lo, hi), clean=self.clean.isChecked(),
                          clean_merge_mm=self.merge_mm.value(),
                          stereo_refine=self.refine.isChecked(),
                          stereo_refine_max_rms_px=self.refine_rms.value(),
                          volume=ScanVolume(height_mm=self.height_mm.value(),
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
        rate = (f" of {st.offered_left} left frames ({100 * st.pairs / st.offered_left:.0f}%)"
                if st.offered_left else "")
        lines = [f"cloud   {st.n_points} points · {st.pairs} pairs{rate}"
                 + (f" · still ×{st.batched}" if st.batched else "")]
        if st.bounds is not None:
            lo, hi = st.bounds
            lines.append(f"extent  x {lo[0]:+.0f}..{hi[0]:+.0f}  "
                         f"y {lo[1]:+.0f}..{hi[1]:+.0f}  z {lo[2]:+.0f}..{hi[2]:+.0f} mm")
        if st.n_confident >= 0:
            lines.append(f"clean   {st.n_confident} confident · dropped {st.n_lonely} lonely, "
                         f"{st.n_flicker} unconfirmed")
        f = st.frame
        if st.note:
            lines.append(f"frame   — {st.note}")
        elif f is not None and f.reason:
            lines.append(f"frame   — {f.reason}")
        elif f is not None:
            lines.append(f"frame   {f.n_kept}/{f.n_scanlines} scanlines kept")
            lines.append(f"pixels  {f.n_confirmed}/{f.n_pixels} confirmed by the right eye")
            if f.pose_gap_deg == f.pose_gap_deg:              # both eyes: not NaN
                fit = (f"joint fit {f.pose_rms_px:.2f} px" if f.pose_rms_px == f.pose_rms_px
                       else "mean of the two")
                lines.append(f"pose    both eyes · their poses differ by {f.pose_gap_deg:.2f}° / "
                             f"{f.pose_gap_mm:.1f} mm · {fit}")
            elif f.pose_source:
                lines.append(f"pose    {f.pose_source} eye only")
            if f.sync_note:
                lines.append(f"sync    {f.sync_note}")
            if f.veto_px == f.veto_px:            # not NaN
                lines.append(f"veto    the eyes disagree by {f.veto_px:+.1f} px "
                             f"about where the stripe is")
            if f.refine_note:
                lines.append(f"refine  — {f.refine_note}")
            elif f.n_refined:
                lines.append(f"refine  {f.n_refined}/{f.n_kept} points · median shift "
                             f"{f.refine_shift_mm:.2f} mm · right eye {100 * f.refine_share:.0f} %")
            lines.append(f"dropped unconfirmed {f.n_rejected_unconfirmed} · blob "
                         f"{f.n_rejected_blob} · reach {f.n_rejected_range} · "
                         f"jump {f.n_rejected_jump} · outside {f.n_rejected_volume}"
                         + (f" · split {f.n_split}" if f.n_split else ""))
            if f.rs_note:
                lines.append(f"rolling — {f.rs_note}")
            else:
                lines.append(f"rolling ≤ {f.rs_max_mm:.2f} mm corrected · board "
                             f"{f.speed_mm_s:.0f} mm/s {f.spin_deg_s:.0f}°/s")
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
