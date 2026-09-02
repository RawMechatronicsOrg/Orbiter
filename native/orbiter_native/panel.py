"""One eye's panel: the live view, its overlay, and a read-only config line.

Read-only on purpose. Which camera is which eye, and how each is oriented, are
set in the web Stereo tab and stored in `model.stereo_rig`; that is the run's
baseline and there should be exactly one place to change it. This app consumes
it — two editors for one setting is how a rig ends up configured differently
depending on which window you looked at last.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from .config import Eye
from .glview import FrameView, Scene
from .scan import CloudOverlay
from .worker import EyeResult, EyeStats


def _fmt(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


class EyePanel(QFrame):
    """Live view plus stats for one side of the pair."""

    def __init__(self, side: str, parent=None) -> None:
        super().__init__(parent)
        self.side = side
        self._eye: Eye | None = None
        self._cloud: CloudOverlay | None = None
        self._scanning = False

        self.setObjectName("panel")   # see the stylesheet in __main__
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(5)

        self.title = QLabel(side.upper())
        self.title.setStyleSheet(
            "color:#7cc4ff; font-weight:600; letter-spacing:2px; font-size:12px;")
        self.subtitle = QLabel("not configured")
        self.subtitle.setStyleSheet("color:#8b9aac; font-family:Consolas; font-size:11px;")
        self.subtitle.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.view = FrameView()

        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.view, 1)

    # ── config ────────────────────────────────────────────────────────────

    def set_eye(self, eye: Eye | None) -> None:
        self._eye = eye
        if eye is None or not eye.configured:
            self.subtitle.setText("no camera assigned — set it in the web Stereo tab")
            self.view.clear_frame("no camera assigned to this eye")
            return
        o = eye.orientation
        bits = [eye.camera_id]
        if o.quarter_turns_cw:
            bits.append(f"rot {o.quarter_turns_cw * 90}°cw")
        if o.flip_h:
            bits.append("flip H")
        if o.flip_v:
            bits.append("flip V")
        if eye.has_intrinsics:
            k = eye.intrinsics_raw or {}
            bits.append(f"K {k.get('width')}x{k.get('height')} "
                        f"rms {k.get('rms_px', float('nan')):.2f}px")
        else:
            bits.append("no intrinsics")
        self.subtitle.setText(" · ".join(bits))

    # ── live updates ──────────────────────────────────────────────────────

    def set_overlay(self, overlay: CloudOverlay | None) -> None:
        """The scanned cloud, drawn over the frame while scanning."""
        self._cloud = overlay

    def set_scanning(self, on: bool) -> None:
        self._scanning = on

    def on_result(self, res: EyeResult) -> None:
        self.view.set_scene(self._scene(res))
        self.view.set_overlay(self._overlay_lines(res))

    def _scene(self, res: EyeResult) -> Scene:
        """The worker's result as the view draws it — every coordinate still
        original; the view orients. The cloud is projected through THIS
        eye's own board pose, so the two overlays disagreeing is itself a
        sign that the board poses do."""
        board = res.board
        stripe = None
        if res.stripe is not None and res.stripe.count:
            stripe = np.stack([res.stripe.x, res.stripe.y], axis=1).astype(np.float32)
        laser = res.laser if res.laser is not None and res.laser.points.size else None
        cloud = (self._cloud.points()
                 if self._scanning and self._cloud is not None else None)
        k = res.intrinsics
        return Scene(
            bgr=res.bgr, orientation=res.orientation, stripe=stripe,
            corners=None if board is None or board.corners is None
            else board.corners.reshape(-1, 2),
            ids=None if board is None or board.ids is None else board.ids.reshape(-1),
            laser=laser,
            hull=None if res.hull is None else res.hull.reshape(-1, 2),
            cloud=cloud,
            R=None if board is None else board.R,
            t=None if board is None else board.t,
            K=None if k is None else k.K,
            D=None if k is None else k.D,
        )

    def on_status(self, error: str | None) -> None:
        if error:
            self.view.clear_frame(f"offline — {error}")

    def _overlay_lines(self, res: EyeResult) -> list[str]:
        s: EyeStats = res.stats
        lines = [
            f"recv   {_fmt(s.recv_fps, 2)} fps" + ("   gpu" if s.gpu else ""),
            f"detect {_fmt(s.detect_fps, 2)} fps   {_fmt(s.detect_ms)} ms",
        ]
        if s.server_age_ms is not None:
            # camserver's own age at send time. This app does not estimate the
            # clock offset between the two machines, so it reports the number
            # camserver vouches for rather than inventing an end-to-end figure
            # from two unsynchronised clocks.
            lines.append(f"age    {_fmt(s.server_age_ms)} ms (server)")
        if s.corners:
            lines.append(f"board  {s.corners} corners · cover {s.coverage * 100:.0f}% · "
                         f"{'tracked' if s.tracked else 'detected'}")
        else:
            lines.append("board  —")
        if res.board is not None and res.board.t is not None:
            t = res.board.t
            lines.append(f"pose   {t[0]:+.0f} {t[1]:+.0f} {t[2]:+.0f} mm")
        elif s.corners and self._eye is not None and not self._eye.has_intrinsics:
            # Say why, rather than leaving a blank the operator has to guess at.
            lines.append("pose   needs per-eye intrinsics")
        lines += self._laser_lines(s)
        return lines

    @staticmethod
    def _laser_lines(s: EyeStats) -> list[str]:
        """The laser fit, or why this frame is not a usable calibration sample.

        The reason is shown rather than a blank line: "no stripe above the
        redness threshold" and "board not detected" call for opposite fixes,
        and an operator staring at an empty row cannot tell them apart.
        """
        if s.laser_reason is not None:
            return [f"laser  — {s.laser_reason}"]
        if not s.laser_inliers:
            return []
        return [
            f"laser  {s.laser_inliers}/{s.laser_points} pts   "
            f"RMS {s.laser_rms_px:.2f} px   {_fmt(s.laser_ms)} ms",
            f"       angle {s.laser_angle_deg:+.2f}°",
        ]
