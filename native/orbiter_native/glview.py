"""GL frame view — draws the newest frame letterboxed, with a text overlay.

Painting happens through `QOpenGLWidget`'s GL-backed `QPainter`, not a manual
texture upload: the compositing, scaling and text all run on the GPU paint
engine, which is what a `QLabel`/`QPixmap` would not give at 2 x 30 fps. The
widget is GL so that the scene viewer planned for stage 2 can draw real
geometry into the same surface without the window being rebuilt around it.

Capture note for anyone verifying this: `QWidget.grab()` on a window holding a
`QOpenGLWidget` reads the backing store and returns colour speckle that is not
what is on screen. Use `FrameView.grabFramebuffer()`.

The image is letterboxed, never cropped and never stretched: a rotated eye is
taller than it is wide, and silently cutting off part of the frame in a tool
used to check framing and coverage would be the worst possible behaviour.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QColor, QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class FrameView(QOpenGLWidget):
    """Displays the newest BGR frame, plus a text overlay drawn over it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._bgr: np.ndarray | None = None      # keeps `_image`'s memory alive
        self._overlay: list[str] = []
        self._placeholder = "waiting for frames…"
        self.setMinimumSize(320, 180)

    # ── content ───────────────────────────────────────────────────────────

    def set_frame(self, bgr: np.ndarray) -> None:
        """Adopt a BGR frame without converting or copying it.

        The array is the worker's fresh output for this frame — orientation
        makes a new one each time and nothing writes to it after publishing —
        so wrapping it is safe as long as this widget holds the reference.
        Measured at 1080p: the BGR→RGB reversal in numpy plus the QImage copy
        this replaces cost 8.6 ms per frame per eye, on the GUI thread.
        """
        h, w = bgr.shape[:2]
        self._bgr = np.ascontiguousarray(bgr)
        self._image = QImage(self._bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.update()

    def set_overlay(self, lines: list[str]) -> None:
        self._overlay = lines
        self.update()

    def clear_frame(self, message: str) -> None:
        self._image = None
        self._placeholder = message
        self.update()

    # ── painting ──────────────────────────────────────────────────────────

    def paintGL(self) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 10, 13))
        if self._image is None:
            self._draw_placeholder(p)
        else:
            self._draw_image(p)
        self._draw_overlay(p)
        p.end()

    def _draw_image(self, p: QPainter) -> None:
        img = self._image
        assert img is not None
        # Fit inside the widget preserving aspect — the letterbox.
        scale = min(self.width() / img.width(), self.height() / img.height())
        w, h = int(img.width() * scale), int(img.height() * scale)
        x, y = (self.width() - w) // 2, (self.height() - h) // 2
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(self.rect().adjusted(x, y, x + w - self.width(), y + h - self.height()),
                    img)

    def _draw_placeholder(self, p: QPainter) -> None:
        p.setPen(QColor(120, 134, 150))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)

    def _draw_overlay(self, p: QPainter) -> None:
        if not self._overlay:
            return
        font = QFont("Consolas")
        font.setPointSize(9)
        p.setFont(font)
        fm = p.fontMetrics()
        line_h = fm.height()
        width = max(fm.horizontalAdvance(s) for s in self._overlay) + 14
        box_h = line_h * len(self._overlay) + 10
        p.fillRect(8, 8, width, box_h, QColor(0, 0, 0, 165))
        p.setPen(QColor(222, 232, 240))
        for i, line in enumerate(self._overlay):
            p.drawText(15, 8 + line_h * (i + 1), line)
