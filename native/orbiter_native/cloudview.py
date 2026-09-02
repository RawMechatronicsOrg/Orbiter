"""The cloud in 3-D: the live scan or a PLY from disk, to turn over by hand.

Points are drawn as the web viewer drew them and better: size-attenuated —
a point's size on screen falls with its distance from the eye, which is what
makes a cloud read as a volume instead of a flat speckle — and round with a
soft edge, shaded by their height above the board through a light-to-dark
gradient. A PLY that carries colour is drawn in it. The board is drawn as
its disc with the three axes, so the cloud always has a floor and an up.

Everything here is GL points from a buffer that changes only when the cloud
does: while scanning, the live view shows the same decimated snapshot the eyes
draw (`scanworker.OVERLAY_MAX` points), and a loaded PLY shows every point it
holds — a million is a 12 MB buffer, drawn in a millisecond.

The camera orbits a target with the board's z up: left-drag turns, wheel
zooms, right-drag pans, double-click fits the cloud again. Nothing is
recorded about the view; it is a look, not a state.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMatrix4x4, QOpenGLFunctions, QPainter, QVector3D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .scan import ScanVolume, read_ply

log = logging.getLogger("orbiter_native.cloudview")

_GL_POINTS = 0x0000
_GL_LINES = 0x0001
_GL_LINE_LOOP = 0x0002
_GL_FLOAT = 0x1406
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_DEPTH_BUFFER_BIT = 0x0100
_GL_DEPTH_TEST = 0x0B71
_GL_BLEND = 0x0BE2
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303
_GL_PROGRAM_POINT_SIZE = 0x8642
#: Compatibility-profile GL hands the fragment shader `gl_PointCoord` only with
#: point sprites enabled; without it every fragment read (0, 0), fell outside
#: the circle and was discarded — a blank view with a clean shader log.
_GL_POINT_SPRITE = 0x8861

# GLSL 1.20 on purpose: `gl_PointCoord` is undefined before it — on this
# driver it read as zero, every fragment fell outside the circle, and nothing
# was drawn without a single error.
_POINT_VS = """#version 120
attribute vec3 xyz;
attribute vec3 rgb;
uniform mat4 mvp;
uniform vec3 eye;
uniform float ref_dist;     // the orbit distance: a point there is `size` px
uniform float size;
uniform float z_lo;
uniform float z_hi;
uniform float use_rgb;
varying vec3 v_color;
void main() {
    gl_Position = mvp * vec4(xyz, 1.0);
    float dist = max(length(xyz - eye), 1.0);
    gl_PointSize = clamp(size * ref_dist / dist, 1.0, 40.0);
    float t = clamp((xyz.z - z_lo) / max(z_hi - z_lo, 1e-6), 0.0, 1.0);
    // Height: deep teal at the floor, amber in the middle, near white on top.
    vec3 lo = vec3(0.10, 0.45, 0.62);
    vec3 mid = vec3(0.98, 0.66, 0.20);
    vec3 hi = vec3(1.00, 0.96, 0.85);
    vec3 shade = t < 0.5 ? mix(lo, mid, t * 2.0) : mix(mid, hi, (t - 0.5) * 2.0);
    v_color = mix(shade, rgb, use_rgb);
}
"""
_POINT_FS = """#version 120
varying vec3 v_color;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float a = smoothstep(0.25, 0.14, r2);
    gl_FragColor = vec4(v_color * (0.78 + 0.22 * (1.0 - 4.0 * r2)), a);
}
"""
_LINE_VS = """#version 120
attribute vec3 xyz;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(xyz, 1.0); }
"""
_LINE_FS = """#version 120
uniform vec4 color;
void main() { gl_FragColor = color; }
"""


class CloudView(QOpenGLWidget):
    """Orbits a point cloud in the board's frame, mm, z up."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self._points = np.empty((0, 3), np.float32)
        self._colors: np.ndarray | None = None
        self._src = None
        self._n = 0
        self._z_range = (0.0, 1.0)
        self.point_px = 3.0
        self.disc_radius_mm = ScanVolume().radius_mm
        # The orbit: target, distance, yaw about z, pitch above the board.
        self._target = np.zeros(3)
        self._dist = 600.0
        self._yaw = math.radians(35.0)
        self._pitch = math.radians(28.0)
        self._drag: tuple[Qt.MouseButton, QPointF] | None = None
        self._gl: QOpenGLFunctions | None = None
        self._pt_prog: QOpenGLShaderProgram | None = None
        self._line_prog: QOpenGLShaderProgram | None = None
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._cbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._lines = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._uploaded = None
        self._caption = "no cloud yet"

    # ── content ───────────────────────────────────────────────────────────

    def set_cloud(self, points: np.ndarray, colors: np.ndarray | None = None,
                  caption: str = "", fit: bool = False) -> None:
        """Adopt a cloud, (N, 3) mm in the board's frame, with optional (N, 3)
        uint8 colours. The same array object again costs nothing."""
        if points is self._src:
            return
        self._src = points
        pts = np.ascontiguousarray(np.asarray(points, np.float32).reshape(-1, 3))
        self._points = pts
        self._colors = (np.ascontiguousarray(np.asarray(colors, np.float32).reshape(-1, 3) / 255.0)
                        if colors is not None and len(colors) == len(pts) else None)
        self._n = len(pts)
        if self._n:
            z = pts[:, 2]
            lo, hi = float(np.percentile(z, 2)), float(np.percentile(z, 98))
            self._z_range = (lo, hi if hi > lo else lo + 1.0)
        self._caption = caption or (f"{self._n} points" if self._n else "no cloud yet")
        if fit and self._n:
            self.fit()
        self.update()

    def fit(self) -> None:
        """Put the whole cloud in view."""
        if not self._n:
            self._target = np.zeros(3)
            self._dist = 600.0
        else:
            lo, hi = self._points.min(axis=0), self._points.max(axis=0)
            self._target = ((lo + hi) / 2.0).astype(np.float64)
            self._dist = max(float(np.linalg.norm(hi - lo)) * 1.2, 100.0)
        self.update()

    # ── the camera ────────────────────────────────────────────────────────

    def eye(self) -> np.ndarray:
        cp = math.cos(self._pitch)
        return self._target + self._dist * np.array(
            [cp * math.cos(self._yaw), cp * math.sin(self._yaw), math.sin(self._pitch)])

    def view_projection(self, w: int | None = None, h: int | None = None) -> np.ndarray:
        """The 4×4 taking board-frame mm to clip space for the current orbit."""
        w = w or max(self.width(), 1)
        h = h or max(self.height(), 1)
        eye = self.eye()
        f = self._target - eye
        f = f / max(np.linalg.norm(f), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        if abs(f @ up) > 0.999:
            up = np.array([0.0, 1.0, 0.0])
        s = np.cross(f, up)
        s = s / max(np.linalg.norm(s), 1e-9)
        u = np.cross(s, f)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = s, u, -f
        view[:3, 3] = -view[:3, :3] @ eye
        near, far = max(self._dist / 200.0, 0.5), self._dist * 50.0
        fov = math.radians(45.0)
        t = 1.0 / math.tan(fov / 2.0)
        proj = np.zeros((4, 4))
        proj[0, 0] = t / (w / h)
        proj[1, 1] = t
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2.0 * far * near / (near - far)
        proj[3, 2] = -1.0
        return proj @ view

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """Widget pixel coordinates of (N, 3) board-frame points — what the
        tests check the drawing against. NaN behind the eye."""
        p = np.column_stack([np.asarray(xyz, np.float64).reshape(-1, 3), np.ones(len(xyz))])
        c = p @ self.view_projection().T
        out = np.full((len(c), 2), np.nan)
        ok = c[:, 3] > 1e-9
        ndc = c[ok, :2] / c[ok, 3:4]
        out[ok, 0] = (ndc[:, 0] + 1.0) * 0.5 * self.width()
        out[ok, 1] = (1.0 - ndc[:, 1]) * 0.5 * self.height()
        return out

    # ── input ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, e) -> None:  # noqa: N802 - Qt naming
        self._drag = (e.button(), e.position())

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802 - Qt naming
        self._drag = None

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802 - Qt naming
        self.fit()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802 - Qt naming
        if self._drag is None:
            return
        button, last = self._drag
        pos = e.position()
        dx, dy = pos.x() - last.x(), pos.y() - last.y()
        self._drag = (button, pos)
        if button == Qt.MouseButton.LeftButton:
            self.orbit(-dx * 0.008, dy * 0.008)
        elif button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.pan(dx, dy)

    def wheelEvent(self, e) -> None:  # noqa: N802 - Qt naming
        steps = e.angleDelta().y() / 120.0
        self.zoom(0.85 ** steps)

    def orbit(self, d_yaw: float, d_pitch: float) -> None:
        self._yaw += d_yaw
        self._pitch = float(np.clip(self._pitch + d_pitch, math.radians(-85), math.radians(85)))
        self.update()

    def zoom(self, factor: float) -> None:
        self._dist = float(np.clip(self._dist * factor, 20.0, 20000.0))
        self.update()

    def pan(self, dx_px: float, dy_px: float) -> None:
        """Slide the target across the view plane by a screen displacement."""
        eye = self.eye()
        f = self._target - eye
        f = f / max(np.linalg.norm(f), 1e-9)
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
        s = s / max(np.linalg.norm(s), 1e-9)
        u = np.cross(s, f)
        per_px = 2.0 * self._dist * math.tan(math.radians(22.5)) / max(self.height(), 1)
        self._target = self._target - s * dx_px * per_px + u * dy_px * per_px
        self.update()

    # ── GL ────────────────────────────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802 - Qt naming
        self._gl = self.context().functions()
        self._pt_prog = _program(self, _POINT_VS, _POINT_FS)
        self._line_prog = _program(self, _LINE_VS, _LINE_FS)
        for buf in (self._vbo, self._cbo, self._lines):
            buf.create()
            buf.setUsagePattern(QOpenGLBuffer.UsagePattern.StreamDraw)
        self._uploaded = None
        self._lines.bind()
        data = _board_lines(self.disc_radius_mm).tobytes()
        self._lines.allocate(data, len(data))
        self._lines.release()

    def paintGL(self) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.beginNativePainting()
        try:
            self._draw()
        finally:
            p.endNativePainting()
        font = QFont("Consolas")
        font.setPointSize(9)
        p.setFont(font)
        p.setPen(QColor(222, 232, 240))
        p.drawText(10, self.height() - 10, self._caption)
        p.end()

    def _draw(self) -> None:
        gl = self._gl
        gl.glClearColor(8 / 255, 10 / 255, 13 / 255, 1.0)
        gl.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        gl.glEnable(_GL_DEPTH_TEST)
        gl.glEnable(_GL_PROGRAM_POINT_SIZE)
        gl.glEnable(_GL_POINT_SPRITE)
        mvp = _qmat4(self.view_projection())

        # The board's disc and axes.
        prog = self._line_prog
        prog.bind()
        prog.setUniformValue("mvp", mvp)
        self._lines.bind()
        prog.enableAttributeArray("xyz")
        prog.setAttributeBuffer("xyz", _GL_FLOAT, 0, 3, 0)
        prog.setUniformValue("color", QColor(90, 104, 120))
        gl.glDrawArrays(_GL_LINE_LOOP, 0, _DISC_SEGMENTS)
        for i, colour in enumerate((QColor(235, 80, 80), QColor(80, 220, 110), QColor(90, 150, 255))):
            prog.setUniformValue("color", colour)
            gl.glDrawArrays(_GL_LINES, _DISC_SEGMENTS + 2 * i, 2)
        prog.disableAttributeArray("xyz")
        self._lines.release()
        prog.release()

        if not self._n:
            return
        if self._src is not self._uploaded:
            self._vbo.bind()
            data = self._points.tobytes()
            self._vbo.allocate(data, len(data))
            self._vbo.release()
            self._cbo.bind()
            cdata = (self._colors if self._colors is not None
                     else np.zeros((self._n, 3), np.float32)).tobytes()
            self._cbo.allocate(cdata, len(cdata))
            self._cbo.release()
            self._uploaded = self._src

        gl.glEnable(_GL_BLEND)
        gl.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)
        prog = self._pt_prog
        prog.bind()
        prog.setUniformValue("mvp", mvp)
        eye = self.eye()
        prog.setUniformValue("eye", QVector3D(float(eye[0]), float(eye[1]), float(eye[2])))
        prog.setUniformValue1f("ref_dist", float(self._dist))
        prog.setUniformValue1f("size", float(self.point_px))
        prog.setUniformValue1f("z_lo", float(self._z_range[0]))
        prog.setUniformValue1f("z_hi", float(self._z_range[1]))
        prog.setUniformValue1f("use_rgb", 1.0 if self._colors is not None else 0.0)
        self._vbo.bind()
        prog.enableAttributeArray("xyz")
        prog.setAttributeBuffer("xyz", _GL_FLOAT, 0, 3, 0)
        self._vbo.release()
        self._cbo.bind()
        prog.enableAttributeArray("rgb")
        prog.setAttributeBuffer("rgb", _GL_FLOAT, 0, 3, 0)
        self._cbo.release()
        gl.glDrawArrays(_GL_POINTS, 0, self._n)
        prog.disableAttributeArray("xyz")
        prog.disableAttributeArray("rgb")
        prog.release()
        gl.glDisable(_GL_BLEND)


_DISC_SEGMENTS = 96


def _board_lines(radius: float) -> np.ndarray:
    """The board's disc as a line loop, then the three axes as line pairs."""
    a = np.linspace(0.0, 2.0 * np.pi, _DISC_SEGMENTS, endpoint=False)
    disc = np.column_stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)])
    L = radius * 0.4
    axes = np.array([[0, 0, 0], [L, 0, 0], [0, 0, 0], [0, L, 0], [0, 0, 0], [0, 0, L]], float)
    return np.vstack([disc, axes]).astype(np.float32)


def _program(owner, vs: str, fs: str) -> QOpenGLShaderProgram:
    prog = QOpenGLShaderProgram(owner)
    if not (prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
            and prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs)
            and prog.link()):
        raise RuntimeError(f"shader build failed: {prog.log()}")
    return prog


def _qmat4(m: np.ndarray) -> QMatrix4x4:
    return QMatrix4x4(*[float(v) for v in np.asarray(m, np.float64).ravel()])


class CloudPanel(QFrame):
    """The view with its few controls: live or a file, and the point size."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.view = CloudView()
        self._file: tuple[np.ndarray, np.ndarray | None, str] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(5)
        title = QLabel("CLOUD")
        title.setStyleSheet(
            "color:#7cc4ff; font-weight:600; letter-spacing:2px; font-size:12px;")
        root.addWidget(title)

        row = QHBoxLayout()
        self.live = QCheckBox("live")
        self.live.setChecked(True)
        self.live.setToolTip("Follow the cloud being scanned. Off: show the opened PLY.")
        self.live.toggled.connect(lambda _on: self._show())
        row.addWidget(self.live)
        self.btn_open = QPushButton("Open PLY")
        self.btn_open.clicked.connect(self._open)
        row.addWidget(self.btn_open)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.view.fit)
        row.addWidget(self.btn_fit)
        row.addWidget(QLabel("size"))
        self.size = QSpinBox()
        self.size.setRange(1, 12)
        self.size.setValue(int(self.view.point_px))
        self.size.setToolTip("Point size in pixels at the orbit distance; nearer points grow.")
        self.size.valueChanged.connect(self._resize_points)
        row.addWidget(self.size)
        row.addStretch(1)
        root.addLayout(row)
        root.addWidget(self.view, 1)

        self.hint = QLabel("drag to turn · wheel to zoom · right-drag to pan · double-click to fit")
        self.hint.setStyleSheet("color:#8b9aac; font-family:Consolas; font-size:10px;")
        root.addWidget(self.hint)

    def set_live_points(self, points: np.ndarray, n_total: int) -> None:
        """The scan's current snapshot; drawn while `live` is on."""
        if self.live.isChecked():
            self.view.set_cloud(points, caption=f"live · {n_total} points"
                                + (f" ({len(points)} shown)" if len(points) < n_total else ""))

    def _resize_points(self, value: int) -> None:
        self.view.point_px = float(value)
        self.view.update()

    def _show(self) -> None:
        if not self.live.isChecked() and self._file is not None:
            pts, rgb, name = self._file
            self.view.set_cloud(pts, rgb, caption=f"{name} · {len(pts)} points", fit=True)

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open point cloud", "", "PLY (*.ply)")
        if not path:
            return
        try:
            pts, rgb = read_ply(path)
        except (OSError, ValueError) as exc:
            self.view._caption = f"could not read {Path(path).name}: {exc}"
            self.view.update()
            return
        self._file = (pts, rgb, Path(path).name)
        self.live.setChecked(False)
        self._show()
