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
from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QMatrix4x4,
    QOpenGLFunctions,
    QPainter,
    QShortcut,
    QSurfaceFormat,
    QVector2D,
    QVector3D,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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
_GL_MULTISAMPLE = 0x809D
_GL_TEXTURE_2D = 0x0DE1
_GL_TEXTURE0 = 0x84C0
_GL_FRAMEBUFFER = 0x8D40
_GL_RGBA16F = 0x881A
_GL_TRIANGLE_STRIP = 0x0005

# GLSL 1.20 on purpose: `gl_PointCoord` is undefined before it — on this
# driver it read as zero, every fragment fell outside the circle, and nothing
# was drawn without a single error.
_POINT_VS = """#version 120
attribute vec3 xyz;
attribute vec3 rgb;
uniform mat4 mv;
uniform mat4 proj;
uniform float ref_dist;     // the orbit distance: a point there is `size` px
uniform float size;
uniform float z_lo;
uniform float z_hi;
uniform float use_rgb;
varying vec3 v_color;
void main() {
    vec4 c = mv * vec4(xyz, 1.0);
    gl_Position = proj * c;
    float dist = max(length(c.xyz), 1.0);
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
uniform mat4 mv;
uniform mat4 proj;
varying float v_depth;
void main() {
    vec4 c = mv * vec4(xyz, 1.0);
    v_depth = -c.z;
    gl_Position = proj * c;
}
"""
_LINE_FS = """#version 120
uniform vec4 color;
uniform float depth_alpha;   // 1: alpha carries the view depth, for the shaded pass
varying float v_depth;
void main() { gl_FragColor = vec4(color.rgb, mix(color.a, v_depth, depth_alpha)); }
"""

# The shaded look: every point a small lit sphere. The fragment shader gives
# each sprite fragment the depth of the sphere's surface there (gl_FragDepth),
# so neighbouring points intersect and occlude like beads rather than
# overlapping discs, and a normal to light. Alpha carries the view depth for
# the eye-dome pass that follows.
_SPHERE_VS = """#version 120
attribute vec3 xyz;
attribute vec3 rgb;
uniform mat4 mv;
uniform mat4 proj;
uniform float ref_dist;
uniform float size;
uniform float focal_px;      // proj[1][1] * viewport height / 2
uniform float z_lo;
uniform float z_hi;
uniform float use_rgb;
varying vec3 v_color;
varying vec3 v_centre;
varying float v_radius;
void main() {
    vec4 c = mv * vec4(xyz, 1.0);
    float dist = max(length(c.xyz), 1.0);
    float px = clamp(size * ref_dist / dist, 2.0, 40.0);
    gl_PointSize = px;
    gl_Position = proj * c;
    v_centre = c.xyz;
    v_radius = 0.5 * px * max(-c.z, 1.0) / focal_px;
    float t = clamp((xyz.z - z_lo) / max(z_hi - z_lo, 1e-6), 0.0, 1.0);
    vec3 lo = vec3(0.10, 0.45, 0.62);
    vec3 mid = vec3(0.98, 0.66, 0.20);
    vec3 hi = vec3(1.00, 0.96, 0.85);
    vec3 shade = t < 0.5 ? mix(lo, mid, t * 2.0) : mix(mid, hi, (t - 0.5) * 2.0);
    v_color = mix(shade, rgb, use_rgb);
}
"""
_SPHERE_FS = """#version 120
uniform mat4 proj;
varying vec3 v_color;
varying vec3 v_centre;
varying float v_radius;
void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    d.y = -d.y;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    vec3 n = vec3(d, sqrt(1.0 - r2));
    vec3 p = v_centre + n * v_radius;
    vec4 clip = proj * vec4(p, 1.0);
    gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
    vec3 L = normalize(vec3(-0.35, 0.55, 1.0));       // a headlight, from upper left
    float diff = max(dot(n, L), 0.0);
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, H), 0.0), 48.0) * 0.2;
    // A high floor: at the 3-4 px these sprites mostly are, the rim is most
    // of the sprite, and a dark rim on every bead reads as a dark cloud.
    vec3 col = v_color * (0.60 + 0.45 * diff) + vec3(spec);
    gl_FragColor = vec4(col, -p.z);
}
"""
# Eye-dome lighting: a pixel whose neighbours are nearer the eye than it is
# sits in the shadow of what stands in front of it. Depth comes in the alpha
# channel of the shaded pass; the background carries none and is left alone.
_EDL_VS = """#version 120
attribute vec2 pos;
varying vec2 uv;
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""
_EDL_FS = """#version 120
uniform sampler2D tex;
uniform vec2 texel;
uniform float radius;
uniform float strength;
varying vec2 uv;
void main() {
    vec4 c = texture2D(tex, uv);
    if (c.a <= 0.0) { gl_FragColor = vec4(c.rgb, 1.0); return; }
    float lz = log2(c.a);
    float resp = 0.0;
    for (int i = 0; i < 8; ++i) {
        float a = float(i) * 0.78539816;
        vec2 o = vec2(cos(a), sin(a)) * texel * radius;
        float zn = texture2D(tex, uv + o).a;
        if (zn > 0.0) resp += max(0.0, lz - log2(zn));
    }
    float shade = exp(-resp / 8.0 * strength);
    gl_FragColor = vec4(c.rgb * shade, 1.0);
}
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
        #: "auto" draws the points' own colour when they have one and height
        #: otherwise; "rgb" and "height" ask for one of them outright.
        self.colour_mode = "auto"
        # Four samples per pixel: point edges and the board's lines stop
        # crawling as the cloud turns. Asked for here, before the widget is
        # shown — the one moment a QOpenGLWidget takes a format.
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setSamples(4)
        self.setFormat(fmt)
        # A slow turntable, for looking at a scan without holding the mouse.
        self._spin = QTimer(self)
        self._spin.setInterval(33)
        self._spin.timeout.connect(self._spin_tick)
        #: Lit spheres with eye-dome lighting (an offscreen pass), or the
        #: soft discs. Falls back to the discs where the shaders or the
        #: framebuffers cannot be had.
        self.shading = True
        #: Eye-dome strength. Gentle on purpose: in a scan's fuzz every pixel
        #: has a slightly nearer neighbour, and a strong setting darkens the
        #: whole cloud instead of its creases.
        self.edl_strength = 40.0
        self._sphere_prog: QOpenGLShaderProgram | None = None
        self._edl_prog: QOpenGLShaderProgram | None = None
        self._quad = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._fbo_ms: QOpenGLFramebufferObject | None = None
        self._fbo: QOpenGLFramebufferObject | None = None

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
        view, proj = self.matrices(w, h)
        return proj @ view

    def matrices(self, w: int | None = None, h: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(view, projection): board-frame mm to camera space, and camera
        space to clip, for the current orbit."""
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
        return view, proj

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

    def set_spinning(self, on: bool) -> None:
        """Turn the cloud on its own, 15 degrees a second, until told not to."""
        if on:
            self._spin.start()
        else:
            self._spin.stop()

    def _spin_tick(self) -> None:
        self._yaw = (self._yaw + math.radians(0.5)) % (2.0 * math.pi)
        self.update()

    # ── the look ──────────────────────────────────────────────────────────

    def set_colour_mode(self, mode: str) -> None:
        """`auto`, `rgb` or `height` — see `colour_mode`."""
        self.colour_mode = mode
        self.update()

    def set_shading(self, on: bool) -> None:
        """Lit spheres with eye-dome lighting, or the soft discs."""
        self.shading = bool(on)
        self.update()

    def use_rgb(self) -> bool:
        """Draw the points in their own colour? Only when they have one and
        the mode does not ask for height; `rgb` without colours is height."""
        return self._colors is not None and self.colour_mode != "height"

    def save_png(self, path: str) -> bool:
        """The view as drawn, at its pixel size."""
        return bool(self.grabFramebuffer().save(path, "PNG"))

    def copy_state_from(self, other: "CloudView") -> None:
        """The same cloud, the same orbit and the same look as `other` — for
        a second view of one cloud, such as the full-screen one. Arrays are
        shared, not copied; neither view writes to them."""
        self._src, self._points, self._colors, self._n = (
            other._src, other._points, other._colors, other._n)
        self._z_range, self._caption = other._z_range, other._caption
        self._target, self._dist = other._target.copy(), other._dist
        self._yaw, self._pitch = other._yaw, other._pitch
        self.point_px, self.colour_mode = other.point_px, other.colour_mode
        self.shading, self.edl_strength = other.shading, other.edl_strength
        self._uploaded = None
        self.update()

    # ── GL ────────────────────────────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802 - Qt naming
        self._gl = self.context().functions()
        self._pt_prog = _program(self, _POINT_VS, _POINT_FS)
        self._line_prog = _program(self, _LINE_VS, _LINE_FS)
        try:
            self._sphere_prog = _program(self, _SPHERE_VS, _SPHERE_FS)
            self._edl_prog = _program(self, _EDL_VS, _EDL_FS)
        except RuntimeError as exc:
            # Discs still draw; the log says why the spheres do not.
            log.warning("shaded cloud unavailable on this GL: %s", exc)
            self._sphere_prog = self._edl_prog = None
        for buf in (self._vbo, self._cbo, self._lines, self._quad):
            buf.create()
            buf.setUsagePattern(QOpenGLBuffer.UsagePattern.StreamDraw)
        self._uploaded = None
        self._fbo_ms = self._fbo = None
        self._lines.bind()
        data = _board_lines(self.disc_radius_mm).tobytes()
        self._lines.allocate(data, len(data))
        self._lines.release()
        self._quad.bind()
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32).tobytes()
        self._quad.allocate(quad, len(quad))
        self._quad.release()

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
        gl.glEnable(_GL_PROGRAM_POINT_SIZE)
        gl.glEnable(_GL_POINT_SPRITE)
        gl.glEnable(_GL_MULTISAMPLE)
        self._upload()
        if self.shading and self._shaded_pass():
            return
        self._flat_pass()

    def _upload(self) -> None:
        """The cloud into its buffers, once per snapshot."""
        if not self._n or self._src is self._uploaded:
            return
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

    def _viewport_px(self) -> tuple[int, int]:
        dpr = float(self.devicePixelRatioF())
        return (max(int(round(self.width() * dpr)), 1),
                max(int(round(self.height() * dpr)), 1))

    def _clear(self, alpha: float) -> None:
        gl = self._gl
        gl.glClearColor(8 / 255, 10 / 255, 13 / 255, alpha)
        gl.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        gl.glEnable(_GL_DEPTH_TEST)

    def _draw_lines(self, mv: QMatrix4x4, proj: QMatrix4x4, depth_alpha: float) -> None:
        """The board's disc and axes."""
        gl = self._gl
        prog = self._line_prog
        prog.bind()
        prog.setUniformValue("mv", mv)
        prog.setUniformValue("proj", proj)
        prog.setUniformValue1f("depth_alpha", depth_alpha)
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

    def _point_uniforms(self, prog: QOpenGLShaderProgram, mv: QMatrix4x4, proj: QMatrix4x4) -> None:
        prog.setUniformValue("mv", mv)
        prog.setUniformValue("proj", proj)
        prog.setUniformValue1f("ref_dist", float(self._dist))
        prog.setUniformValue1f("size", float(self.point_px))
        prog.setUniformValue1f("z_lo", float(self._z_range[0]))
        prog.setUniformValue1f("z_hi", float(self._z_range[1]))
        prog.setUniformValue1f("use_rgb", 1.0 if self.use_rgb() else 0.0)

    def _draw_points(self, prog: QOpenGLShaderProgram) -> None:
        gl = self._gl
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

    def _flat_pass(self) -> None:
        """Soft discs blended over the board — the look without shading."""
        gl = self._gl
        self._clear(1.0)
        view, proj = self.matrices()
        mv, pm = _qmat4(view), _qmat4(proj)
        self._draw_lines(mv, pm, 0.0)
        if not self._n:
            return
        gl.glEnable(_GL_BLEND)
        gl.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)
        prog = self._pt_prog
        prog.bind()
        self._point_uniforms(prog, mv, pm)
        self._draw_points(prog)
        prog.release()
        gl.glDisable(_GL_BLEND)

    def _ensure_fbos(self, w: int, h: int) -> bool:
        """The offscreen pair for the shaded pass at this size: a
        multisampled target to draw into, and a plain one to sample from."""
        if (self._fbo is not None and self._fbo_ms is not None
                and self._fbo.size() == QSize(w, h) and self._fbo.isValid()):
            return True
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fmt.setSamples(4)
        fmt.setInternalTextureFormat(_GL_RGBA16F)
        plain = QOpenGLFramebufferObjectFormat()
        plain.setInternalTextureFormat(_GL_RGBA16F)
        self._fbo_ms = QOpenGLFramebufferObject(QSize(w, h), fmt)
        self._fbo = QOpenGLFramebufferObject(QSize(w, h), plain)
        return bool(self._fbo_ms.isValid() and self._fbo.isValid())

    def _shaded_pass(self) -> bool:
        """Lit spheres that occlude one another, then eye-dome lighting.
        False when this GL cannot do it; the caller draws the discs."""
        gl = self._gl
        w, h = self._viewport_px()
        if (self._sphere_prog is None or self._edl_prog is None
                or not self._ensure_fbos(w, h)):
            return False
        view, proj = self.matrices()
        mv, pm = _qmat4(view), _qmat4(proj)
        # Pass 1: the scene into the multisampled target, alpha = view depth.
        self._fbo_ms.bind()
        gl.glViewport(0, 0, w, h)
        gl.glDisable(_GL_BLEND)
        self._clear(0.0)
        self._draw_lines(mv, pm, 1.0)
        if self._n:
            prog = self._sphere_prog
            prog.bind()
            self._point_uniforms(prog, mv, pm)
            prog.setUniformValue1f("focal_px", float(proj[1, 1]) * h * 0.5)
            self._draw_points(prog)
            prog.release()
        self._fbo_ms.release()
        QOpenGLFramebufferObject.blitFramebuffer(self._fbo, self._fbo_ms)
        # Pass 2: eye-dome lighting over the resolved image, into the widget.
        gl.glBindFramebuffer(_GL_FRAMEBUFFER, int(self.defaultFramebufferObject()))
        gl.glViewport(0, 0, w, h)
        gl.glDisable(_GL_DEPTH_TEST)
        gl.glActiveTexture(_GL_TEXTURE0)
        gl.glBindTexture(_GL_TEXTURE_2D, int(self._fbo.texture()))
        prog = self._edl_prog
        prog.bind()
        prog.setUniformValue1i("tex", 0)
        prog.setUniformValue("texel", QVector2D(1.0 / w, 1.0 / h))
        prog.setUniformValue1f("radius", 1.0)
        prog.setUniformValue1f("strength", float(self.edl_strength))
        self._quad.bind()
        prog.enableAttributeArray("pos")
        prog.setAttributeBuffer("pos", _GL_FLOAT, 0, 2, 0)
        gl.glDrawArrays(_GL_TRIANGLE_STRIP, 0, 4)
        prog.disableAttributeArray("pos")
        self._quad.release()
        prog.release()
        gl.glBindTexture(_GL_TEXTURE_2D, 0)
        gl.glEnable(_GL_DEPTH_TEST)
        return True


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


class _FullscreenCloud(QWidget):
    """The cloud alone on the whole screen.

    A second CloudView fed the same arrays, not the panel's view moved out: a
    QOpenGLWidget reparented into another top-level window loses its GL
    context and rebuilds it, and the dance around that is worse than one more
    context that lives only while the screen is taken. Esc or F11 gives the
    screen back.
    """

    def __init__(self, panel: "CloudPanel") -> None:
        super().__init__(None, Qt.WindowType.Window)
        self._panel = panel
        self.setWindowTitle("Orbiter — cloud")
        self.setStyleSheet("background:#0d1013;")
        self.view = CloudView(self)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.view, 1)
        hint = QLabel("Esc or F11 to come back · drag to turn · wheel to zoom · "
                      "right-drag to pan · double-click to fit")
        hint.setStyleSheet(
            "color:#8b9aac; font-family:Consolas; font-size:10px; padding:2px 8px;")
        root.addWidget(hint)

    def keyPressEvent(self, e) -> None:  # noqa: N802 - Qt naming
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_F11):
            self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e) -> None:  # noqa: N802 - Qt naming
        self._panel._fullscreen_closed(self)
        super().closeEvent(e)


class CloudPanel(QFrame):
    """The view with its controls: live or a file, the point size, the
    colour, a turntable, a PNG of the view, and the whole screen."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.view = CloudView()
        self._file: tuple[np.ndarray, np.ndarray | None, str] | None = None
        self._big: _FullscreenCloud | None = None

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

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("colour"))
        self.colour = QComboBox()
        self.colour.addItems(["auto", "rgb", "height"])
        self.colour.setToolTip("auto: the points' own colour when the scan has one, "
                               "height shading otherwise.")
        self.colour.currentTextChanged.connect(self._set_colour_mode)
        row2.addWidget(self.colour)
        self.spin = QCheckBox("spin")
        self.spin.setToolTip("Turn the cloud slowly on its own, like a turntable.")
        self.spin.toggled.connect(self._set_spinning)
        row2.addWidget(self.spin)
        self.shade = QCheckBox("shade")
        self.shade.setToolTip("Every point a small lit sphere, occluding its neighbours, "
                              "with eye-dome lighting; off: soft discs.")
        self.shade.setChecked(self.view.shading)
        self.shade.toggled.connect(self._set_shading)
        row2.addWidget(self.shade)
        self.btn_png = QPushButton("PNG")
        self.btn_png.setToolTip("Save the view as drawn to a PNG.")
        self.btn_png.clicked.connect(self._save_png)
        row2.addWidget(self.btn_png)
        self.btn_full = QPushButton("Full screen")
        self.btn_full.setToolTip("The cloud alone on the whole screen (F11); Esc comes back.")
        self.btn_full.clicked.connect(self.expand)
        row2.addWidget(self.btn_full)
        row2.addStretch(1)
        root.addLayout(row2)
        QShortcut(QKeySequence(Qt.Key.Key_F11), self, activated=self.expand)
        root.addWidget(self.view, 1)

        self.hint = QLabel("drag to turn · wheel to zoom · right-drag to pan · double-click to fit")
        self.hint.setStyleSheet("color:#8b9aac; font-family:Consolas; font-size:10px;")
        root.addWidget(self.hint)

    def _views(self) -> list[CloudView]:
        """The panel's view and, while the screen is taken, the big one."""
        return [self.view] + ([self._big.view] if self._big is not None else [])

    def _feed(self, points, colors, caption: str, fit: bool) -> None:
        for v in self._views():
            v.set_cloud(points, colors, caption=caption, fit=fit)

    def set_live_points(self, points: np.ndarray, n_total: int,
                        colors: np.ndarray | None = None) -> None:
        """The scan's current snapshot, with its colours when the scan has
        them; drawn while `live` is on."""
        if self.live.isChecked():
            self._feed(points, colors, f"live · {n_total} points"
                       + (f" ({len(points)} shown)" if len(points) < n_total else ""), False)

    def _resize_points(self, value: int) -> None:
        for v in self._views():
            v.point_px = float(value)
            v.update()

    def _set_colour_mode(self, mode: str) -> None:
        for v in self._views():
            v.set_colour_mode(mode)

    def _set_spinning(self, on: bool) -> None:
        for v in self._views():
            v.set_spinning(on)

    def _set_shading(self, on: bool) -> None:
        for v in self._views():
            v.set_shading(on)

    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the view", "cloud.png", "PNG (*.png)")
        if not path:
            return
        ok = self.view.save_png(path)
        self.hint.setText(f"saved {Path(path).name}" if ok else f"could not write {path}")

    def expand(self) -> None:
        """The cloud on the whole screen — or back, if it already is."""
        if self._big is not None:
            self._big.close()
            return
        big = _FullscreenCloud(self)
        big.view.copy_state_from(self.view)
        big.view.set_spinning(self.spin.isChecked())
        screen = self.screen()
        if screen is not None:
            big.move(screen.geometry().topLeft())       # the screen this panel is on
        self._big = big
        big.showFullScreen()

    def _fullscreen_closed(self, big: "_FullscreenCloud") -> None:
        if self._big is big:
            self._big = None

    def _show(self) -> None:
        if not self.live.isChecked() and self._file is not None:
            pts, rgb, name = self._file
            self._feed(pts, rgb, f"{name} · {len(pts)} points", True)

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
