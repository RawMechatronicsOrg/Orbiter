"""GL frame view — the frame as a texture, the overlays as GL points, the
annotations through QPainter, all oriented and letterboxed on the GPU.

**The detector detects; the view draws.** Before this, each detector thread
oriented a copy of its 6 MB frame (1.5 ms), wrote the stripe pixels and the
board corners into it, and — while scanning — projected the cloud through the
eye's pose with `cv2.projectPoints` and wrote 2×2 px per point: 7 ms for 40k
points, per eye, per frame, before the GUI thread copied it again. Now the
worker publishes the frame as decoded plus the geometry it found, in original
pixel coordinates, and this widget:

  * uploads the frame as a texture and draws it on a quad whose texture
    coordinates carry the eye's orientation — no oriented copy exists;
  * draws the stripe pixels and the laser fit's points as GL points, straight
    from the worker's arrays;
  * draws the cloud as GL points whose vertex shader does the projection —
    board frame → camera through the eye's own pose, the pinhole, OpenCV's
    Brown distortion, the orientation and the letterbox. The cloud is
    uploaded once per snapshot, not per frame;
  * draws the board corners, their IDs, the fitted laser line and the board
    hull with QPainter, on top — a few dozen items, where text is easy.

The orientation is applied through one 3×3 matrix derived from
`orient.map_points`, the function the flip-then-rotate contract's tests pin,
so the GL view and the CPU mapping cannot disagree. The same matrix, in
widget pixels, places the QPainter annotations (`to_widget`).

Capture note for anyone verifying this: `QWidget.grab()` on a window holding a
`QOpenGLWidget` reads the backing store and returns colour speckle that is not
what is on screen. Use `FrameView.grabFramebuffer()`; `test_glview.py` does.

The image is letterboxed, never cropped and never stretched: a rotated eye is
taller than it is wide, and silently cutting off part of the frame in a tool
used to check framing and coverage would be the worst possible behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QMatrix3x3,
    QOpenGLFunctions,
    QPainter,
    QPolygonF,
    QVector2D,
    QVector3D,
    QVector4D,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLPixelTransferOptions,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .laser import LaserLine
from .orient import Orientation, map_points

# The few GL constants used, by value: QOpenGLFunctions binds the calls but not
# the enums, and pulling in PyOpenGL for four numbers would be a dependency.
_GL_POINTS = 0x0000
_GL_TRIANGLE_STRIP = 0x0005
_GL_FLOAT = 0x1406
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_DEPTH_TEST = 0x0B71
_GL_BLEND = 0x0BE2
_GL_PROGRAM_POINT_SIZE = 0x8642

# Colours as the CPU overlay had them (BGR there), as RGB here.
_STRIPE = (80 / 255, 235 / 255, 80 / 255, 1.0)
_CLOUD = (255 / 255, 150 / 255, 0.0, 1.0)
_INLIER = (80 / 255, 235 / 255, 80 / 255, 1.0)
_OUTLIER = (235 / 255, 60 / 255, 60 / 255, 1.0)
_CORNER = QColor(120, 235, 0)
_LINE = QColor(60, 235, 235)
_HULL = QColor(255, 190, 0)

_IMAGE_VS = """
attribute vec2 pos;
attribute vec2 uv;
varying vec2 v_uv;
void main() { v_uv = uv; gl_Position = vec4(pos, 0.0, 1.0); }
"""
# The texture holds the frame's BGR bytes uploaded as RGB; the swizzle puts
# them right, and saves a 6 MB channel reversal on the CPU per frame.
_IMAGE_FS = """
uniform sampler2D tex;
varying vec2 v_uv;
void main() { gl_FragColor = vec4(texture2D(tex, v_uv).bgr, 1.0); }
"""
_POINT_FS = """
uniform vec4 color;
void main() { gl_FragColor = color; }
"""
# 2-D points in original pixel indices; M takes continuous original pixel
# coordinates (index + 0.5 is the pixel's centre) to clip space.
_POINTS2D_VS = """
attribute vec2 xy;
uniform mat3 M;
uniform float size;
void main() {
    vec3 c = M * vec3(xy + 0.5, 1.0);
    gl_Position = vec4(c.xy, 0.0, 1.0);
    gl_PointSize = size;
}
"""
# Cloud points in the board's frame: the pose, the pinhole, OpenCV's Brown
# model exactly as `cv2.projectPoints` applies it, then M like any pixel.
# Points behind the camera go outside the clip volume, where nothing draws.
_POINTS3D_VS = """
attribute vec3 xyz;
uniform mat3 R;
uniform vec3 t;
uniform vec4 fc;        // fx, fy, cx, cy
uniform vec3 k;         // k1, k2, k3
uniform vec2 p;         // p1, p2
uniform mat3 M;
uniform float size;
void main() {
    vec3 c = R * xyz + t;
    if (c.z <= 1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; return; }
    float x = c.x / c.z;
    float y = c.y / c.z;
    float r2 = x * x + y * y;
    float radial = 1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z));
    float xd = x * radial + 2.0 * p.x * x * y + p.y * (r2 + 2.0 * x * x);
    float yd = y * radial + p.x * (r2 + 2.0 * y * y) + 2.0 * p.y * x * y;
    vec3 px = vec3(fc.x * xd + fc.z + 0.5, fc.y * yd + fc.w + 0.5, 1.0);
    vec3 cl = M * px;
    gl_Position = vec4(cl.xy, 0.0, 1.0);
    gl_PointSize = size;
}
"""


@dataclass
class Scene:
    """What one eye's view shows: the frame as decoded, and geometry found on
    it, every coordinate in ORIGINAL pixels. The view orients."""

    bgr: np.ndarray
    orientation: Orientation = Orientation()
    #: The frame's true size (width, height) when `bgr` is a smaller copy
    #: of it — geometry is mapped in these coordinates, not the texture's.
    size: tuple[int, int] | None = None
    #: Stripe pixels, (N, 2) as (x, y).
    stripe: np.ndarray | None = None
    #: ChArUco corners (N, 2) and their IDs (N,).
    corners: np.ndarray | None = None
    ids: np.ndarray | None = None
    #: The calibration-mode laser fit and the board hull it was confined to.
    laser: LaserLine | None = None
    hull: np.ndarray | None = None
    #: The cloud (M, 3) in the board's frame, mm, with what projects it into
    #: this eye: the board's pose here and the eye's K and D. Any of them
    #: missing, and no cloud is drawn.
    cloud: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    K: np.ndarray | None = None
    D: np.ndarray | None = None

    @property
    def wh(self) -> tuple[int, int]:
        return self.size if self.size is not None else (self.bgr.shape[1], self.bgr.shape[0])


def orientation_matrix(w: int, h: int, o: Orientation) -> np.ndarray:
    """The 3×3 affine taking CONTINUOUS original pixel coordinates — pixel
    (i, j) centred on (i + 0.5, j + 0.5), the frame spanning [0, w] × [0, h]
    — to continuous oriented ones. Fitted on `orient.map_points`, which maps
    pixel INDICES, so the two agree by construction: the quad's corners land
    on the oriented frame's edges and a mapped pixel on its oriented self."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dst = map_points(src, w, h, o)
    a = np.column_stack([dst[1] - dst[0], dst[2] - dst[0]])   # the linear part
    b = (dst[0] + 0.5) - a @ (src[0] + 0.5)
    m = np.eye(3)
    m[:2, :2] = a
    m[:2, 2] = b
    return m


class FrameView(QOpenGLWidget):
    """Displays one eye's `Scene`, plus a text overlay drawn over it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene: Scene | None = None
        self._overlay: list[str] = []
        self._placeholder = "waiting for frames…"
        self.setMinimumSize(320, 180)
        # GL resources, made in initializeGL.
        self._gl: QOpenGLFunctions | None = None
        self._image_prog: QOpenGLShaderProgram | None = None
        self._pt2_prog: QOpenGLShaderProgram | None = None
        self._pt3_prog: QOpenGLShaderProgram | None = None
        self._quad = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._scratch = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._cloud_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._texture: QOpenGLTexture | None = None
        # What is uploaded: the frame in the texture, the cloud in its VBO.
        self._frame_src: np.ndarray | None = None
        self._cloud_src: np.ndarray | None = None
        self._cloud_n = 0

    # ── content ───────────────────────────────────────────────────────────

    def set_scene(self, scene: Scene) -> None:
        """Adopt a scene without copying anything: the arrays are the worker's
        fresh output for this frame, and nothing writes to them after
        publishing, so holding references is safe."""
        self._scene = scene
        self.update()

    def set_overlay(self, lines: list[str]) -> None:
        self._overlay = lines
        self.update()

    def clear_frame(self, message: str) -> None:
        self._scene = None
        self._placeholder = message
        self.update()

    # ── geometry ──────────────────────────────────────────────────────────

    def _layout(self, scene: Scene):
        """(oriented width, oriented height, scale, x offset, y offset): the
        letterbox of the oriented frame in this widget, in logical pixels."""
        w, h = scene.wh
        ow, oh = (h, w) if scene.orientation.swaps_axes else (w, h)
        scale = min(self.width() / ow, self.height() / oh)
        ox = (self.width() - ow * scale) / 2.0
        oy = (self.height() - oh * scale) / 2.0
        return ow, oh, scale, ox, oy

    def _widget_matrix(self, scene: Scene) -> np.ndarray:
        """Continuous original pixel coordinates → widget pixels."""
        w, h = scene.wh
        _, _, s, ox, oy = self._layout(scene)
        v = np.array([[s, 0.0, ox], [0.0, s, oy], [0.0, 0.0, 1.0]])
        return v @ orientation_matrix(w, h, scene.orientation)

    def _clip_matrix(self, scene: Scene) -> np.ndarray:
        """Continuous original pixel coordinates → GL clip space."""
        to_clip = np.array([[2.0 / self.width(), 0.0, -1.0],
                            [0.0, -2.0 / self.height(), 1.0],
                            [0.0, 0.0, 1.0]])
        return to_clip @ self._widget_matrix(scene)

    def to_widget(self, pts: np.ndarray) -> np.ndarray:
        """Original pixel INDICES (N, 2) → widget pixel coordinates (N, 2):
        where the QPainter annotations go, and what the tests check the GL
        drawing against."""
        if self._scene is None:
            return np.empty((0, 2))
        m = self._widget_matrix(self._scene)
        p = np.asarray(pts, np.float64).reshape(-1, 2) + 0.5
        out = p @ m[:2, :2].T + m[:2, 2]
        return out

    # ── GL ────────────────────────────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802 - Qt naming
        self._gl = self.context().functions()
        self._image_prog = self._program(_IMAGE_VS, _IMAGE_FS)
        self._pt2_prog = self._program(_POINTS2D_VS, _POINT_FS)
        self._pt3_prog = self._program(_POINTS3D_VS, _POINT_FS)
        for buf in (self._quad, self._scratch, self._cloud_vbo):
            buf.create()
            buf.setUsagePattern(QOpenGLBuffer.UsagePattern.StreamDraw)
        self._frame_src = None
        self._cloud_src = None
        self._cloud_n = 0

    def _program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram(self)
        if not (prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
                and prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs)
                and prog.link()):
            raise RuntimeError(f"shader build failed: {prog.log()}")
        return prog

    def paintGL(self) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        scene = self._scene
        if scene is None or self._gl is None:
            p.fillRect(self.rect(), QColor(8, 10, 13))
            self._draw_placeholder(p)
        else:
            p.beginNativePainting()
            try:
                self._draw_scene(scene)
            finally:
                p.endNativePainting()
            self._annotate(p, scene)
        self._draw_overlay(p)
        p.end()

    def _draw_scene(self, scene: Scene) -> None:
        gl = self._gl
        gl.glClearColor(8 / 255, 10 / 255, 13 / 255, 1.0)
        gl.glClear(_GL_COLOR_BUFFER_BIT)
        gl.glDisable(_GL_DEPTH_TEST)
        gl.glDisable(_GL_BLEND)
        gl.glEnable(_GL_PROGRAM_POINT_SIZE)
        self._draw_image(scene)
        clip = self._clip_matrix(scene)
        if (scene.cloud is not None and len(scene.cloud)
                and scene.R is not None and scene.t is not None
                and scene.K is not None and scene.D is not None):
            self._draw_cloud(scene, clip)
        if scene.laser is not None and scene.laser.points.size:
            pts = scene.laser.points
            inl = scene.laser.inliers if scene.laser.inliers.size == len(pts) \
                else np.zeros(len(pts), bool)
            self._draw_points(pts[~inl], clip, _OUTLIER, 2.0)
            self._draw_points(pts[inl], clip, _INLIER, 2.0)
        if scene.stripe is not None and len(scene.stripe):
            self._draw_points(scene.stripe, clip, _STRIPE, 1.0)

    def _draw_image(self, scene: Scene) -> None:
        w, h = scene.wh                       # the frame, for the geometry
        th, tw = scene.bgr.shape[:2]          # the texture, possibly smaller
        tex = self._texture
        if tex is None or tex.width() != tw or tex.height() != th:
            if tex is not None:
                tex.destroy()
            tex = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            tex.setSize(tw, th)
            tex.setFormat(QOpenGLTexture.TextureFormat.RGB8_UNorm)
            tex.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
            tex.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
            tex.allocateStorage()
            self._texture = tex
            self._frame_src = None
        if scene.bgr is not self._frame_src:
            opts = QOpenGLPixelTransferOptions()
            opts.setAlignment(1)
            # The array goes in through the buffer protocol; `tobytes()` here
            # cost 0.9 ms per 1080p frame for nothing. `setData` copies before
            # it returns, so the array is not referenced afterwards.
            tex.setData(QOpenGLTexture.PixelFormat.RGB, QOpenGLTexture.PixelType.UInt8,
                        np.ascontiguousarray(scene.bgr), opts)
            self._frame_src = scene.bgr

        # The oriented frame's rectangle, in clip space, with each corner's
        # texture coordinate found by mapping it BACK to the original frame.
        ow, oh, _, _, _ = self._layout(scene)
        orient = orientation_matrix(w, h, scene.orientation)
        back = np.linalg.inv(orient)
        clip = self._clip_matrix(scene)
        verts = []
        for x, y in ((0, 0), (ow, 0), (0, oh), (ow, oh)):            # strip order
            src = back @ np.array([x, y, 1.0])                        # original, continuous
            c = clip @ src                                            # clip carries the orientation
            verts += [c[0], c[1], src[0] / w, src[1] / h]
        data = np.asarray(verts, np.float32).tobytes()

        prog = self._image_prog
        prog.bind()
        tex.bind(0)
        prog.setUniformValue1i("tex", 0)
        self._quad.bind()
        self._quad.allocate(data, len(data))
        prog.enableAttributeArray("pos")
        prog.enableAttributeArray("uv")
        prog.setAttributeBuffer("pos", _GL_FLOAT, 0, 2, 16)
        prog.setAttributeBuffer("uv", _GL_FLOAT, 8, 2, 16)
        self._gl.glDrawArrays(_GL_TRIANGLE_STRIP, 0, 4)
        prog.disableAttributeArray("pos")
        prog.disableAttributeArray("uv")
        self._quad.release()
        tex.release(0)
        prog.release()

    def _draw_points(self, xy: np.ndarray, clip: np.ndarray, color, size: float) -> None:
        if xy is None or not len(xy):
            return
        data = np.ascontiguousarray(xy, np.float32).tobytes()
        prog = self._pt2_prog
        prog.bind()
        prog.setUniformValue("M", _qmatrix(clip))
        prog.setUniformValue1f("size", float(size))
        prog.setUniformValue("color", QVector4D(*color))
        self._scratch.bind()
        self._scratch.allocate(data, len(data))
        prog.enableAttributeArray("xy")
        prog.setAttributeBuffer("xy", _GL_FLOAT, 0, 2, 0)
        self._gl.glDrawArrays(_GL_POINTS, 0, len(xy))
        prog.disableAttributeArray("xy")
        self._scratch.release()
        prog.release()

    def _draw_cloud(self, scene: Scene, clip: np.ndarray) -> None:
        cloud = scene.cloud
        self._cloud_vbo.bind()
        if cloud is not self._cloud_src:
            # Uploaded once per snapshot the scan thread publishes, not per
            # frame: 40k points is 480 kB, and the snapshot changes only when
            # a pair added points.
            data = np.ascontiguousarray(cloud, np.float32).tobytes()
            self._cloud_vbo.allocate(data, len(data))
            self._cloud_src = cloud
            self._cloud_n = len(cloud)
        k = np.asarray(scene.K, np.float64)
        d = np.zeros(5)
        dd = np.asarray(scene.D, np.float64).ravel()
        d[: min(5, len(dd))] = dd[:5]
        prog = self._pt3_prog
        prog.bind()
        prog.setUniformValue("R", _qmatrix(np.asarray(scene.R, np.float64)))
        t = np.asarray(scene.t, np.float64).ravel()
        prog.setUniformValue("t", QVector3D(float(t[0]), float(t[1]), float(t[2])))
        prog.setUniformValue("fc", QVector4D(float(k[0, 0]), float(k[1, 1]),
                                             float(k[0, 2]), float(k[1, 2])))
        prog.setUniformValue("k", QVector3D(float(d[0]), float(d[1]), float(d[4])))
        prog.setUniformValue("p", QVector2D(float(d[2]), float(d[3])))
        prog.setUniformValue("M", _qmatrix(clip))
        prog.setUniformValue1f("size", 2.0)
        prog.setUniformValue("color", QVector4D(*_CLOUD))
        prog.enableAttributeArray("xyz")
        prog.setAttributeBuffer("xyz", _GL_FLOAT, 0, 3, 0)
        self._gl.glDrawArrays(_GL_POINTS, 0, self._cloud_n)
        prog.disableAttributeArray("xyz")
        self._cloud_vbo.release()
        prog.release()

    # ── QPainter annotations ──────────────────────────────────────────────

    def _annotate(self, p: QPainter, scene: Scene) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if scene.hull is not None and len(scene.hull) >= 2:
            pts = self.to_widget(scene.hull)
            p.setPen(_HULL)
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in pts])
            poly.append(poly[0])
            p.drawPolyline(poly)
        line = scene.laser
        if line is not None and line.point is not None and line.direction is not None:
            a, b = line.endpoints()
            (ax, ay), (bx, by) = self.to_widget(np.array([a, b], np.float64))
            p.setPen(_LINE)
            p.drawLine(QLineF(ax, ay, bx, by))
        if scene.corners is not None and len(scene.corners):
            pts = self.to_widget(scene.corners)
            p.setPen(_CORNER)
            font = QFont("Consolas")
            font.setPointSize(7)
            p.setFont(font)
            ids = None if scene.ids is None else np.asarray(scene.ids).ravel()
            for i, (x, y) in enumerate(pts):
                p.drawRect(int(round(x)) - 3, int(round(y)) - 3, 6, 6)
                if ids is not None and i < len(ids):
                    p.drawText(int(round(x)) + 5, int(round(y)) - 4, str(int(ids[i])))

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


def _qmatrix(m: np.ndarray) -> QMatrix3x3:
    """A numpy 3×3 as Qt's row-major QMatrix3x3; Qt transposes on upload."""
    return QMatrix3x3([float(v) for v in np.asarray(m, np.float64).ravel()])
