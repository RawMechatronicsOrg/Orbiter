"""The GL view against the CPU mapping it replaces.

`orientation_matrix` and `to_widget` are checked as pure functions against
`orient.map_points` for every orientation. Then one real frame is rendered
through the widget's own GL path and read back with `grabFramebuffer()`: the
frame must land oriented, a stripe pixel must land where `map_points` puts it,
and a cloud point must land where `cv2.projectPoints` puts it — the vertex
shader's distortion against OpenCV's.

The rendering tests need a window system with OpenGL; they are skipped where
a QOpenGLWidget cannot come up.
"""

from __future__ import annotations

import os

import numpy as np
import cv2
import pytest

from orbiter_native.glview import FrameView, Scene, orientation_matrix
from orbiter_native.orient import Orientation, map_points

ORIENTATIONS = [
    Orientation(q, fh, fv)
    for q in range(4) for fh in (False, True) for fv in (False, True)
]


@pytest.mark.parametrize("o", ORIENTATIONS)
def test_orientation_matrix_agrees_with_map_points(o: Orientation) -> None:
    w, h = 320, 200
    m = orientation_matrix(w, h, o)
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, [w - 1, h - 1], (50, 2))
    expect = map_points(pts, w, h, o)
    got = (np.column_stack([pts + 0.5, np.ones(50)]) @ m.T)[:, :2] - 0.5
    assert np.allclose(got, expect, atol=1e-9)
    # The frame's own corners land on the oriented frame's corners.
    ow, oh = (h, w) if o.swaps_axes else (w, h)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ m.T
    assert set(map(tuple, np.rint(corners[:, :2]).astype(int))) == {
        (0, 0), (ow, 0), (ow, oh), (0, oh)}


# ── rendering ────────────────────────────────────────────────────────────

W, H = 320, 200
K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]])
D = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])


def _app():
    # Logical pixels are device pixels then, and the read-back framebuffer
    # is the widget's size — first effective when the app is made here.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _render(scene: Scene, size: tuple[int, int]):
    """(framebuffer as an (H, W, 3) RGB array, the view, its pixel ratio), or
    None when no GL context can be had here."""
    from PySide6.QtGui import QImage
    app = _app()
    view = FrameView()
    view.setMinimumSize(1, 1)
    view.resize(*size)
    view.show()
    app.processEvents()
    if not view.isValid():
        view.close()
        return None
    view.set_scene(scene)
    app.processEvents()
    img = view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes())
    out = arr.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()
    return out, view, float(view.devicePixelRatioF())


def _scene(o: Orientation) -> Scene:
    bgr = np.full((H, W, 3), 30, np.uint8)
    bgr[100:120, 40:60] = (0, 0, 255)                      # a red block, BGR
    stripe = np.array([[200, 20], [201, 20], [202, 20]], np.float32)
    cloud = np.array([[100.0, 50.0, 500.0], [-80.0, -30.0, 640.0]])
    return Scene(bgr=bgr, orientation=o, stripe=stripe, cloud=cloud,
                 R=np.eye(3), t=np.zeros(3), K=K, D=D)


def _near(img: np.ndarray, xy: np.ndarray, dpr: float, rgb, reach: int = 1) -> bool:
    """Is `rgb` within `reach` device pixels of oriented pixel `xy`?"""
    cx, cy = (np.asarray(xy, float) + 0.5) * dpr
    x0, y0 = int(np.floor(cx)), int(np.floor(cy))
    patch = img[max(y0 - reach, 0): y0 + reach + 1, max(x0 - reach, 0): x0 + reach + 1]
    return bool((patch == rgb).all(axis=2).any())


@pytest.mark.parametrize("o", [Orientation(1), Orientation(3, flip_h=True)])
def test_the_frame_and_the_geometry_land_where_the_cpu_mapping_says(o) -> None:
    scene = _scene(o)
    ow, oh = (H, W) if o.swaps_axes else (W, H)
    got = _render(scene, (ow, oh))            # scale 1, no letterbox offset
    if got is None:
        pytest.skip("no OpenGL context here")
    img, view, dpr = got
    try:
        assert img.shape == (round(oh * dpr), round(ow * dpr), 3), img.shape

        # The red block, through the orientation; just outside it, background.
        inside = map_points(np.array([[50.0, 110.0]]), W, H, o)[0]
        x, y = ((inside + 0.5) * dpr).astype(int)
        assert tuple(img[y, x]) == (255, 0, 0), img[y, x]
        outside = map_points(np.array([[70.0, 130.0]]), W, H, o)[0]
        x, y = ((outside + 0.5) * dpr).astype(int)
        assert tuple(img[y, x]) == (30, 30, 30), img[y, x]

        # A stripe pixel, drawn 1 px where map_points puts it.
        assert _near(img, map_points(np.array([[201.0, 20.0]]), W, H, o)[0], dpr, (80, 235, 80))

        # A cloud point, through the shader's pinhole and distortion, against
        # OpenCV's: orange within the 2 px point around the projected pixel.
        px, _ = cv2.projectPoints(scene.cloud, np.zeros(3), np.zeros(3), K, D)
        for pt in px.reshape(-1, 2):
            assert _near(img, map_points(pt.reshape(1, 2), W, H, o)[0], dpr,
                         (255, 150, 0), reach=2), pt
    finally:
        view.close()


def test_to_widget_is_the_letterboxed_mapping() -> None:
    """A wide widget on a rotated frame: bars left and right, the mapping
    scaled and offset to match."""
    got = _render(_scene(Orientation(1)), (500, 400))     # oriented frame is 200×320
    if got is None:
        pytest.skip("no OpenGL context here")
    img, view, dpr = got
    try:
        scale = 400 / 320
        ox = (500 - 200 * scale) / 2
        # Original pixel (0, 0) → oriented (199, 0) → widget.
        x, y = view.to_widget(np.array([[0.0, 0.0]]))[0]
        assert np.isclose(x, ox + (199 + 0.5) * scale) and np.isclose(y, 0.5 * scale)
        # The bars are background, the frame is not.
        assert tuple(img[int(200 * dpr), int(10 * dpr)]) == (8, 10, 13)
        assert tuple(img[int(200 * dpr), int(250 * dpr)]) == (30, 30, 30)
    finally:
        view.close()
