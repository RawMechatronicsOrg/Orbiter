"""The cloud view: its camera against its own drawing, and the PLY round trip.

The orbit camera's projection is checked as a pure function — a point at the
target lands in the middle, the board's z is up — and then a small cloud is
rendered through the widget and read back: its points must be lit where
`project` puts them, in the colour the height gives them, and turning the
orbit must move them. Rendering needs a window system with OpenGL and is
skipped where a QOpenGLWidget cannot come up.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from orbiter_native.cloudview import CloudView
from orbiter_native.scan import PointCloud, read_ply


def test_ply_round_trip_binary(tmp_path) -> None:
    cloud = PointCloud()
    pts = np.array([[1.5, -2.0, 30.0], [100.0, 20.0, 5.25], [-40.0, 0.0, 0.0]])
    cloud.add(pts)
    path = tmp_path / "c.ply"
    assert cloud.write_ply(str(path)) == 3
    got, rgb = read_ply(str(path))
    assert np.allclose(got, pts, atol=1e-6) and rgb is None


def test_ply_ascii_with_colour_and_faces(tmp_path) -> None:
    path = tmp_path / "a.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "1 2 3 255 0 10\n4 5 6 0 128 255\n3 0 1 1\n")
    got, rgb = read_ply(str(path))
    assert np.allclose(got, [[1, 2, 3], [4, 5, 6]])
    assert rgb.tolist() == [[255, 0, 10], [0, 128, 255]]


def test_ply_refuses_what_is_not_a_cloud(tmp_path) -> None:
    path = tmp_path / "b.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n")
    with pytest.raises(ValueError):
        read_ply(str(path))


# ── the camera ───────────────────────────────────────────────────────────

def _app():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_target_is_in_the_middle_and_z_is_up() -> None:
    _app()
    view = CloudView()
    view.resize(400, 300)
    view._target = np.array([10.0, -5.0, 40.0])
    centre = view.project(np.array([[10.0, -5.0, 40.0]]))[0]
    assert np.allclose(centre, [200.0, 150.0], atol=1e-6)
    above = view.project(np.array([[10.0, -5.0, 140.0]]))[0]
    assert above[1] < centre[1]                     # higher z: higher on screen
    view.orbit(np.pi / 2, 0.0)
    turned = view.project(np.array([[10.0, -5.0, 40.0]]))[0]
    assert np.allclose(turned, centre, atol=1e-6)   # the target stays put


def _render(view):
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    img = view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes())
    return arr.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()


def test_points_are_drawn_where_the_camera_says() -> None:
    app = _app()
    view = CloudView()
    view.resize(400, 300)
    view.show()
    app.processEvents()
    if not view.isValid():
        view.close()
        pytest.skip("no OpenGL context here")
    try:
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(-60, 60, 400), rng.uniform(-60, 60, 400),
                               rng.uniform(0, 80, 400)])
        view.point_px = 6.0
        view.set_cloud(pts, fit=True)
        img = _render(view)
        assert img.shape == (300, 400, 3)
        background = np.array([8, 10, 13])
        lit = ~(np.abs(img.astype(int) - background) <= 2).all(axis=2)
        assert lit.sum() > 400, lit.sum()
        # The highest point lands near-white where `project` puts it.
        top = pts[np.argmax(pts[:, 2])]
        x, y = view.project(top.reshape(1, 3))[0]
        patch = img[int(y) - 4: int(y) + 5, int(x) - 4: int(x) + 5].astype(int)
        assert (patch.min(axis=2) > 150).any(), patch.max(axis=(0, 1))
        # Turn the orbit: the picture changes.
        view.orbit(0.9, 0.2)
        assert (lit != ~(np.abs(_render(view).astype(int) - background) <= 2).all(axis=2)).mean() > 0.01
    finally:
        view.close()
