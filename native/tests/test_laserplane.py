"""Laser-plane calibration and its two uses, against a known synthetic plane."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from orbiter_native.cvcore import Intrinsics
from orbiter_native.laserplane import (
    MAX_RMS_MM,
    LaserPlane,
    PlaneCollector,
    fit,
    from_config,
    points_on_board,
    rays,
)

WH = (1280, 720)
K = Intrinsics(fx=960.0, fy=960.0, cx=640.0, cy=360.0,
               dist=(-0.28, 0.09, 0.0, 0.0, 0.0))

#: A plane tilted away from the camera, 400 mm out along its own normal.
N_TRUE = np.array([0.30, 0.10, 0.95])
N_TRUE = N_TRUE / np.linalg.norm(N_TRUE)
D_TRUE = 400.0


def _project(xyz: np.ndarray) -> np.ndarray:
    img, _ = cv2.projectPoints(np.asarray(xyz, np.float64), np.zeros(3), np.zeros(3),
                               K.K, K.D)
    return img.reshape(-1, 2)


def _board(rvec, tvec):
    return cv2.Rodrigues(np.asarray(rvec, float))[0], np.asarray(tvec, float)


def _stripe_on_board(board_R, board_t, n=400, noise=0.0, rng=None):
    """Where the true laser plane meets that board pose, and its pixels.

    400 samples per pose, matching what the detector reports on a real frame
    from this rig (300-600 stripe centroids), so the point-count gate is
    exercised at the density it was chosen for.
    """
    # Intersection of two planes: the laser sheet and the board.
    nb, pb = board_R[:, 2], board_t
    db = float(nb @ pb)
    direction = np.cross(N_TRUE, nb)
    direction = direction / np.linalg.norm(direction)
    # A point on both planes.
    A = np.stack([N_TRUE, nb, direction])
    p0 = np.linalg.solve(A, np.array([D_TRUE, db, 0.0]))
    t = np.linspace(-70.0, 70.0, n)
    xyz = p0 + t[:, None] * direction
    xyz = xyz[xyz[:, 2] > 50.0]
    px = _project(xyz)
    if noise and rng is not None:
        px = px + rng.normal(0.0, noise, px.shape)
    inside = ((px[:, 0] >= 0) & (px[:, 0] < WH[0])
              & (px[:, 1] >= 0) & (px[:, 1] < WH[1]))
    return px[inside], xyz[inside]


#: Board poses whose stripe lands inside the frame for this synthetic sheet.
#: Several, at varied tilts and depths, so a plane is determined and the
#: three-pose gate is met with margin.
GOOD_POSES = (
    ((0.25, -0.15, 0.0), (0, 0, 520)),
    ((-0.2, 0.3, 0.1), (30, -20, 600)),
    ((0.05, -0.35, -0.1), (-25, 15, 460)),
    ((0.15, 0.1, 0.0), (10, 5, 550)),
    ((-0.1, -0.2, 0.05), (-10, -10, 500)),
    ((0.2, 0.25, -0.05), (20, 10, 640)),
)


def test_rays_are_unit_and_point_forward() -> None:
    px = np.array([[640.0, 360.0], [100.0, 80.0], [1200.0, 700.0]])
    d = rays(px, K)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert (d[:, 2] > 0).all()
    # The principal point looks straight down the optical axis.
    assert np.allclose(d[0], [0.0, 0.0, 1.0], atol=1e-6)


def test_points_on_board_recovers_the_true_3d() -> None:
    """A known board pose turns stripe pixels back into the points that made them."""
    board_R, board_t = _board((0.25, -0.15, 0.05), (10.0, -5.0, 520.0))
    px, truth = _stripe_on_board(board_R, board_t)
    assert len(px) > 30
    got = points_on_board(px, K, board_R, board_t)
    assert len(got) == len(truth)
    assert np.allclose(got, truth, atol=1e-6)


def test_fit_recovers_the_known_plane() -> None:
    rng = np.random.default_rng(3)
    col = PlaneCollector()
    # More poses than strictly needed: not every synthetic pose lands the
    # stripe inside the frame, and the fit now requires three that do.
    for rv, tv in GOOD_POSES:
        board_R, board_t = _board(rv, tv)
        px, _ = _stripe_on_board(board_R, board_t, noise=0.15, rng=rng)
        col.add_frame(px, K, board_R, board_t)
    assert len(col) > 200
    assert col.frames >= 3
    plane, why = col.fit(WH)
    assert plane is not None, why
    assert abs(abs(float(plane.normal @ N_TRUE)) - 1.0) < 1e-3
    assert abs(plane.d - D_TRUE) < 1.0
    assert plane.rms_mm < 0.5
    # Not every synthetic pose puts the stripe inside the frame; what matters
    # is that several distinct poses contributed, since one pose alone samples
    # a single line and cannot determine a plane.
    assert plane.n_frames >= 2
    assert plane.n_points > 400


def test_fit_refuses_points_that_are_not_one_sheet() -> None:
    rng = np.random.default_rng(1)
    pts = rng.normal(0.0, 40.0, (600, 3)) + np.array([0.0, 0.0, 500.0])
    plane, why = fit(pts, WH)
    assert plane is None and "sheet" in why


def test_fit_needs_enough_points() -> None:
    plane, why = fit(np.zeros((10, 3)), WH)
    assert plane is None and "need" in why


def test_fit_survives_a_few_strays() -> None:
    """A handful of glints must not tilt a plane the rest determines."""
    rng = np.random.default_rng(5)
    col = PlaneCollector()
    for rv, tv in GOOD_POSES:
        board_R, board_t = _board(rv, tv)
        px, _ = _stripe_on_board(board_R, board_t, noise=0.1, rng=rng)
        col.add_frame(px, K, board_R, board_t)
    clean, why = col.fit(WH)
    assert clean is not None, why

    pts = col.points()
    strays = pts[:15] + np.array([0.0, 0.0, 30.0])     # 30 mm off the sheet
    dirty, why = fit(np.vstack([pts, strays]), WH)
    assert dirty is not None, why
    assert abs(float(dirty.normal @ clean.normal)) > 0.9999
    assert abs(dirty.d - clean.d) < 0.3


def test_distance_and_ray_intersection_are_consistent() -> None:
    """The single-camera path must land exactly on the plane."""
    plane = LaserPlane(normal=N_TRUE, d=D_TRUE, wh=WH)
    board_R, board_t = _board((0.2, -0.1, 0.0), (0.0, 0.0, 520.0))
    px, truth = _stripe_on_board(board_R, board_t)

    d = rays(px, K)
    origins = np.zeros_like(d)
    got = plane.intersect_rays(origins, d)
    assert np.isfinite(got).all()
    assert np.allclose(got, truth, atol=1e-6)
    assert np.abs(plane.distance(got)).max() < 1e-6


def test_distance_rejects_a_point_off_the_sheet() -> None:
    """The check that stereo correspondence cannot make for itself."""
    plane = LaserPlane(normal=N_TRUE, d=D_TRUE, wh=WH)
    on = np.array([N_TRUE * D_TRUE])
    off = on + N_TRUE * 7.5
    assert abs(float(plane.distance(on)[0])) < 1e-9
    assert abs(float(plane.distance(off)[0]) - 7.5) < 1e-6


def test_rays_parallel_to_the_plane_yield_nothing() -> None:
    plane = LaserPlane(normal=np.array([0.0, 0.0, 1.0]), d=500.0, wh=WH)
    parallel = np.array([[1.0, 0.0, 0.0]])
    assert not np.isfinite(plane.intersect_rays(np.zeros((1, 3)), parallel)).any()
    # And a ray pointing away from the plane must not produce a point behind
    # the camera.
    away = np.array([[0.0, 0.0, -1.0]])
    assert not np.isfinite(plane.intersect_rays(np.zeros((1, 3)), away)).any()


def test_stored_plane_is_refused_at_another_resolution() -> None:
    plane = LaserPlane(normal=N_TRUE, d=D_TRUE, rms_mm=0.2, n_points=900, wh=WH)
    cfg = plane.as_config()
    back = from_config(cfg, WH)
    assert back is not None
    assert np.allclose(back.normal, N_TRUE) and abs(back.d - D_TRUE) < 1e-9
    assert from_config(cfg, (1920, 1080)) is None
    assert from_config(None, WH) is None
    assert from_config({"n": [0, 0, 0], "d": 1, "width": WH[0], "height": WH[1]},
                       WH) is None


def test_fit_handles_a_large_collection_without_exhausting_memory() -> None:
    """Regression: the SVD default builds an (N, N) matrix.

    A live collection reached 424452 points, and `np.linalg.svd` without
    full_matrices=False asks for a 424452-squared array — roughly 1.4 TB. It
    took the machine down. Only the three right-singular vectors are needed.
    """
    rng = np.random.default_rng(11)
    n = 120_000
    pts = np.column_stack([rng.uniform(-80, 80, n), rng.uniform(-80, 80, n),
                           np.zeros(n)]) + np.array([0.0, 0.0, 500.0])
    pts[:, 2] += rng.normal(0.0, 0.05, n)
    plane, why = fit(pts, WH)
    assert plane is not None, why
    assert abs(abs(float(plane.normal @ np.array([0.0, 0.0, 1.0]))) - 1.0) < 1e-3


def test_collector_is_bounded() -> None:
    """Unbounded growth is what produced the 424452-point collection."""
    from orbiter_native.laserplane import MAX_POINTS

    col = PlaneCollector()
    board_R, board_t = _board((0.2, -0.1, 0.0), (0.0, 0.0, 500.0))
    step = 0.0
    for _ in range(400):
        step += 10.0                      # always a new pose, so nothing is skipped
        R, t = _board((0.2, -0.1, 0.0), (0.0, 0.0, 500.0 + step))
        px, _ = _stripe_on_board(R, t)
        if len(px):
            col.add_frame(px, K, R, t)
    assert len(col) <= MAX_POINTS
    assert len(col.points()) == len(col)


def test_fit_refuses_a_single_pose() -> None:
    """One pose is one line; every plane through it has zero residual.

    Regression: a 267-point single-frame collection passed the point gate,
    fit at rms 0.000 mm, and overwrote a good 337-frame plane on the server.
    """
    board_R, board_t = _board((0.2, -0.1, 0.0), (0.0, 0.0, 520.0))
    px, _ = _stripe_on_board(board_R, board_t)
    col = PlaneCollector()
    col.add_frame(px, K, board_R, board_t)
    assert len(col) >= 200
    plane, why = col.fit(WH)
    assert plane is None
    assert "pose" in why or "collinear" in why


def test_fit_refuses_collinear_points_even_across_frames() -> None:
    """Frame count alone is not enough: three identical poses are still one line."""
    board_R, board_t = _board((0.2, -0.1, 0.0), (0.0, 0.0, 520.0))
    px, _ = _stripe_on_board(board_R, board_t)
    pts = points_on_board(px, K, board_R, board_t)
    plane, why = fit(np.vstack([pts, pts, pts]), WH, n_frames=3)
    assert plane is None and "collinear" in why
