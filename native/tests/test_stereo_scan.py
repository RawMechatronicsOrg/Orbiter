"""Stereo geometry and the scan pipeline, against a known synthetic rig.

Everything here is checkable because the rig, the board poses and the 3D points
are all chosen rather than measured — which is the only way to test
triangulation without a calibrated pair of cameras on a bench.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from orbiter_native.cvcore import BoardSpec, Intrinsics, build_board
from orbiter_native.intrinsics import EyeView, PairSample, describe
from orbiter_native.laser import LaserLine
from orbiter_native.scan import ScanParams, ScanVolume, PointCloud, scan_frame
from orbiter_native.stereo import StereoRig, calibrate, result_from_config

WH = (1280, 720)
KL = Intrinsics(fx=900.0, fy=905.0, cx=646.0, cy=358.0, dist=(0.0,) * 5)
KR = Intrinsics(fx=898.0, fy=903.0, cx=634.0, cy=364.0, dist=(0.0,) * 5)

#: Right eye sits 200 mm to the left of the left eye's origin (so the object
#: appears shifted), toed in slightly — a plausible 20 cm pair.
R_TRUE = cv2.Rodrigues(np.array([0.0, -0.06, 0.0]))[0]
T_TRUE = np.array([-200.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def board():
    return build_board(BoardSpec(8, 8, 36.0, 26.64, cv2.aruco.DICT_5X5_100))


def _project(K: Intrinsics, R, t, xyz):
    rvec = cv2.Rodrigues(np.asarray(R, float))[0]
    img, _ = cv2.projectPoints(np.asarray(xyz, np.float64), rvec,
                               np.asarray(t, float).reshape(3, 1), K.K, K.D)
    return img.reshape(-1, 2)


def _pair_views(board, rvec, tvec_mm, rng):
    """One simultaneous view of the board by both eyes."""
    obj_m = board.getChessboardCorners().astype(np.float64)      # metres
    obj_mm = obj_m * 1000.0
    ids = np.arange(len(obj_mm), dtype=np.int32).reshape(-1, 1)
    Rb = cv2.Rodrigues(np.asarray(rvec, float))[0]
    in_left = (Rb @ obj_mm.T).T + np.asarray(tvec_mm, float)

    li = _project(KL, np.zeros(3), np.zeros(3), in_left)
    ri = _project(KR, cv2.Rodrigues(R_TRUE)[0].ravel(), T_TRUE, in_left)

    def keep(img):
        return ((img[:, 0] >= 0) & (img[:, 0] < WH[0])
                & (img[:, 1] >= 0) & (img[:, 1] < WH[1]))

    m = keep(li) & keep(ri)
    if m.sum() < 10:
        return None
    lc = li[m].reshape(-1, 1, 2).astype(np.float32)
    rc = ri[m].reshape(-1, 1, 2).astype(np.float32)
    kept = ids[m]
    return PairSample(
        left=EyeView(lc, kept, WH, describe(lc, kept, board, WH)),
        right=EyeView(rc, kept, WH, describe(rc, kept, board, WH)),
    )


def _sweep(board, n=12, seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = _pair_views(
            board,
            (rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(-.3, .3)),
            (rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(500, 800)),
            rng,
        )
        if p is not None:
            out.append(p)
    return out


def test_stereo_calibrate_recovers_the_known_rig(board) -> None:
    pairs = _sweep(board)
    res, why = calibrate(pairs, board, KL, KR, WH)
    assert res is not None, why
    assert res.rms_px < 0.05
    assert np.allclose(res.R, R_TRUE, atol=1e-3)
    assert np.allclose(res.T, T_TRUE, atol=0.5)
    assert abs(res.baseline_mm - 200.0) < 0.5


def test_stereo_needs_simultaneous_views(board) -> None:
    """Half-populated pairs carry no stereo information at all."""
    pairs = [PairSample(left=p.left) for p in _sweep(board)]
    res, why = calibrate(pairs, board, KL, KR, WH)
    assert res is None and "both eyes" in why


def test_stored_geometry_is_refused_at_another_resolution(board) -> None:
    pairs = _sweep(board)
    res, _ = calibrate(pairs, board, KL, KR, WH)
    cfg = res.as_config()
    assert result_from_config(cfg, WH) is not None
    assert result_from_config(cfg, (1920, 1080)) is None
    assert result_from_config(None, WH) is None


#: A board FACING the camera: its own z axis points back out of the printed
#: face, towards the lens, so a thing standing on it has positive board-z.
#: Identity would mean the board's z ran away from the camera and everything
#: above it scored negative — an orientation no physical board has when the
#: cameras can read it.
BOARD_R = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
BOARD_T = np.array([0.0, 0.0, 600.0])


def _rig() -> StereoRig:
    from orbiter_native.stereo import StereoResult
    geom = StereoResult(R=R_TRUE, T=T_TRUE, E=np.zeros((3, 3)), F=np.zeros((3, 3)),
                        rms_px=0.1, n_views=12, wh=WH)
    return StereoRig(KL, KR, geom)


def test_triangulation_recovers_known_points() -> None:
    rig = _rig()
    truth = np.array([[0.0, 0.0, 600.0], [40.0, -25.0, 520.0], [-60.0, 30.0, 700.0]])
    lp = _project(KL, np.zeros(3), np.zeros(3), truth)
    rp = _project(KR, cv2.Rodrigues(R_TRUE)[0].ravel(), T_TRUE, truth)
    got = rig.triangulate(lp, rp)
    assert np.allclose(got, truth, atol=1e-3)
    assert np.all(rig.reprojection_error(got, lp, rp) < 1e-3)


def _laser_on_plane(rig: StereoRig, z_mm: float, y_mm: float):
    """A stripe at a fixed depth, as both eyes would see it.

    It runs mostly ACROSS the baseline, not along it. A stripe parallel to the
    baseline lies along the epipolar lines, and then a point in one image has
    no unique match in the other — the intersection is degenerate no matter how
    good the calibration is. This synthetic rig has a horizontal baseline, so
    the stripe is close to vertical. The physical rig gets the same property
    from the other direction: its sensors are mounted rotated 90 degrees, so
    its near-horizontal stripe crosses a baseline that is vertical in sensor
    coordinates.
    """
    t = np.linspace(-80.0, 80.0, 120)
    truth = np.stack([0.15 * t, y_mm + t, np.full_like(t, z_mm)], axis=1)
    lp = _project(KL, np.zeros(3), np.zeros(3), truth)
    rp = _project(KR, cv2.Rodrigues(R_TRUE)[0].ravel(), T_TRUE, truth)

    def line(pts):
        c = pts.mean(axis=0)
        d = np.linalg.svd(pts - c)[2][0]
        return LaserLine(points=pts.astype(np.float32),
                         inliers=np.ones(len(pts), bool),
                         point=c, direction=d / np.linalg.norm(d),
                         rms_px=0.0, reason=None)

    return line(lp), line(rp), truth


def test_scan_triangulates_a_stripe_into_the_board_frame() -> None:
    """A stripe 100 mm above the board lands 100 mm above it, in board units."""
    rig = _rig()
    left, right, truth = _laser_on_plane(rig, z_mm=500.0, y_mm=0.0)

    out = scan_frame(rig, left, right, BOARD_R, BOARD_T)
    assert out.reason is None, out.reason
    assert out.n_kept > 100
    # Board frame: z is distance from the board towards the camera.
    assert np.allclose(out.points_board[:, 2], 100.0, atol=0.5)
    assert np.ptp(out.points_board[:, 1]) > 140      # the stripe spans the box


def test_scan_drops_points_outside_the_volume() -> None:
    """The box is what removes the bench and the far wall, with no recognition."""
    rig = _rig()
    board_R, board_t = BOARD_R, BOARD_T

    # 100 mm above the board: inside a 400 mm box.
    inside, r_in, _ = _laser_on_plane(rig, z_mm=500.0, y_mm=0.0)
    assert scan_frame(rig, inside, r_in, board_R, board_t).n_kept > 100

    # 500 mm off the board, i.e. beyond the 400 mm ceiling.
    above, r_ab, _ = _laser_on_plane(rig, z_mm=100.0, y_mm=0.0)
    out = scan_frame(rig, above, r_ab, board_R, board_t)
    assert out.n_kept == 0
    assert out.n_rejected_volume > 0

    # On the board's own surface — the calibration target, not the subject.
    on_board, r_on, _ = _laser_on_plane(rig, z_mm=600.0, y_mm=0.0)
    assert scan_frame(rig, on_board, r_on, board_R, board_t).n_kept == 0


def test_scan_requires_both_eyes_and_a_board_pose() -> None:
    rig = _rig()
    left, right, _ = _laser_on_plane(rig, 500.0, 0.0)
    empty = LaserLine(reason="no stripe")
    assert scan_frame(rig, empty, right, BOARD_R, BOARD_T).n_kept == 0
    assert scan_frame(rig, left, empty, BOARD_R, BOARD_T).n_kept == 0
    r = scan_frame(rig, left, right, None, None)
    assert r.n_kept == 0 and "board pose" in r.reason


def test_scan_rejects_points_the_two_eyes_disagree_about() -> None:
    """A right stripe that is not the same physical line must not be accepted.

    Note what does NOT catch this: reprojection error. The match is built as a
    point on the left observation's epipolar line, so the rays meet exactly by
    construction and every point reprojects to ~1e-13 px however wrong the
    match is. Only support on the right eye's actually-detected points notices.
    """
    rig = _rig()
    board_R, board_t = BOARD_R, BOARD_T
    left, right, _ = _laser_on_plane(rig, z_mm=500.0, y_mm=0.0)
    good = scan_frame(rig, left, right, board_R, board_t).n_kept
    assert good > 100

    shifted = LaserLine(points=right.points, inliers=right.inliers,
                        point=right.point + np.array([25.0, 0.0]),
                        direction=right.direction, rms_px=0.0, reason=None)
    out = scan_frame(rig, left, shifted, board_R, board_t,
                     ScanParams(max_reproj_px=1.5))
    assert out.n_kept == 0
    assert out.n_rejected_unconfirmed > 0


def test_volume_edges() -> None:
    v = ScanVolume(height_mm=400.0, half_width_mm=200.0, floor_mm=3.0)
    pts = np.array([
        [0.0, 0.0, 200.0],      # inside
        [0.0, 0.0, 1.0],        # below the floor: the board's own surface
        [0.0, 0.0, 401.0],      # above the ceiling
        [201.0, 0.0, 200.0],    # outside in x
        [0.0, -201.0, 200.0],   # outside in y
    ])
    assert v.contains(pts).tolist() == [True, False, False, False, False]


def test_point_cloud_accumulates_and_writes(tmp_path) -> None:
    pc = PointCloud()
    assert len(pc) == 0 and pc.bounds() is None
    pc.add(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    pc.add(np.empty((0, 3)))
    pc.add(np.array([[-1.0, 0.0, 2.0]]))
    assert len(pc) == 3
    lo, hi = pc.bounds()
    assert lo.tolist() == [-1.0, 0.0, 2.0] and hi.tolist() == [4.0, 5.0, 6.0]

    path = tmp_path / "cloud.ply"
    assert pc.write_ply(str(path)) == 3
    text = path.read_text()
    assert "element vertex 3" in text and "1.0000 2.0000 3.0000" in text

    pc.clear()
    assert len(pc) == 0
