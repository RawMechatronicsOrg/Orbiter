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
from orbiter_native.laser import LaserPoints
from orbiter_native.scan import (
    PointCloud, ScanParams, ScanVolume, _cross_polyline, scan_frame,
)
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


def _stripe(rig: StereoRig, xyz_truth: np.ndarray):
    """Project a 3D stripe into both eyes as shape-free centroid sets."""
    lp = _project(KL, np.zeros(3), np.zeros(3), xyz_truth)
    rp = _project(KR, cv2.Rodrigues(R_TRUE)[0].ravel(), T_TRUE, xyz_truth)

    def pts(a):
        # Sample order along the axis the detector would have scanned.
        along_x = np.ptp(a[:, 0]) >= np.ptp(a[:, 1])
        order = np.argsort(a[:, 0] if along_x else a[:, 1])
        return LaserPoints(points=a[order].astype(np.float32),
                           along_x=bool(along_x), reason=None)

    return pts(lp), pts(rp)


def _laser_on_plane(rig: StereoRig, z_mm: float, y_mm: float):
    """A straight stripe at a fixed depth, running ACROSS the baseline.

    A stripe parallel to the baseline lies along the epipolar lines and has no
    unique match in the other image, whatever the calibration quality. This
    synthetic rig has a horizontal baseline, so the stripe is close to
    vertical.
    """
    t = np.linspace(-80.0, 80.0, 160)
    truth = np.stack([0.15 * t, y_mm + t, np.full_like(t, z_mm)], axis=1)
    left, right = _stripe(rig, truth)
    return left, right, truth


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

    # The same stripe against a ceiling below it. Shrinking the box rather than
    # moving the stripe far from the cameras keeps the projection geometry
    # workable, so the VOLUME is demonstrably what rejects the points and not
    # the stereo overlap running out at close range.
    low = ScanParams(volume=ScanVolume(height_mm=50.0))
    out = scan_frame(rig, inside, r_in, board_R, board_t, low)
    assert out.n_kept == 0
    assert out.n_rejected_volume > 100

    # Same for the horizontal extent. The stripe runs +/-80 mm in y, so a
    # 10 mm half-width keeps only the short span across the middle.
    narrow = ScanParams(volume=ScanVolume(half_width_mm=10.0))
    kept = scan_frame(rig, inside, r_in, board_R, board_t, narrow).n_kept
    assert 0 < kept < 40

    # On the board's own surface — the calibration target, not the subject.
    on_board, r_on, _ = _laser_on_plane(rig, z_mm=600.0, y_mm=0.0)
    assert scan_frame(rig, on_board, r_on, board_R, board_t).n_kept == 0


def test_scan_follows_a_stripe_that_is_not_a_line() -> None:
    """The case a line fit destroys, and the reason scanning stopped using one.

    On a real frame from this rig — a drill on the bench — the straight-line
    fit kept 126 of 649 stripe points; the 523 it discarded WERE the object.
    Here the stripe steps 40 mm partway along, and both the near and the far
    part must survive with their own depths.
    """
    rig = _rig()
    t = np.linspace(-80.0, 80.0, 160)
    z = np.where(t < 0.0, 500.0, 460.0)          # a 40 mm step in depth
    truth = np.stack([0.15 * t, t, z], axis=1)
    left, right = _stripe(rig, truth)

    out = scan_frame(rig, left, right, BOARD_R, BOARD_T)
    assert out.reason is None, out.reason
    assert out.n_kept > 120

    z_board = out.points_board[:, 2]
    near = z_board[z_board < 120.0]
    far = z_board[z_board >= 120.0]
    assert len(near) > 50 and len(far) > 50, "both sides of the step must survive"
    assert abs(np.median(near) - 100.0) < 1.0     # 600 - 500
    assert abs(np.median(far) - 140.0) < 1.0      # 600 - 460


def test_scan_does_not_bridge_a_gap_in_the_stripe() -> None:
    """Where the stripe is interrupted, no surface may be invented across it."""
    rig = _rig()
    t = np.concatenate([np.linspace(-80.0, -30.0, 60),
                        np.linspace(30.0, 80.0, 60)])       # a hole in the middle
    truth = np.stack([0.15 * t, t, np.full_like(t, 500.0)], axis=1)
    left, right = _stripe(rig, truth)
    out = scan_frame(rig, left, right, BOARD_R, BOARD_T)
    assert out.reason is None, out.reason
    ys = np.sort(out.points_board[:, 1])
    # Nothing reconstructed inside the gap the detector never saw.
    assert not ((ys > -25.0) & (ys < 25.0)).any()


def test_scan_requires_both_eyes_and_a_board_pose() -> None:
    rig = _rig()
    left, right, _ = _laser_on_plane(rig, 500.0, 0.0)
    empty = LaserPoints(reason="no stripe")
    assert scan_frame(rig, empty, right, BOARD_R, BOARD_T).n_kept == 0
    assert scan_frame(rig, left, empty, BOARD_R, BOARD_T).n_kept == 0
    r = scan_frame(rig, left, right, None, None)
    assert r.n_kept == 0 and "board pose" in r.reason


def test_a_shifted_right_stripe_moves_the_depth_and_is_not_otherwise_caught() -> None:
    """Documents a real limit of two-camera stripe matching, not a bug.

    Nothing purely geometric distinguishes "both eyes saw the same physical
    point" from "both eyes saw laser, at points that do not correspond". The
    match is built as a crossing of the left point's epipolar line with an
    observed right segment, so the rays meet exactly by construction and
    reprojection error is ~1e-13 px however wrong the pairing is. A uniformly
    shifted stripe is self-consistent in both directions too, so a mutual
    left-right check would not catch it either.

    What it does do is change the reconstructed DEPTH. Here a 40 px shift moves
    the surface from 100 mm above the board to about 38 mm — plausible, and
    wrong. Only a gross error leaves the scan volume.

    The independent check that would catch this is the laser plane: a
    triangulated point must lie on the plane the laser sweeps. That is not
    calibrated yet. In practice both eyes look at one physical stripe, so the
    stripes cannot drift apart the way this test makes them.
    """
    rig = _rig()
    left, right, _ = _laser_on_plane(rig, z_mm=500.0, y_mm=0.0)
    truthful = scan_frame(rig, left, right, BOARD_R, BOARD_T)
    assert truthful.n_kept > 100
    assert abs(np.median(truthful.points_board[:, 2]) - 100.0) < 1.0

    shifted = LaserPoints(points=(right.points + np.array([40.0, 0.0], np.float32)),
                          along_x=right.along_x, reason=None)
    out = scan_frame(rig, shifted and left, shifted, BOARD_R, BOARD_T)
    assert out.n_kept > 0                       # still "agrees" — that is the point
    moved = float(np.median(out.points_board[:, 2]))
    assert abs(moved - 100.0) > 30.0, f"depth should have moved, got {moved:.1f}"


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
    blob = path.read_bytes()
    header, _, body = blob.partition(b"end_header\n")
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 3" in header
    xyz = np.frombuffer(body, "<f4").reshape(-1, 3)
    assert xyz.tolist() == [[1, 2, 3], [4, 5, 6], [-1, 0, 2]]

    pc.clear()
    assert len(pc) == 0


def test_stereo_drops_pairs_from_before_the_rig_moved(board) -> None:
    """A set holding two rig geometries must yield the majority one.

    Seen on the rig: the cameras were adjusted after the first captures. The
    first 20 of 140 pairs solved at 132.8 px against 2.0 px for the rest, and
    all together refused at 62.8 px. Correspondence was correct in every pair
    and freeing the intrinsics made it worse — the data, not the model.
    """
    good = _sweep(board, n=16, seed=4)
    # The same board poses seen by a rig whose right eye sat somewhere else.
    global R_TRUE, T_TRUE
    saved = (R_TRUE, T_TRUE)
    try:
        R_TRUE = cv2.Rodrigues(np.array([0.0, 0.25, 0.0]))[0]
        T_TRUE = np.array([-120.0, 30.0, 0.0])
        stale = _sweep(board, n=4, seed=9)
    finally:
        R_TRUE, T_TRUE = saved

    res, why = calibrate(stale + good, board, KL, KR, WH)
    assert res is not None, why
    assert res.rms_px < 0.1
    assert abs(res.baseline_mm - 200.0) < 1.0
    assert sorted(res.dropped) == list(range(len(stale)))


def test_point_cloud_bounds_track_every_add() -> None:
    """Running bounds must equal a min/max over everything added."""
    rng = np.random.default_rng(3)
    pc = PointCloud()
    for _ in range(25):
        pc.add(rng.normal(0, 100, (rng.integers(1, 40), 3)))
    lo, hi = pc.bounds()
    allp = pc.points()
    assert np.allclose(lo, allp.min(axis=0)) and np.allclose(hi, allp.max(axis=0))
    pc.clear()
    assert pc.bounds() is None
    pc.add(np.array([[5.0, 6.0, 7.0]]))
    lo, hi = pc.bounds()
    assert lo.tolist() == [5, 6, 7] and hi.tolist() == [5, 6, 7]


def test_point_cloud_decimated_samples_every_chunk() -> None:
    """The overlay must show old sweeps as well as new ones, and stay small."""
    pc = PointCloud()
    for k in range(50):
        pc.add(np.full((200, 3), float(k)))          # 10000 points, 50 sweeps
    d = pc.decimated(1000)
    assert len(d) <= 1000 + 50                         # at most one extra per chunk
    assert set(np.unique(d[:, 0]).tolist()) == set(float(k) for k in range(50))
    small = PointCloud()
    small.add(np.arange(9.0).reshape(3, 3))
    assert np.array_equal(small.decimated(1000), small.points())


def _cross_polyline_reference(lines, pts, along_x, p):
    """The original all-float64 formulation, kept as the definition."""
    n_lines = len(lines)
    if len(pts) < 2:
        return (np.full((n_lines, 2), np.nan), np.zeros(n_lines, int),
                np.zeros(n_lines))
    a, b = pts[:-1], pts[1:]
    scan_axis = 0 if along_x else 1
    contiguous = np.abs(b[:, scan_axis] - a[:, scan_axis]) <= p.max_segment_gap_px
    ha = np.hstack([a, np.ones((len(a), 1))])
    hb = np.hstack([b, np.ones((len(b), 1))])
    da = lines @ ha.T
    db = lines @ hb.T
    straddles = ((da <= 0) & (db >= 0)) | ((da >= 0) & (db <= 0))
    straddles &= contiguous[None, :]
    denom = da - db
    straddles &= np.abs(denom) > 1e-12
    n_cross = straddles.sum(axis=1)
    match = np.full((n_lines, 2), np.nan)
    angle = np.zeros(n_lines)
    hit = n_cross > 0
    if hit.any():
        first = np.argmax(straddles, axis=1)
        rows = np.flatnonzero(hit)
        seg = first[rows]
        t = da[rows, seg] / denom[rows, seg]
        match[rows] = a[seg] + t[:, None] * (b[seg] - a[seg])
        d = b[seg] - a[seg]
        d_norm = np.linalg.norm(d, axis=1)
        ln = lines[rows, :2]
        l_norm = np.linalg.norm(ln, axis=1)
        good = (d_norm > 1e-9) & (l_norm > 1e-9)
        cosang = np.zeros(len(rows))
        cosang[good] = np.abs(np.einsum("ij,ij->i", ln[good], d[good])) / (
            l_norm[good] * d_norm[good])
        angle[rows] = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    return match, n_cross, angle


def test_cross_polyline_matches_the_float64_definition() -> None:
    """The chunked float32 sweep must agree with the reference on none, one
    and many crossings, across chunk boundaries, with breaks in the stripe."""
    rng = np.random.default_rng(11)
    xs = np.arange(0, 1500, dtype=float)
    ys = 500 + 120 * np.sin(xs / 90) + rng.normal(0, 0.3, len(xs))
    ys[600:640] += 80                                   # a step in the subject
    pts = np.stack([xs, ys], 1)
    pts = np.delete(pts, np.s_[900:960], axis=0)        # a gap: no segment there
    n = 700
    # Lines of every slope: near-vertical ones cross once or not at all,
    # near-horizontal ones cross the wave many times.
    ang = rng.uniform(-np.pi / 2, np.pi / 2, n)
    nx, ny = np.cos(ang), np.sin(ang)
    px = rng.uniform(-100, 1600, n)
    py = rng.uniform(300, 700, n)
    lines = np.stack([nx, ny, -(nx * px + ny * py)], 1)
    lines[::7] *= rng.uniform(0.01, 50, (len(lines[::7]), 1))   # unnormalised too
    p = ScanParams()

    m_ref, c_ref, a_ref = _cross_polyline_reference(lines, pts, True, p)
    m_new, c_new, a_new = _cross_polyline(lines, pts, True, p)

    assert (c_ref == 0).any() and (c_ref == 1).any() and (c_ref > 1).any()
    assert np.array_equal(c_ref, c_new)
    assert np.array_equal(np.isnan(m_ref), np.isnan(m_new))
    assert np.allclose(np.nan_to_num(m_ref), np.nan_to_num(m_new), atol=1e-6)
    assert np.allclose(a_ref, a_new, atol=1e-6)


def _eye_result(side: str, t: float):
    from orbiter_native.worker import EyeResult, EyeStats
    return EyeResult(side, np.zeros((2, 2, 3), np.uint8), EyeStats(),
                     capture_mono=t, wh=(2, 2))


def test_scan_worker_pairs_oldest_first_by_capture_clock() -> None:
    """Pairs come from the capture clock, oldest first, each frame used once;
    a frame whose partner may still be on the way is not given up on."""
    from orbiter_native.scanworker import ScanWorker

    sw = ScanWorker()
    sw.set_active(True)
    sw.offer(_eye_result("left", 0.000))
    sw.offer(_eye_result("right", 0.0005))
    a, b, _, _ = sw._take_pair()
    assert (a.capture_mono, b.capture_mono) == (0.000, 0.0005)

    sw.offer(_eye_result("left", 0.033))
    sw.offer(_eye_result("left", 0.066))
    sw.offer(_eye_result("right", 0.040))
    a, b, _, _ = sw._take_pair()
    assert (a.capture_mono, b.capture_mono) == (0.033, 0.040)
    # 0.066 has no partner yet, and none can be ruled out: wait.
    assert sw._take_pair() is None

    sw.offer(_eye_result("right", 0.100))
    # Now every right that could have matched 0.066 has arrived: skip it.
    assert sw._take_pair() is None
    sw.offer(_eye_result("left", 0.1002))
    a, b, _, _ = sw._take_pair()
    assert (a.capture_mono, b.capture_mono) == (0.1002, 0.100)
    assert not sw._hist["left"] and not sw._hist["right"]

    sw.set_active(False)
    sw.offer(_eye_result("left", 0.2))
    assert sw._take_pair() is None                     # inactive: nothing kept


def test_board_pose_frame_is_centred_with_z_toward_the_camera() -> None:
    """OpenCV's raw board frame has its origin at a corner and z pointing INTO
    the board (measured on a straight-on view). Scanning keeps what is above
    the board, so the pose handed out must be the centred, face-out frame."""
    from orbiter_native.cvcore import charuco_detect, estimate_pose

    spec = BoardSpec(squares_x=8, squares_y=8, square_length_mm=36.0,
                     marker_length_mm=26.64, aruco_dict_id=5)
    board = build_board(spec)
    img = board.generateImage((1600, 1600), marginSize=80)
    corners, ids = charuco_detect(img, board)
    k = Intrinsics(fx=1500.0, fy=1500.0, cx=800.0, cy=800.0, dist=(0.0,) * 5)
    R, t, _ = estimate_pose(corners, ids, board, k)

    # z out of the printed face: toward the camera looking at it.
    assert R[2, 2] < -0.99
    # A point above the board is nearer the camera than the board.
    assert (R @ np.array([0.0, 0.0, 100.0]) + t)[2] < t[2]

    rvec = cv2.Rodrigues(R)[0]

    def px(p):
        return cv2.projectPoints(np.asarray(p, float).reshape(1, 3), rvec,
                                 t.reshape(3, 1), k.K, k.D)[0].ravel()

    # Origin at the board's centre, x to the right, y up in the image.
    bbox = corners.reshape(-1, 2)
    assert np.allclose(px([0, 0, 0]), (bbox.min(axis=0) + bbox.max(axis=0)) / 2, atol=1.5)
    assert px([50, 0, 0])[0] > px([0, 0, 0])[0]
    assert px([0, 50, 0])[1] < px([0, 0, 0])[1]

    # The temporal prior speaks this frame too, and comes back in it from the
    # server's disambiguated solver. The tolerance is solver noise: IPPE and
    # the iterative solve differ by ~0.1° on a straight-on view, where the two
    # planar twins are as alike as they ever get. A frame mismatch would be
    # 180° or 144 mm.
    R2, t2, _ = estimate_pose(corners, ids, board, k, R_predicted=R)
    assert np.allclose(R2, R, atol=0.02) and np.allclose(t2, t, atol=2.0)
