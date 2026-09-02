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
from orbiter_native.laser import StripePixels
from orbiter_native.laserplane import from_config as plane_from_config
from orbiter_native.scan import PointCloud, ScanParams, ScanVolume, scan_frame
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


def test_project_right_matches_opencv_with_distortion() -> None:
    from orbiter_native.stereo import StereoResult
    kr = Intrinsics(fx=898.0, fy=903.0, cx=634.0, cy=364.0,
                    dist=(-0.28, 0.09, 0.001, -0.0005, 0.02))
    geom = StereoResult(R=R_TRUE, T=T_TRUE, E=np.zeros((3, 3)), F=np.zeros((3, 3)),
                        rms_px=0.1, n_views=12, wh=WH)
    rig = StereoRig(KL, kr, geom)
    rng = np.random.default_rng(2)
    xyz = np.stack([rng.uniform(-150, 150, 200), rng.uniform(-100, 100, 200),
                    rng.uniform(400, 800, 200)], axis=1)
    want = _project(kr, R_TRUE, T_TRUE, xyz)
    assert np.allclose(rig.project_right(xyz), want, atol=1e-6)
    assert np.isnan(rig.project_right(np.array([[0.0, 0.0, -50.0]]))).all()


#: The laser sheet of the synthetic rig: normal along y, 74 mm from the left
#: camera and containing its optical axis — the geometry measured on the real
#: rig, where the sheet's offset is the one-eye baseline.
def _plane():
    return plane_from_config({"n": [0.0, 1.0, 0.0], "d": 74.0, "rms_mm": 0.3,
                              "points": 1000, "frames": 10,
                              "width": WH[0], "height": WH[1]}, WH)


#: These tests are about the veto, the sheet and the cylinder; the reach gate
#: has its own tests, so it is opened wide here.
WIDE = ScanParams(range_mm=(0.0, 1e9))

#: A board 600 mm out, facing the camera, its centre under the sheet: the
#: centred, face-out frame `cvcore.estimate_pose` hands out.
BOARD2_R = np.diag([1.0, -1.0, -1.0])
BOARD2_T = np.array([0.0, 74.0, 600.0])


def _curve(z_of_x, n: int = 600) -> np.ndarray:
    """A 3D curve lying on the sheet, in the left camera's frame."""
    x = np.linspace(-80.0, 80.0, n)
    return np.stack([x, np.full_like(x, 74.0), z_of_x(x)], axis=1)


def _pixels(K: Intrinsics, R, t, xyz, wh=WH, sigma: float = 1.0) -> StripePixels:
    """The stripe as one eye reports it: a soft profile of rows per column."""
    img = _project(K, R, t, xyz)
    xs, ys, ws = [], [], []
    for u, v in img:
        col = int(round(u))
        for row in range(int(np.floor(v)) - 2, int(np.floor(v)) + 4):
            w = 255.0 * np.exp(-0.5 * ((row - v) / sigma) ** 2)
            if w >= 40 and 0 <= col < wh[0] and 0 <= row < wh[1]:
                xs.append(col)
                ys.append(row)
                ws.append(int(w))
    return StripePixels(x=np.array(xs, np.int32), y=np.array(ys, np.int32),
                        w=np.array(ws, np.uint8), wh=wh, along_x=True, reason=None)


def _join(*parts: StripePixels) -> StripePixels:
    return StripePixels(x=np.concatenate([p.x for p in parts]),
                        y=np.concatenate([p.y for p in parts]),
                        w=np.concatenate([p.w for p in parts]),
                        wh=parts[0].wh, along_x=True, reason=None)


def test_scan_recovers_a_curve_on_the_plane() -> None:
    """Points come from the sheet, at sub-pixel centroids, in the board frame."""
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: 500.0 + 30.0 * np.sin(x / 25.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)

    out = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T, WIDE)
    assert out.reason is None, out.reason
    assert out.n_kept > 250
    assert out.n_confirmed > 0.95 * out.n_pixels
    x, y, z = out.points_camera.T
    assert np.abs(y - 74.0).max() < 1e-9                 # on the sheet by construction
    # The end columns of a synthetic stripe are sampled from one side only
    # and sit a fraction of a pixel off; the algorithm is judged inside them.
    mid = np.abs(x) < 78.0
    assert np.abs(z - (500.0 + 30.0 * np.sin(x / 25.0)))[mid].max() < 0.5
    # Board frame: 600 - z above a board facing the camera.
    xb, _, zb = out.points_board.T
    assert np.allclose(zb, 100.0 - 30.0 * np.sin(xb / 25.0), atol=0.5)


def test_scan_vetoes_what_the_right_eye_did_not_see() -> None:
    """A red wire in the left eye lands on the sheet somewhere — but not where
    the right eye saw stripe. And a flank hidden from the right eye yields
    nothing, rather than an unverified point."""
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: np.full_like(x, 500.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)

    # A wire 44 mm nearer the camera than the sheet, running alongside the
    # stripe in the image. Its rays meet the sheet far out — at about 1.2 m.
    wire = np.stack([np.linspace(-80.0, 80.0, 600), np.full(600, 30.0),
                     np.full(600, 500.0)], axis=1)
    both = _join(left, _pixels(KL, np.eye(3), np.zeros(3), wire))
    out = scan_frame(rig, plane, both, right, BOARD2_R, BOARD2_T, WIDE)
    assert out.reason is None, out.reason
    assert out.n_kept > 250
    # The wire's pixels are out; the stripe's — every row of it — are in.
    assert 0.95 * left.count <= out.n_confirmed <= 1.05 * left.count
    assert np.abs(out.points_camera[:, 2] - 500.0).max() < 0.5   # not averaged in

    # The right eye misses the stretch x in [10, 40]: those scanlines go.
    keep = ~((truth[:, 0] > 10.0) & (truth[:, 0] < 40.0))
    right_gap = _pixels(KR, R_TRUE, T_TRUE, truth[keep])
    out = scan_frame(rig, plane, left, right_gap, BOARD2_R, BOARD2_T, WIDE)
    assert out.n_rejected_unconfirmed > 30
    # Up to the confirmation slack: 3 px in the right eye is 1.7 mm here, and
    # a stripe row off-centre is another 2.6 px — about 3 mm at each edge.
    x = out.points_camera[:, 0]
    assert not ((x > 14.0) & (x < 36.0)).any()


def test_scan_needs_the_plane_a_board_pose_and_both_eyes() -> None:
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: np.full_like(x, 500.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)
    r = scan_frame(rig, None, left, right, BOARD2_R, BOARD2_T, WIDE)
    assert r.n_kept == 0 and "laser plane" in r.reason
    r = scan_frame(rig, plane, left, right, None, None, WIDE)
    assert r.n_kept == 0 and "board pose" in r.reason
    empty = StripePixels(wh=WH, reason="no stripe")
    assert scan_frame(rig, plane, empty, right, BOARD2_R, BOARD2_T, WIDE).n_kept == 0
    assert scan_frame(rig, plane, left, empty, BOARD2_R, BOARD2_T, WIDE).n_kept == 0


def test_scan_drops_points_outside_the_cylinder() -> None:
    """The cylinder is what removes the bench and the wall, with no recognition."""
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: np.full_like(x, 500.0))        # 100 mm above the board
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)
    assert scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T, WIDE).n_kept > 250

    low = ScanParams(range_mm=(0.0, 1e9), volume=ScanVolume(height_mm=50.0))
    out = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T, low)
    assert out.n_kept == 0 and out.n_rejected_volume > 250

    # The curve spans +/-80 mm in x; a 10 mm radius keeps the middle only.
    narrow = ScanParams(range_mm=(0.0, 1e9), volume=ScanVolume(radius_mm=10.0))
    kept = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T, narrow).n_kept
    assert 0 < kept < 60

    # On the board's own surface: the calibration target, not the subject.
    on_board = _curve(lambda x: np.full_like(x, 600.0))
    lb = _pixels(KL, np.eye(3), np.zeros(3), on_board)
    rb = _pixels(KR, R_TRUE, T_TRUE, on_board)
    assert scan_frame(rig, plane, lb, rb, BOARD2_R, BOARD2_T, WIDE).n_kept == 0


def test_volume_edges() -> None:
    v = ScanVolume(height_mm=400.0, radius_mm=200.0, floor_mm=3.0)
    pts = np.array([
        [0.0, 0.0, 200.0],      # inside
        [0.0, 0.0, 1.0],        # below the floor: the board's own surface
        [0.0, 0.0, 401.0],      # above the ceiling
        [201.0, 0.0, 200.0],    # outside the radius
        [150.0, 150.0, 200.0],  # inside a 200 mm box, outside the disc
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
