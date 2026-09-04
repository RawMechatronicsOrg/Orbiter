"""The optimisations rolled in together: the reach band in image space, the
still-frame average, the voxel merge, the right eye's pose from the left's,
the stripe gate, and the half-size display path."""

from __future__ import annotations

import numpy as np
import pytest

from orbiter_native.laser import find_stripe_pixels
from orbiter_native.scan import PointCloud, ScanFrame, stripe_rows
from orbiter_native.scanworker import STILL_BATCH, ScanWorker, _still, average_still
from orbiter_native.stereo import compose_right_pose
from orbiter_native.worker import EyeWorker, board_wanted, stripe_wanted

from test_pure import _stripe_frame
from test_stereo_scan import KL, _plane, _rig


# ── the reach band ───────────────────────────────────────────────────────

def _row_of_reach(reach: float, eye: str = "left") -> float:
    """The row where a sheet point at that reach, straight ahead, appears."""
    rig = _rig()
    z = np.sqrt(reach ** 2 - 74.0 ** 2)
    xyz = np.array([[0.0, 74.0, z]])
    if eye == "left":
        return float(KL.fy * 74.0 / z + KL.cy)
    return float(rig.project_right(xyz)[0, 1])


@pytest.mark.parametrize("eye", ["left", "right"])
def test_stripe_rows_hold_the_reach_and_nothing_far_from_it(eye) -> None:
    rig, plane = _rig(), _plane()
    band = stripe_rows(plane, rig, (150.0, 450.0), (1280, 720), eye)
    assert band is not None
    y0, y1 = band
    # The near end of the reach falls below this synthetic frame (row ~870 of
    # 720), so the band is checked from where the sheet comes into view.
    for reach in (220.0, 300.0, 450.0):
        assert y0 <= _row_of_reach(reach, eye) < y1, (eye, reach, band)
    for reach in (80.0, 900.0):
        row = _row_of_reach(reach, eye)
        assert not (y0 <= row < y1), (eye, reach, band, row)


def test_stripe_rows_is_none_when_the_sheet_cannot_be_seen() -> None:
    rig, plane = _rig(), _plane()
    assert stripe_rows(plane, rig, (5.0, 20.0), (1280, 720)) is None
    assert stripe_rows(plane, rig, (150.0, 450.0), (0, 0)) is None


def test_the_search_stays_inside_the_band_in_frame_coordinates() -> None:
    bgr, _ = _stripe_frame(y0=60.0)
    inside = find_stripe_pixels(bgr, rows=(40, 100))
    assert inside.ok and 55 <= inside.y.min() and inside.y.max() <= 80
    outside = find_stripe_pixels(bgr, rows=(100, 200))
    assert not outside.ok
    assert not find_stripe_pixels(bgr, rows=(150, 100)).ok


# ── averaging while still ────────────────────────────────────────────────

def _frame(rng, noise: float, n: int = 200) -> ScanFrame:
    scan = np.arange(n)
    truth = np.column_stack([scan * 3.0, np.zeros(n), 100.0 + 20.0 * np.sin(scan / 30.0)])
    pts = truth + rng.normal(0, noise, truth.shape)
    return ScanFrame(points_board=pts, points_camera=pts.copy(), scanlines=scan)


def test_average_still_cuts_noise_and_drops_flickers() -> None:
    truth = np.column_stack([np.arange(200) * 3.0, np.zeros(200),
                             100.0 + 20.0 * np.sin(np.arange(200) / 30.0)])
    # The trimmed mean of five keeps ~85% of the plain mean's noise
    # reduction (sqrt(5) = 2.24x -> ~1.9x); judged over seeds, not one draw.
    gains = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        one = _frame(rng, 1.0)
        batch = [_frame(rng, 1.0) for _ in range(STILL_BATCH)]
        single = np.linalg.norm(one.points_board - truth, axis=1).std()
        averaged = average_still(batch)
        assert len(averaged) == 200
        gains.append(single / np.linalg.norm(averaged - truth, axis=1).std())
    assert np.median(gains) > 1.8, gains
    rng = np.random.default_rng(2)
    one = _frame(rng, 1.0)
    batch = [_frame(rng, 1.0) for _ in range(STILL_BATCH)]
    # A scanline seen in one frame of five is a flicker.
    extra = _frame(rng, 1.0, n=201)
    assert len(average_still(batch[:-1] + [extra])) == 200
    assert average_still([one]) is one.points_board


def test_still_is_a_pose_within_half_a_millimetre_and_a_tenth_of_a_degree() -> None:
    from scipy.spatial.transform import Rotation
    R, t = np.eye(3), np.array([0.0, 0.0, 500.0])
    assert _still((R, t), (R, t + [0.3, 0.0, 0.0]))
    assert not _still((R, t), (R, t + [0.8, 0.0, 0.0]))
    turned = Rotation.from_rotvec([0.0, 0.0, np.radians(0.3)]).as_matrix() @ R
    assert not _still((R, t), (turned, t))


def _bank(worker, frame, R, t) -> bool:
    """What one pair does: sort the frame into the batch under the offer
    lock, then merge whatever the batch gave up outside it."""
    return worker._merge(worker._bank(frame, R, t))


def test_the_worker_batches_still_frames_and_flushes_on_motion() -> None:
    rng = np.random.default_rng(3)
    worker = ScanWorker()
    R, t = np.eye(3), np.array([0.0, 0.0, 500.0])
    for i in range(STILL_BATCH - 1):
        assert not _bank(worker, _frame(rng, 1.0), R, t)      # held
    assert len(worker._batch) == STILL_BATCH - 1
    assert _bank(worker, _frame(rng, 1.0), R, t)              # the fifth flushes
    assert 190 <= len(worker.cloud) <= 200 and len(worker._batch) == 0   # voxel-merged
    _bank(worker, _frame(rng, 1.0), R, t)                     # a new batch begins
    assert len(worker._batch) == 1
    assert _bank(worker, _frame(rng, 1.0), R, t + [5.0, 0.0, 0.0])   # motion: flush
    assert len(worker._batch) == 1


def test_a_batch_ends_where_the_scanline_axis_flips() -> None:
    """A scanline id counts along columns in one frame and along rows in
    another, and `along_x` is decided per frame from the lit pixels' extent.
    Averaged across a flip, the same id would put a column onto a row."""
    rng = np.random.default_rng(11)
    worker = ScanWorker()
    R, t = np.eye(3), np.array([0.0, 0.0, 500.0])
    down = _frame(rng, 1.0)
    across = _frame(rng, 1.0)
    across.along_x = not down.along_x
    _bank(worker, down, R, t)
    assert len(worker._batch) == 1
    assert _bank(worker, across, R, t)                # the flip ends the batch
    assert len(worker._batch) == 1 and worker._batch[0] is across


def test_a_new_session_is_not_judged_against_the_last_one_s_pose() -> None:
    """`set_active(False)` used to leave the batch pose behind, so the first
    frame of the next scan was measured for stillness against wherever the
    board stood when the last one ended."""
    rng = np.random.default_rng(12)
    worker = ScanWorker()
    R, t = np.eye(3), np.array([0.0, 0.0, 500.0])
    worker.set_active(True)
    _bank(worker, _frame(rng, 1.0), R, t)
    assert worker._batch_pose is not None
    worker.set_active(False)
    assert worker._batch_pose is None and not worker._batch


def test_nearest_gaps_is_what_the_pairing_gets_to_choose_from() -> None:
    """The two cameras free-run, so what matters is not the offset but its
    distribution: for each left frame, how close the nearest right one is."""
    from orbiter_native.rigcheck import nearest_gaps

    left = np.arange(0.0, 0.33, 0.033)
    assert np.allclose(nearest_gaps(left, left), 0.0)
    # A steady 11 ms offset: every frame is 11 ms from its nearest partner,
    # which a 4 ms window pairs never and a 20 ms window pairs always.
    gaps = nearest_gaps(left, left + 0.011)
    assert np.allclose(gaps, 0.011)
    assert (gaps <= 0.004).mean() == 0.0 and (gaps <= 0.020).mean() == 1.0
    # The nearest partner can be the one before, not only the one after.
    assert np.allclose(nearest_gaps(np.array([0.100]), np.array([0.090, 0.140])), 0.010)
    assert not len(nearest_gaps(np.empty(0), left))


# ── the voxel merge ──────────────────────────────────────────────────────

def test_voxel_merge_collapses_repeats_to_their_mean() -> None:
    rng = np.random.default_rng(4)
    truth = rng.uniform(-50, 50, (300, 3))
    cloud = PointCloud(voxel_mm=0.5)
    for _ in range(10):
        cloud.add(truth + rng.normal(0, 0.01, truth.shape))
    assert 300 <= len(cloud) <= 420                    # some straddle a voxel edge
    pts = cloud.points()
    d = np.linalg.norm(pts[:, None, :] - truth[None, :, :], axis=2).min(axis=1)
    assert np.median(d) < 0.02
    lo, hi = cloud.bounds()
    assert np.all(lo <= truth.min(axis=0)) and np.all(hi >= truth.max(axis=0))
    cloud.clear()
    assert len(cloud) == 0 and cloud.bounds() is None


def test_voxel_merge_keeps_first_appearance_order() -> None:
    cloud = PointCloud()
    cloud.add(np.array([[10.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    assert cloud.points()[:, 0].tolist() == [10.0, -3.0, 5.0]
    cloud.add(np.array([[10.1, 0.0, 0.0]]))            # same voxel as the first
    assert len(cloud) == 3 and np.isclose(cloud.points()[0, 0], 10.05)


# ── the right eye's pose from the left's ─────────────────────────────────

def test_composed_right_pose_projects_like_the_rig() -> None:
    from test_stereo_scan import BOARD2_R, BOARD2_T, KR, _project
    rig = _rig()
    pts_board = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0], [30.0, -40.0, 50.0]])
    xyz_left = pts_board @ BOARD2_R.T + BOARD2_T
    want = rig.project_right(xyz_left)
    R_r, t_r = compose_right_pose(BOARD2_R, BOARD2_T, rig.geom)
    got = _project(KR, R_r, t_r, pts_board)
    assert np.allclose(got, want, atol=1e-6)


# ── the stripe gate ──────────────────────────────────────────────────────

def test_the_right_eye_skips_the_board_only_while_scanning() -> None:
    assert board_wanted("left", True) and board_wanted("left", False)
    assert board_wanted("right", False) and not board_wanted("right", True)


def test_stripe_is_wanted_only_where_it_can_be_placed() -> None:
    assert stripe_wanted("left", True, False)
    assert not stripe_wanted("left", False, True)
    assert stripe_wanted("right", False, True)
    assert not stripe_wanted("right", True, False)


def test_a_glint_in_one_frame_of_five_does_not_move_the_average() -> None:
    rng = np.random.default_rng(8)
    batch = [_frame(rng, 0.1) for _ in range(STILL_BATCH)]
    batch[2].points_board[50] += [0.0, 0.0, 40.0]           # one frame, one glint
    truth = np.column_stack([np.arange(200) * 3.0, np.zeros(200),
                             100.0 + 20.0 * np.sin(np.arange(200) / 30.0)])
    averaged = average_still(batch)
    assert abs(averaged[50, 2] - truth[50, 2]) < 0.5      # a mean would be 8 mm off


def test_the_worker_asks_for_full_frames_only_for_the_line_fit() -> None:
    w = EyeWorker("left", gpu=True)
    assert not w._needs_full_bgr()                        # laser off
    w.set_laser(True)
    assert w._needs_full_bgr()                            # calibration-mode line fit
    w.set_scan_mode(True)
    assert not w._needs_full_bgr()                        # scanning: the GPU scores it
    w.set_scan_mode(False)
    assert w._needs_full_bgr()
    w.set_laser(False)
    assert not w._needs_full_bgr()
