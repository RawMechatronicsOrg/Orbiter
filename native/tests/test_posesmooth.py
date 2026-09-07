"""The board's pose taken as the median of its neighbours in time: a lone
jump is outvoted, a steady sweep passes through, and the worker places a
frame's points through the pose that came back."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orbiter_native.posesmooth import WINDOW, PoseSmoother, median_pose


def _rot(deg_z: float) -> np.ndarray:
    return Rotation.from_euler("z", deg_z, degrees=True).as_matrix()


def test_a_lone_jump_is_outvoted() -> None:
    poses = [(np.eye(3), np.array([0.0, 0.0, 500.0])) for _ in range(7)]
    poses[3] = (_rot(0.6), np.array([3.0, -2.0, 503.0]))              # the middle frame jumped
    r, t = median_pose(poses, 3)
    assert np.allclose(t, [0.0, 0.0, 500.0])
    assert np.allclose(r, np.eye(3), atol=1e-9)


def test_a_steady_sweep_passes_through_its_middle() -> None:
    poses = [(_rot(0.05 * k), np.array([1.0 * k, 0.0, 500.0])) for k in range(7)]
    r, t = median_pose(poses, 3)
    assert np.allclose(t, [3.0, 0.0, 500.0])                         # no lag: the middle
    assert np.allclose(r, _rot(0.15), atol=1e-9)


def test_frames_come_back_from_the_middle_of_the_window_then_the_tail() -> None:
    sm = PoseSmoother()
    out = []
    for k in range(WINDOW + 2):
        out += sm.push(k, np.eye(3), np.array([1.0 * k, 0.0, 500.0]), 0.033 * k)
    # The first frame back is the window's middle, once the window is full;
    # then one per push, each with the median of its own window.
    assert [item for item, _, _ in out] == [3, 4, 5]
    assert [float(t[0]) for _, _, t in out] == [3.0, 4.0, 5.0]
    assert sm.pending == 3
    tail = sm.flush()
    assert [item for item, _, _ in tail] == [6, 7, 8]
    assert sm.pending == 0 and sm.flush() == []
    # A frame whose neighbours are gone is placed with the trailing median.
    assert float(tail[-1][2][0]) == 5.0                                 # median of 2..8 is 5


def test_a_gap_in_time_ends_the_window() -> None:
    sm = PoseSmoother()
    for k in range(4):
        sm.push(k, np.eye(3), np.array([0.0, 0.0, 500.0]), 0.033 * k)
    out = sm.push(9, np.eye(3), np.array([0.0, 0.0, 600.0]), 5.0)     # five seconds later
    assert [item for item, _, _ in out] == [0, 1, 2, 3]                 # the old window, flushed
    assert all(float(t[2]) == 500.0 for _, _, t in out)                 # placed among their own
    assert sm.pending == 1


def test_the_window_must_have_a_middle() -> None:
    with pytest.raises(ValueError):
        PoseSmoother(window=4)


def test_the_worker_places_a_frame_through_the_smoothed_pose() -> None:
    from orbiter_native.scan import ScanFrame, ScanParams
    from orbiter_native.scanworker import ScanWorker

    xyz_cam = np.array([[0.0, 0.0, 400.0], [10.0, 0.0, 400.0], [0.0, 0.0, 900.0]])
    frame = ScanFrame(points_board=xyz_cam.copy(), points_camera=xyz_cam.copy(),
                      pixels_left=np.array([[5.0, 10.0], [6.0, 10.0], [7.0, 10.0]]),
                      weights=np.ones(3), scanlines=np.array([5, 6, 7]), colours=None)
    own_t = np.array([0.0, 0.0, 500.0])
    face_out = np.diag([1.0, -1.0, -1.0])            # the board's frame: z toward the camera
    # Placed through a pose 2 mm from its own: the points move by those 2 mm
    # (board frame is R^T (x - t)), and what leaves the volume is dropped.
    ScanWorker._place(frame, None, own_t, face_out, np.array([0.0, 0.0, 502.0]),
                      ScanParams())
    assert frame.pose_smooth_mm == pytest.approx(2.0)
    assert frame.n_kept == 2 and frame.n_rejected_volume == 1           # 900 mm out: under the board
    assert np.allclose(frame.points_board, [[0.0, 0.0, 102.0], [10.0, 0.0, 102.0]])
    assert len(frame.pixels_left) == 2 and len(frame.weights) == 2 and len(frame.scanlines) == 2


def test_the_pose_comes_from_the_corners_seen_in_every_recent_frame(monkeypatch) -> None:
    """A corner that blinks in on a full detection pass must not move the
    pose: the eye is solved through the corners it has had all along."""
    from orbiter_native import scanworker
    from orbiter_native.scanworker import ScanInput, ScanWorker

    calls = []

    def fake_estimate(corners, ids, board, k, R_prev=None):
        calls.append(sorted(int(i) for i in np.asarray(ids).ravel()))
        return np.eye(3), np.array([0.0, 0.0, 500.0]), 0.0

    monkeypatch.setattr(scanworker, "estimate_pose", fake_estimate)
    w = ScanWorker()

    def inp(ids):
        ids = np.array(ids, np.int32).reshape(-1, 1)
        return ScanInput(1.0, None, np.eye(3), np.array([1.0, 2.0, 3.0]), (64, 48),
                         corners=np.zeros((len(ids), 1, 2), np.float32), ids=ids)

    base = list(range(20))
    first = w._steady("left", inp(base), None, object())
    assert first.board_t.tolist() == [1.0, 2.0, 3.0] and not calls       # nothing to compare with
    out = w._steady("left", inp(base + [99]), None, object())             # one blinks in
    assert calls[-1] == base and len(out.ids) == 20 and out.board_t[2] == 500.0
    n = len(calls)
    same = w._steady("left", inp(base), None, object())                   # nothing dropped
    assert len(calls) == n and same.board_t.tolist() == [1.0, 2.0, 3.0]
    few = w._steady("left", inp(base[:5]), None, object())                # too few in common
    assert len(calls) == n and len(few.ids) == 5
    # The eyes keep their own histories.
    assert w._steady("right", inp(base + [7, 8]), None, object()).board_t.tolist() == [1.0, 2.0, 3.0]
