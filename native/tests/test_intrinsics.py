"""Intrinsics solve and the diversity rules that decide whether to trust it.

These run on synthetic views projected through a KNOWN camera matrix, so the
answer is checkable — which is the only way to test a calibration without a
camera and an operator waving a board at it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from orbiter_native.cvcore import BoardSpec, build_board
from orbiter_native.intrinsics import (
    MIN_TILT_SPREAD,
    EyeView,
    PairSample,
    SampleSet,
    describe,
    solve,
)

WH = (1280, 720)
K_TRUE = np.array([[900.0, 0.0, 646.0], [0.0, 905.0, 358.0], [0.0, 0.0, 1.0]])
D_TRUE = np.array([-0.28, 0.09, 0.001, -0.0015, 0.0])   # k3 = 0, as FIX_K3 assumes


@pytest.fixture(scope="module")
def board():
    return build_board(BoardSpec(8, 8, 36.0, 26.64, cv2.aruco.DICT_5X5_100))


def _project(board, tvec, rvec, noise: float, rng) -> EyeView | None:
    obj = board.getChessboardCorners().astype(np.float32)
    ids = np.arange(len(obj), dtype=np.int32).reshape(-1, 1)
    img, _ = cv2.projectPoints(obj, np.asarray(rvec, float), np.asarray(tvec, float),
                               K_TRUE, D_TRUE)
    img = img.reshape(-1, 2)
    if noise:
        img = img + rng.normal(0.0, noise, img.shape)
    inside = ((img[:, 0] >= 0) & (img[:, 0] < WH[0])
              & (img[:, 1] >= 0) & (img[:, 1] < WH[1]))
    if inside.sum() < 10:
        return None
    corners = img[inside].reshape(-1, 1, 2).astype(np.float32)
    kept = ids[inside]
    return EyeView(corners, kept, WH, describe(corners, kept, board, WH))


def _sweep(board, n: int, max_tilt: float, noise: float = 0.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = _project(
            board,
            (rng.uniform(-0.09, 0.09), rng.uniform(-0.06, 0.06), rng.uniform(0.35, 0.75)),
            (rng.uniform(-max_tilt, max_tilt), rng.uniform(-max_tilt, max_tilt),
             rng.uniform(-0.4, 0.4)),
            noise, rng,
        )
        if v is not None:
            out.append(v)
    return out


def _spread(views) -> float:
    ss = SampleSet()
    for v in views:
        ss.add(PairSample(left=v))
    return ss.tilt_spread("left")


def test_solve_recovers_a_known_camera_matrix(board) -> None:
    views = _sweep(board, 16, max_tilt=0.5)
    res, why = solve(views, board, tilt_spread=_spread(views))
    assert res is not None, why
    i = res.intrinsics
    assert abs(i.fx - K_TRUE[0, 0]) < 1.0
    assert abs(i.fy - K_TRUE[1, 1]) < 1.0
    assert abs(i.cx - K_TRUE[0, 2]) < 1.0
    assert abs(i.cy - K_TRUE[1, 2]) < 1.0
    assert np.allclose(i.dist[:4], D_TRUE[:4], atol=2e-3)
    assert res.rms_px < 0.05
    assert res.wh == WH


def test_a_flat_on_set_is_refused_however_good_its_rms(board) -> None:
    """The failure this gate exists for, demonstrated.

    A set of head-on views solves to a near-zero reprojection RMS and a focal
    length wrong by a quarter. Nothing in the result says so — only the tilt
    spread does, which is why it is a separate gate and not a warning.
    """
    views = _sweep(board, 14, max_tilt=0.0)
    spread = _spread(views)
    assert spread < MIN_TILT_SPREAD

    unguarded, why = solve(views, board)          # no tilt_spread passed
    assert unguarded is not None, why
    assert unguarded.rms_px < 0.05                # looks perfect
    assert abs(unguarded.intrinsics.fx - K_TRUE[0, 0]) > 100.0   # and is not

    guarded, reason = solve(views, board, tilt_spread=spread)
    assert guarded is None
    assert "tilt" in reason


def test_tilt_spread_separates_good_sets_from_degenerate_ones(board) -> None:
    assert _spread(_sweep(board, 14, max_tilt=0.0)) < MIN_TILT_SPREAD
    assert _spread(_sweep(board, 14, max_tilt=0.5)) > MIN_TILT_SPREAD


def test_too_few_views_is_refused(board) -> None:
    views = _sweep(board, 3, max_tilt=0.5)
    res, why = solve(views, board, tilt_spread=99.0)
    assert res is None and "views" in why


def test_mixed_frame_sizes_are_refused(board) -> None:
    """One camera matrix cannot describe two resolutions."""
    views = _sweep(board, 10, max_tilt=0.5)
    views[3] = EyeView(views[3].corners, views[3].ids, (1920, 1080), views[3].descriptor)
    res, why = solve(views, board, tilt_spread=99.0)
    assert res is None and "frame sizes" in why


def test_novelty_rejects_a_repeat_and_accepts_a_move(board) -> None:
    rng = np.random.default_rng(3)
    a = _project(board, (0.0, 0.0, 0.5), (0.3, -0.2, 0.1), 0.0, rng)
    ss = SampleSet()
    assert ss.novelty("left", a.descriptor) == float("inf")   # nothing held yet
    ss.add(PairSample(left=a))
    assert ss.novelty("left", a.descriptor) == pytest.approx(0.0, abs=1e-9)

    moved = _project(board, (0.06, 0.04, 0.45), (-0.35, 0.3, -0.2), 0.0, rng)
    assert ss.novelty("left", moved.descriptor) >= SampleSet.novelty_threshold


def test_novelty_is_not_swamped_by_tilt_noise(board) -> None:
    """A board sitting still must not keep looking like a new view.

    Before the tilt terms were weighted, their measurement noise alone cleared
    the threshold and a stationary board accumulated seven "new" views in
    twelve seconds on the live rig.
    """
    rng = np.random.default_rng(5)
    held = _project(board, (0.0, 0.0, 0.5), (0.25, -0.15, 0.05), 0.0, rng)
    ss = SampleSet()
    ss.add(PairSample(left=held))
    for _ in range(20):
        jittered = _project(board, (0.0, 0.0, 0.5), (0.25, -0.15, 0.05), 0.2, rng)
        assert ss.novelty("left", jittered.descriptor) < SampleSet.novelty_threshold


def test_coverage_marks_where_the_board_went(board) -> None:
    """One view lights exactly the cell its corner centroid falls in.

    Checked against the descriptor rather than a hardcoded cell: the board's
    origin is at one of its corners, not its middle, so a view at translation
    (0, 0, z) does not sit centred in the frame.
    """
    rng = np.random.default_rng(1)
    ss = SampleSet()
    v = _project(board, (0.0, 0.0, 0.9), (0.2, 0.1, 0.0), 0.0, rng)
    ss.add(PairSample(left=v))
    cells = ss.coverage("left", grid=6)
    assert cells.sum() == 1
    gx = min(int(v.descriptor.cx * 6), 5)
    gy = min(int(v.descriptor.cy * 6), 5)
    assert cells[gy, gx]

    # A second view elsewhere in the frame lights a second cell.
    far = _project(board, (-0.14, -0.09, 0.5), (0.2, 0.1, 0.0), 0.0, rng)
    ss.add(PairSample(left=far))
    assert ss.coverage("left", grid=6).sum() == 2


def test_pairs_are_kept_for_the_stereo_solve(board) -> None:
    """Both-eye samples must be identifiable — stereoCalibrate needs them."""
    rng = np.random.default_rng(2)
    v = _project(board, (0.0, 0.0, 0.5), (0.2, 0.1, 0.0), 0.0, rng)
    ss = SampleSet()
    ss.add(PairSample(left=v, right=v))
    ss.add(PairSample(left=v))
    assert len(ss) == 2
    assert len(ss.paired()) == 1
    assert len(ss.views("left")) == 2 and len(ss.views("right")) == 1


def test_as_config_carries_the_resolution(board) -> None:
    """Stored intrinsics without their frame size are unusable, so refuse to
    produce them that way."""
    views = _sweep(board, 16, max_tilt=0.5)
    res, why = solve(views, board, tilt_spread=_spread(views))
    assert res is not None, why
    cfg = res.as_config()
    assert cfg["width"] == WH[0] and cfg["height"] == WH[1]
    assert cfg["views"] == res.n_views
    assert len(cfg["dist"]) == 5
