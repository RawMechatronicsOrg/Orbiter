"""A degenerate IPPE answer costs the frame, not the detector.

Seen on the rig with the 8x10 board partly in view: solvePnPGeneric handed
back NaN for a thin corner set, scoring it against the prior raised out of
scipy's SVD, and the worker logged "right detector raised" for the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from orbiter_native.cvcore import BoardSpec, Intrinsics, build_board
from orbiter_server import calibration

K = Intrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=(0.0,) * 5)


def _view(board, n: int = 12):
    """`n` chessboard corners of `board` seen from straight ahead, as the
    detector would hand them over: (N, 1, 2) float32 and (N, 1) int32."""
    obj = np.asarray(board.getChessboardCorners(), np.float64)[:n]
    rvec = np.array([0.15, -0.1, 0.05])
    tvec = np.array([-obj[:, 0].mean(), -obj[:, 1].mean(), 400.0])
    img, _ = cv2.projectPoints(obj, rvec, tvec, K.K, K.D)
    return (np.asarray(img, np.float32).reshape(-1, 1, 2),
            np.arange(n, dtype=np.int32).reshape(-1, 1))


def test_a_sound_view_still_solves() -> None:
    board = build_board(BoardSpec(8, 10, 44.1, 32.634, cv2.aruco.DICT_5X5_100))
    corners, ids = _view(board)
    out = calibration.estimate_board_pose_disambiguated(corners, ids, board, K, np.eye(3))
    assert out is not None
    R, t, _ = out
    assert np.isfinite(R).all() and np.isfinite(t).all()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_a_nan_candidate_is_no_candidate(monkeypatch) -> None:
    board = build_board(BoardSpec(8, 10, 44.1, 32.634, cv2.aruco.DICT_5X5_100))
    corners, ids = _view(board)
    nan3 = np.full((3, 1), np.nan)

    def broken(*_args, **_kwargs):
        return 2, [nan3, nan3], [nan3, nan3], np.zeros((2, 1))

    monkeypatch.setattr(cv2, "solvePnPGeneric", broken)
    assert calibration.estimate_board_pose_disambiguated(
        corners, ids, board, K, np.eye(3)) is None


def test_one_good_candidate_beside_a_nan_one_is_kept(monkeypatch) -> None:
    board = build_board(BoardSpec(8, 10, 44.1, 32.634, cv2.aruco.DICT_5X5_100))
    corners, ids = _view(board)
    real = cv2.solvePnPGeneric
    nan3 = np.full((3, 1), np.nan)

    def half_broken(*args, **kwargs):
        n, rvecs, tvecs, err = real(*args, **kwargs)
        return n + 1, list(rvecs) + [nan3], list(tvecs) + [nan3], err

    monkeypatch.setattr(cv2, "solvePnPGeneric", half_broken)
    out = calibration.estimate_board_pose_disambiguated(corners, ids, board, K, np.eye(3))
    assert out is not None and np.isfinite(out[0]).all()
