"""Rolling shutter: the readout solve on synthetic frames with a true rolling
shutter, and the per-row correction it enables.

The synthetic board follows a smooth trajectory; each corner of each frame is
rendered at its own row's instant (the row depends on the instant, so it is
found by iteration), then given 0.2 px of noise. The solve sees only corners,
IDs and capture instants — what the app collects — and must recover the
readout time, its sign, and refuse when the board barely moved.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orbiter_native.cvcore import BoardSpec, Intrinsics, build_board
from orbiter_native.laser import StripePixels
from orbiter_native.laserplane import LaserPlane
from orbiter_native.rolling import (
    MIN_SKEW_PX,
    Motion,
    MotionView,
    Readout,
    _board_points,
    _project,
    poses_at,
    solve_readout,
    to_board,
    twist,
)
from orbiter_native.scan import ScanParams, ScanVolume, scan_frame
from orbiter_native.stereo import StereoRig, StereoResult

SPEC = BoardSpec(squares_x=8, squares_y=8, square_length_mm=36.0,
                 marker_length_mm=26.64, aruco_dict_id=5)
W, H = 1920, 1080
K = Intrinsics(fx=1400.0, fy=1400.0, cx=960.0, cy=540.0, dist=(-0.3, 0.1, 0.0, 0.0, 0.0))
T_TRUE = 0.0213


@pytest.fixture(scope="module")
def board():
    return build_board(SPEC)


_BASE = Rotation.from_euler("xyz", [np.pi, 0.15, 0.0]).as_matrix()


def _trajectory(kind: str):
    """Board→camera pose (cvcore frame) as a function of time."""
    def pose(t: float):
        if kind == "twist+tilt":
            rv = np.array([0.25 * np.sin(2 * np.pi * 1.3 * t),
                           0.2 * np.sin(2 * np.pi * 0.9 * t + 1),
                           0.3 * np.sin(2 * np.pi * 1.1 * t + 2)])
            tr = np.array([40 * np.sin(2 * np.pi * 0.8 * t),
                           25 * np.sin(2 * np.pi * 0.6 * t + 0.5),
                           500.0 + 30 * np.sin(2 * np.pi * 0.5 * t)])
        elif kind == "slide":
            rv = np.zeros(3)
            tr = np.array([120 * np.sin(2 * np.pi * 1.0 * t), 0.0, 500.0])
        else:                                                   # barely moving
            rv = np.array([0.01 * np.sin(2 * np.pi * 0.5 * t), 0.0, 0.0])
            tr = np.array([3 * np.sin(2 * np.pi * 0.5 * t), 0.0, 500.0])
        return Rotation.from_rotvec(rv).as_matrix() @ _BASE, tr
    return pose


def _render(board, kind: str, T: float, n: int = 60, fps: float = 30.0,
            noise: float = 0.2, seed: int = 0, gap_after: int | None = None,
            isolate: int | None = None) -> list[MotionView]:
    """Frames of the moving board with a true rolling shutter of `T`.
    `gap_after` puts a one-second pause after that frame — two bursts —
    and `isolate` sets one frame half a second apart from both sides."""
    rng = np.random.default_rng(seed)
    pts, ids = _board_points(board)
    pose = _trajectory(kind)
    views = []
    for k in range(n):
        t0 = k / fps + rng.normal(0, 0.0005)
        if gap_after is not None and k > gap_after:
            t0 += 1.0
        if isolate is not None:
            t0 += 0.5 if k == isolate else (1.0 if k > isolate else 0.0)
        R, t = pose(t0)
        y = _project(pts @ R.T + t, K)[:, 1]
        for _ in range(6):            # each corner at its own row's instant
            Rs = np.stack([pose(t0 + yi * T / H)[0] for yi in y])
            ts = np.stack([pose(t0 + yi * T / H)[1] for yi in y])
            cam = np.einsum("mij,mj->mi", Rs, pts) + ts
            xy = _project(cam, K)
            y = xy[:, 1]
        xy = xy + rng.normal(0, noise, xy.shape)
        ok = ((xy[:, 0] > 5) & (xy[:, 0] < W - 5) & (xy[:, 1] > 5) & (xy[:, 1] < H - 5)
              & (cam[:, 2] > 10) & (rng.uniform(size=len(xy)) < 0.85))
        views.append(MotionView(xy[ok].astype(np.float32), ids[ok].astype(np.int32),
                                t0, (W, H)))
    return views


@pytest.mark.parametrize("kind", ["twist+tilt", "slide"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_readout_is_recovered_with_its_sign(board, kind, sign) -> None:
    r, why = solve_readout(_render(board, kind, sign * T_TRUE), board, K)
    assert r is not None, why
    assert abs(r.seconds - sign * T_TRUE) < 0.03 * T_TRUE, (r.seconds, why)
    assert r.sigma_s < 0.02 * T_TRUE
    assert r.skew_px >= MIN_SKEW_PX and r.rms_px < 0.3
    assert (r.width, r.height) == (W, H) and r.views == 60


def test_bursts_are_solved_as_runs_and_lone_frames_dropped(board) -> None:
    """Brisk motion comes in bursts; across a pause the neighbouring poses
    say nothing about the motion inside a frame, so each burst is its own
    run, and a frame with no neighbours at all is left out."""
    r, why = solve_readout(_render(board, "twist+tilt", T_TRUE, n=70, gap_after=34),
                           board, K)
    assert r is not None, why
    assert abs(r.seconds - T_TRUE) < 0.03 * T_TRUE and r.views == 70
    r, why = solve_readout(_render(board, "twist+tilt", T_TRUE, n=61, isolate=30),
                           board, K)
    assert r is not None, why
    assert r.views == 60 and abs(r.seconds - T_TRUE) < 0.03 * T_TRUE


def test_too_little_motion_is_refused(board) -> None:
    r, why = solve_readout(_render(board, "slow", T_TRUE), board, K)
    assert r is None and "too little motion" in why


def test_too_few_frames_are_refused(board) -> None:
    r, why = solve_readout(_render(board, "twist+tilt", T_TRUE, n=10), board, K)
    assert r is None and "need" in why


def test_readout_config_round_trip_and_size_gate() -> None:
    r = Readout(0.0213, W, H, sigma_s=0.0002, skew_px=12.0, rms_px=0.2, views=150)
    back = Readout.from_config(r.as_config(), (W, H))
    assert back == r
    assert Readout.from_config(r.as_config(), (1280, 720)) is None
    assert Readout.from_config({"seconds": "x"}, (W, H)) is None
    assert np.allclose(r.row_offset([0, H / 2, H]), [0.0, 0.01065, 0.0213])


def test_twist_and_poses_at_round_trip() -> None:
    R0 = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    t0 = np.array([10.0, 20.0, 500.0])
    omega_true = np.array([0.5, -0.3, 0.8])
    v_true = np.array([100.0, -50.0, 20.0])
    (R1,), (t1,) = poses_at(R0, t0, omega_true, v_true, np.array([0.033]))
    omega, v = twist(R0, t0, 1.0, R1, t1, 1.033)
    assert np.allclose(omega, omega_true, atol=1e-9)
    assert np.allclose(v, v_true, atol=1e-9)
    (R_half,), _ = poses_at(R0, t0, omega, v, np.array([0.0165]))
    assert np.allclose(R_half, Rotation.from_rotvec(omega * 0.0165).as_matrix() @ R0)


def test_to_board_undoes_the_motion_within_a_frame() -> None:
    """Points on a fixed board surface, read at different rows while the board
    turns: through the corners' pose they scatter, through their own rows'
    poses they land back on the surface."""
    R0 = _BASE
    t0 = np.array([0.0, 0.0, 500.0])
    omega = np.array([0.0, 0.0, 1.2])           # 69°/s roll, a brisk twist
    v = np.array([60.0, 0.0, 0.0])
    truth = np.array([[50.0, 20.0, 30.0], [-40.0, 60.0, 10.0], [0.0, -30.0, 80.0]])
    readout = Readout(0.0213, W, H)
    rows = np.array([100.0, 600.0, 1050.0])
    ref_row = 400.0
    dts = readout.row_offset(rows - ref_row)
    R_m, t_m = poses_at(R0, t0, omega, v, dts)
    xyz_cam = np.einsum("mij,mj->mi", R_m, truth) + t_m
    static = (R0.T @ (xyz_cam - t0).T).T
    # Up to 12.8 ms of that motion on the far rows: half a millimetre wrong.
    assert np.linalg.norm(static - truth, axis=1).max() > 0.3
    motion = Motion(omega=omega, v=v, readout=readout, ref_row=ref_row)
    corrected = motion.to_board(xyz_cam, rows, R0, t0)
    assert np.linalg.norm(corrected - truth, axis=1).max() < 1e-6
    assert np.allclose(to_board(xyz_cam, dts, R0, t0, omega, v), corrected)


def test_motion_between_frames_orders_by_row_instant() -> None:
    readout = Readout(0.02, W, H)
    R0, t0 = _BASE, np.array([0.0, 0.0, 500.0])
    (R1,), (t1,) = poses_at(R0, t0, np.array([0.0, 0.0, 1.0]), np.array([30.0, 0.0, 0.0]),
                            np.array([0.0333]))
    m = Motion.between(R0, t0, 0.0, 540.0, R1, t1, 0.0333, 540.0, readout)
    assert m is not None
    assert np.isclose(m.speed_mm_s, 30.0) and np.isclose(m.spin_deg_s, np.degrees(1.0))
    assert Motion.between(R0, t0, 0.0333, 540.0, R1, t1, 0.0, 540.0, readout) is None


# ── through scan_frame ───────────────────────────────────────────────────

def _rig() -> StereoRig:
    geom = StereoResult(R=np.eye(3), T=np.array([-144.0, 0.0, 0.0]),
                        E=np.zeros((3, 3)), F=np.zeros((3, 3)), rms_px=0.5,
                        n_views=10, wh=(W, H))
    return StereoRig(K, K, geom)


def _stripe(px: np.ndarray) -> StripePixels:
    """Stripe pixels at the rounded positions of (N, 2) projections; rows are
    the scanlines, so a near-vertical stripe gives one centroid per row."""
    xi = np.rint(px[:, 0]).astype(np.int32)
    yi = np.rint(px[:, 1]).astype(np.int32)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return StripePixels(x=xi[ok], y=yi[ok], w=np.full(int(ok.sum()), 200, np.uint8),
                        wh=(W, H), along_x=False, reason=None)


def test_scan_frame_applies_the_motion_per_row() -> None:
    """A stripe whose rows span half the frame, on a board that turns during
    the readout: with a `Motion` the kept points shift by their rows' share
    of the readout, and the frame reports how far."""
    plane = LaserPlane(normal=np.array([0.0, 1.0, 0.0]), d=74.0, rms_mm=0.3,
                       n_points=1000, n_frames=10, wh=(W, H))
    R0, t0 = _BASE, np.array([0.0, 0.0, 480.0])
    zs = np.linspace(220.0, 900.0, 300)                 # depth along the sheet
    xs = 20.0 * np.sin(zs / 60.0)
    xyz_cam = np.column_stack([xs, np.full_like(zs, 74.0), zs])      # on the sheet
    left = _stripe(_project(xyz_cam, K))
    right = _stripe(_rig().project_right(xyz_cam))
    params = ScanParams(range_mm=(0.0, 1e9), jump_mm=0.0, blob_width_px=(1, 24),
                        volume=ScanVolume(height_mm=2000.0, radius_mm=2000.0, floor_mm=-2000.0))
    still = scan_frame(_rig(), plane, left, right, R0, t0, params)
    assert still.n_kept > 100 and still.rs_note == "no motion estimate"
    motion = Motion(omega=np.array([0.0, 0.0, 2.0]), v=np.array([80.0, 0.0, 0.0]),
                    readout=Readout(0.0213, W, H), ref_row=float(left.y.mean()))
    moved = scan_frame(_rig(), plane, left, right, R0, t0, params, motion=motion)
    assert moved.rs_note is None and moved.n_kept == still.n_kept
    assert moved.rs_max_mm > 0.2
    assert np.isclose(moved.speed_mm_s, 80.0)
    # The shift grows with the row's distance from the pose's reference row.
    shift = np.linalg.norm(moved.points_board - still.points_board, axis=1)
    rows = np.rint(_project(still.points_camera, K)[:, 1])
    far = np.abs(rows - motion.ref_row)
    assert np.corrcoef(far, shift)[0, 1] > 0.9
