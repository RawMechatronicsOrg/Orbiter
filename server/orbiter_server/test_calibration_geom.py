"""Hand-eye geometry math tests for `calibration.py`.

Pure math — no rig, no camera, no network. Covers:
  * the `build_rig_graph`-based A matrix reduces to the legacy pure-rotation
    pose when `turntable_axis is None`, and adds exactly `C - Rz(az)*C` when set;
  * the TSAI->PARK fix: a hand-eye round-trip recovers a mount planted at the
    rig's NOMINAL look-back orientation `diag(-1,1,-1)` (where TSAI is singular);
  * the corrected G1 claim: an AZ<->EL offset along the EL axis is absorbed
    exactly (residual ~0), a transverse offset is NOT and inflates the AX=XB
    residual.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import calibration
from geom.transforms import el_matrix, homogeneous

# All round-trip tests need the real solver.
pytest.importorskip("cv2")

# Diverse sweep for well-conditioned synthetic solves.
_WIDE_POSES = [(az, el) for az in (-60.0, -30.0, 0.0, 30.0, 60.0)
               for el in (15.0, 40.0, 65.0)]
_NOMINAL = np.diag([-1.0, 1.0, -1.0])   # camera "look-back" == pan=tilt=0
# Explicit full azimuth ring (8 az × 2 el), independent of the module's
# DEFAULT_POSES (which is a reduced DEBUG sweep) — used by the tests that
# exercise full-ring behaviour (cx solve, eccentricity mean-cancellation).
_FULL_RING = [(float(az), el) for az in range(0, 360, 45) for el in (20.0, 50.0)]


def _Rz(az_deg: float) -> np.ndarray:
    return Rotation.from_euler("z", az_deg, degrees=True).as_matrix()


def _se3(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=float)
    return M


def _legacy_arm_pose(az_deg: float, el_deg: float):
    """Verbatim copy of the pre-refactor arm_pose_in_world (pure rotation, t=0)."""
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    Rz = np.array([[np.cos(az), -np.sin(az), 0.0],
                   [np.sin(az),  np.cos(az), 0.0],
                   [0.0,         0.0,        1.0]])
    Ry = np.array([[ np.cos(-el), 0.0, np.sin(-el)],
                   [ 0.0,         1.0, 0.0],
                   [-np.sin(-el), 0.0, np.cos(-el)]])
    return Rz @ Ry, np.zeros(3)


def _make_samples(X, Z, poses, C=None, p_el=None):
    """Fabricate CaptureSamples consistent with mount X and a board fixed in
    world at Z, via `world_target = A_true @ X @ B`  =>  B = inv(A_true @ X) @ Z.

    With `p_el` set, the TRUE EL<-AZ edge carries that translation (a real
    AZ<->EL arm offset) that the solver's A (arm_pose_in_world) does NOT model.
    """
    samples = []
    for az, el in poses:
        if p_el is None:
            R, t = calibration.arm_pose_in_world(az, el, C)
            A_true = _se3(R, t)
        else:
            c = np.array([C[0], C[1], 0.0]) if C else np.zeros(3)
            Az = _se3(_Rz(az), c - _Rz(az) @ c)
            A_true = Az @ homogeneous(el_matrix(el), np.asarray(p_el, float))
        B = np.linalg.inv(A_true @ X) @ Z
        samples.append(calibration.CaptureSample(
            az_deg=az, el_deg=el,
            board_R_cam=B[:3, :3], board_t_cam=B[:3, 3]))
    return samples


def test_arm_pose_reduces_to_legacy_when_C_none():
    for az, el in calibration.DEFAULT_POSES + [(180.0, 90.0), (-180.0, -10.0)]:
        R, t = calibration.arm_pose_in_world(az, el, None)
        R0, _ = _legacy_arm_pose(az, el)
        assert np.allclose(R, R0, atol=1e-12)
        assert np.allclose(t, 0.0, atol=1e-12)


def test_arm_pose_eccentricity_is_C_minus_RzC():
    C = (12.0, -5.0)
    c = np.array([C[0], C[1], 0.0])
    for az, el in [(0.0, 0.0), (30.0, 20.0), (90.0, 45.0), (200.0, 70.0)]:
        R, t = calibration.arm_pose_in_world(az, el, C)
        R0, _ = _legacy_arm_pose(az, el)
        assert np.allclose(R, R0, atol=1e-12)            # rotation independent of C
        assert np.allclose(t, c - _Rz(az) @ c, atol=1e-9)


def test_handeye_roundtrip_recovers_nominal_mount_with_park():
    # Mount planted EXACTLY at the nominal look-back orientation — the point
    # where the old TSAI method returned a 180-flipped X with negative radius.
    X = _se3(_NOMINAL, [150.0, 3.0, 40.0])
    Z = _se3(Rotation.from_euler("xyz", [5, -8, 12], degrees=True).as_matrix(),
             [20.0, -10.0, 300.0])
    samples = _make_samples(X, Z, _WIDE_POSES, C=None)

    R_rec, t_rec = calibration.solve_hand_eye(samples, None)
    assert np.allclose(R_rec, _NOMINAL, atol=1e-6)
    assert np.allclose(t_rec, [150.0, 3.0, 40.0], atol=1e-3)

    result = calibration.derive_geometry(R_rec, t_rec)
    assert result.arm_radius_mm == pytest.approx(150.0, abs=1e-2)
    assert result.camera_offset_mm == pytest.approx(40.0, abs=1e-2)
    assert result.camera_pan_deg == pytest.approx(0.0, abs=1e-2)
    assert result.camera_tilt_deg == pytest.approx(0.0, abs=1e-2)

    rt, rr = calibration._handeye_residual(samples, R_rec, t_rec, None)
    assert rt < 1e-6 and rr < 1e-6                       # fabrication is self-consistent


def test_eccentricity_recovered_with_correct_C_biased_without():
    C = (12.0, -5.0)
    X = _se3(_NOMINAL, [150.0, 8.0, 40.0])
    Z = _se3(np.eye(3), [0.0, 0.0, 300.0])
    samples = _make_samples(X, Z, _WIDE_POSES, C=C)

    _, t_ok = calibration.solve_hand_eye(samples, C)         # correct C
    assert np.allclose(t_ok, [150.0, 8.0, 40.0], atol=1e-2)

    _, t_bad = calibration.solve_hand_eye(samples, None)     # wrong C -> biased
    assert not np.allclose(t_bad, [150.0, 8.0, 40.0], atol=1.0)


def test_transverse_offset_inflates_residual_el_axis_does_not():
    X = _se3(_NOMINAL, [150.0, 0.0, 40.0])
    Z = _se3(np.eye(3), [0.0, 0.0, 300.0])

    # Offset ALONG the EL axis (+Y) is absorbed into a constant X -> ~0 residual.
    s_axis = _make_samples(X, Z, _WIDE_POSES, C=None, p_el=[0.0, 30.0, 0.0])
    Ra, ta = calibration.solve_hand_eye(s_axis, None)
    rt_axis, _ = calibration._handeye_residual(s_axis, Ra, ta, None)
    assert rt_axis < 1e-3

    # TRANSVERSE (X,Z) offset is NOT absorbable -> material misfit.
    s_tr = _make_samples(X, Z, _WIDE_POSES, C=None, p_el=[40.0, -15.0, 25.0])
    Rt, tt = calibration.solve_hand_eye(s_tr, None)
    rt_tr, _ = calibration._handeye_residual(s_tr, Rt, tt, None)
    assert rt_tr > 5.0


def test_board_in_world_mean_recovers_centre_and_eccentricity():
    # Board glued at an eccentric centre; with the correct C the averaged
    # board-in-world centre is exact and eccentricity = centre - C.
    C = (12.0, -5.0)
    X = _se3(_NOMINAL, [150.0, 8.0, 40.0])
    centre = [35.0, -20.0, 12.0]
    Z = _se3(np.eye(3), centre)
    samples = _make_samples(X, Z, _WIDE_POSES, C=C)
    t_z = calibration.board_in_world_mean(samples, X[:3, :3], X[:3, 3], C)
    assert np.allclose(t_z, centre, atol=1e-6)
    assert np.allclose((t_z[0] - C[0], t_z[1] - C[1]), [23.0, -15.0], atol=1e-6)


def test_eccentricity_read_exact_on_full_ring_without_C():
    # The operator leaves turntable_axis unset though the true AZ axis is
    # offset by C_true. Solving in the C=0 gauge puts the world origin ON the
    # axis, so the averaged board centre comes out RELATIVE TO THE AXIS — i.e.
    # directly the eccentricity = centre - C_true. On the full DEFAULT_POSES
    # ring this is exact even though the recovered mount X is biased (the
    # wrong-axis (I-Rz)*C variation mean-cancels over the symmetric ring), so
    # the free eccentricity read needs no known C.
    C_true = (12.0, -5.0)
    X = _se3(_NOMINAL, [150.0, 8.0, 40.0])
    centre = [35.0, -20.0, 12.0]
    Z = _se3(np.eye(3), centre)
    ecc = (centre[0] - C_true[0], centre[1] - C_true[1])       # = (23, -15)
    samples = _make_samples(X, Z, _FULL_RING, C=C_true)
    R_bad, t_bad = calibration.solve_hand_eye(samples, None)   # biased X
    t_z = calibration.board_in_world_mean(samples, R_bad, t_bad, None)
    assert np.allclose(t_z[:2], ecc, atol=1e-3)                # ring read exact


def test_wedge_eccentricity_read_is_biased_without_C():
    # Same eccentric board on the OLD narrow ±45° wedge: the wrong-axis
    # variation does NOT mean-cancel over a narrow span, so the C=0-gauge
    # eccentricity read drifts. This is precisely why DEFAULT_POSES must be a
    # full ring rather than a wedge.
    C_true = (12.0, -5.0)
    X = _se3(_NOMINAL, [150.0, 8.0, 40.0])
    centre = [35.0, -20.0, 12.0]
    Z = _se3(np.eye(3), centre)
    ecc = (centre[0] - C_true[0], centre[1] - C_true[1])
    wedge = [(-40.0, 20.0), (-40.0, 50.0), (0.0, 20.0), (0.0, 45.0),
             (0.0, 70.0), (40.0, 20.0), (40.0, 50.0)]
    samples = _make_samples(X, Z, wedge, C=C_true)
    R_bad, t_bad = calibration.solve_hand_eye(samples, None)
    t_z = calibration.board_in_world_mean(samples, R_bad, t_bad, None)
    assert not np.allclose(t_z[:2], ecc, atol=0.5)             # wedge read drifts


# ── P1: intrinsics from the same ChArUco photos (cv2.calibrateCamera) ────────

def _project_board_views(board, K, poses_rt, W=1920, H=1080):
    """Build CaptureSamples by projecting the board's inner corners through a
    known K at each (rvec_deg, tvec_m) pose — synthetic ChArUco detections."""
    import cv2
    objp = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    ids = np.arange(len(objp), dtype=np.int32).reshape(-1, 1)
    samples = []
    for i, (rvec_deg, tvec_m, el) in enumerate(poses_rt):
        rvec = np.radians(np.asarray(rvec_deg, float))
        tvec = np.asarray(tvec_m, float)
        imgp, _ = cv2.projectPoints(objp, rvec, tvec, K, np.zeros(5))
        samples.append(calibration.CaptureSample(
            az_deg=float(i * 40), el_deg=float(el),
            charuco_corners=imgp.reshape(-1, 1, 2).astype(np.float32),
            charuco_ids=ids, image_wh=(W, H)))
    return samples


def test_calibrate_intrinsics_recovers_K_and_Bi():
    import cv2
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1480.0, 0.0, 955.0], [0.0, 1490.0, 545.0], [0.0, 0.0, 1.0]])
    poses = [   # (rvec_deg, tvec_m, el_deg) — depth/tilt diverse, >=3 elevations
        ((8.0, -5.0, 10.0),   (0.00, -0.02, 0.45), 10.0),
        ((20.0, 5.0, -8.0),   (0.03,  0.01, 0.55), 30.0),
        ((-15.0, 12.0, 5.0),  (-0.02, 0.02, 0.50), 50.0),
        ((5.0, 25.0, 0.0),    (0.01, -0.01, 0.65), 70.0),
        ((30.0, -10.0, 15.0), (0.00,  0.00, 0.40), 10.0),
        ((-25.0, -18.0, -5.0),(0.02,  0.03, 0.60), 30.0),
    ]
    samples = _project_board_views(board, K, poses)
    K0 = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    out = calibration.calibrate_intrinsics(samples, board, K0, np.zeros(5))
    assert out is not None
    intr, rms, kept, bposes = out
    assert rms < 1.0
    assert abs(intr.fx - 1480.0) < 5.0 and abs(intr.fy - 1490.0) < 5.0
    assert abs(intr.cx - 955.0) < 5.0 and abs(intr.cy - 545.0) < 5.0
    _, t0_mm = bposes[0]                       # first tvec [0,-0.02,0.45] m
    assert np.allclose(t0_mm, [0.0, -20.0, 450.0], atol=2.0)   # m->mm


def test_calibrate_intrinsics_gate_rejects_underdiverse():
    import cv2
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1480.0, 0.0, 960.0], [0.0, 1480.0, 540.0], [0.0, 0.0, 1.0]])
    # Only 2 views at one elevation — below the >=6-view / >=3-elevation gate.
    samples = _project_board_views(board, K, [((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), 20.0)] * 2)
    assert calibration.calibrate_intrinsics(samples, board, K.copy(), np.zeros(5)) is None


# ── P2: turntable-axis world-X from photos; cy is a structural gauge null ────

def test_solve_turntable_cx_recovers_planted_cx():
    cx_true = 18.0
    X = _se3(_NOMINAL, [150.0, 6.0, 40.0])
    Z = _se3(np.eye(3), [25.0, -12.0, 8.0])      # eccentric board placement
    samples = _make_samples(X, Z, _FULL_RING, C=(cx_true, 0.0))
    assert abs(calibration.solve_turntable_cx(samples) - cx_true) < 0.5


def test_cy_is_flat_null_and_cx_observable():
    cx_true = 18.0
    X = _se3(_NOMINAL, [150.0, 6.0, 40.0])
    Z = _se3(np.eye(3), [25.0, -12.0, 8.0])
    samples = _make_samples(X, Z, _FULL_RING, C=(cx_true, 0.0))
    # cy is a flat structural null: ANY cy at the true cx fits to ~0 residual,
    # so a 2-D (cx,cy) search would wander cy — only 1-D-over-cx is well-posed.
    for cy in (-30.0, -5.0, 0.0, 5.0, 40.0):
        R, t = calibration.solve_hand_eye(samples, (cx_true, cy))
        rt, _ = calibration._handeye_residual(samples, R, t, (cx_true, cy))
        assert rt < 1e-3
    # cx, in contrast, IS observable: a wrong cx inflates the residual.
    Rw, tw = calibration.solve_hand_eye(samples, (cx_true + 20.0, 0.0))
    rtw, _ = calibration._handeye_residual(samples, Rw, tw, (cx_true + 20.0, 0.0))
    assert rtw > 1.0


def test_solve_turntable_cx_returns_none_when_centered():
    # Axis already at the origin → the search cannot beat the cx=0 baseline, so
    # it returns None (caller keeps the origin gauge) rather than a spurious cx.
    X = _se3(_NOMINAL, [150.0, 6.0, 40.0])
    Z = _se3(np.eye(3), [25.0, -12.0, 8.0])
    samples = _make_samples(X, Z, _FULL_RING, C=(0.0, 0.0))
    assert calibration.solve_turntable_cx(samples) is None


def test_is_full_azimuth_ring_detects_ring_vs_wedge():
    ring = [calibration.CaptureSample(az_deg=az, el_deg=el)
            for az, el in _FULL_RING]
    assert calibration._is_full_azimuth_ring(ring)
    wedge = [calibration.CaptureSample(az_deg=az, el_deg=el) for az, el in
             [(-40, 20), (-40, 50), (0, 20), (0, 45), (0, 70), (40, 20), (40, 50)]]
    assert not calibration._is_full_azimuth_ring(wedge)        # gap > 90°
    ring4 = [calibration.CaptureSample(az_deg=az, el_deg=20.0)
             for az in (0, 90, 180, 270)]
    assert not calibration._is_full_azimuth_ring(ring4)        # < 6 + 90°-periodic


def test_render_gauge_correction_gated_on_extrinsic():
    # With the manual model (extrinsic=None) a set turntable_axis must NOT shift
    # the rendered camera — only the calibrated 6-DOF path applies the gauge.
    from geom.machine_model import compute_rig
    kw = dict(el_deg=30.0, arm_radius_mm=150.0, camera_offset_mm=40.0,
              camera_tilt_deg=0.0, camera_pan_deg=0.0)
    base = compute_rig(extrinsic=None, turntable_axis=None, **kw)
    with_axis = compute_rig(extrinsic=None, turntable_axis=(20.0, -7.0), **kw)
    assert np.allclose(base.camera_pos, with_axis.camera_pos)


# ── P3: EL-axis rocker (tilt + transverse offset) — full-model refine ────────

from geom.rig import MountTransform  # noqa: E402
from geom.transforms import matrix_to_rotvec  # noqa: E402


def _project_chain_samples(board, K, X_true, Z_true, poses, rocker_true):
    """Fabricate samples the way the real pipeline sees them: B_i from the
    TRUE chain (with rocker), corners = exact projections of the board
    through K at B_i. PnP is assumed perfect (board_R/t = B_true)."""
    import cv2
    objp = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    samples = []
    for az, el in poses:
        R, t = calibration.arm_pose_in_world(az, el, None, rocker_true)
        A_true = _se3(R, t)
        B = np.linalg.inv(A_true @ X_true) @ Z_true
        assert B[2, 3] > 0, "board must be in front of the camera"
        rvec, _ = cv2.Rodrigues(B[:3, :3])
        proj, _ = cv2.projectPoints(objp, rvec, B[:3, 3] / 1000.0, K, np.zeros(5))
        samples.append(calibration.CaptureSample(
            az_deg=az, el_deg=el,
            board_R_cam=B[:3, :3], board_t_cam=B[:3, 3],
            obj_points=objp.reshape(-1, 1, 3),
            img_points=proj.astype(np.float32),
        ))
    return samples


def test_rocker_identity_matches_legacy_pose():
    ident = MountTransform(t=(0.0, 0.0, 0.0), rvec=(0.0, 0.0, 0.0))
    for az, el in [(0.0, 0.0), (120.0, 45.0), (300.0, 65.0)]:
        R0, t0 = calibration.arm_pose_in_world(az, el, None, None)
        R1, t1 = calibration.arm_pose_in_world(az, el, None, ident)
        assert np.allclose(R0, R1, atol=1e-12)
        assert np.allclose(t0, t1, atol=1e-12)


def test_refine_recovers_rocker_tilt_and_offset():
    import cv2
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    intr = calibration.Intrinsics(fx=1500.0, fy=1500.0, cx=960.0, cy=540.0,
                                  dist=(0.0, 0.0, 0.0, 0.0, 0.0))
    # Real-rig-like truth (from an actual calibration), plus a planted rocker.
    X_true = _se3(Rotation.from_rotvec([2.3732, -0.1072, -1.9269]).as_matrix(),
                  [293.28, -26.20, 23.70])
    Z_true = _se3(Rotation.from_rotvec([3.1181, -0.2770, -0.0061]).as_matrix(),
                  [-129.04, 163.86, -196.0])
    # Truth: EL-axis tilt 0.06 rad + a 12 mm RADIAL offset of the EL axis.
    # The radial offset is exactly degenerate with the turntable axis (only
    # px − cx is observable — see the gauge note in refine_full_model), so
    # the refine must return it as cx ≈ −12 with the rocker carrying tilt only.
    rocker_true = MountTransform(t=(12.0, 0.0, 0.0), rvec=(0.06, 0.0, 0.0))
    poses = [(float(az), el) for az in range(0, 360, 45)
             for el in (15.0, 40.0, 65.0)]
    samples = _project_chain_samples(board, K, X_true, Z_true, poses, rocker_true)

    # PARK under the IDEAL-axis model: must misfit (that's the symptom).
    R0, t0 = calibration.solve_hand_eye(samples, None)
    stats0 = calibration.board_world_stats(samples, R0, t0, None)
    assert stats0.rms_translation_mm > 1.0

    out = calibration.refine_full_model(samples, intr, R0, t0, stats0, 0.0)
    assert out is not None
    R_X, t_X, z_ref, cx, rocker, az_harm, el_scale, rms_b, rms_a = out
    assert rms_a < 0.05 and rms_a < rms_b * 0.1     # pixels: essentially exact
    assert rocker is not None
    assert rocker.rvec[0] == pytest.approx(0.06, abs=5e-3)
    assert (rocker.t[0] - cx) == pytest.approx(12.0, abs=1.0)  # px − cx gauge
    assert np.allclose(t_X, X_true[:3, 3], atol=2.0)
    # Final chain must be self-consistent at sub-mm level.
    stats1 = calibration.board_world_stats(samples, R_X, t_X, (cx, 0.0), rocker)
    assert stats1.rms_translation_mm < 0.5
    assert stats1.rms_rotation_deg < 0.05


def test_refine_rejected_when_rig_is_ideal():
    # No rocker in the truth: the refine must either return None (no gain)
    # or come back with an essentially-zero rocker — never invent structure.
    import cv2
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    intr = calibration.Intrinsics(fx=1500.0, fy=1500.0, cx=960.0, cy=540.0,
                                  dist=(0.0, 0.0, 0.0, 0.0, 0.0))
    X_true = _se3(Rotation.from_rotvec([2.3732, -0.1072, -1.9269]).as_matrix(),
                  [293.28, -26.20, 23.70])
    Z_true = _se3(Rotation.from_rotvec([3.1181, -0.2770, -0.0061]).as_matrix(),
                  [-129.04, 163.86, -196.0])
    poses = [(float(az), el) for az in range(0, 360, 45)
             for el in (15.0, 40.0, 65.0)]
    samples = _project_chain_samples(board, K, X_true, Z_true, poses, None)
    R0, t0 = calibration.solve_hand_eye(samples, None)
    stats0 = calibration.board_world_stats(samples, R0, t0, None)
    out = calibration.refine_full_model(samples, intr, R0, t0, stats0, 0.0)
    if out is not None:
        rocker = out[4]
        assert rocker is None or (
            np.degrees(np.linalg.norm(rocker.rvec)) < 0.2
            and np.linalg.norm(rocker.t) < 2.0
        )


def test_per_view_diagnostics_shape():
    import cv2
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    intr = calibration.Intrinsics(fx=1500.0, fy=1500.0, cx=960.0, cy=540.0,
                                  dist=(0.0, 0.0, 0.0, 0.0, 0.0))
    X_true = _se3(Rotation.from_rotvec([2.3732, -0.1072, -1.9269]).as_matrix(),
                  [293.28, -26.20, 23.70])
    Z_true = _se3(Rotation.from_rotvec([3.1181, -0.2770, -0.0061]).as_matrix(),
                  [-129.04, 163.86, -196.0])
    poses = [(0.0, 15.0), (120.0, 40.0), (240.0, 65.0)]
    samples = _project_chain_samples(board, K, X_true, Z_true, poses, None)
    z_ref = MountTransform(
        t=tuple(Z_true[:3, 3]), rvec=tuple(matrix_to_rotvec(Z_true[:3, :3])))
    rows = calibration.per_view_diagnostics(
        samples, intr, X_true[:3, :3], X_true[:3, 3], z_ref, None, None)
    assert len(rows) == 3
    for r in rows:
        assert {"az", "el", "dpos_mm", "drot_deg", "reproj_px"} <= set(r)
        assert r["dpos_mm"] < 1e-6 and r["reproj_px"] < 1e-6


def test_disambiguated_pnp_picks_branch_near_prediction():
    """The planar-PnP tie-break must return the pose consistent with the FK
    prediction, and flag how ambiguous the view was."""
    import cv2
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    intr = calibration.Intrinsics(1500.0, 1500.0, 960.0, 540.0,
                                  (0.0, 0.0, 0.0, 0.0, 0.0))
    objp = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    ids = np.arange(len(objp), dtype=np.int32).reshape(-1, 1)
    # An oblique board pose (where the planar ambiguity bites).
    R_true = Rotation.from_euler("xyz", [35.0, 12.0, 4.0], degrees=True).as_matrix()
    t_true_m = np.array([0.01, -0.02, 0.45])
    rvec, _ = cv2.Rodrigues(R_true)
    proj, _ = cv2.projectPoints(objp, rvec, t_true_m, K, np.zeros(5))
    cc = proj.reshape(-1, 1, 2).astype(np.float32)

    out = calibration.estimate_board_pose_disambiguated(cc, ids, board, intr, R_true)
    assert out is not None
    R, t_mm, amb = out
    # Predicted near truth → returns the true branch (sub-degree, mm-accurate).
    assert np.degrees(np.linalg.norm(
        calibration.matrix_to_rotvec(R_true.T @ R))) < 1.0
    assert np.allclose(t_mm, t_true_m * 1000.0, atol=3.0)
    assert amb >= 0.0


def test_refine_recovers_az_encoder_harmonic(monkeypatch):
    """Stage B: a planted AZ-encoder first harmonic (AS5600 eccentricity) is
    recovered on a full ring, and the chain self-consistency collapses.

    The harmonic stage ships default-OFF (see config.az_harmonic_enabled) —
    enable it for this test's scope; the solver machinery must still work."""
    import cv2
    from config import settings
    monkeypatch.setattr(settings, "az_harmonic_enabled", True)
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    intr = calibration.Intrinsics(1500.0, 1500.0, 960.0, 540.0,
                                  (0.0, 0.0, 0.0, 0.0, 0.0))
    X_true = _se3(Rotation.from_rotvec([2.3732, -0.1072, -1.9269]).as_matrix(),
                  [293.28, -26.20, 23.70])
    Z_true = _se3(Rotation.from_rotvec([3.1181, -0.2770, -0.0061]).as_matrix(),
                  [-129.04, 163.86, -196.0])
    harm_true = (1.2, -0.7)            # a_c, a_s in degrees
    objp = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    samples = []
    for az_meas in range(0, 360, 45):
        for el in (15.0, 40.0, 65.0):
            az_true = calibration.az_encoder_corrected(float(az_meas), harm_true)
            R, t = calibration.arm_pose_in_world(az_true, el, None, None)
            A_true = _se3(R, t)
            B = np.linalg.inv(A_true @ X_true) @ Z_true
            rvec, _ = cv2.Rodrigues(B[:3, :3])
            proj, _ = cv2.projectPoints(objp, rvec, B[:3, 3] / 1000.0, K,
                                        np.zeros(5))
            samples.append(calibration.CaptureSample(
                az_deg=float(az_meas), el_deg=el,
                board_R_cam=B[:3, :3], board_t_cam=B[:3, 3],
                obj_points=objp.reshape(-1, 1, 3),
                img_points=proj.astype(np.float32)))

    R0, t0 = calibration.solve_hand_eye(samples, None)
    st0 = calibration.board_world_stats(samples, R0, t0, None)
    out = calibration.refine_full_model(samples, intr, R0, t0, st0, 0.0)
    assert out is not None
    R_X, t_X, z_ref, cx, rocker, az_harm, el_scale, rms_b, rms_a = out
    assert az_harm is not None, "harmonic stage must engage on a full ring"
    assert az_harm[0] == pytest.approx(harm_true[0], abs=0.05)
    assert az_harm[1] == pytest.approx(harm_true[1], abs=0.05)
    assert rms_a < 0.3                              # pixels: essentially exact
    assert np.allclose(t_X, X_true[:3, 3], atol=2.0)   # geometry stays physical


def test_disambiguation_recovers_flipped_views():
    """Plant REAL IPPE twins on a subset of views (the bench failure mode:
    near-frontal flat board) and verify flip-descent restores rigidity."""
    import cv2
    board = calibration._build_board(
        calibration.BoardSpec(5, 7, 30.0, 15.0, cv2.aruco.DICT_4X4_50))
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    intr = calibration.Intrinsics(1500.0, 1500.0, 960.0, 540.0,
                                  (0.0, 0.0, 0.0, 0.0, 0.0))
    X_true = _se3(Rotation.from_rotvec([2.3732, -0.1072, -1.9269]).as_matrix(),
                  [293.28, -26.20, 23.70])
    Z_true = _se3(Rotation.from_rotvec([3.1181, -0.2770, -0.0061]).as_matrix(),
                  [-129.04, 163.86, -196.0])
    objp = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    # Near-frontal elevations (70/78): the polished twin reprojects within
    # noise of the true branch — exactly the bench failure mode. el=15 views
    # keep PARK well-conditioned.
    poses = [(float(az), el) for az in range(0, 360, 45)
             for el in (15.0, 70.0, 78.0)]
    rng = np.random.default_rng(0)   # detector noise makes twins viable,
    samples = []                     # exactly like on the bench
    for az, el in poses:
        R, t = calibration.arm_pose_in_world(az, el, None, None)
        B = np.linalg.inv(_se3(R, t) @ X_true) @ Z_true
        rvec, _ = cv2.Rodrigues(B[:3, :3])
        proj, _ = cv2.projectPoints(objp, rvec, B[:3, 3] / 1000.0, K, np.zeros(5))
        noisy = proj + rng.normal(0.0, 0.4, size=proj.shape)
        samples.append(calibration.CaptureSample(
            az_deg=az, el_deg=el,
            board_R_cam=B[:3, :3], board_t_cam=B[:3, 3],
            obj_points=objp.reshape(-1, 1, 3),
            img_points=noisy.astype(np.float32)))

    # Corrupt a third of the views with their REAL IPPE twin (when it exists).
    # samples are ordered (az, el) groups of 3 — [1::3] picks the el=70 views,
    # the near-frontal ones where twins exist.
    n_twins = calibration.attach_pose_twins(samples, intr, board)
    corrupted = 0
    for s in samples[1::3]:
        if s.board_pose_alt is not None:
            R_alt, t_alt, err = s.board_pose_alt
            s.board_pose_alt = (s.board_R_cam, s.board_t_cam, err)
            s.board_R_cam, s.board_t_cam = R_alt, t_alt
            corrupted += 1
    if corrupted == 0:
        pytest.skip("no view produced a viable IPPE twin at this geometry")

    # The corruption must be visible…
    R0, t0 = calibration.solve_hand_eye(samples, None)
    spread_bad = calibration.board_world_stats(samples, R0, t0, None).rms_rotation_deg
    assert spread_bad > 1.0

    # …and flip-descent must undo it.
    n_flipped = calibration.disambiguate_board_poses(samples)
    assert n_flipped >= corrupted - 1            # allow one twin ≈ truth
    R1, t1 = calibration.solve_hand_eye(samples, None)
    spread_ok = calibration.board_world_stats(samples, R1, t1, None).rms_rotation_deg
    assert spread_ok < 0.2
    assert np.allclose(t1, X_true[:3, 3], atol=2.0)
