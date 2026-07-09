"""ChArUco hand-eye geometry calibration.

Drives the rig through a sweep of poses, captures a photo at each, detects
the ChArUco calibration board, then runs `cv2.calibrateHandEye` to derive
the camera's pose in the arm-end frame. From that we read off the rig's
scalar geometry:

  * `arm_radius_mm`     — distance from arm pivot to camera along the arm
  * `camera_offset_mm`  — vertical offset of the camera above the arm pivot
                          (the camera does not sit on the arm centreline)
  * `camera_tilt_deg`   — pitch correction of the optical axis vs nominal
                          (diagnostic only — see `apply_result`)
  * `camera_pan_deg`    — yaw correction (diagnostic only)

`base_height_mm` (world height of the arm pivot above the platform) is
**not** derived from hand-eye — the standard solver is invariant to
arm-pivot translation in world. The user enters it once by tape-measure
when assembling the rig; default is `45 mm`.

# Setup

The ChArUco board is mounted **on** the rotating platform — it is glued to
the turntable, so it co-rotates with AZ and is fixed in the platform
(object) frame. That is the frame this solver works in: `arm_pose_in_world`
gives the camera pose relative to the platform, and the board, being rigid
in that frame, is the stationary hand-eye target.

The camera arm is azimuth-static — only EL moves it; AZ spins the platform
(see `platform_spin` vs `orbit_spin` in scene_graph.py). So mounting the
board on the platform is exactly what gives the sweep its azimuthal
diversity: as the platform turns, the camera sees the board from every
azimuth. A board placed *off* the platform (fixed in the lab) would NOT
work here — with the camera's azimuth fixed, a lab-fixed board's bearing
never changes across the AZ sweep.

Board defaults are tuned for the physical 8×8 ChArUco at 36 mm squares /
26.64 mm markers using `cv2.aruco.DICT_5X5_100` (square measured by ruler,
marker = 36 × 22.2/30). Override the board geometry
from the Machine config panel if you print a different one — e.g.
https://calib.io/pages/camera-calibration-pattern-generator.

Camera intrinsics are calibrated from the SAME ChArUco photos — one
`cv2.calibrateCamera` over the swept views yields `K`, distortion, and the
per-view board poses, which `apply_result` writes back to
`model.camera_fx/fy/cx/cy/distortion`. The model defaults
(`fx=fy=1500, cx=960, cy=540`) are only a seed / fallback used when too few
views are usable for an intrinsics solve.

# Solver notes

The hand-eye solve uses `cv2.calibrateHandEye` with the **PARK** method.
TSAI is avoided: its separable Rodrigues rotation step is singular at the
rig's nominal 180° look-back orientation (the pan=tilt=0 operating point),
where it returns a 180°-flipped mount with a negative `arm_radius`.

The per-pose "gripper" pose `A = arm_pose_in_world(az, el, turntable_axis)`
is built from the shared rig frame graph (`geom.rig.build_rig_graph`), so it
stays in lock-step with the live-pose math. `turntable_axis = (cx, cy)` is
the AZ-rotation-axis eccentricity in the platform XY plane; it defaults to
`None` (axis through the origin), in which case `A` is the old pure-rotation
pose and the result is identical to before. Set it (operator-measured) by
editing `orbiter_state.json` when the platform axis is off-centre.

The AZ↔EL arm offset is **not** a separate parameter — it is absorbed into
the recovered mount X. Only its component along the EL axis is absorbed
exactly; a transverse offset is not representable by a single constant X and
instead inflates the `AX=XB` residual (`rms_translation_mm` /
`rms_rotation_deg`), rising with the EL span. An el-correlated residual is
the signal that the rig has a transverse offset the v0.1 model doesn't fit.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

import camera_io
from esp_proxy import esp
from geom.rig import FRAME_EL, FRAME_WORLD, MountTransform, build_rig_graph
from geom.transforms import homogeneous, matrix_to_rotvec, rotvec_to_matrix
from orbiter_model import model

log = logging.getLogger("orbiter.calibration")


def _ui_log(level: str, msg: str) -> None:
    """Mirror a calibration progress line to BOTH the server log and the UI
    LogPanel (via the WS hub's `log` broadcast), so the operator can watch the
    sweep + solve unfold live. Best-effort — a no-op when the hub isn't running
    (e.g. unit tests). `level` ∈ {'I','W','E'}.
    """
    {"W": log.warning, "E": log.error}.get(level, log.info)(msg)
    try:
        from ws_hub import hub
        hub.emit_log({"level": level, "source": "api", "tag": "calib", "msg": msg})
    except Exception:  # noqa: BLE001
        pass

#: Identity camera mount used when building the hand-eye "gripper" pose A.
#: The real mount X is the unknown the solver recovers, so it must NOT be
#: baked into A — see `arm_pose_in_world`.
_IDENTITY_MOUNT = MountTransform(t=(0.0, 0.0, 0.0), rvec=(0.0, 0.0, 0.0))


# ── data classes ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    aruco_dict_id: int   # one of cv2.aruco.DICT_* int constants


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...]   # k1, k2, p1, p2, k3

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def D(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64)


@dataclass
class CaptureSample:
    az_deg: float
    el_deg: float
    # board→camera pose (3x3 R, (3,) t in mm). Filled AFTER the sweep — either
    # by the one-shot intrinsics calibration (calibrate_intrinsics) or, on
    # fallback, the per-view solvePnP (estimate_board_pose). None until then.
    board_R_cam: np.ndarray | None = None
    board_t_cam: np.ndarray | None = None
    # Raw ChArUco detection retained from the sweep so intrinsics can be
    # calibrated from the same photos (board.matchImagePoints needs these).
    charuco_corners: np.ndarray | None = None
    charuco_ids: np.ndarray | None = None
    image_wh: tuple[int, int] | None = None
    # Matched 3D board points (metres, board frame) ↔ 2D detections (px) —
    # retained for the nonlinear reprojection refine.
    obj_points: np.ndarray | None = None
    img_points: np.ndarray | None = None
    # The OTHER branch of the planar-PnP two-fold ambiguity for this view
    # (IPPE twin): (R, t_mm, reproj_rms_px). None when the view was
    # unambiguous or the twin reprojects much worse. Used by
    # disambiguate_board_poses to flip flipped views.
    board_pose_alt: tuple[np.ndarray, np.ndarray, float] | None = None
    # Phone-IMU readouts at capture time (rig-frame elevation estimate and
    # bank about the lens axis) — a free strain gauge on the camera bracket.
    # None when the IMU was offline during the sweep.
    phone_el_deg: float | None = None
    phone_roll_deg: float | None = None


@dataclass
class CalibrationResult:
    arm_radius_mm: float
    camera_offset_mm: float
    camera_tilt_deg: float
    camera_pan_deg: float
    n_views: int
    n_attempted: int
    rms_translation_mm: float | None
    rms_rotation_deg: float | None
    # Full 6-DOF camera mount X (camera-in-EL frame) from the hand-eye solve,
    # carrying the lateral component the three scalars drop. Additive: lets a
    # caller persist X for the compute_camera_pose_x 6-DOF path. None until set.
    extrinsic: MountTransform | None = None
    # Board-in-world reference pose Z_ref = mean(A·X·B) — the constant board
    # placement for a glued board. Persisted for the post-calibration "Test
    # accuracy" check (predict vs observe at the live pose). None until set.
    board_world: MountTransform | None = None
    # Board placement read off the azimuth-averaged board-in-world mean:
    # eccentricity = board centre − turntable_axis (xy), plus the board-centre
    # height. DIAGNOSTIC ONLY (not persisted) — the y-component is gauge-
    # dependent when turntable_axis is unset, and the read is exact only with a
    # full-ring sweep (see DEFAULT_POSES). None until set.
    board_eccentricity_mm: tuple[float, float] | None = None
    board_height_mm: float | None = None
    # Camera intrinsics solved from the same calibration photos (P1). Written
    # back to the model by apply_result only when `intrinsics_from_photos`;
    # otherwise the prior model intrinsics were used (fallback path).
    camera_fx: float | None = None
    camera_fy: float | None = None
    camera_cx: float | None = None
    camera_cy: float | None = None
    camera_distortion: list[float] | None = None
    intrinsics_rms_px: float | None = None
    n_intrinsic_views: int | None = None
    intrinsics_from_photos: bool = False
    # World-X of the AZ axis solved from the photos (cy is a gauge, held 0).
    # `turntable_axis_solved` is True only when run_calibration estimated it
    # (an operator-set axis is respected and not overwritten).
    turntable_cx_mm: float | None = None
    turntable_axis_solved: bool = False
    # EL-axis correction (tilt about X/Z + transverse offset) recovered by the
    # nonlinear reprojection refine. None when the refine was skipped or
    # rejected — the ideal-axis model is then kept.
    rocker: MountTransform | None = None
    # AZ-encoder first-harmonic coefficients (a_c, a_s) in degrees —
    # az_true = az + a_c·sin(az) + a_s·cos(az). None unless stage B earned it.
    az_harm: tuple[float, float] | None = None
    # EL encoder scale k (el_true = k·el); 1.0 = exact. Also absorbs a linear
    # bracket sag about the EL axis (identical signature).
    el_scale: float = 1.0
    # Reprojection RMS (px) over all views before/after the refine — the
    # honest measure of how much unmodelled structure the rocker absorbed.
    refine_rms_before_px: float | None = None
    refine_rms_after_px: float | None = None
    # Per-view residual table: [{az, el, dpos_mm, drot_deg, reproj_px}, ...]
    # — az/el-correlated patterns here are the fingerprint of whatever
    # structure is STILL unmodelled.
    diagnostics: list[dict[str, float]] | None = None


# ── lazy cv2 import ─────────────────────────────────────────────────────────
# Import cv2 inside functions so the module is importable even without
# opencv installed (e.g., during dev when deps aren't synced yet). Only
# actual calibration runs require cv2.

def _cv2():
    import cv2
    return cv2


# ── detection ───────────────────────────────────────────────────────────────


def _build_board(spec: BoardSpec):
    cv2 = _cv2()
    aruco_dict = cv2.aruco.getPredefinedDictionary(spec.aruco_dict_id)
    # CharucoBoard expects sizes in metres; we work in mm so convert.
    return cv2.aruco.CharucoBoard(
        size=(spec.squares_x, spec.squares_y),
        squareLength=spec.square_length_mm / 1000.0,
        markerLength=spec.marker_length_mm / 1000.0,
        dictionary=aruco_dict,
    )


def detect_board(image_bgr: np.ndarray, board) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (charuco_corners, charuco_ids) or (None, None) if not detected."""
    cv2 = _cv2()
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if corners is None or len(corners) < 4:
        return None, None
    return corners, ids


def estimate_board_pose(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board,
    intrinsics: Intrinsics,
) -> tuple[np.ndarray, np.ndarray] | None:
    """solvePnP from detected charuco corners → (R, t) board→camera, t in mm."""
    cv2 = _cv2()
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsics.K, intrinsics.D)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # solvePnP returned t in board-units (metres because we built board in m);
    # convert to mm for consistency with the rest of the model.
    return R, tvec.flatten() * 1000.0


def estimate_board_pose_disambiguated(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board,
    intrinsics: Intrinsics,
    R_predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Board→camera pose for a NEAR-PLANAR target, with the two-fold planar
    ambiguity broken against a prediction. Returns `(R, t_mm, ambiguity_deg)`
    or None.

    A flat ChArUco board admits TWO poses that reproject almost identically
    (the classic planar PnP ambiguity); at oblique elevations a plain
    `cv2.solvePnP` flips between them, which shows up as a ~10-30° rotation
    error with only a modest translation error. `cv2.solvePnPGeneric` with
    IPPE returns BOTH; we pick the one whose rotation is closest to
    `R_predicted` (the board orientation the calibrated forward kinematics
    expects at this pose). `ambiguity_deg` is the geodesic angle between the
    two candidate rotations — large means the view was genuinely ambiguous and
    the tie-break did real work; ~0 means PnP was unambiguous anyway."""
    cv2 = _cv2()
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None
    try:
        n, rvecs, tvecs, _err = cv2.solvePnPGeneric(
            obj_pts, img_pts, intrinsics.K, intrinsics.D,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None
    if not n:
        return None
    cands = []
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        score = float(np.linalg.norm(matrix_to_rotvec(R_predicted.T @ R)))
        cands.append((score, R, tv.flatten() * 1000.0))
    cands.sort(key=lambda c: c[0])
    ambiguity_deg = (
        float(np.degrees(np.linalg.norm(
            matrix_to_rotvec(cands[0][1].T @ cands[1][1]))))
        if len(cands) > 1 else 0.0
    )
    return cands[0][1], cands[0][2], ambiguity_deg


def attach_pose_twins(samples: Sequence[CaptureSample], intrinsics: Intrinsics,
                      board) -> int:
    """For every sample with retained correspondences, compute BOTH IPPE
    branches of the planar pose, LM-polish each, and store the branch FARTHER
    from the current `board_R_cam` as `board_pose_alt` (the candidate twin).

    Twins are kept RAW (no LM polish at this stage): polishing a twin from a
    clean view collapses it back into the dominant branch (measured: both
    branches → 0.2° apart after LM), which would hide exactly the candidates
    we need. The reprojection gate is deliberately loose (best+10 px) — the
    real arbiter is board rigidity in `disambiguate_board_poses`, where a
    wrong twin can only LOSE (it inflates the rotation spread and the flip is
    rejected). Accepted flips are polished afterwards. Returns how many
    samples got a twin."""
    cv2 = _cv2()
    n_twins = 0
    for s in samples:
        s.board_pose_alt = None
        if (s.obj_points is None or s.img_points is None
                or s.board_R_cam is None):
            continue
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                s.obj_points, s.img_points, intrinsics.K, intrinsics.D,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error as exc:
            log.warning("calibration: IPPE failed on az=%.0f el=%.0f (%s)",
                        s.az_deg, s.el_deg, exc)
            continue
        if not n or len(rvecs) < 2:
            continue
        branches = []
        for rv, tv, er in zip(rvecs, tvecs, np.asarray(errs).ravel()):
            R, _ = cv2.Rodrigues(rv)
            branches.append((R, tv.flatten() * 1000.0, float(er)))
        best_err = min(b[2] for b in branches)
        # The twin = branch farther in rotation from the CURRENT pose.
        far = max(branches, key=lambda b: float(np.linalg.norm(
            matrix_to_rotvec(s.board_R_cam.T @ b[0]))))
        sep = float(np.degrees(np.linalg.norm(
            matrix_to_rotvec(s.board_R_cam.T @ far[0]))))
        if sep < 3.0:                       # both branches ≈ current: unambiguous
            continue
        # Only reject outright garbage — rigidity in disambiguate_board_poses
        # is the real arbiter, and on real noisy frames the raw-IPPE twin of a
        # genuinely flipped view easily sits 10-20 px above the best branch.
        if far[2] > best_err + 50.0:
            continue
        s.board_pose_alt = far
        n_twins += 1
    return n_twins


def _polish_flipped(s: CaptureSample, intrinsics: Intrinsics) -> None:
    """LM-polish a freshly-flipped view from its new (raw-IPPE) branch.
    Kept only if LM stays in that branch — on real noisy data the twin
    branch has a genuine local minimum; if LM slides back toward the old
    branch, the raw twin is retained."""
    cv2 = _cv2()
    old_R = s.board_pose_alt[0] if s.board_pose_alt is not None else None
    rv0, _ = cv2.Rodrigues(s.board_R_cam)
    ok, rv1, tv1 = cv2.solvePnP(
        s.obj_points, s.img_points, intrinsics.K, intrinsics.D,
        rv0, (s.board_t_cam / 1000.0).reshape(3, 1),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return
    R1, _ = cv2.Rodrigues(rv1)
    if old_R is not None:
        d_new = float(np.linalg.norm(matrix_to_rotvec(s.board_R_cam.T @ R1)))
        d_old = float(np.linalg.norm(matrix_to_rotvec(old_R.T @ R1)))
        if d_old < d_new:
            return                      # LM slid back to the old branch
    s.board_R_cam, s.board_t_cam = R1, tv1.flatten() * 1000.0


def disambiguate_board_poses(
    samples: Sequence[CaptureSample],
    intrinsics: Intrinsics | None = None,
) -> int:
    """Resolve the planar-PnP two-fold ambiguity ACROSS the sweep.

    A flat board admits two near-identical-reprojection poses per view; on
    near-frontal views `calibrateCamera` can converge to the WRONG branch for
    a subset of views, which poisons the hand-eye with a large constant
    rotation spread (bench 2026-06-11: mean Δrot ≈ 14°, X lateral +124 mm,
    rocker pinned at its bound).

    The board is rigid on the platform, so the per-view board-in-world
    rotations must agree. Flip-descent: try each view's twin; keep the flip
    if the ROTATION spread of `Z_i = A_i·X·B_i` (X re-solved by PARK each
    time) drops. Rotation spread is independent of cx and of all the
    translation gauges, so the criterion needs no axis/rocker knowledge.
    Returns the number of flipped views; samples are edited in place.
    """
    flippable = [s for s in samples if s.board_pose_alt is not None]
    if not flippable:
        return 0

    def rot_spread() -> float:
        try:
            R_x, t_x = solve_hand_eye(samples, None)
            return board_world_stats(samples, R_x, t_x, None).rms_rotation_deg
        except (ValueError, np.linalg.LinAlgError, _cv2().error):
            return float("inf")

    base = rot_spread()
    n_flipped = 0
    flipped_views: list[CaptureSample] = []
    for _ in range(4):                       # flip-descent passes
        changed = False
        for s in flippable:
            saved = (s.board_R_cam, s.board_t_cam)
            alt = s.board_pose_alt
            s.board_R_cam, s.board_t_cam = alt[0], alt[1]
            trial = rot_spread()
            if trial < base - 1e-3:
                base = trial
                s.board_pose_alt = (saved[0], saved[1], alt[2])
                if s not in flipped_views:
                    flipped_views.append(s)
                n_flipped += 1
                changed = True
            else:
                s.board_R_cam, s.board_t_cam = saved
        if not changed:
            break
    # Polish accepted flips from their new branch (kept only if LM stays
    # there — see _polish_flipped).
    if intrinsics is not None:
        for s in flipped_views:
            if s.obj_points is not None and s.img_points is not None:
                _polish_flipped(s, intrinsics)
    return n_flipped


# ── kinematic arm pose (encoder-only, no geometry priors) ───────────────────


def arm_pose_in_world(
    az_deg: float,
    el_deg: float,
    turntable_axis: tuple[float, float] | None = None,
    rocker: MountTransform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """EL("gripper") pose in the rotating-platform world frame — the hand-eye
    A matrix, built from the shared rig frame graph so it stays in lock-step
    with the live-pose math (`geom.rig.build_rig_graph`).

      * R = Rz(az)·Ry(-el)   — the encoder rotation (unchanged).
      * t = C - Rz(az)·C     — where C = `turntable_axis` is the AZ-axis
        eccentricity in the platform XY plane. `turntable_axis=None` gives
        t = 0, i.e. exactly the old pure-rotation pose, so the solve is
        byte-for-byte the previous behaviour when no eccentricity is set.

    The mount passed to the graph is IDENTITY: the camera mount X is the
    hand-eye unknown and must not be baked into A. The AZ↔EL arm offset has
    no edge to live on here (the EL←AZ graph edge carries zero translation),
    so it is folded into X — only its EL-axis component is absorbed exactly;
    a transverse offset surfaces as an el-dependent `AX=XB` residual.

    `base_height_mm` does NOT enter (a constant world-vertical translation
    cancels in the relative AX=XB motions). Returns (R, t) with t in mm.

    `rocker` is the EL-axis tilt/offset correction (see rig.build_rig_graph);
    None reproduces the ideal-axis pose exactly.
    """
    graph = build_rig_graph(az_deg, el_deg, _IDENTITY_MOUNT, turntable_axis,
                            rocker=rocker)
    T = graph.matrix(FRAME_EL, FRAME_WORLD)
    return T[:3, :3], T[:3, 3]


# ── solver ──────────────────────────────────────────────────────────────────


def solve_hand_eye(
    samples: Sequence[CaptureSample],
    turntable_axis: tuple[float, float] | None = None,
    rocker: MountTransform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`cv2.calibrateHandEye` for the eye-in-hand setup.

    Method is PARK, not TSAI. TSAI's separable Rodrigues rotation step is
    singular when the recovered camera rotation equals the rig's nominal
    180° look-back (`diag(-1, 1, -1)`) — exactly the pan=tilt=0 operating
    point — where it returns a 180°-flipped X with a *negative* arm_radius.
    PARK/HORAUD recover the true X there; PARK is OpenCV's default.

    `turntable_axis` (the AZ-axis eccentricity) is threaded into the A
    matrices; None reproduces the legacy pure-rotation A exactly.

    Returns (R_cam_arm, t_cam_arm) — camera pose in the EL/arm frame, t in mm.
    """
    cv2 = _cv2()
    R_arm_world: list[np.ndarray] = []
    t_arm_world: list[np.ndarray] = []
    R_board_cam: list[np.ndarray] = []
    t_board_cam: list[np.ndarray] = []
    for s in samples:
        R, t = arm_pose_in_world(s.az_deg, s.el_deg, turntable_axis, rocker)
        R_arm_world.append(R)
        t_arm_world.append(t.reshape(3, 1))
        R_board_cam.append(s.board_R_cam)
        t_board_cam.append(s.board_t_cam.reshape(3, 1))
    R_cam_arm, t_cam_arm = cv2.calibrateHandEye(
        R_arm_world, t_arm_world,
        R_board_cam, t_board_cam,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    return R_cam_arm, t_cam_arm.flatten()


@dataclass(frozen=True)
class BoardWorldStats:
    """Statistics of the per-view board-in-world poses `Z_i = A_i · X · B_i`.

    The board is glued to the platform, so `Z_i` must be the SAME rigid
    transform for every view. `R_ref` / `t_ref` are the mean pose (rotation
    averaged on SO(3)) — the board's constant placement, persisted as the
    "Test accuracy" reference. The RMS spreads about that mean are the
    hand-eye consistency residual.

    The residual is the diagnostic for the one approximation in the model: a
    transverse AZ↔EL arm offset is not absorbable into the single constant X,
    so instead of silently biasing the recovered scalars it leaks into this
    spread — and it climbs with elevation. An el-correlated residual is the
    tell that the rig has a transverse offset the v0.1 solve does not fit
    (or that `turntable_axis` is wrong).

    Subtracting `turntable_axis` from `t_ref`'s XY gives the board's
    ECCENTRICITY — how far its centre sits from the AZ rotation axis. On a
    full, non-90°-periodic ring that read is exact even when `turntable_axis`
    is unset (the wrong-axis variation mean-cancels over a symmetric ring).
    """

    R_ref: np.ndarray
    t_ref: np.ndarray
    rms_translation_mm: float
    rms_rotation_deg: float


def board_world_stats(
    samples: Sequence[CaptureSample],
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    turntable_axis: tuple[float, float] | None = None,
    rocker: MountTransform | None = None,
) -> BoardWorldStats:
    """ONE pass over `Z_i = A_i · X · B_i` yielding the mean board-in-world
    pose AND the residual spreads about it. The single source for what used
    to be three near-identical loops (residual / mean centre / reference
    pose)."""
    if not samples:
        raise ValueError("board_world_stats: no samples")
    X = homogeneous(R_cam_arm, t_cam_arm)
    zs: list[np.ndarray] = []
    for s in samples:
        A = homogeneous(*arm_pose_in_world(s.az_deg, s.el_deg, turntable_axis,
                                           rocker))
        B = homogeneous(s.board_R_cam, s.board_t_cam)
        zs.append(A @ X @ B)             # board-in-world; constant for a perfect fit
    t_ref = np.mean([Z[:3, 3] for Z in zs], axis=0)
    R_ref = Rotation.from_matrix(np.array([Z[:3, :3] for Z in zs])).mean().as_matrix()
    t_errs = [float(np.linalg.norm(Z[:3, 3] - t_ref)) for Z in zs]
    r_errs = [
        float(np.degrees(np.linalg.norm(matrix_to_rotvec(R_ref.T @ Z[:3, :3]))))
        for Z in zs
    ]
    return BoardWorldStats(
        R_ref=R_ref,
        t_ref=t_ref,
        rms_translation_mm=float(np.sqrt(np.mean(np.square(t_errs)))),
        rms_rotation_deg=float(np.sqrt(np.mean(np.square(r_errs)))),
    )


def _handeye_residual(
    samples: Sequence[CaptureSample],
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    turntable_axis: tuple[float, float] | None = None,
) -> tuple[float | None, float | None]:
    """RMS hand-eye consistency residual (translation mm, rotation deg) —
    thin view over `board_world_stats`; see its docstring."""
    if not samples:
        return None, None
    s = board_world_stats(samples, R_cam_arm, t_cam_arm, turntable_axis)
    return s.rms_translation_mm, s.rms_rotation_deg


def board_in_world_mean(
    samples: Sequence[CaptureSample],
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    turntable_axis: tuple[float, float] | None = None,
) -> np.ndarray:
    """Azimuth-averaged board centre in the world (platform) frame — thin
    view over `board_world_stats`; see its docstring."""
    return board_world_stats(samples, R_cam_arm, t_cam_arm, turntable_axis).t_ref


def derive_geometry(R_cam_arm: np.ndarray, t_cam_arm: np.ndarray) -> CalibrationResult:
    """Translate the camera-in-arm-end SE3 into the rig's scalar geometry.

    Convention (matches scene_graph.py):
      * arm-end frame at the pivot, +X = arm direction, +Z = up at el=0
      * at (az=0, el=0) the camera sits at (arm_radius, 0, camera_offset);
        the NOMINAL mount rotation is Ry(180°) = diag(−1, 1, −1), i.e. the
        optical axis (+Z_cam) points along −Z_arm — straight DOWN at el=0.
        That matches the physical mount: the phone lens is ⊥ to the arm, so
        el=0 (arm horizontal) ⇔ the lens points straight down.
        Historically this matrix is dubbed "look-back" in tests/docstrings —
        keep in mind the actual axis is lens-down, not −X.

    From those:
      * arm_radius     = t_cam_arm[0]
      * camera_offset  = t_cam_arm[2]
      * camera_tilt    = pitch correction vs the nominal mount rotation
      * camera_pan     = yaw correction
    """
    arm_radius_mm    = float(t_cam_arm[0])
    camera_offset_mm = float(t_cam_arm[2])

    # Nominal mount rotation Ry(180°): +X_cam→−X_arm, +Z_cam→−Z_arm
    # (lens straight down at el=0 — the phone mounts with lens ⊥ arm).
    nominal = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ])
    correction = nominal.T @ R_cam_arm
    # Decompose correction as Rz(pan) · Ry(tilt) (small-angle ZYX-ish).
    # pitch (Y) from r20, yaw (Z) from r10/r00. Sign convention matches
    # `model.camera_tilt_deg` / `camera_pan_deg` already used elsewhere.
    pitch = float(np.arctan2(-correction[2, 0],
                             np.sqrt(correction[2, 1] ** 2 + correction[2, 2] ** 2)))
    yaw   = float(np.arctan2(correction[1, 0], correction[0, 0]))

    # Full 6-DOF mount X (camera-in-EL), carrying the lateral t[1] the scalars
    # drop, so the caller can persist it for the compute_camera_pose_x path.
    extrinsic = MountTransform(
        t=(float(t_cam_arm[0]), float(t_cam_arm[1]), float(t_cam_arm[2])),
        rvec=tuple(float(v) for v in matrix_to_rotvec(R_cam_arm)),
    )

    return CalibrationResult(
        arm_radius_mm=arm_radius_mm,
        camera_offset_mm=camera_offset_mm,
        camera_tilt_deg=float(np.degrees(pitch)),
        camera_pan_deg=float(np.degrees(yaw)),
        n_views=0,           # filled by caller
        n_attempted=0,
        rms_translation_mm=None,
        rms_rotation_deg=None,
        extrinsic=extrinsic,
    )


# ── orchestration (capture sweep + solve + apply) ───────────────────────────


#: Default sweep: a full 360° azimuth RING at 45° steps × 4 elevations. The
#: board co-rotates with the platform, so each azimuth shows the camera a
#: different face of the board — a full ring gives the solve its azimuthal
#: diversity AND makes the free `board_eccentricity_mm` read-off exact even
#: when `turntable_axis` is left unset: the wrong-axis variation mean-cancels
#: over a symmetric ring, whereas a narrow wedge biases it (~6–7 mm).
#:
#: The 45° step is deliberate. A coarse 90°-periodic ring (only 0/90/180/270)
#: aliases the (I−Rz)·C eccentricity structure and collapses the PARK rotation
#: diversity — it breaks the solve outright (verified: ΔR≈180°). 45° steps (8
#: azimuths) include the intermediate bearings and avoid that aliasing.
#:
#: The elevations span a wide out-of-plane tilt range, which is what
#: conditions the intrinsics solve (P1) — a narrow {20,50} span leaves focal
#: length and the principal point measurably noisier.
#:
#: ⚠ EL is CAPPED at 55°. Above that the camera sees the flat board nearly
#: frontally and the planar-PnP two-fold ambiguity becomes live (bench: a 26°
#: twin at el=65, flipped views poisoning the solve with a ~14° constant
#: rotation spread). Staying ≥ ~35° away from frontal kills the twin branch
#: geometrically — same effect as wedging the board, with no hardware change.
#:
#: ⚠ DEBUG SWEEP (reduced for speed while iterating — accuracy traded for
#: turnaround, per the operator). 3 azimuths × 3 elevations = 9 poses. Azimuths
#: use 120° steps (NOT 90°-periodic, which collapses PARK). This keeps the
#: intrinsics solve alive (≥6 views, ≥3 elevations) but is too sparse for the
#: turntable-axis cx solve (that needs the full ≥6-azimuth ring) — cx stays 0.
#: RESTORE the full accurate ring for production:
#:   [(float(az), el) for az in range(0, 360, 45) for el in (10., 25., 40., 55.)]
DEFAULT_POSES: list[tuple[float, float]] = [
    (float(az), el)
    for az in range(0, 360, 120)          # 0, 120, 240 — non-90°-periodic
    for el in (15.0, 35.0, 50.0)
]


def poses_for_preset(preset: str) -> list[tuple[float, float]]:
    """Sweep pose list for a UI accuracy preset. Higher accuracy = denser
    azimuth ring + more elevations (longer sweep):

      * `fast`   — DEFAULT_POSES: 3 az (120° steps) × 3 el = 9 poses. The debug
                   sweep — alive enough for intrinsics but too sparse for the
                   turntable-axis cx solve (needs the full ≥6-azimuth ring).
      * `normal` — 6 az (60° steps) × 4 el = 24 poses. A full ring, so the cx
                   solve and the free eccentricity read-off both engage.
      * `full`   — 8 az (45° steps) × 4 el = 32 poses, widest EL span. The
                   production sweep this module's docstrings describe.

    All presets cap EL at 55° — higher elevations view the flat board nearly
    frontally, where the planar-PnP twin pose goes live (see DEFAULT_POSES).

    An unknown preset falls back to `fast`.
    """
    if preset == "normal":
        return [
            (float(az), el)
            for az in range(0, 360, 60)
            for el in (15.0, 30.0, 45.0, 55.0)
        ]
    if preset == "full":
        return [
            (float(az), el)
            for az in range(0, 360, 45)
            for el in (10.0, 25.0, 40.0, 55.0)
        ]
    return list(DEFAULT_POSES)

#: Settle time after a move before fetching the photo — gives the camera
#: auto-exposure / auto-focus a moment to catch up.
_SETTLE_S = 0.8


def _board_spec_from_model() -> BoardSpec:
    cv2 = _cv2()
    return BoardSpec(
        squares_x=int(model.charuco_squares_x),
        squares_y=int(model.charuco_squares_y),
        square_length_mm=float(model.charuco_square_length_mm),
        marker_length_mm=float(model.charuco_marker_length_mm),
        aruco_dict_id=int(getattr(model, "aruco_dict_id", cv2.aruco.DICT_4X4_50)),
    )


def _intrinsics_from_model() -> Intrinsics:
    return Intrinsics(
        fx=float(model.camera_fx),
        fy=float(model.camera_fy),
        cx=float(model.camera_cx),
        cy=float(model.camera_cy),
        dist=tuple(model.camera_distortion or [0.0] * 5),
    )


#: Minimum ChArUco inner corners a view must show to enter the intrinsics solve.
_MIN_INTRINSIC_CORNERS = 6
#: Minimum number of views and distinct elevations for an intrinsics solve.
#: Distinct elevations matter: out-of-plane tilt is what conditions focal length.
_MIN_INTRINSIC_VIEWS = 6
_MIN_INTRINSIC_ELEVATIONS = 3
#: Reject an intrinsics solve whose reprojection RMS exceeds this — but the gate
#: is SCALED by frame width (RMS in px scales ~linearly with resolution), so a
#: 4080-wide phone still gets ~2× this. The base is debug-lenient: a cropped
#: board over a short sweep runs a bit higher, and a slightly-loose REAL K beats
#: the catastrophic fallback to a wrong-resolution guess.
_MAX_INTRINSIC_RMS_PX = 2.5


def calibrate_intrinsics(
    samples: Sequence[CaptureSample],
    board,
    k0: np.ndarray,
    dist0: np.ndarray,
) -> tuple[Intrinsics, float, list[CaptureSample], list[tuple[np.ndarray, np.ndarray]]] | None:
    """Calibrate camera intrinsics from the swept ChArUco photos in ONE
    `cv2.calibrateCamera` call (cv2 4.x removed `calibrateCameraCharuco`).

    Returns `(intrinsics, rms_px, kept_samples, board_poses)` where
    `board_poses[i]` is the board→camera `(R, t_mm)` of `kept_samples[i]` — the
    per-view rvecs/tvecs the SAME call returns, so the hand-eye `B_i` come for
    free and no second `solvePnP` is needed.

    Returns ``None`` (caller falls back to the model intrinsics + per-view
    `estimate_board_pose`) when there are too few / under-diverse views, the
    image sizes disagree, the solve raises, or the reprojection RMS is poor.
    """
    cv2 = _cv2()
    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    kept: list[CaptureSample] = []
    for s in samples:
        if s.charuco_corners is None or s.charuco_ids is None:
            continue
        obj, imgp = board.matchImagePoints(s.charuco_corners, s.charuco_ids)
        if obj is None or len(obj) < _MIN_INTRINSIC_CORNERS:
            continue
        obj_pts.append(obj.astype(np.float32))
        img_pts.append(imgp.astype(np.float32))
        kept.append(s)

    if (len(kept) < _MIN_INTRINSIC_VIEWS
            or len({round(s.el_deg, 1) for s in kept}) < _MIN_INTRINSIC_ELEVATIONS):
        return None
    wh = kept[0].image_wh
    if wh is None or any(s.image_wh != wh for s in kept):
        return None   # cv2.calibrateCamera needs one image size for all views

    # USE_INTRINSIC_GUESS seeds K from the model; ZERO_TANGENT_DIST|FIX_K3 is
    # the most stable choice for a near-distortion-free phone lens (verified:
    # best principal-point stability, no focal penalty). NEVER FIX_ASPECT_RATIO
    # and never FIX_PRINCIPAL_POINT — those freeze terms we want to measure.
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS
             | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K3)
    try:
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, wh, k0.copy(), dist0.copy(), flags=flags,
        )
    except cv2.error as exc:
        log.warning("calibration: intrinsics solve raised (%s)", exc)
        return None
    rms_gate = _MAX_INTRINSIC_RMS_PX * max(1.0, wh[0] / 1920.0)
    if not np.isfinite(rms) or rms > rms_gate:
        log.warning(
            "calibration: intrinsics rms %.2f px > gate %.2f px (frame %dx%d) "
            "— rejecting", rms, rms_gate, wh[0], wh[1],
        )
        return None

    intrinsics = Intrinsics(
        fx=float(k[0, 0]), fy=float(k[1, 1]),
        cx=float(k[0, 2]), cy=float(k[1, 2]),
        dist=tuple(float(x) for x in dist.ravel()[:5]),   # (k1, k2, p1, p2, k3)
    )
    poses = [
        (cv2.Rodrigues(rv)[0], tv.ravel() * 1000.0)   # board→camera, m→mm
        for rv, tv in zip(rvecs, tvecs)
    ]
    # Retain the matched 3D↔2D points on each kept sample — the nonlinear
    # reprojection refine re-uses exactly the correspondences this solve saw.
    for s, obj, imgp in zip(kept, obj_pts, img_pts):
        s.obj_points, s.img_points = obj, imgp
    return intrinsics, float(rms), kept, poses


#: Bound for the 1-D turntable-axis search (mm). A larger eccentricity than
#: this is a build error, not a calibration target.
_CX_BOUND_MM = 150.0


def _cx_cost(cx: float, samples: Sequence[CaptureSample]) -> float:
    """Hand-eye translation residual at axis ``(cx, 0)`` — the objective for the
    1-D world-X axis search. ``cy`` is pinned to the gauge 0: it is a structural
    null absorbed by the mount, so a 2-D search would wander it."""
    cv2 = _cv2()
    try:
        R, t = solve_hand_eye(samples, (float(cx), 0.0))
        rt = board_world_stats(samples, R, t, (float(cx), 0.0)).rms_translation_mm
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        # A degenerate board-in-world cloud makes calibrateHandEye return a NaN
        # rotation (no raise) → Rotation.mean()/matrix_to_rotvec raise
        # ValueError/LinAlgError; calibrateHandEye itself can raise cv2.error.
        # Treat any of these as an infeasible axis.
        return 1e9
    return float(rt) if np.isfinite(rt) else 1e9


def _is_full_azimuth_ring(samples: Sequence[CaptureSample]) -> bool:
    """True when the captured azimuths cover a full ring densely enough to
    estimate the turntable axis: ≥ 6 distinct bearings, largest gap ≤ 90°, and
    not a purely 90°-periodic set (which collapses PARK). Used instead of an
    exact pose-list match so an equivalent-but-not-identical ring (reordered,
    int vs float, a different comprehension) isn't silently rejected."""
    az = sorted({round(s.az_deg % 360.0, 1) for s in samples})
    if len(az) < 6:
        return False
    gaps = [az[i + 1] - az[i] for i in range(len(az) - 1)]
    gaps.append(az[0] + 360.0 - az[-1])
    if max(gaps) > 90.0 + 1e-6:
        return False
    if all(abs(a - round(a / 90.0) * 90.0) < 1e-6 for a in az):
        return False   # purely 90°-periodic — would collapse PARK
    return True


def solve_turntable_cx(samples: Sequence[CaptureSample]) -> float | None:
    """Estimate the OBSERVABLE world-X of the AZ axis from the photos by
    minimising the azimuth-variance of board-in-world (= ``rms_translation_mm``).
    ``cy`` is NOT solved — it is a structural gauge null and stays 0.

    Returns ``None`` when the axis is not meaningfully observable — the search
    hit the all-infeasible sentinel or could not beat the cx=0 baseline — so the
    caller keeps the origin gauge instead of a bound-pinned garbage value."""
    res = minimize_scalar(
        _cx_cost, args=(samples,), method="bounded",
        bounds=(-_CX_BOUND_MM, _CX_BOUND_MM), options={"xatol": 1e-3},
    )
    if not np.isfinite(res.fun) or res.fun >= 1e8:
        return None                                  # all axes infeasible
    if res.fun >= _cx_cost(0.0, samples) - 1e-9:     # no improvement over origin
        return None
    return float(res.x)


# ── EL-axis rocker refine: 2-DOF outer search, PARK inner ────────────────────
#
# PARK solves AX=XB under the IDEAL kinematic model A = Az_C·Ry(-el): the AZ
# and EL axes intersect and EL is exactly horizontal. This rig's history says
# otherwise (the laser-era machine_solve settled on a ~6.8° EL-axis tilt), and
# any such structure makes PARK smear the error across X — consistent within
# the sweep, but wrong at any new pose (exactly what "Test accuracy" measures).
#
# A naive fix — bundle-adjust X(6), Z(6), cx and the rocker against the pixels
# all at once — is ILL-CONDITIONED on a sparse sweep: X and Z slide together
# along the viewing direction (a near-planar target weakly constrains it),
# fitting the corners while ballooning arm_radius from 293 to 415 mm and
# camera_offset to 246 mm. Reprojection looks great; the absolute geometry is
# garbage, and a fresh-pose check is WORSE. (Observed on the bench, 2026-06-11.)
#
# So the search is reduced to the only two NEW observable parameters:
#   θ = [cx, rx]   — turntable-axis world-X and the EL-axis out-of-horizontal
#                    tilt — while X and Z are NOT free: at every (cx, rx) they
#                    come from the linear PARK hand-eye + board mean. PARK pins
#                    arm_radius from the board's metric square size, so the
#                    geometry can't wander; the outer search only places the
#                    axis. Excluded by symmetry (see the older note that stood
#                    here, condensed): rocker ry/py and cy are absorbed by X /
#                    are gauges, a rocker z-rotation/z-translation commute with
#                    Az and are absorbed by Z, and the radial offset px is
#                    EXACTLY degenerate with cx (only px−cx observable). What
#                    remains is rx, the EL axis tilting out of horizontal.

# With (cx, rx) the RIGID two-revolute model is complete: the camera observes
# only the board, so "AZ-axis tilt" relative to the lab/gravity is a pure
# gauge — only the axes' MUTUAL angle (rx) and offset (cx) are observable.
# What can still produce an az-correlated residual is NON-rigid structure;
# the dominant deterministic candidate on this hardware is the AZ encoder's
# first harmonic (AS5600 magnet eccentricity: az_true = az_meas +
# a_c·sin(az) + a_s·cos(az), classically up to a few degrees). Stage B of the
# refine solves those two coefficients — gated hard, so they are only kept
# when the data genuinely demands them. (A constant az offset is the board-
# phase gauge and stays excluded by construction: the harmonic has no DC term.)

#: Hard limit for the refine — beyond this it's a build error, not geometry.
_ROCKER_TILT_BOUND_RAD = 0.35      # ±20°
#: AZ-encoder first-harmonic coefficient limit (deg) — beyond this the magnet
#: is physically misassembled, not calibratable.
_AZ_HARM_BOUND_DEG = 5.0
#: Accept the refined model only if it beats the ideal-axis reprojection RMS
#: by at least this factor (guards against overfitting noise with new DOF).
_REFINE_MIN_GAIN = 0.97
#: Stage B (encoder harmonic, +2 DOF) must earn a further material gain over
#: stage A AND be solvable at all (full ring) — stricter than stage A's gate.
_AZ_HARM_MIN_GAIN = 0.93
#: Ignore a solved harmonic smaller than this (deg) — below detection noise.
_AZ_HARM_MIN_AMP_DEG = 0.05
#: EL encoder scale (el_true = k·el) limits — also covers a linear bracket
#: sag about the EL axis (mathematically the same parameter). Beyond ±15 %
#: something is mechanically broken, not calibratable.
_EL_SCALE_MIN = 0.85
_EL_SCALE_MAX = 1.15


def _rocker_from_params(rx: float) -> MountTransform | None:
    return MountTransform(t=(0.0, 0.0, 0.0), rvec=(rx, 0.0, 0.0)) if rx else None


def az_encoder_corrected(az_deg: float,
                         harm: Sequence[float] | None) -> float:
    """Apply the AZ-encoder harmonic correction
    `az_true = az + Σ_n (h[2n]·sin(n·az) + h[2n+1]·cos(n·az))`, n = 1, 2
    (coefficients in degrees; a 2-element harm is first-harmonic-only for
    backward compatibility). An AS5600 with an eccentric magnet produces the
    1st harmonic; a tilted/misseated magnet adds a strong 2nd (bench
    2026-06-11: 9.1° + 7.6°). None / zeros → identity. THE single
    definition — calibration, FK and test_accuracy must all route through it.
    """
    if not harm or not any(harm):
        return az_deg
    r = math.radians(az_deg)
    out = az_deg + harm[0] * math.sin(r) + harm[1] * math.cos(r)
    if len(harm) >= 4:
        out += harm[2] * math.sin(2 * r) + harm[3] * math.cos(2 * r)
    return out


def _reproj_residuals(
    samples: Sequence[CaptureSample],
    K: np.ndarray,
    D: np.ndarray,
    X44: np.ndarray,
    Z44: np.ndarray,
    turntable_axis: tuple[float, float],
    rocker: MountTransform | None,
    az_harm: tuple[float, float] | None = None,
    el_scale: float = 1.0,
) -> np.ndarray:
    """Stacked pixel residuals of all corners in all views, for the model
    B_pred = X⁻¹ · A⁻¹ · Z (board→camera). Units: px."""
    cv2 = _cv2()
    X_inv = np.linalg.inv(X44)
    out: list[np.ndarray] = []
    for s in samples:
        az = az_encoder_corrected(s.az_deg, az_harm)
        A = homogeneous(*arm_pose_in_world(az, s.el_deg * el_scale,
                                           turntable_axis, rocker))
        B = X_inv @ np.linalg.inv(A) @ Z44
        rvec, _ = cv2.Rodrigues(B[:3, :3])
        tvec = B[:3, 3] / 1000.0            # mm → m (obj points are metres)
        proj, _ = cv2.projectPoints(s.obj_points, rvec, tvec, K, D)
        out.append((proj - s.img_points).reshape(-1))
    return np.concatenate(out)


def _solve_xz_at(
    samples: Sequence[CaptureSample], cx: float, rx: float,
    az_harm: tuple[float, float] | None = None,
    el_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Inner solve: PARK hand-eye + board-mean under the (cx, rx, harmonic,
    el-scale) model. Returns (X44, Z44) so X and Z are always the
    linear-optimal pair for that axis — they cannot wander. None when the
    hand-eye is degenerate. Encoder corrections are applied by substituting
    corrected angles into throwaway sample copies (solve_hand_eye /
    board_world_stats read angles from samples)."""
    rocker = _rocker_from_params(rx)
    axis = (float(cx), 0.0)
    if (az_harm and (az_harm[0] or az_harm[1])) or el_scale != 1.0:
        samples = [
            CaptureSample(
                az_deg=az_encoder_corrected(s.az_deg, az_harm),
                el_deg=s.el_deg * el_scale,
                board_R_cam=s.board_R_cam, board_t_cam=s.board_t_cam,
            )
            for s in samples
        ]
    try:
        R_x, t_x = solve_hand_eye(samples, axis, rocker)
        st = board_world_stats(samples, R_x, t_x, axis, rocker)
    except (ValueError, np.linalg.LinAlgError, _cv2().error):
        return None
    return homogeneous(R_x, t_x), homogeneous(st.R_ref, st.t_ref)


def refine_full_model(
    samples: Sequence[CaptureSample],
    intrinsics: Intrinsics,
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    stats: BoardWorldStats,
    cx0: float,
    freeze_cx: bool = False,
) -> tuple[np.ndarray, np.ndarray, MountTransform, float, MountTransform | None,
           tuple[float, float] | None, float, float] | None:
    """Place the deterministic machine model by minimising corner reprojection,
    with X and Z taken from the linear PARK solve at each step — so the
    recovered geometry stays physical.

    Stage A: θ = (cx, rx, k) — turntable axis, EL-axis tilt and the EL scale
    factor `el_true = k·el` (a linear bracket sag about the EL axis is
    mathematically identical to an encoder scale error — one parameter covers
    both; the complete rigid two-revolute model + the dominant el-trend).
    Stage B: θ = (cx, rx, k, a_c, a_s) — adds the AZ-encoder first harmonic,
    attempted only on a full azimuth ring and kept only on a further material
    reprojection gain (its own stricter gate).

    Returns `(R_X, t_X, Z_ref, cx, rocker, az_harm, el_scale, rms_before_px,
    rms_after_px)` or None when unavailable / rejected. `az_harm` is
    `(a_c, a_s)` in degrees or None; `el_scale` is k (1.0 = exact encoder)."""
    from scipy.optimize import least_squares

    usable = [s for s in samples if s.obj_points is not None
              and s.img_points is not None]
    if len(usable) < 3:
        return None
    K, D = intrinsics.K, intrinsics.D
    big_residual = np.full(sum(s.obj_points.shape[0] for s in usable) * 2, 1e4)

    def residual(theta):
        cx, rx, k = float(theta[0]), float(theta[1]), float(theta[2])
        harm = (tuple(float(v) for v in theta[3:])
                if len(theta) > 3 else None)
        xz = _solve_xz_at(usable, cx, rx, harm, k)
        if xz is None:
            return big_residual
        X44, Z44 = xz
        return _reproj_residuals(usable, K, D, X44, Z44, (cx, 0.0),
                                 _rocker_from_params(rx), harm, k)

    rms_before = float(np.sqrt(np.mean(np.square(residual([cx0, 0.0, 1.0])))))
    cx_lo, cx_hi = ((cx0 - 1e-3, cx0 + 1e-3) if freeze_cx
                    else (-_CX_BOUND_MM, _CX_BOUND_MM))

    # ── stage A: rigid model + el scale (cx, rx, k) ──
    try:
        sol_a = least_squares(
            residual, [cx0, 0.0, 1.0],
            bounds=([cx_lo, -_ROCKER_TILT_BOUND_RAD, _EL_SCALE_MIN],
                    [cx_hi, _ROCKER_TILT_BOUND_RAD, _EL_SCALE_MAX]),
            x_scale=[10.0, 0.02, 0.02],
            diff_step=[1e-2, 1e-3, 1e-3], max_nfev=250,
        )
    except Exception as exc:  # noqa: BLE001 — refine is best-effort
        log.warning("calibration: refine raised (%s) — keeping PARK result", exc)
        return None
    rms_a = float(np.sqrt(np.mean(np.square(sol_a.fun))))
    if not np.isfinite(rms_a) or rms_a > rms_before * _REFINE_MIN_GAIN:
        return None                          # no material gain — don't add DOF
    best_x = [float(sol_a.x[0]), float(sol_a.x[1]), float(sol_a.x[2])]
    rms_after = rms_a
    az_harm: tuple[float, float] | None = None

    # ── stage B: + AZ-encoder harmonics 1+2 (4 coeffs), full ring only ──
    # 1st harmonic = magnet eccentricity; 2nd = magnet tilt/misseating. The
    # bench map needed BOTH (9.1° + 7.6°) — a 1st-only model left R²≈0.58.
    from config import settings
    if settings.az_harmonic_enabled and _is_full_azimuth_ring(usable):
        b = _AZ_HARM_BOUND_DEG
        try:
            sol_b = least_squares(
                residual, best_x + [0.0, 0.0, 0.0, 0.0],
                bounds=([cx_lo, -_ROCKER_TILT_BOUND_RAD, _EL_SCALE_MIN,
                         -b, -b, -b, -b],
                        [cx_hi, _ROCKER_TILT_BOUND_RAD, _EL_SCALE_MAX,
                         b, b, b, b]),
                x_scale=[10.0, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2],
                diff_step=[1e-2, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2],
                max_nfev=400,
            )
            rms_b = float(np.sqrt(np.mean(np.square(sol_b.fun))))
            amp1 = float(np.hypot(sol_b.x[3], sol_b.x[4]))
            amp2 = float(np.hypot(sol_b.x[5], sol_b.x[6]))
            if (np.isfinite(rms_b) and rms_b < rms_a * _AZ_HARM_MIN_GAIN
                    and max(amp1, amp2) >= _AZ_HARM_MIN_AMP_DEG):
                best_x = [float(sol_b.x[0]), float(sol_b.x[1]),
                          float(sol_b.x[2])]
                az_harm = tuple(float(v) for v in sol_b.x[3:7])
                rms_after = rms_b
                _ui_log("I", (
                    f"calibration: AZ-encoder harmonics accepted — 1st "
                    f"{amp1:.2f}°, 2nd {amp2:.2f}° (reproj {rms_a:.2f} → "
                    f"{rms_b:.2f} px). AS5600 magnet eccentricity + tilt."
                ))
        except Exception as exc:  # noqa: BLE001 — stage B is optional
            log.warning("calibration: harmonic stage raised (%s) — skipped", exc)

    cx, rx, el_scale = best_x
    xz = _solve_xz_at(usable, cx, rx, az_harm, el_scale)
    if xz is None:
        return None
    X44, Z44 = xz
    tilt_deg = float(np.degrees(abs(rx)))
    if abs(rx) > 0.98 * _ROCKER_TILT_BOUND_RAD:
        _ui_log("W", (
            f"calibration: refine hit its tilt bound ({tilt_deg:.1f}°) — "
            "inspect the rig, this is implausibly large for axis geometry"
        ))
    Z_ref = MountTransform(
        t=tuple(float(v) for v in Z44[:3, 3]),
        rvec=tuple(float(v) for v in matrix_to_rotvec(Z44[:3, :3])),
    )
    # Drop a numerically-zero rocker so the FK chain stays on the exact
    # legacy path when the rig really is ideal.
    use_rocker = _rocker_from_params(rx) if tilt_deg >= 0.05 else None
    # An el-scale within encoder noise of exact is dropped (keeps FK on the
    # untouched path for a healthy rig).
    if abs(el_scale - 1.0) < 5e-4:
        el_scale = 1.0
    return (X44[:3, :3], X44[:3, 3], Z_ref, cx, use_rocker, az_harm,
            el_scale, rms_before, rms_after)


def per_view_diagnostics(
    samples: Sequence[CaptureSample],
    intrinsics: Intrinsics,
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    Z_ref: MountTransform,
    turntable_axis: tuple[float, float] | None,
    rocker: MountTransform | None,
    az_harm: tuple[float, float] | None = None,
    el_scale: float = 1.0,
) -> list[dict[str, float]]:
    """Per-view residual table against the FINAL model: how far each view's
    board-in-world pose sits from Z_ref (mm / deg), plus its reprojection RMS
    (px) when the matched points were retained. An az/el-correlated pattern
    here is the fingerprint of still-unmodelled structure."""
    cv2 = _cv2()
    X44 = homogeneous(R_cam_arm, t_cam_arm)
    Z44 = Z_ref.as_matrix()
    R_ref_T = Z44[:3, :3].T
    t_ref = Z44[:3, 3]
    X_inv = np.linalg.inv(X44)
    out: list[dict[str, float]] = []
    for s in samples:
        az = az_encoder_corrected(s.az_deg, az_harm)
        A = homogeneous(*arm_pose_in_world(az, s.el_deg * el_scale,
                                           turntable_axis, rocker))
        Z_i = A @ X44 @ homogeneous(s.board_R_cam, s.board_t_cam)
        rv = np.degrees(matrix_to_rotvec(R_ref_T @ Z_i[:3, :3]))
        row: dict[str, float] = {
            "az": float(s.az_deg),
            "el": float(s.el_deg),
            "dpos_mm": float(np.linalg.norm(Z_i[:3, 3] - t_ref)),
            "drot_deg": float(np.linalg.norm(rv)),
            # Signed rotation-vector components (deg) — the vector residual,
            # for structure analysis (a |Δrot| mean alone cannot distinguish
            # a smooth model defect from a flipped subset of views).
            "drot_x": float(rv[0]), "drot_y": float(rv[1]),
            "drot_z": float(rv[2]),
        }
        if s.phone_el_deg is not None:
            row["phone_el_dev"] = float(s.phone_el_deg - s.el_deg)
        if s.phone_roll_deg is not None:
            row["phone_roll"] = float(s.phone_roll_deg)
        if s.obj_points is not None and s.img_points is not None:
            B = X_inv @ np.linalg.inv(A) @ Z44
            rvec, _ = cv2.Rodrigues(B[:3, :3])
            proj, _ = cv2.projectPoints(s.obj_points, rvec, B[:3, 3] / 1000.0,
                                        intrinsics.K, intrinsics.D)
            row["reproj_px"] = float(np.sqrt(np.mean(
                np.square((proj - s.img_points).reshape(-1)))))
        out.append(row)
    return out


def residual_structure(rows: list[dict[str, float]]) -> str | None:
    """Deterministic verdict on the REMAINING per-view rotation residual.

    Fits the VECTOR rotation residual (signed rotvec components, not the
    modulus — a |Δrot| mean cannot distinguish a smooth model defect from a
    flipped subset of views) onto [1, sin az, cos az, el] per component, and
    additionally screens for bimodality (a cluster of outlier views ≫ the
    median = surviving planar-PnP flips). None when too few views."""
    if len(rows) < 5 or "drot_x" not in rows[0]:
        return None
    azr = np.radians([r["az"] for r in rows])
    el = np.array([r["el"] for r in rows], float)
    mag = np.array([r["drot_deg"] for r in rows], float)
    V = np.column_stack([[r["drot_x"] for r in rows],
                         [r["drot_y"] for r in rows],
                         [r["drot_z"] for r in rows]])
    M = np.column_stack([np.ones_like(azr), np.sin(azr), np.cos(azr),
                         (el - el.mean()) / 30.0])
    coef, *_ = np.linalg.lstsq(M, V, rcond=None)   # (4, 3)
    fit = M @ coef
    ss_tot = float(np.sum((V - V.mean(axis=0)) ** 2))
    r2 = (1.0 - float(np.sum((V - fit) ** 2)) / ss_tot
          if ss_tot > 1e-12 else 0.0)
    const_amp = float(np.linalg.norm(coef[0]))
    az_amp = float(np.linalg.norm(coef[1:3]))
    el_amp = float(np.linalg.norm(coef[3]))
    # Bimodality screen: views whose |Δrot| dwarfs the median are a flip
    # cluster, not a smooth defect.
    med = float(np.median(mag))
    n_out = int(np.sum(mag > max(2.5 * med, med + 5.0)))
    verdict = (
        f"{n_out}/{len(rows)} views are far-outliers vs the median "
        f"({med:.2f}°) → looks like SURVIVING planar-PnP flips; rerun with "
        "the board wedged 10-15° off flat." if n_out >= 2 else
        "az-harmonic dominates → AZ bearing wobble or residual encoder error."
        if az_amp > 2 * el_amp and az_amp > 0.2 else
        "el-trend dominates → el-dependent flex (arm sag) or EL-encoder scale."
        if el_amp > 2 * az_amp and el_amp > 0.2 else
        "constant offset dominates — board moved mid-sweep or a global "
        "reference inconsistency." if const_amp > 2 * max(az_amp, el_amp)
        and const_amp > 0.5 else
        "no dominant structure — remaining residual looks like noise."
    )
    return (
        f"residual structure (vector): const {const_amp:.2f}° | az-harmonic "
        f"{az_amp:.2f}° | el-trend {el_amp:.2f}°/30°el | R²={r2:.2f} | "
        f"median |Δrot| {med:.2f}°. {verdict}"
    )


async def _capture_one(
    az: float,
    el: float,
    board,
) -> CaptureSample | None:
    """Move to (az, el), grab a photo, detect the ChArUco board, and RETAIN the
    raw corners + image size. The board→camera pose is filled later — either by
    the one-shot intrinsics calibration or the per-view solvePnP fallback.

    The sample records the ACTUAL settled encoder angles, not the commanded
    pose — the closed-loop accepts a settle tolerance, and feeding commanded
    angles into the hand-eye silently injects that tolerance as model error.
    The phone-IMU readouts at capture time are retained too: a free strain
    gauge on the camera bracket (flex shows up as an el-correlated deviation
    between phone_el and the encoder el)."""
    cv2 = _cv2()
    await esp.move_and_await(azimuth_deg=az, elevation_deg=el, timeout_ms=15000)
    await asyncio.sleep(_SETTLE_S)
    az_act, el_act = float(model.az), float(model.el)
    phone_el = (float(model.phone_el_deg)
                if model.phone_sensor_online and model.phone_el_deg is not None
                else None)
    phone_roll = (float(model.phone_roll_deg)
                  if model.phone_sensor_online and model.phone_roll_deg is not None
                  else None)
    raw = await camera_io.fetch_photo(el_deg=el)
    if not raw:
        log.warning("calibration: no photo at az=%.1f el=%.1f", az, el)
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("calibration: undecodable photo at az=%.1f el=%.1f", az, el)
        return None
    corners, ids = detect_board(img, board)
    if corners is None:
        log.warning("calibration: board not detected at az=%.1f el=%.1f", az, el)
        return None
    h, w = img.shape[:2]
    if abs(az_act - az) > 0.5 or abs(el_act - el) > 0.5:
        log.info("calibration: settled %.2f°/%.2f° off the commanded pose "
                 "(az %.1f→%.1f, el %.1f→%.1f) — using actuals",
                 az_act - az, el_act - el, az, az_act, el, el_act)
    # Per-pose progress is logged by run_calibration (it has the pose index),
    # so we don't double-log the corner count here.
    return CaptureSample(
        az_deg=az_act, el_deg=el_act,
        charuco_corners=corners, charuco_ids=ids, image_wh=(w, h),
        phone_el_deg=phone_el, phone_roll_deg=phone_roll,
    )


async def run_calibration(
    poses: Sequence[tuple[float, float]] | None = None,
    solve_axis: bool = True,
) -> CalibrationResult:
    """Sweep poses, calibrate intrinsics from the same photos, solve hand-eye,
    derive geometry — everything geometric comes from the photos.

    Pipeline: capture+detect (retaining ChArUco corners) → ONE
    `cv2.calibrateCamera` (intrinsics + per-view board poses = the hand-eye
    `B_i`) → turntable-axis world-X solve (when `model.turntable_axis` is unset
    and the sweep is the full ring; gated by `solve_axis`) → PARK hand-eye →
    `derive_geometry` + eccentricity read-off. `cy` and `base_height` are
    gauges and stay 0 / their default; the only user input is the board spec.

    Does NOT persist the result — caller decides via `apply_result()`.
    Raises `RuntimeError` if too few views detected the board (< 3) or if the
    hand-eye solve is degenerate. Intrinsics are solved from the photos only
    when ≥ 6 views across ≥ 3 elevations are usable, else they fall back to the
    model guess (logged).
    """
    if poses is None:
        poses = DEFAULT_POSES
    poses = list(poses)
    board = _build_board(_board_spec_from_model())

    _ui_log("I", (
        f"calibration: starting — {len(poses)} poses, board "
        f"{int(model.charuco_squares_x)}×{int(model.charuco_squares_y)} @ "
        f"{float(model.charuco_square_length_mm):.0f} mm; camera_url="
        f"{model.camera_url or '(unset)'}"
    ))

    samples: list[CaptureSample] = []
    for i, (az, el) in enumerate(poses, 1):
        sample = await _capture_one(az, el, board)
        if sample is not None:
            n = (len(sample.charuco_corners)
                 if sample.charuco_corners is not None else 0)
            samples.append(sample)
            _ui_log("I", f"calibration: [{i}/{len(poses)}] "
                         f"az={az:.0f}° el={el:.0f}° → {n} corners ✓")
        else:
            _ui_log("W", f"calibration: [{i}/{len(poses)}] "
                         f"az={az:.0f}° el={el:.0f}° → board NOT detected ✗")

    _ui_log("I", f"calibration: sweep done — {len(samples)}/{len(poses)} "
                 "views detected the board")

    if len(samples) < 3:
        raise RuntimeError(
            f"calibration failed: only {len(samples)}/{len(poses)} views "
            f"detected the board (need ≥ 3 for hand-eye; ≥ {_MIN_INTRINSIC_VIEWS} "
            f"across ≥ {_MIN_INTRINSIC_ELEVATIONS} elevations to also solve "
            "intrinsics from photos). Check board placement, lighting, and "
            "framing across the sweep."
        )

    # ── intrinsics from the same photos (P1), with a graceful fallback ──
    w, h = samples[0].image_wh or (
        int(round(2 * model.camera_cx)), int(round(2 * model.camera_cy))
    )
    k0 = np.array(
        [[float(model.camera_fx), 0.0, w / 2.0],
         [0.0, float(model.camera_fy), h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    _ui_log("I", f"calibration: solving camera intrinsics from {len(samples)} views…")
    ci = calibrate_intrinsics(samples, board, k0, np.zeros(5))
    if ci is not None:
        intrinsics, intr_rms, samples, board_poses = ci
        for s, (R, t) in zip(samples, board_poses):
            s.board_R_cam, s.board_t_cam = R, t
        intrinsics_from_photos = True
        _ui_log("I", (
            f"calibration: intrinsics from {len(samples)} photos — "
            f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
            f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f} (rms {intr_rms:.3f} px)"
        ))
    else:
        # Model intrinsics are a 1920-era guess; centre the principal point on
        # the ACTUAL frame and rescale the focal so per-view solvePnP isn't
        # wildly off when the frame is a different size/orientation (e.g. a
        # 4080×3060 portrait still vs landscape 960/540 defaults). The real fix
        # is the solve above passing — this is just a less-catastrophic fallback.
        _m = _intrinsics_from_model()
        _s = w / 1920.0
        intrinsics = Intrinsics(
            fx=_m.fx * _s, fy=_m.fy * _s, cx=w / 2.0, cy=h / 2.0,
            dist=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        intr_rms = None
        intrinsics_from_photos = False
        _ui_log("W", (
            f"calibration: intrinsics NOT solved from photos (need ≥"
            f"{_MIN_INTRINSIC_VIEWS} views across ≥{_MIN_INTRINSIC_ELEVATIONS} "
            "elevations) — falling back to the model-guess intrinsics + per-view "
            "solvePnP"
        ))
        for s in samples:
            pose = estimate_board_pose(
                s.charuco_corners, s.charuco_ids, board, intrinsics,
            )
            if pose is not None:
                s.board_R_cam, s.board_t_cam = pose
                # Retain the correspondences so the refine can still run on
                # the fallback path (K is a guess, but the rocker geometry is
                # observable regardless).
                obj, imgp = board.matchImagePoints(s.charuco_corners,
                                                   s.charuco_ids)
                if obj is not None:
                    s.obj_points = obj.astype(np.float32)
                    s.img_points = imgp.astype(np.float32)
        samples = [s for s in samples if s.board_R_cam is not None]
        if len(samples) < 3:
            raise RuntimeError(
                "calibration failed: board poses could not be recovered for "
                "≥ 3 views even with the fallback intrinsics."
            )

    # ── planar-PnP ambiguity: flip flipped views BEFORE anything uses B_i ──
    # A flat board admits a twin pose per view; near-frontal views can come
    # out of the intrinsics solve on the wrong branch and poison the
    # hand-eye with a large constant rotation spread. The board's rigidity
    # is the arbiter.
    n_twins = attach_pose_twins(samples, intrinsics, board)
    n_flipped = disambiguate_board_poses(samples, intrinsics) if n_twins else 0
    _ui_log("I" if n_flipped == 0 else "W", (
        f"calibration: planar-ambiguity check — {n_twins}/{len(samples)} "
        f"views had a viable twin pose, {n_flipped} flipped to the "
        "rigid-board-consistent branch"
    ))

    # ── turntable axis (P2). cy is a structural gauge null. ──
    axis = getattr(model, "turntable_axis", None)
    axis_solved = False
    if axis is not None:
        # Operator-set axis is honoured AS GIVEN (incl. cy) so the solver and
        # the renderer agree on where the axis is.
        cx = float(axis[0])
        cy = float(axis[1]) if len(axis) > 1 else 0.0
        _ui_log("I", f"calibration: using operator-set turntable axis "
                     f"({cx:.1f}, {cy:.1f}) mm")
    elif solve_axis and _is_full_azimuth_ring(samples):
        _ui_log("I", "calibration: solving turntable axis (world-X) from the ring…")
        solved = solve_turntable_cx(samples)
        if solved is not None:
            cx, cy, axis_solved = solved, 0.0, True
            _ui_log("I", f"calibration: solved turntable axis world-X cx={cx:.1f} mm")
        else:
            cx, cy = 0.0, 0.0
            _ui_log("I", "calibration: turntable axis not observable — using origin")
    else:
        cx, cy = 0.0, 0.0
        _ui_log("I", "calibration: turntable-axis solve skipped (axis unset & "
                     "sweep is not a full ring) — using origin")
    turntable_axis = (cx, cy)

    # ── hand-eye (PARK) under the ideal-axis model; guard degenerate solves ──
    _ui_log("I", f"calibration: solving hand-eye (PARK) from {len(samples)} "
                 f"views, axis=({cx:.1f}, {cy:.1f})…")
    try:
        R_cam_arm, t_cam_arm = solve_hand_eye(samples, turntable_axis)
        stats = board_world_stats(samples, R_cam_arm, t_cam_arm, turntable_axis)
    except (ValueError, np.linalg.LinAlgError, _cv2().error) as exc:
        raise RuntimeError(
            f"calibration failed: degenerate hand-eye solve ({exc}). The "
            "captured views are too few or too co-linear — widen the sweep."
        ) from exc

    # ── EL-axis rocker refine: place (cx, rx) by reprojection, X/Z from PARK ──
    # PARK assumes the AZ/EL axes intersect and EL is exactly horizontal; on
    # this rig they don't (laser-era history: ~6.8° rocker tilt). The refine
    # searches only the turntable axis cx and the EL-axis tilt rx against the
    # raw corner reprojections, with X and Z taken from PARK at each step so
    # the recovered geometry stays physical (see refine_full_model).
    rocker: MountTransform | None = None
    az_harm: tuple[float, float] | None = None
    el_scale = 1.0
    refine_before: float | None = None
    refine_after: float | None = None
    _ui_log("I", "calibration: refining EL-axis tilt + turntable axis + EL "
                 "scale against corner reprojections…")
    refined = refine_full_model(
        samples, intrinsics, R_cam_arm, t_cam_arm, stats, cx,
        freeze_cx=(axis is not None),
    )
    if refined is not None:
        (R_cam_arm, t_cam_arm, z_ref_mt, cx, rocker, az_harm, el_scale,
         refine_before, refine_after) = refined
        turntable_axis = (cx, cy)
        if az_harm or el_scale != 1.0:
            # Re-evaluate the SE3 stats with corrected encoder angles.
            corr = [CaptureSample(
                az_deg=az_encoder_corrected(s.az_deg, az_harm),
                el_deg=s.el_deg * el_scale,
                board_R_cam=s.board_R_cam, board_t_cam=s.board_t_cam,
            ) for s in samples]
            stats = board_world_stats(corr, R_cam_arm, t_cam_arm,
                                      turntable_axis, rocker)
        else:
            stats = board_world_stats(samples, R_cam_arm, t_cam_arm,
                                      turntable_axis, rocker)
        if axis is None:
            axis_solved = True
        tilt_deg = (float(np.degrees(np.linalg.norm(rocker.rvec)))
                    if rocker is not None else 0.0)
        _ui_log("I", (
            f"calibration: refine accepted — reproj rms {refine_before:.2f} → "
            f"{refine_after:.2f} px | EL-axis tilt {tilt_deg:.2f}° | "
            f"el-scale {el_scale:.4f} | cx={cx:.1f} mm"
        ))
    else:
        _ui_log("I", "calibration: refine skipped/rejected (no material gain) "
                     "— keeping the ideal-axis PARK solve")
        z_ref_mt = MountTransform(
            t=tuple(float(v) for v in stats.t_ref),
            rvec=tuple(float(x) for x in matrix_to_rotvec(stats.R_ref)),
        )

    result = derive_geometry(R_cam_arm, t_cam_arm)
    result.n_views = len(samples)
    result.n_attempted = len(poses)
    result.rms_translation_mm = stats.rms_translation_mm
    result.rms_rotation_deg = stats.rms_rotation_deg
    result.rocker = rocker
    result.az_harm = az_harm
    result.el_scale = el_scale
    result.refine_rms_before_px = refine_before
    result.refine_rms_after_px = refine_after
    # Free eccentricity read-off (diagnostic only; exact on a full ring).
    # rms_translation_mm is the quality flag.
    t_z = np.asarray(z_ref_mt.t, float)
    result.board_eccentricity_mm = (float(t_z[0] - cx), float(t_z[1] - cy))
    result.board_height_mm = float(t_z[2])
    # Reference board-in-world pose Z_ref — persisted for "Test accuracy".
    result.board_world = z_ref_mt

    # ── per-view residual table: the fingerprint of remaining structure ──
    result.diagnostics = per_view_diagnostics(
        samples, intrinsics, R_cam_arm, t_cam_arm, z_ref_mt,
        turntable_axis, rocker, az_harm, el_scale,
    )
    for row in result.diagnostics:
        px = (f" | reproj {row['reproj_px']:.2f} px"
              if 'reproj_px' in row else "")
        _ui_log("I", (
            f"calibration:   view az={row['az']:.0f}° el={row['el']:.0f}° → "
            f"Δpos {row['dpos_mm']:.1f} mm · Δrot {row['drot_deg']:.2f}°{px}"
        ))
    # Deterministic verdict on whatever is left — names the next defect.
    structure = residual_structure(result.diagnostics)
    if structure:
        _ui_log("I", f"calibration: {structure}")

    # Machine-readable dump for offline analysis (overwritten every run) —
    # everything needed to re-derive the solve without asking for logs.
    try:
        import json as _json

        from config import settings
        views = []
        for s, row in zip(samples, result.diagnostics or []):
            v = dict(row)
            v["B_rvec"] = [float(x) for x in matrix_to_rotvec(s.board_R_cam)]
            v["B_t_mm"] = [float(x) for x in s.board_t_cam]
            v["had_twin"] = s.board_pose_alt is not None
            views.append(v)
        dump = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "n_views": result.n_views, "n_attempted": result.n_attempted,
            "board": {
                "squares_x": int(model.charuco_squares_x),
                "squares_y": int(model.charuco_squares_y),
                "square_mm": float(model.charuco_square_length_mm),
                "marker_mm": float(model.charuco_marker_length_mm),
                "dict_id": int(model.aruco_dict_id),
            },
            "intrinsics": {
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "cx": intrinsics.cx, "cy": intrinsics.cy,
                "dist": list(intrinsics.dist),
                "from_photos": intrinsics_from_photos,
            },
            "X": {"rvec": [float(x) for x in matrix_to_rotvec(R_cam_arm)],
                  "t_mm": [float(x) for x in t_cam_arm]},
            "Z_ref": {"rvec": list(z_ref_mt.rvec), "t_mm": list(z_ref_mt.t)},
            "cx_mm": float(cx),
            "rocker_rvec": list(rocker.rvec) if rocker else None,
            "az_harm_deg": list(az_harm) if az_harm else None,
            "el_scale": el_scale,
            "rms_translation_mm": result.rms_translation_mm,
            "rms_rotation_deg": result.rms_rotation_deg,
            "refine_rms_px": [refine_before, refine_after],
            "residual_structure": structure,
            "views": views,
        }
        dump_path = settings.storage_dir / "calib_debug_last.json"
        dump_path.write_text(_json.dumps(dump, indent=1), encoding="utf-8")
        _ui_log("I", f"calibration: debug dump → {dump_path}")
    except Exception:  # noqa: BLE001 — diagnostics must never kill a solve
        log.exception("calibration: debug dump failed")

    # ── carry the photo-solved intrinsics + axis for apply_result/result_dict ──
    result.intrinsics_from_photos = intrinsics_from_photos
    if intrinsics_from_photos:
        result.camera_fx, result.camera_fy = intrinsics.fx, intrinsics.fy
        result.camera_cx, result.camera_cy = intrinsics.cx, intrinsics.cy
        result.camera_distortion = list(intrinsics.dist)
        result.intrinsics_rms_px = intr_rms
        result.n_intrinsic_views = len(samples)
    result.turntable_cx_mm = cx
    result.turntable_axis_solved = axis_solved

    _ui_log("I", (
        f"calibration: DONE — {result.n_views}/{result.n_attempted} views | "
        f"arm_radius={result.arm_radius_mm:.1f} mm "
        f"camera_offset={result.camera_offset_mm:.1f} mm "
        f"pan={result.camera_pan_deg:.2f}° tilt={result.camera_tilt_deg:.2f}° | "
        f"residual {result.rms_translation_mm or 0.0:.2f} mm / "
        f"{result.rms_rotation_deg or 0.0:.3f}° | axis cx={cx:.1f} mm | "
        f"eccentricity=({result.board_eccentricity_mm[0]:.1f}, "
        f"{result.board_eccentricity_mm[1]:.1f}) mm"
    ))
    return result


def apply_result(result: CalibrationResult) -> None:
    """Persist the derived geometry into `model` (config-like, survives restart).

    Writes the hand-eye geometry, the photo-solved camera intrinsics (only when
    `intrinsics_from_photos`), and the solved turntable axis (only when WE
    solved it — an operator-set axis is left untouched; `cy` stays the gauge 0).

    `camera_pan_deg` and `camera_tilt_deg` are intentionally NOT applied (reset
    to the nominal 0): the camera is assumed to look ALONG the arm. For a
    portrait-mounted phone the recovered pan/tilt are dominated by the ~90°
    portrait ROLL decomposed against the landscape look-back nominal — applying
    them would mis-AIM the live frustum (yaw it ~110° off the platform) even
    though the position is right. They remain in `result_dict` as diagnostics.
    `base_height_mm` is a gauge and is not derived (tape-measure default).
    """
    model.update(
        arm_radius_mm=result.arm_radius_mm,
        camera_offset_mm=result.camera_offset_mm,
        camera_pan_deg=0.0,
        camera_tilt_deg=0.0,
        # Flips the UI's "not calibrated" warning off — a calibration has now
        # been applied.
        calibrated=True,
    )
    if result.intrinsics_from_photos and result.camera_fx is not None:
        model.update(
            camera_fx=result.camera_fx,
            camera_fy=result.camera_fy,
            camera_cx=result.camera_cx,
            camera_cy=result.camera_cy,
            camera_distortion=list(result.camera_distortion or []),
        )
    if result.turntable_axis_solved and result.turntable_cx_mm is not None:
        model.update(turntable_axis=(result.turntable_cx_mm, 0.0))
    # Persist the hand-eye X + board-in-world reference so "Test accuracy" can
    # predict the board pose at the live encoder angles after a restart. The
    # rocker (EL-axis correction) is written alongside — and explicitly reset
    # to None when this calibration didn't solve one, so a stale correction
    # from a previous run can't haunt the FK chain.
    if result.extrinsic is not None and result.board_world is not None:
        model.update(
            calib_extrinsic={
                "rvec": list(result.extrinsic.rvec),
                "t": list(result.extrinsic.t),
            },
            calib_board_world={
                "rvec": list(result.board_world.rvec),
                "t": list(result.board_world.t),
            },
            rocker_correction=(
                {"rvec": list(result.rocker.rvec), "t": list(result.rocker.t)}
                if result.rocker is not None else None
            ),
            az_encoder_correction=(
                list(result.az_harm) if result.az_harm is not None else None
            ),
            el_encoder_scale=(
                result.el_scale if result.el_scale != 1.0 else None
            ),
        )


async def test_accuracy() -> dict[str, Any]:
    """One-shot accuracy check at the CURRENT rig pose (no move).

    Captures a photo, detects the ChArUco board, and forms the board-in-world
    pose the optics imply, `Z_obs = A(az,el)·X·B_obs`, then compares it to the
    calibrated reference `Z_ref` (persisted at calibration). The disagreement —
    rotation° + translation mm — is how far the optics drift from what the
    ENCODER angles predict via the calibrated model. Sets `model.calib_test_msg`
    (human string, timestamped so the UI always sees a fresh value) and returns
    `{ok, detected, delta_deg, delta_mm, az, el}`.
    """
    cv2 = _cv2()
    _ts = datetime.now().strftime("%H:%M:%S")  # stamp every msg so it's unique
    ex = getattr(model, "calib_extrinsic", None)
    zr = getattr(model, "calib_board_world", None)
    if not ex or not zr:
        msg = f"Test accuracy ({_ts}): no calibration reference — run a calibration first."
        model.update(calib_test_msg=msg)
        _ui_log("W", msg)
        return {"ok": True, "detected": False, "message": msg}

    az_raw, el_raw = float(model.az), float(model.el)
    from config import settings
    harm = (getattr(model, "az_encoder_correction", None)
            if settings.az_harmonic_enabled else None)
    az = az_encoder_corrected(
        az_raw, tuple(harm) if harm else None)
    k = getattr(model, "el_encoder_scale", None)
    el = el_raw * float(k) if k else el_raw
    board = _build_board(_board_spec_from_model())
    intr = _intrinsics_from_model()
    raw = await camera_io.fetch_photo(el_deg=el)
    img = (cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
           if raw else None)
    corners, ids = detect_board(img, board) if img is not None else (None, None)
    if corners is None:
        msg = f"Test accuracy ({_ts}): board NOT detected @ az={az_raw:.0f}° el={el_raw:.0f}°"
        model.update(calib_test_msg=msg)
        _ui_log("W", msg)
        return {"ok": True, "detected": False, "message": msg}

    # The calibrated forward kinematics predict the board's pose at this pose;
    # use it to break the planar PnP two-fold ambiguity so a flipped twin
    # solution can't masquerade as a calibration error.
    A = homogeneous(*arm_pose_in_world(
        az, el, getattr(model, "turntable_axis", None),
        MountTransform.from_dict(getattr(model, "rocker_correction", None)),
    ))
    X = homogeneous(rotvec_to_matrix(np.asarray(ex["rvec"], float)),
                    np.asarray(ex["t"], float))
    Zr_R = rotvec_to_matrix(np.asarray(zr["rvec"], float))
    Zr_t = np.asarray(zr["t"], float)
    Zr = homogeneous(Zr_R, Zr_t)
    B_pred = np.linalg.inv(X) @ np.linalg.inv(A) @ Zr      # board→camera, predicted

    pose = estimate_board_pose_disambiguated(
        corners, ids, board, intr, B_pred[:3, :3])
    if pose is None:
        msg = f"Test accuracy ({_ts}): board pose unsolvable @ az={az_raw:.0f}° el={el_raw:.0f}°"
        model.update(calib_test_msg=msg)
        _ui_log("W", msg)
        return {"ok": True, "detected": False, "message": msg}

    R_bc, t_bc, ambiguity_deg = pose
    Z = A @ X @ homogeneous(R_bc, t_bc)
    delta_mm = float(np.linalg.norm(Z[:3, 3] - Zr_t))
    delta_deg = float(np.degrees(
        np.linalg.norm(matrix_to_rotvec(Zr_R.T @ Z[:3, :3])),
    ))
    amb = f" (planar-PnP ambiguity {ambiguity_deg:.0f}°)" if ambiguity_deg > 5 else ""
    msg = (f"Test accuracy ({_ts}) @ az={az_raw:.0f}° el={el_raw:.0f}°: "
           f"Δrot {delta_deg:.2f}° · Δpos {delta_mm:.1f} mm{amb}")
    model.update(calib_test_msg=msg)
    _ui_log("I", msg)
    return {"ok": True, "detected": True, "delta_deg": delta_deg,
            "delta_mm": delta_mm, "az": az_raw, "el": el_raw,
            "ambiguity_deg": ambiguity_deg}


def result_dict(result: CalibrationResult) -> dict[str, Any]:
    """JSON-able view of a result for WS command responses."""
    return {
        "arm_radius_mm":      result.arm_radius_mm,
        "camera_offset_mm":   result.camera_offset_mm,
        "camera_tilt_deg":    result.camera_tilt_deg,
        "camera_pan_deg":     result.camera_pan_deg,
        "n_views":            result.n_views,
        "n_attempted":        result.n_attempted,
        "rms_translation_mm": result.rms_translation_mm,
        "rms_rotation_deg":   result.rms_rotation_deg,
        "board_eccentricity_mm": (
            list(result.board_eccentricity_mm)
            if result.board_eccentricity_mm is not None else None
        ),
        "board_height_mm":    result.board_height_mm,
        "intrinsics_from_photos": result.intrinsics_from_photos,
        "camera_fx":          result.camera_fx,
        "camera_fy":          result.camera_fy,
        "camera_cx":          result.camera_cx,
        "camera_cy":          result.camera_cy,
        "camera_distortion":  (
            list(result.camera_distortion)
            if result.camera_distortion is not None else None
        ),
        "intrinsics_rms_px":  result.intrinsics_rms_px,
        "n_intrinsic_views":  result.n_intrinsic_views,
        "turntable_cx_mm":    result.turntable_cx_mm,
        "turntable_axis_solved": result.turntable_axis_solved,
        "extrinsic": (
            {"t": list(result.extrinsic.t), "rvec": list(result.extrinsic.rvec)}
            if result.extrinsic is not None else None
        ),
        "rocker": (
            {"t": list(result.rocker.t), "rvec": list(result.rocker.rvec)}
            if result.rocker is not None else None
        ),
        "az_encoder_harmonic_deg": (
            list(result.az_harm) if result.az_harm is not None else None
        ),
        "el_encoder_scale":     result.el_scale,
        "refine_rms_before_px": result.refine_rms_before_px,
        "refine_rms_after_px":  result.refine_rms_after_px,
        "diagnostics":          result.diagnostics,
    }
