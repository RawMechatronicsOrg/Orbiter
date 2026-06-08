"""ChArUco hand-eye geometry calibration.

Drives the rig through a sweep of poses, captures a photo at each, detects
the ChArUco calibration board, then runs `cv2.calibrateHandEye` to derive
the camera's position in the arm-end frame. From that we read off the
rig's three-scalar geometry:

  * `arm_radius_mm`     — distance from arm pivot to camera along the arm
  * `camera_offset_mm`  — vertical offset of the camera above the arm pivot
                          (the camera does not sit on the arm centreline)
  * `camera_tilt_deg`   — pitch correction of the optical axis vs nominal
  * `camera_pan_deg`    — yaw correction

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

Board defaults are tuned for an A4-printable 5×7 ChArUco at 30 mm squares /
15 mm markers using `cv2.aruco.DICT_4X4_50`. Override the board geometry
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
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

import camera_io
from esp_proxy import esp
from geom.rig import FRAME_EL, FRAME_WORLD, MountTransform, build_rig_graph
from geom.transforms import matrix_to_rotvec
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


# ── kinematic arm pose (encoder-only, no geometry priors) ───────────────────


def arm_pose_in_world(
    az_deg: float,
    el_deg: float,
    turntable_axis: tuple[float, float] | None = None,
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
    """
    graph = build_rig_graph(az_deg, el_deg, _IDENTITY_MOUNT, turntable_axis)
    T = graph.matrix(FRAME_EL, FRAME_WORLD)
    return T[:3, :3], T[:3, 3]


# ── solver ──────────────────────────────────────────────────────────────────


def solve_hand_eye(
    samples: Sequence[CaptureSample],
    turntable_axis: tuple[float, float] | None = None,
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
        R, t = arm_pose_in_world(s.az_deg, s.el_deg, turntable_axis)
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


def _handeye_residual(
    samples: Sequence[CaptureSample],
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    turntable_axis: tuple[float, float] | None = None,
) -> tuple[float | None, float | None]:
    """RMS hand-eye consistency residual, split into translation (mm) and
    rotation (deg).

    The board is glued to the platform, so its pose in the world frame,
    `Z_i = A_i · X · B_i`, must be the SAME rigid transform for every view.
    The residual is how much these per-view estimates disagree — the RMS
    spread of `Z_i`'s translation about its mean and of its rotation about
    its mean rotation.

    This is the diagnostic for the one approximation in the model: a
    transverse AZ↔EL arm offset is not absorbable into the single constant X,
    so instead of silently biasing the recovered scalars it leaks into this
    spread — and it climbs with elevation. An el-correlated residual is the
    tell that the rig has a transverse offset the v0.1 solve does not fit
    (or that `turntable_axis` is wrong).
    """
    if not samples:
        return None, None
    X = np.eye(4)
    X[:3, :3] = R_cam_arm
    X[:3, 3] = t_cam_arm
    zs: list[np.ndarray] = []
    for s in samples:
        R, t = arm_pose_in_world(s.az_deg, s.el_deg, turntable_axis)
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
        B = np.eye(4)
        B[:3, :3] = s.board_R_cam
        B[:3, 3] = s.board_t_cam
        zs.append(A @ X @ B)             # board-in-world; constant for a perfect fit
    t_ref = np.mean([Z[:3, 3] for Z in zs], axis=0)
    R_ref = Rotation.from_matrix(np.array([Z[:3, :3] for Z in zs])).mean().as_matrix()
    t_errs = [float(np.linalg.norm(Z[:3, 3] - t_ref)) for Z in zs]
    r_errs = [
        float(np.degrees(np.linalg.norm(matrix_to_rotvec(R_ref.T @ Z[:3, :3]))))
        for Z in zs
    ]
    return (
        float(np.sqrt(np.mean(np.square(t_errs)))),
        float(np.sqrt(np.mean(np.square(r_errs)))),
    )


def board_in_world_mean(
    samples: Sequence[CaptureSample],
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    turntable_axis: tuple[float, float] | None = None,
) -> np.ndarray:
    """Azimuth-averaged board centre in the world (platform) frame.

    `Z_i = A_i · X · B_i` is the board's placement in the platform frame; for
    a board glued to the platform it is the same for every view, so the mean
    translation is the board CENTRE in world. Subtracting `turntable_axis`
    gives the board's ECCENTRICITY — how far its centre sits from the AZ
    rotation axis. This is FREE: the very `A_i · X · B_i` `_handeye_residual`
    already forms. On a full, non-90°-periodic ring the mean is exact even
    when `turntable_axis` is unset (the wrong-axis variation mean-cancels over
    a symmetric ring); the spread about it is exactly `rms_translation_mm`.
    """
    X = np.eye(4)
    X[:3, :3] = R_cam_arm
    X[:3, 3] = t_cam_arm
    centres: list[np.ndarray] = []
    for s in samples:
        R, t = arm_pose_in_world(s.az_deg, s.el_deg, turntable_axis)
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
        B = np.eye(4)
        B[:3, :3] = s.board_R_cam
        B[:3, 3] = s.board_t_cam
        centres.append((A @ X @ B)[:3, 3])
    return np.mean(centres, axis=0)


def board_world_pose(
    samples: Sequence[CaptureSample],
    R_cam_arm: np.ndarray,
    t_cam_arm: np.ndarray,
    turntable_axis: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full board-in-world reference pose (R_ref, t_ref) = mean of `Z_i =
    A_i·X·B_i` over the sweep — the constant board placement the "Test
    accuracy" check compares a live observation against. Same `Z` the
    residual forms; here we keep the rotation too (averaged on SO(3))."""
    X = np.eye(4)
    X[:3, :3] = R_cam_arm
    X[:3, 3] = t_cam_arm
    zs: list[np.ndarray] = []
    for s in samples:
        R, t = arm_pose_in_world(s.az_deg, s.el_deg, turntable_axis)
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
        B = np.eye(4)
        B[:3, :3] = s.board_R_cam
        B[:3, 3] = s.board_t_cam
        zs.append(A @ X @ B)
    t_ref = np.mean([Z[:3, 3] for Z in zs], axis=0)
    R_ref = Rotation.from_matrix(np.array([Z[:3, :3] for Z in zs])).mean().as_matrix()
    return R_ref, t_ref


def derive_geometry(R_cam_arm: np.ndarray, t_cam_arm: np.ndarray) -> CalibrationResult:
    """Translate the camera-in-arm-end SE3 into the rig's three scalars.

    Convention (matches scene_graph.py):
      * arm-end frame at the pivot, +X = arm direction, +Z = up at el=0
      * at (az=0, el=0) the camera sits at (arm_radius, 0, camera_offset)
        with optical axis nominally pointing back at the platform centre
        (i.e. along −X in arm frame)

    From those:
      * arm_radius     = t_cam_arm[0]
      * camera_offset  = t_cam_arm[2]
      * camera_tilt    = pitch correction vs the nominal "look-back" axis
      * camera_pan     = yaw correction
    """
    arm_radius_mm    = float(t_cam_arm[0])
    camera_offset_mm = float(t_cam_arm[2])

    # Nominal rotation: camera optical axis +Z_cam points along −X_arm
    # (looking back at the platform). That's a 180° rotation about the
    # arm-Y axis applied to bring camera-frame to arm-frame.
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
#: The four elevations span a wide out-of-plane tilt range, which is what
#: conditions the intrinsics solve (P1) — a narrow {20,50} span leaves focal
#: length and the principal point measurably noisier. Adjust the set if the
#: rig cannot physically reach 10° or 70° EL.
#:
#: ⚠ DEBUG SWEEP (reduced for speed while iterating — accuracy traded for
#: turnaround, per the operator). 3 azimuths × 3 elevations = 9 poses. Azimuths
#: use 120° steps (NOT 90°-periodic, which collapses PARK). This keeps the
#: intrinsics solve alive (≥6 views, ≥3 elevations) but is too sparse for the
#: turntable-axis cx solve (that needs the full ≥6-azimuth ring) — cx stays 0.
#: RESTORE the full accurate ring for production:
#:   [(float(az), el) for az in range(0, 360, 45) for el in (10., 30., 50., 70.)]
DEFAULT_POSES: list[tuple[float, float]] = [
    (float(az), el)
    for az in range(0, 360, 120)          # 0, 120, 240 — non-90°-periodic
    for el in (15.0, 45.0, 65.0)
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

    An unknown preset falls back to `fast`.
    """
    if preset == "normal":
        return [
            (float(az), el)
            for az in range(0, 360, 60)
            for el in (15.0, 35.0, 55.0, 70.0)
        ]
    if preset == "full":
        return [
            (float(az), el)
            for az in range(0, 360, 45)
            for el in (10.0, 30.0, 50.0, 70.0)
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
        rt, _ = _handeye_residual(samples, R, t, (float(cx), 0.0))
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        # A degenerate board-in-world cloud makes calibrateHandEye return a NaN
        # rotation (no raise) → Rotation.mean()/matrix_to_rotvec raise
        # ValueError/LinAlgError; calibrateHandEye itself can raise cv2.error.
        # Treat any of these as an infeasible axis.
        return 1e9
    return float(rt) if (rt is not None and np.isfinite(rt)) else 1e9


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


async def _capture_one(
    az: float,
    el: float,
    board,
) -> CaptureSample | None:
    """Move to (az, el), grab a photo, detect the ChArUco board, and RETAIN the
    raw corners + image size. The board→camera pose is filled later — either by
    the one-shot intrinsics calibration or the per-view solvePnP fallback."""
    cv2 = _cv2()
    await esp.move_and_await(azimuth_deg=az, elevation_deg=el, timeout_ms=15000)
    await asyncio.sleep(_SETTLE_S)
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
    # Per-pose progress is logged by run_calibration (it has the pose index),
    # so we don't double-log the corner count here.
    return CaptureSample(
        az_deg=az, el_deg=el,
        charuco_corners=corners, charuco_ids=ids, image_wh=(w, h),
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
        samples = [s for s in samples if s.board_R_cam is not None]
        if len(samples) < 3:
            raise RuntimeError(
                "calibration failed: board poses could not be recovered for "
                "≥ 3 views even with the fallback intrinsics."
            )

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

    # ── hand-eye + geometry (existing path); guard the degenerate solve ──
    _ui_log("I", f"calibration: solving hand-eye (PARK) from {len(samples)} "
                 f"views, axis=({cx:.1f}, {cy:.1f})…")
    try:
        R_cam_arm, t_cam_arm = solve_hand_eye(samples, turntable_axis)
        result = derive_geometry(R_cam_arm, t_cam_arm)
        rms_t, rms_r = _handeye_residual(
            samples, R_cam_arm, t_cam_arm, turntable_axis,
        )
        t_z = board_in_world_mean(samples, R_cam_arm, t_cam_arm, turntable_axis)
    except (ValueError, np.linalg.LinAlgError, _cv2().error) as exc:
        raise RuntimeError(
            f"calibration failed: degenerate hand-eye solve ({exc}). The "
            "captured views are too few or too co-linear — widen the sweep."
        ) from exc
    result.n_views = len(samples)
    result.n_attempted = len(poses)
    result.rms_translation_mm, result.rms_rotation_deg = rms_t, rms_r
    # Free eccentricity read-off (diagnostic only; exact on a full ring).
    # rms_translation_mm is the quality flag.
    result.board_eccentricity_mm = (float(t_z[0] - cx), float(t_z[1] - cy))
    result.board_height_mm = float(t_z[2])
    # Reference board-in-world pose Z_ref — persisted for "Test accuracy".
    _bw_R, _bw_t = board_world_pose(samples, R_cam_arm, t_cam_arm, turntable_axis)
    result.board_world = MountTransform(
        t=(float(_bw_t[0]), float(_bw_t[1]), float(_bw_t[2])),
        rvec=tuple(float(x) for x in matrix_to_rotvec(_bw_R)),
    )

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
    # predict the board pose at the live encoder angles after a restart.
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
    import camera_io
    from datetime import datetime

    cv2 = _cv2()
    _ts = datetime.now().strftime("%H:%M:%S")  # stamp every msg so it's unique
    ex = getattr(model, "calib_extrinsic", None)
    zr = getattr(model, "calib_board_world", None)
    if not ex or not zr:
        msg = f"Test accuracy ({_ts}): no calibration reference — run a calibration first."
        model.update(calib_test_msg=msg)
        _ui_log("W", msg)
        return {"ok": True, "detected": False, "message": msg}

    az, el = float(model.az), float(model.el)
    board = _build_board(_board_spec_from_model())
    intr = _intrinsics_from_model()
    raw = await camera_io.fetch_photo(el_deg=el)
    img = (cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
           if raw else None)
    corners, ids = detect_board(img, board) if img is not None else (None, None)
    pose = (estimate_board_pose(corners, ids, board, intr)
            if corners is not None else None)
    if pose is None:
        msg = f"Test accuracy ({_ts}): board NOT detected @ az={az:.0f}° el={el:.0f}°"
        model.update(calib_test_msg=msg)
        _ui_log("W", msg)
        return {"ok": True, "detected": False, "message": msg}

    R_bc, t_bc = pose
    Ra, ta = arm_pose_in_world(az, el, getattr(model, "turntable_axis", None))
    A = np.eye(4); A[:3, :3] = Ra; A[:3, 3] = ta
    X = np.eye(4)
    X[:3, :3] = Rotation.from_rotvec(np.asarray(ex["rvec"], float)).as_matrix()
    X[:3, 3] = np.asarray(ex["t"], float)
    B = np.eye(4); B[:3, :3] = R_bc; B[:3, 3] = t_bc
    Z = A @ X @ B

    Zr_R = Rotation.from_rotvec(np.asarray(zr["rvec"], float)).as_matrix()
    Zr_t = np.asarray(zr["t"], float)
    delta_mm = float(np.linalg.norm(Z[:3, 3] - Zr_t))
    delta_deg = float(np.degrees(
        np.linalg.norm(matrix_to_rotvec(Zr_R.T @ Z[:3, :3])),
    ))
    msg = (f"Test accuracy ({_ts}) @ az={az:.0f}° el={el:.0f}°: "
           f"Δrot {delta_deg:.2f}° · Δpos {delta_mm:.1f} mm")
    model.update(calib_test_msg=msg)
    _ui_log("I", msg)
    return {"ok": True, "detected": True, "delta_deg": delta_deg,
            "delta_mm": delta_mm, "az": az, "el": el}


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
    }
