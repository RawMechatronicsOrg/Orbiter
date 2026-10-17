"""Rig forward kinematics — port of the dynamic parts of `MachineModel.tsx`.

The rig is drawn in the static world frame (visual az = 0); only the platform
disc rotates with the live azimuth. This module computes the moving pieces —
the arm-end and the camera pose — in the math frame (origin at the AZ-EL axes
intersection) and returns them as a plain dataclass; `scene_graph.py` turns
them into nodes. The arm itself is the CAD model's (`geom/rig_model.py`).

Pure: depends only on `geom.pose`, no FastAPI / model imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .pose import GeomParams, MountTransform, camera_pose_at
from .transforms import Quat, Vec3


@dataclass(frozen=True)
class Cylinder:
    """A cylinder primitive placed between two points (axis = +Y native)."""
    mid: Vec3
    quat: Quat
    length: float


@dataclass(frozen=True)
class RigKinematics:
    arm_end: Vec3                  # arm-tip (camera_offset = 0 point)
    camera_pos: Vec3
    camera_quat: Quat              # three.js object-frame quaternion


def compute_rig(
    el_deg: float,
    arm_radius_mm: float,
    camera_offset_mm: float,
    camera_tilt_deg: float,
    camera_pan_deg: float,
    extrinsic: MountTransform | None,
    turntable_axis: tuple[float, float] | None,
    rocker: MountTransform | None = None,
) -> RigKinematics:
    """Compute the moving rig geometry at the static visual frame (az = 0)."""
    # Arm-tip: +X rotated by elevation only (az_vis = 0).
    r = max(arm_radius_mm, 1.0)
    er = math.radians(el_deg)
    arm_end: Vec3 = (r * math.cos(er), 0.0, r * math.sin(er))

    # Camera pose at az_vis = 0 — same math the frustum / pose marker use.
    geom = GeomParams(
        extrinsic=extrinsic,
        arm_radius=r,
        camera_offset=camera_offset_mm,
        camera_tilt=camera_tilt_deg,
        camera_pan=camera_pan_deg,
        turntable_axis=turntable_axis,
        rocker=rocker,
    )
    pose = camera_pose_at(0.0, el_deg, geom)
    camera_quat = pose.camera_quat or (0.0, 0.0, 0.0, 1.0)

    # ── render-time gauge correction for turntable_axis eccentricity ──
    # The CAD renderer draws the disc at world origin, NOT at the eccentric
    # `turntable_axis = (cx, cy)`. So `pose.camera_xyz_mm` lives in the
    # solver-world (disc at C), while the rest of the scene lives in CAD
    # world (disc at origin). At az_vis = 0 the kinematic chain reduces to
    # `El*X` (no C), so the camera position is just `pose.camera_xyz_mm`.
    # Subtracting (cx, cy, 0) gives the camera position in disc-centred
    # CAD coords — what the renderer needs.
    camera_xyz = pose.camera_xyz_mm
    look_at_xyz = pose.look_at_xyz_mm
    # Gate on `extrinsic`: only the calibrated 6-DOF path (compute_camera_pose_x)
    # places the camera in solver-world with the axis at C. The manual model
    # ignores C, so applying the correction there would shift the camera by −C
    # with nothing to compensate.
    if extrinsic is not None and turntable_axis is not None:
        cx, cy = turntable_axis
        if abs(cx) > 1e-3 or abs(cy) > 1e-3:
            camera_xyz = (camera_xyz[0] - cx, camera_xyz[1] - cy, camera_xyz[2])
            look_at_xyz = (look_at_xyz[0] - cx, look_at_xyz[1] - cy, look_at_xyz[2])

    return RigKinematics(
        arm_end=arm_end,
        camera_pos=camera_xyz,
        camera_quat=camera_quat,
    )
