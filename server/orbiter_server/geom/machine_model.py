"""Rig forward kinematics — port of the dynamic parts of `MachineModel.tsx`.

The schematic rig is drawn in the static world frame (visual az = 0); only the
platform disc rotates with the live azimuth. This module computes the moving
pieces — arm-end, the two-segment L-arm, the stem, the camera pose — and
returns them as plain dataclasses; `scene_graph.py` turns those into nodes.

Pure: depends only on `geom.pose`, no FastAPI / model imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pose import GeomParams, MountTransform, camera_pose_at
from .transforms import Quat, Vec3, quat_from_unit_vectors

# Fixed yoke half-length along the EL axis (mm), from MachineModel.tsx.
YOKE_HALF = 160.0


@dataclass(frozen=True)
class Cylinder:
    """A cylinder primitive placed between two points (axis = +Y native)."""
    mid: Vec3
    quat: Quat
    length: float


@dataclass(frozen=True)
class RigKinematics:
    has_geom: bool                 # True once arm radius or an extrinsic is known
    arm_end: Vec3                  # arm-tip (camera_offset = 0 point)
    camera_pos: Vec3
    camera_quat: Quat              # three.js object-frame quaternion
    yoke_ball: Vec3
    arm_joint: Vec3 | None
    seg1: Cylinder | None          # yoke-ball -> joint
    seg2: Cylinder | None          # joint -> arm-tip
    stem: Cylinder | None          # arm-tip -> camera body
    stem_start: Vec3 | None


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
    has_geom = arm_radius_mm > 0 or extrinsic is not None

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

    # Yoke ball at az_vis = 0: (160·sin0, −160·cos0, 0) = (0, −160, 0).
    yoke_ball: Vec3 = (0.0, -YOKE_HALF, 0.0)

    arm_joint: Vec3 | None = None
    seg1: Cylinder | None = None
    seg2: Cylinder | None = None
    if has_geom and arm_radius_mm > 0:
        yb = np.array(yoke_ball)
        tip = np.array(arm_end)
        joint = yb + tip                       # mirror right-angle corner
        s1 = joint - yb
        s2 = tip - joint
        l1 = float(np.linalg.norm(s1))
        l2 = float(np.linalg.norm(s2))
        if l1 >= 1.0 or l2 >= 1.0:
            arm_joint = tuple(joint)
            d1 = s1 / l1 if l1 > 0 else np.array([0.0, 1.0, 0.0])
            d2 = s2 / l2 if l2 > 0 else np.array([0.0, 1.0, 0.0])
            seg1 = Cylinder(tuple(yb + s1 * 0.5), quat_from_unit_vectors((0, 1, 0), tuple(d1)), l1)
            seg2 = Cylinder(tuple(joint + s2 * 0.5), quat_from_unit_vectors((0, 1, 0), tuple(d2)), l2)

    stem: Cylinder | None = None
    stem_start: Vec3 | None = None
    if camera_offset_mm > 0 and arm_radius_mm > 0:
        start = np.array(arm_end)
        end = np.array(camera_xyz)
        d = end - start
        length = float(np.linalg.norm(d))
        if length >= 1.0:
            stem = Cylinder(
                tuple((start + end) * 0.5),
                quat_from_unit_vectors((0, 1, 0), tuple(d / length)),
                length,
            )
            stem_start = arm_end

    return RigKinematics(
        has_geom=has_geom,
        arm_end=arm_end,
        camera_pos=camera_xyz,
        camera_quat=camera_quat,
        yoke_ball=yoke_ball,
        arm_joint=arm_joint,
        seg1=seg1,
        seg2=seg2,
        stem=stem,
        stem_start=stem_start,
    )
