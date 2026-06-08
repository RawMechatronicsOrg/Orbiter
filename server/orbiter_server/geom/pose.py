"""Camera-pose math. See COORDINATES.md §5.

  p_local(el=0) = (r_arm, 0, 0) + cameraOffset · stem_dir
  R_el rotates around -Y_W so +el lifts the camera toward +Z_W
  R_az rotates around +Z_W
  camera = R_az(az) · R_el(el) · p_local

Two pose models:
  * compute_camera_pose   — 4-param tape-measure model (position + optical axis,
    no roll).
  * compute_camera_pose_x — full 6-DOF model from a calibrated mount transform X.
    `T_cam->world = Az_C(az)·El(el)·X`, evaluated through the `rig.py` frame
    graph (pytransform3d).
`camera_pose_at` picks X when available, else the manual model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .angles import deg2rad
from .rig import (
    FRAME_CAMERA,
    FRAME_CAMERA_OBJ,
    MountTransform,
    build_rig_graph,
)
from .transforms import Quat, Vec3, matrix_to_quat

__all__ = [
    "Vec3", "Quat", "MountTransform", "Pose", "GeomParams",
    "compute_camera_pose", "compute_camera_pose_x", "camera_pose_at",
]


@dataclass(frozen=True)
class Pose:
    """Camera pose. `camera_quat` is a three.js object-frame quaternion (x,y,z,w)
    whose -Z axis lies along the optical axis (set by the 6-DOF path; the manual
    fallback fills it with a level-camera assumption)."""

    camera_xyz_mm: Vec3
    look_at_xyz_mm: Vec3
    optical_axis_unit: Vec3
    camera_quat: Quat | None = None


@dataclass(frozen=True)
class GeomParams:
    """Inputs for `camera_pose_at` — the calibrated X plus the manual model."""

    extrinsic: MountTransform | None
    arm_radius: float
    camera_offset: float
    camera_tilt: float = 0.0
    camera_pan: float = 0.0
    # Turntable rotation axis (world XY, mm); used only on the calibrated path.
    turntable_axis: tuple[float, float] | None = None


def _js_round(x: float, decimals: int) -> float:
    """Round half toward +Inf, matching JavaScript `Math.round` (pose.ts uses
    `Math.round(x * 10**d) / 10**d`)."""
    f = 10.0**decimals
    return math.floor(x * f + 0.5) / f


def compute_camera_pose(
    az_deg: float,
    el_deg: float,
    arm_radius: float,
    camera_offset: float,
    camera_tilt_deg: float = 0.0,
    camera_pan_deg: float = 0.0,
) -> Pose:
    """4-parameter manual pose model (no roll). Port of `computeCameraPose`."""
    az, el = deg2rad(az_deg), deg2rad(el_deg)
    tilt, pan = deg2rad(camera_tilt_deg), deg2rad(camera_pan_deg)
    s_t, c_t = math.sin(tilt), math.cos(tilt)
    s_p, c_p = math.sin(pan), math.cos(pan)
    s_e, c_e = math.sin(el), math.cos(el)
    s_a, c_a = math.sin(az), math.cos(az)

    # Camera body vectors after the 2-axis joint Rc = R_z(pan)·R_y(tilt).
    stem_x = -s_t * c_p
    stem_y = -s_t * s_p
    stem_z = c_t

    ax = arm_radius + camera_offset * stem_x
    ay = camera_offset * stem_y
    azl = camera_offset * stem_z

    # R_el (about -Y): rotates X-Z, keeps Y.
    x_l = ax * c_e - azl * s_e
    y_l = ay
    z_l = ax * s_e + azl * c_e

    # R_az (about +Z): rotates X-Y.
    camera = (
        _js_round(x_l * c_a - y_l * s_a, 3),
        _js_round(x_l * s_a + y_l * c_a, 3),
        _js_round(z_l, 3),
    )

    # Optical axis: Rc·(-1,0,0), then R_el, then R_az.
    o_x, o_y, o_z = -c_t * c_p, -c_t * s_p, -s_t
    ox_e = o_x * c_e - o_z * s_e
    oy_e = o_y
    oz_e = o_x * s_e + o_z * c_e
    optical_x = ox_e * c_a - oy_e * s_a
    optical_y = ox_e * s_a + oy_e * c_a
    optical_z = oz_e

    r = max(arm_radius, 1.0)
    look_at = (
        _js_round(camera[0] + optical_x * r, 3),
        _js_round(camera[1] + optical_y * r, 3),
        _js_round(camera[2] + optical_z * r, 3),
    )
    return Pose(
        camera_xyz_mm=camera,
        look_at_xyz_mm=look_at,
        optical_axis_unit=(
            _js_round(optical_x, 6),
            _js_round(optical_y, 6),
            _js_round(optical_z, 6),
        ),
    )


def compute_camera_pose_x(
    az_deg: float,
    el_deg: float,
    x: MountTransform,
    turntable_axis: tuple[float, float] | None = None,
) -> Pose:
    """Full 6-DOF pose from the calibrated mount transform X, via the rig graph.
    `T_cam->world = Az_C(az)·El(el)·X` — Az_C rotates about the vertical line
    through the turntable axis, so an eccentric board placement is handled."""
    graph = build_rig_graph(az_deg, el_deg, x, turntable_axis)
    c = np.asarray(graph.position(FRAME_CAMERA), dtype=float)
    optical = graph.rotation(FRAME_CAMERA) @ np.array([0.0, 0.0, 1.0])
    r = max(float(np.linalg.norm(c)), 1.0)
    return Pose(
        camera_xyz_mm=(_js_round(c[0], 3), _js_round(c[1], 3), _js_round(c[2], 3)),
        look_at_xyz_mm=(
            _js_round(c[0] + optical[0] * r, 3),
            _js_round(c[1] + optical[1] * r, 3),
            _js_round(c[2] + optical[2] * r, 3),
        ),
        optical_axis_unit=(
            _js_round(optical[0], 6),
            _js_round(optical[1], 6),
            _js_round(optical[2], 6),
        ),
        camera_quat=graph.quat(FRAME_CAMERA_OBJ),
    )


def _level_camera_quat(camera: Vec3, look_at: Vec3) -> Quat:
    """Quaternion for the manual model's level-camera assumption (world +Z up).
    Replicates `THREE.Object3D.lookAt` for a non-camera object: the object's
    +Z axis points toward the subject."""
    cam = np.asarray(camera, dtype=float)
    tgt = np.asarray(look_at, dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    z = tgt - cam
    if float(z @ z) == 0.0:
        z = np.array([0.0, 0.0, 1.0])
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if float(x @ x) == 0.0:
        if abs(up[2]) == 1.0:
            z[0] += 1e-4
        else:
            z[2] += 1e-4
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.column_stack((x, y, z))       # columns = local axes
    return matrix_to_quat(rot)


def camera_pose_at(az_deg: float, el_deg: float, geom: GeomParams) -> Pose:
    """Unified camera-pose accessor. Uses the calibrated extrinsic X when
    present (exact, roll included), else the manual model with a level-camera
    roll assumption."""
    if geom.extrinsic is not None:
        return compute_camera_pose_x(
            az_deg, el_deg, geom.extrinsic, geom.turntable_axis,
        )
    p = compute_camera_pose(
        az_deg, el_deg,
        geom.arm_radius, geom.camera_offset, geom.camera_tilt, geom.camera_pan,
    )
    return Pose(
        camera_xyz_mm=p.camera_xyz_mm,
        look_at_xyz_mm=p.look_at_xyz_mm,
        optical_axis_unit=p.optical_axis_unit,
        camera_quat=_level_camera_quat(p.camera_xyz_mm, p.look_at_xyz_mm),
    )
