"""Rig kinematics as a frame graph (pytransform3d `TransformManager`).

The kinematic chain of the turntable, described **once** and queried for any
frame's pose:

    world ──Az(az) about C──▶ az ──El(-el)──▶ el ──X──▶ camera ──RX180──▶ camera_obj

  * `Az` is the azimuth rotation about the *vertical line through the
    turntable axis* `C = (cx, cy)` — so an eccentric board placement is
    handled exactly (`Translate(C)·Rz(az)·Translate(-C)`).
  * `El` is the elevation rotation about -Y.
  * `X` is the constant camera mount transform (`MountTransform`), in the
    manual model derived from the arm/offset/tilt/pan parameters.
  * `camera` is the OpenCV camera frame (X right, Y down, Z forward);
    `camera_obj` is the three.js object frame the renderer wants.

`T_cam->world = Az_C(az)·El(el)·X` — the same relation `pose.py` documents,
now expressed as graph edges instead of hand-composed matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation

from .transforms import (
    RX180,
    Quat,
    Vec3,
    el_matrix,
    homogeneous,
    matrix_to_quat,
    rotvec_to_matrix,
)

# Frame names — use the constants, not bare strings, so a typo is a NameError.
FRAME_WORLD = "world"
FRAME_AZ = "az"
FRAME_EL = "el"
FRAME_CAMERA = "camera"          # OpenCV camera frame (Z forward)
FRAME_CAMERA_OBJ = "camera_obj"  # three.js object frame (renderer quaternion)
FRAME_BOARD = "board"


@dataclass(frozen=True)
class MountTransform:
    """A constant 6-DOF transform: translation (mm) + rotation vector (rad)."""

    t: Vec3
    rvec: Vec3

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous transform."""
        return homogeneous(rotvec_to_matrix(self.rvec), self.t)


class RigGraph:
    """Thin query wrapper around a `TransformManager`. Every accessor takes a
    frame name (see the `FRAME_*` constants) and an optional reference frame."""

    def __init__(self, tm: TransformManager) -> None:
        self.tm = tm

    def matrix(self, frame: str, ref: str = FRAME_WORLD) -> np.ndarray:
        """4x4 transform mapping points in `frame` into `ref`."""
        return self.tm.get_transform(frame, ref)

    def position(self, frame: str, ref: str = FRAME_WORLD) -> Vec3:
        """Origin of `frame` expressed in `ref`."""
        m = self.tm.get_transform(frame, ref)
        return (float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))

    def rotation(self, frame: str, ref: str = FRAME_WORLD) -> np.ndarray:
        """3x3 rotation of `frame` relative to `ref`."""
        return self.tm.get_transform(frame, ref)[:3, :3]

    def quat(self, frame: str, ref: str = FRAME_WORLD) -> Quat:
        """Orientation of `frame` relative to `ref` as a quaternion (x,y,z,w)."""
        return matrix_to_quat(self.rotation(frame, ref))


def _az_about_axis(az_deg: float, axis_xy: tuple[float, float]) -> np.ndarray:
    """Azimuth rotation about the vertical line through `axis_xy`. The composite
    `Translate(C)·Rz(az)·Translate(-C)` has rotation `Rz` and translation
    `C − Rz·C`."""
    c = np.array([axis_xy[0], axis_xy[1], 0.0])
    rz = Rotation.from_euler("z", az_deg, degrees=True).as_matrix()
    return homogeneous(rz, c - rz @ c)


def build_rig_graph(
    az_deg: float,
    el_deg: float,
    mount: MountTransform,
    turntable_axis: tuple[float, float] | None = None,
    board: MountTransform | None = None,
) -> RigGraph:
    """Build the rig frame graph at a given encoder pose.

    `turntable_axis` offsets the azimuth rotation axis; `board`, when given,
    adds a `board` frame at its world placement so board-relative queries work.
    """
    tm = TransformManager(strict_check=False)

    tm.add_transform(FRAME_AZ, FRAME_WORLD,
                     _az_about_axis(az_deg, turntable_axis or (0.0, 0.0)))
    tm.add_transform(FRAME_EL, FRAME_AZ,
                     homogeneous(el_matrix(el_deg), (0.0, 0.0, 0.0)))
    tm.add_transform(FRAME_CAMERA, FRAME_EL, mount.as_matrix())
    tm.add_transform(FRAME_CAMERA_OBJ, FRAME_CAMERA,
                     homogeneous(RX180, (0.0, 0.0, 0.0)))

    if board is not None:
        tm.add_transform(FRAME_BOARD, FRAME_WORLD, board.as_matrix())

    return RigGraph(tm)
