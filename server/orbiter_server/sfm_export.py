"""SfM-priors exporter — write per-photo camera poses for a stored scan.

Output: a `sfm_priors.json` next to the scan manifest, in the schema
documented in `docs/COLMAP.md`:

    {
      "schema": "orbiter.sfm_priors.v1",
      "camera_intrinsics": {
        "model": "PINHOLE",
        "width":  1920, "height": 1080,
        "fx": 1500,     "fy": 1500,
        "cx":  960,     "cy":  540
      },
      "images": [
        {"file": "c_001/photo.jpg",
         "qw":  0.707, "qx": 0, "qy": 0.707, "qz": 0,
         "tx":   220, "ty": 0,  "tz":  45}
      ]
    }

Conventions (matching COLMAP):
  * quaternions Hamilton (w, x, y, z);
  * translations in millimetres;
  * the transform takes world points into camera space.

The OpenCV / COLMAP camera frame is +Z forward, +Y down. The Capture
record on disk carries `camera_xyz_mm` (camera position in the scan's
world frame, mm) and `camera_quat` (the renderer's three.js object-frame
quaternion, with -Z down the optical axis). Convert via `pose.py` from
the same `GeomParams` used at capture time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import storage
from config import settings


_SCHEMA = "orbiter.sfm_priors.v1"
_OUTPUT_NAME = "sfm_priors.json"

# Default IP-Webcam intrinsics — overridden once the operator runs a
# camera-config tool. Listed here so the priors file is always self-
# contained even without a calibrated intrinsic.
_DEFAULT_INTRINSICS = {
    "model": "PINHOLE",
    "width": 1920,
    "height": 1080,
    "fx": 1500.0,
    "fy": 1500.0,
    "cx": 960.0,
    "cy": 540.0,
}


def _three_object_quat_to_world_R(quat_xyzw: list[float]) -> np.ndarray:
    """The capture's `camera_quat` is a three.js object-frame quaternion
    (-Z along the optical axis). Return the camera-to-world rotation matrix
    for an OpenCV camera frame (+Z down the optical axis, +Y down).

    The three.js object frame and the OpenCV camera frame differ by a
    180-deg rotation about +X (R_X180 = diag(1, -1, -1)): a point with
    OpenCV-camera coords X_cv is the same physical point as a three.js
    object-frame coord X_obj = R_X180 * X_cv. So if R_obj is the rotation
    that maps the object frame to world, then R_world<-cam = R_obj * R_X180.
    """
    R_obj = Rotation.from_quat(quat_xyzw).as_matrix()
    R_x180 = np.diag([1.0, -1.0, -1.0])
    return R_obj @ R_x180


def _world_to_camera_quat_t(
    camera_xyz_mm: tuple[float, float, float],
    camera_quat_xyzw: list[float] | None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Return `(qw, qx, qy, qz), (tx, ty, tz)` taking world -> camera.

    COLMAP stores the world->camera transform: a point X_world maps to
    X_cam = R_w2c @ X_world + t_w2c. Camera position C is given in world
    coordinates by C = -R_w2c.T @ t_w2c, so t_w2c = -R_w2c @ C.
    """
    if camera_quat_xyzw is None:
        # No 6-DOF quaternion stored — fall back to an identity orientation.
        R_w2c = np.eye(3)
    else:
        R_c2w = _three_object_quat_to_world_R(list(camera_quat_xyzw))
        R_w2c = R_c2w.T

    C = np.asarray(camera_xyz_mm, dtype=float).reshape(3)
    t_w2c = -R_w2c @ C

    # scipy returns scalar-last (x, y, z, w); COLMAP wants scalar-first.
    q_xyzw = Rotation.from_matrix(R_w2c).as_quat()
    qw, qx, qy, qz = float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])
    return (qw, qx, qy, qz), (float(t_w2c[0]), float(t_w2c[1]), float(t_w2c[2]))


def build_sfm_priors(scan_id: str) -> dict:
    """Build the priors JSON for a stored scan, without writing it."""
    manifest = storage.read_manifest(scan_id)

    images: list[dict] = []
    for cap in manifest.captures:
        xyz = (cap.camera_xyz_mm.x, cap.camera_xyz_mm.y, cap.camera_xyz_mm.z)
        (qw, qx, qy, qz), (tx, ty, tz) = _world_to_camera_quat_t(
            xyz, cap.camera_quat,
        )
        # File name relative to the scan ZIP / archive — the same scheme
        # storage.build_scan_archive uses.
        from camera_adapter import photo_basename

        file_name = f"photos/{photo_basename(cap.index, cap.az_deg, cap.el_deg)}"

        images.append({
            "file": file_name,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": tx, "ty": ty, "tz": tz,
        })

    return {
        "schema": _SCHEMA,
        "camera_intrinsics": dict(_DEFAULT_INTRINSICS),
        "images": images,
    }


def write_sfm_priors(scan_id: str) -> Path:
    """Write the priors JSON for a stored scan, return the path."""
    payload = build_sfm_priors(scan_id)
    out_path = settings.scans_dir / scan_id / _OUTPUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_path
