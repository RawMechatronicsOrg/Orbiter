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
from models import Manifest


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

# Same mapping as tools/sfm_priors_to_colmap.py — keep priors, database and
# feature_extractor flags aligned.
_COLMAP_CAMERA_MODELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "PINHOLE": ("PINHOLE", ("fx", "fy", "cx", "cy")),
    "SIMPLE_PINHOLE": ("SIMPLE_PINHOLE", ("f", "cx", "cy")),
    "SIMPLE_RADIAL": ("SIMPLE_RADIAL", ("f", "cx", "cy", "k")),
    "RADIAL": ("RADIAL", ("f", "cx", "cy", "k1", "k2")),
    "OPENCV": ("OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")),
}


def _load_calibration_state() -> dict:
    """Calibration snapshot persisted by the server.

    The calibrated intrinsics (``calibrated``, ``camera_fx`` ...) are read
    from ``orbiter_state.json`` — the same file the model persists them to —
    rather than the in-memory ``model`` singleton, so this module stays
    import-light and usable from CLI tooling against a data dir.
    Missing / unreadable file → ``{}`` (the uncalibrated fallback)."""
    path = settings.storage_dir / "orbiter_state.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _stored_image_size(manifest: Manifest) -> tuple[int, int] | None:
    """Pixel size of stored ``original.jpg`` files (after ``camera_preset``)."""
    for cap in manifest.captures:
        if cap.stored_width and cap.stored_height:
            return int(cap.stored_width), int(cap.stored_height)
    return None


def _preset_quarter_turns(manifest: Manifest) -> int:
    """CW quarter-turns the captures' preset applied to the stored pixels.

    The calibrated intrinsics in ``orbiter_state.json`` are solved on RAW
    sensor frames (calibration fetches photos without the preset rotation),
    so when the stored JPEGs are rotated (e.g. sm22 -> 1 turn) the intrinsics
    must be rotated to match. Presets don't mix within a scan in practice;
    take the first capture's.
    """
    from camera_adapter import get_preset

    for cap in manifest.captures:
        return get_preset(cap.camera_preset).extra_cw_quarter_turns % 4
    return 0


def _rotate_intrinsics_cw(
    fx: float, fy: float, cx: float, cy: float,
    dist: list[float],
    sensor_w: float, sensor_h: float,
    turns: int,
) -> tuple[float, float, float, float, list[float]]:
    """Map SENSOR-frame pinhole intrinsics onto pixels rotated N x 90 deg CW.

    One CW turn sends pixel (u, v) -> (H - v, u), so per turn:
    fx' = fy, fy' = fx, cx' = H - cy, cy' = cx; radial k1/k2 are rotation-
    invariant, tangential (p1, p2) -> (p2, -p1). Applied iteratively.
    """
    k1, k2, p1, p2 = (list(dist) + [0.0] * 4)[:4]
    w, h = sensor_w, sensor_h
    for _ in range(turns % 4):
        fx, fy = fy, fx
        cx, cy = h - cy, cx
        p1, p2 = p2, -p1
        w, h = h, w
    rest = list(dist[4:])
    return fx, fy, cx, cy, [k1, k2, p1, p2, *rest]


def build_camera_intrinsics(manifest: Manifest) -> dict:
    """COLMAP/OpenCV intrinsics block for ``sfm_priors.json``.

    When the rig is calibrated, uses the ``camera_*`` intrinsics from the
    ChArUco solve (read from the device-persisted ``orbiter_state.json``).
    Frame ``width``/``height`` come from the scan's stored photos (e.g.
    4080×3060), not the 1920×1080 IP-Webcam guess defaults.
    """
    wh = _stored_image_size(manifest)
    if wh is None:
        width = int(_DEFAULT_INTRINSICS["width"])
        height = int(_DEFAULT_INTRINSICS["height"])
    else:
        width, height = wh

    state = _load_calibration_state()
    if state.get("calibrated"):
        fx = float(state.get("camera_fx", _DEFAULT_INTRINSICS["fx"]))
        fy = float(state.get("camera_fy", _DEFAULT_INTRINSICS["fy"]))
        cx = float(state.get("camera_cx", _DEFAULT_INTRINSICS["cx"]))
        cy = float(state.get("camera_cy", _DEFAULT_INTRINSICS["cy"]))
        dist = list(state.get("camera_distortion") or [0.0, 0.0, 0.0, 0.0, 0.0])
        # Calibration solves on raw SENSOR frames; the stored JPEGs may be
        # preset-rotated (sm22 -> 90 deg CW). Rotate the intrinsics so they
        # describe the pixels COLMAP will actually see.
        turns = _preset_quarter_turns(manifest)
        if turns:
            sensor_w, sensor_h = (
                (float(height), float(width)) if turns % 2 else
                (float(width), float(height))
            )
            fx, fy, cx, cy, dist = _rotate_intrinsics_cw(
                fx, fy, cx, cy, dist, sensor_w, sensor_h, turns,
            )
        if any(abs(d) > 1e-8 for d in dist[:4]):
            return {
                "model": "OPENCV",
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": float(dist[0]),
                "k2": float(dist[1]),
                "p1": float(dist[2]),
                "p2": float(dist[3]),
            }
        return {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    # Uncalibrated rig — scale the IP-Webcam guess to the actual frame size.
    ref_w = float(_DEFAULT_INTRINSICS["width"])
    ref_h = float(_DEFAULT_INTRINSICS["height"])
    sx = width / ref_w
    sy = height / ref_h
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "fx": float(_DEFAULT_INTRINSICS["fx"]) * sx,
        "fy": float(_DEFAULT_INTRINSICS["fy"]) * sy,
        "cx": float(_DEFAULT_INTRINSICS["cx"]) * sx,
        "cy": float(_DEFAULT_INTRINSICS["cy"]) * sy,
    }


def format_image_reader_flags(intrinsics: dict) -> str:
    """``feature_extractor`` flags that match ``sparse_priors/cameras.txt``.

    Without this COLMAP defaults to SIMPLE_RADIAL in the database while our
    priors say PINHOLE — ``point_triangulator`` then aborts on model mismatch.
    """
    model = intrinsics.get("model", "PINHOLE")
    if model not in _COLMAP_CAMERA_MODELS:
        raise ValueError(f"unsupported camera model for COLMAP: {model!r}")
    colmap_name, param_keys = _COLMAP_CAMERA_MODELS[model]
    try:
        params = ",".join(f"{float(intrinsics[k]):.10g}" for k in param_keys)
    except KeyError as e:
        raise ValueError(f"camera_intrinsics missing key {e}") from e
    return (
        f"--ImageReader.camera_model {colmap_name} "
        f"--ImageReader.camera_params {params}"
    )


def build_turntable_info() -> dict | None:
    """Rig geometry the mask pipeline stamps deterministically (optional
    ``turntable`` block of ``sfm_priors.json``).

    Everything here lives in the same co-rotating world frame as the image
    poses (the platform frame — the board is glued to the platform, so its
    calibrated reference pose ``calib_board_world`` is CONSTANT in it):

      * ``axis_xy_mm`` — turntable rotation axis (= disc centre);
      * ``board`` — ChArUco board→world pose (Rodrigues ``rvec``, ``t`` mm)
        plus its physical extent. Stamped from this NOMINAL calibrated pose:
        during an object scan the board is often occluded, so per-frame
        detection is deliberately not relied upon.

    ``None`` when the rig was never calibrated (uncalibrated scans keep a
    fully working priors file without the block).
    """
    state = _load_calibration_state()
    if not state.get("calibrated"):
        return None

    axis = state.get("turntable_axis") or [0.0, 0.0]
    info: dict = {
        "axis_xy_mm": [float(axis[0]), float(axis[1])],
    }
    zr = state.get("calib_board_world")
    sx = state.get("charuco_squares_x")
    sy = state.get("charuco_squares_y")
    sq = state.get("charuco_square_length_mm")
    if (
        isinstance(zr, dict) and zr.get("rvec") is not None
        and zr.get("t") is not None and sx and sy and sq
    ):
        info["board"] = {
            "rvec": [float(v) for v in zr["rvec"]],
            "t": [float(v) for v in zr["t"]],
            "width_mm": float(sx) * float(sq),
            "height_mm": float(sy) * float(sq),
        }
    return info


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
        try:
            storage.capture_path(cap.capture_id, "full")
        except FileNotFoundError:
            continue
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

    payload = {
        "schema": _SCHEMA,
        "camera_intrinsics": build_camera_intrinsics(manifest),
        "images": images,
    }
    turntable = build_turntable_info()
    if turntable is not None:
        payload["turntable"] = turntable
    return payload


def write_sfm_priors(scan_id: str) -> Path:
    """Write the priors JSON for a stored scan, return the path."""
    payload = build_sfm_priors(scan_id)
    out_path = settings.scans_dir / scan_id / _OUTPUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_path
