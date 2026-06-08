"""Low-level rotation / transform primitives, built on `scipy.spatial.transform`.

This replaces the hand-rolled Rodrigues / quaternion / matrix code that used to
live in `pose.py`. Everything downstream (`pose.py`, `rig.py`, `machine_model.py`)
builds on these.

Conventions:
  * quaternions are scalar-last `(x, y, z, w)` — the three.js / scipy order;
  * rotation vectors ("Rodrigues") are axis·angle in radians;
  * the encoder rotation is `Az·El` with Az about +Z and El about -Y (so a
    positive elevation lifts the camera toward +Z) — see COORDINATES.md §5.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# OpenCV camera axes (X right, Y down, Z forward) -> three.js object axes
# (a 180° turn about X). Used to convert a camera->world rotation into the
# object-frame quaternion three.js wants.
RX180: np.ndarray = np.diag([1.0, -1.0, -1.0]).astype(float)


# ── conversions ─────────────────────────────────────────────────────────────

def rotvec_to_matrix(rvec: np.ndarray | Vec3) -> np.ndarray:
    """Rotation vector (axis·angle, rad) -> 3x3 rotation matrix."""
    return Rotation.from_rotvec(np.asarray(rvec, dtype=float)).as_matrix()


def matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector (rad). Robust near 180°."""
    return Rotation.from_matrix(np.asarray(m, dtype=float)).as_rotvec()


def matrix_to_quat(m: np.ndarray) -> Quat:
    """3x3 rotation matrix -> quaternion (x, y, z, w)."""
    q = Rotation.from_matrix(np.asarray(m, dtype=float)).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def quat_to_matrix(q: np.ndarray | Quat) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    return Rotation.from_quat(np.asarray(q, dtype=float)).as_matrix()


# ── quaternion algebra (three.js (x,y,z,w) order) ────────────────────────────

def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product of two `(x, y, z, w)` quaternions. `quat_mul(a, b)`
    applies `b` FIRST then `a` (a∘b), so post-multiplying by a local-frame
    rotation (e.g. `roll_about_lens_quat`) banks in the body frame. Shared by
    the live-frustum bank and the stored-capture bank so they agree exactly."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def roll_about_lens_quat(roll_deg: float) -> Quat:
    """Quaternion rotating by `roll_deg` about a three.js camera's lens axis
    (local -Z). Post-multiply onto a camera quaternion to bank it the way the
    phone reports the bracket roll."""
    half = math.radians(roll_deg) * 0.5
    s, c = math.sin(half), math.cos(half)
    return (0.0, 0.0, -s, c)


# ── rig rotations ───────────────────────────────────────────────────────────

def el_matrix(el_deg: float) -> np.ndarray:
    """Elevation rotation about -Y: +el lifts the camera toward +Z."""
    return Rotation.from_euler("y", -el_deg, degrees=True).as_matrix()


def az_el_matrix(az_deg: float, el_deg: float) -> np.ndarray:
    """The encoder-driven rotation `Az(az)·El(el)` — Az about +Z, El about -Y."""
    return Rotation.from_euler("ZY", [az_deg, -el_deg], degrees=True).as_matrix()


# ── alignment / projection ──────────────────────────────────────────────────

def quat_from_unit_vectors(v_from: Vec3, v_to: Vec3) -> Quat:
    """Shortest-arc quaternion rotating `v_from` onto `v_to` (both need not be
    unit length). Equivalent to three.js `Quaternion.setFromUnitVectors`."""
    a = np.asarray(v_from, dtype=float).reshape(1, 3)
    b = np.asarray(v_to, dtype=float).reshape(1, 3)
    rot, _ = Rotation.align_vectors(b, a)
    q = rot.as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def procrustes_rotation(cov: np.ndarray) -> np.ndarray:
    """Best rotation R ∈ SO(3) from a 3x3 cross-covariance matrix
    (orthogonal-Procrustes / Wahba / Kabsch closure)."""
    c = np.asarray(cov, dtype=float).reshape(3, 3)
    u, _s, vt = np.linalg.svd(c)
    d = 1.0 if np.linalg.det(u) * np.linalg.det(vt) >= 0 else -1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def rotation_angle(m: np.ndarray) -> float:
    """Geodesic rotation angle (rad) of a 3x3 rotation matrix."""
    return float(Rotation.from_matrix(np.asarray(m, dtype=float)).magnitude())


# ── homogeneous (SE3) helpers ───────────────────────────────────────────────

def homogeneous(rotation: np.ndarray, translation: np.ndarray | Vec3) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation + translation."""
    m = np.eye(4)
    m[:3, :3] = np.asarray(rotation, dtype=float)
    m[:3, 3] = np.asarray(translation, dtype=float)
    return m
