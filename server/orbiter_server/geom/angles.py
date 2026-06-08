"""Angle utilities — port of `ui/src/scanner/math/angles.ts`.

Elevation limits mirror the firmware physical range (COORDINATES.md §4).
"""

from __future__ import annotations

import math

# Physical elevation travel range, degrees. Mirrors the firmware limits.
EL_MIN = -36.0
EL_MAX = 90.0


def normalize_az(deg: float) -> float:
    """Map any float azimuth into [0, 360)."""
    return ((deg % 360.0) + 360.0) % 360.0


def clamp_el(deg: float) -> float:
    """Clamp elevation to the physical range [EL_MIN, EL_MAX]."""
    if deg < EL_MIN:
        return EL_MIN
    if deg > EL_MAX:
        return EL_MAX
    return deg


def is_el_valid(deg: float) -> bool:
    """True if elevation is finite and within the physical range."""
    return math.isfinite(deg) and EL_MIN <= deg <= EL_MAX


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi
