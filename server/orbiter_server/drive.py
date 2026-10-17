"""Per-axis drive facts — the one place the server keeps how a driver step
rate maps to output speed, the speed-knob band, and the elevation travel.

The firmware takes speeds as driver step rates (Hz, in 1/16 microsteps) and
enforces the EL travel itself (`POST /move` answers 400 outside it). The UI
shows the output speed next to every Hz value and clamps its EL controls to
the travel from these numbers; it reads them from `model.axes`
(`axes_info`), never from literals of its own.

Steps per output degree, v2 mechanics — 1.8° motors (200 full steps/rev) at
1/16 microstepping:

    AZ  belt 7:1   200 · 16 · 7  / 360 = 62.2 µsteps/°
    EL  10:1       200 · 16 · 10 / 360 = 88.9 µsteps/°

Measured on the bench on 2026-10-17: AZ 61.1, EL 90.1 µsteps/°. The readouts
use the measured values. The motor self-calibration being added to the
firmware will report its own steps/° and top speeds; they replace these then.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from geom.angles import EL_MAX, EL_MIN

_USTEPS_PER_MOTOR_REV = 200 * 16


@dataclass(frozen=True)
class AxisDrive:
    #: µsteps per output degree used to turn Hz into °/s.
    steps_per_deg: float
    #: The same from the gear ratio alone.
    steps_per_deg_nominal: float
    #: Where `steps_per_deg` comes from.
    steps_per_deg_source: str
    #: Band of the per-axis top step rate the operator may set (Hz). Tighter
    #: than the firmware's own clamp (CL_HZ_MIN..CL_HZ_OVERRIDE_MAX).
    hz_min: int
    hz_max: int
    #: The firmware's compiled top step rate (CL_HZ_MAX_AZ / _EL).
    hz_default: int

    def deg_per_s(self, hz: float) -> float:
        return hz / self.steps_per_deg


#: AZ is the light axis and runs fast; EL carries the whole arm, so its band
#: is lower to keep sweeps from over-shooting.
AXES: dict[str, AxisDrive] = {
    "az": AxisDrive(
        steps_per_deg=61.1,
        steps_per_deg_nominal=_USTEPS_PER_MOTOR_REV * 7 / 360,
        steps_per_deg_source="bench 2026-10-17",
        hz_min=100, hz_max=4000, hz_default=2100,
    ),
    "el": AxisDrive(
        steps_per_deg=90.1,
        steps_per_deg_nominal=_USTEPS_PER_MOTOR_REV * 10 / 360,
        steps_per_deg_source="bench 2026-10-17",
        hz_min=50, hz_max=1000, hz_default=350,
    ),
}


def clamp_hz(axis: str, hz: float) -> int:
    """`hz` rounded into the axis' speed-knob band."""
    a = AXES[axis]
    return max(a.hz_min, min(a.hz_max, int(round(hz))))


def axes_info() -> dict[str, Any]:
    """`model.axes` — what the UI needs: the drive facts per axis plus the EL
    travel (AZ turns without limit)."""
    out = {name: asdict(a) for name, a in AXES.items()}
    out["el"].update(min_deg=EL_MIN, max_deg=EL_MAX)
    return out
