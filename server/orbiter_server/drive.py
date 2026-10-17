"""Per-axis drive facts — the one place the server keeps how a driver step
rate maps to output speed, the speed-knob band, and the elevation travel.

The firmware takes speeds as driver step rates (Hz, in 1/16 microsteps) and
enforces the EL travel itself (`POST /move` answers 400 outside it). The UI
shows the output speed next to every Hz value and clamps its EL controls to
the travel from these numbers; it reads them from `model.axes`
(`axes_info`), never from literals of its own.

Steps per output degree come from the firmware's motor self-calibration
(`GET /state` → `motor_cal`, measured per axis) once the axis has one, and
from the v2 gear ratios until then — 1.8° motors (200 full steps/rev) at
1/16 microstepping:

    AZ  belt 7:1   200 · 16 · 7  / 360 = 62.2 µsteps/°
    EL  10:1       200 · 16 · 10 / 360 = 88.9 µsteps/°
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from geom.angles import EL_MAX, EL_MIN

_USTEPS_PER_MOTOR_REV = 200 * 16


@dataclass(frozen=True)
class AxisDrive:
    #: µsteps per output degree from the gear ratio alone.
    steps_per_deg_nominal: float
    #: Band of the per-axis top step rate the operator may set (Hz). Tighter
    #: than the firmware's own clamp (CL_HZ_MIN..CL_HZ_OVERRIDE_MAX).
    hz_min: int
    hz_max: int
    #: The firmware's compiled top step rate (CL_HZ_MAX_AZ / _EL).
    hz_default: int


#: AZ is the light axis and runs fast; EL carries the whole arm, so its band
#: is lower to keep sweeps from over-shooting.
AXES: dict[str, AxisDrive] = {
    "az": AxisDrive(steps_per_deg_nominal=_USTEPS_PER_MOTOR_REV * 7 / 360,
                    hz_min=100, hz_max=4000, hz_default=2100),
    "el": AxisDrive(steps_per_deg_nominal=_USTEPS_PER_MOTOR_REV * 10 / 360,
                    hz_min=50, hz_max=1000, hz_default=350),
}


def clamp_hz(axis: str, hz: float) -> int:
    """`hz` rounded into the axis' speed-knob band."""
    a = AXES[axis]
    return max(a.hz_min, min(a.hz_max, int(round(hz))))


def axis_calibrated(motor_cal: dict[str, Any] | None, axis: str) -> bool | None:
    """Whether the firmware's motor self-calibration holds a result for
    `axis`; None when the firmware does not report one (older builds)."""
    if not motor_cal:
        return None
    m = motor_cal.get(axis) or {}
    return bool(m.get("valid")) and float(m.get("steps_per_deg") or 0.0) > 0.0


def axes_info(motor_cal: dict[str, Any] | None = None) -> dict[str, Any]:
    """`model.axes` — what the UI needs per axis: the speed band, the steps
    per degree to convert with (measured when the axis is motor-calibrated,
    nominal otherwise), the measured top speed, and for EL the travel (AZ
    turns without limit)."""
    out: dict[str, Any] = {}
    for name, a in AXES.items():
        info = asdict(a)
        calibrated = axis_calibrated(motor_cal, name)
        m = (motor_cal or {}).get(name) or {}
        info.update(
            motor_calibrated=calibrated,
            steps_per_deg=float(m["steps_per_deg"]) if calibrated else a.steps_per_deg_nominal,
            steps_per_deg_source="motor calibration" if calibrated else "nominal (gear ratio)",
            max_speed_dps=float(m.get("max_speed_dps") or 0.0) if calibrated else None,
        )
        out[name] = info
    out["el"].update(min_deg=EL_MIN, max_deg=EL_MAX)
    return out
