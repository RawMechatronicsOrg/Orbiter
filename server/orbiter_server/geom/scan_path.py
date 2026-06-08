"""Scan-path planning — port of `ui/src/scanner/math/scanPath.ts`.

Spherically-uniform sampling density: the number of points on a ring scales
as |cos(el)|, so the arc length between neighbouring cameras on a ring is
roughly constant at all elevations. `az_step_deg` is the step AT THE EQUATOR
(el=0); rings nearer the pole are thinned automatically. Odd rings are
staggered by half their own ring step (checkerboard).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Minimum points per ring near the pole — keeps a ring from degenerating.
MIN_AZ_PER_RING = 6


@dataclass(frozen=True)
class ScanPathPoint:
    index: int
    az_deg: float
    el_deg: float


def _js_round(x: float, decimals: int = 0) -> float:
    """Round half toward +Inf, matching JavaScript `Math.round`."""
    f = 10.0**decimals
    return math.floor(x * f + 0.5) / f


def plan_scan_path(
    el_start_deg: float,
    el_max_deg: float,
    el_steps: int,
    az_step_deg: float,
) -> list[ScanPathPoint]:
    """Plan the scan trajectory in execution order. Returns [] for invalid params."""
    if az_step_deg <= 0 or el_steps < 0:
        return []

    el_interval = (el_max_deg - el_start_deg) / el_steps if el_steps > 0 else 0.0
    base_az_count = max(1, math.floor(360.0 / az_step_deg))

    out: list[ScanPathPoint] = []
    idx = 0
    for ei in range(el_steps + 1):
        el_deg = _js_round(el_start_deg + ei * el_interval, 4)

        c = abs(math.cos(el_deg * math.pi / 180.0))
        ring_count = max(MIN_AZ_PER_RING, int(_js_round(base_az_count * c)))
        step_ring = 360.0 / ring_count
        az_phase = step_ring / 2.0 if (ei % 2) == 1 else 0.0

        for ai in range(ring_count):
            az_raw = ai * step_ring + az_phase
            az_norm = ((az_raw % 360.0) + 360.0) % 360.0
            out.append(ScanPathPoint(
                index=idx, az_deg=_js_round(az_norm, 4), el_deg=el_deg,
            ))
            idx += 1
    return out
