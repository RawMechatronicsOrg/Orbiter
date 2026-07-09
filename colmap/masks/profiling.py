"""Phase profiler for the mask tool — find where a run spends its time.

The masking loop runs the same handful of phases per frame (read → geometric
prompt → SAM2 encode → SAM2 decode → post-process → write). To answer "what is
the bottleneck?" across many runs we accumulate wall time PER PHASE (total +
count) and, optionally, per-frame so a single slow frame is locatable.

Stdlib-only on purpose: the tool must run anywhere — the COLMAP container, a
bare host venv — so this keeps the same dependency-free footprint as its
sibling modules. ``time.perf_counter`` is the
right clock here — a monotonic, highest-resolution timer unaffected by wall
clock jumps.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class PhaseProfiler:
    """Accumulate elapsed time per named phase.

    Use as ``with prof.phase("encode"): ...`` anywhere; wrap a frame's work in
    ``with prof.frame() as t: ...`` to ALSO collect that frame's per-phase
    breakdown into ``t`` (for the manifest row)."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._frame: dict[str, float] | None = None

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1
            if self._frame is not None:
                self._frame[name] = round(self._frame.get(name, 0.0) + dt, 4)

    @contextmanager
    def frame(self):
        """Scope one frame. Phases entered inside also sum into a fresh
        per-frame dict, which is yielded so the caller can stash it on the
        frame's manifest row. Nesting is not supported (one frame at a time)."""
        self._frame = {}
        try:
            yield self._frame
        finally:
            self._frame = None

    def summary(self, wall_sec: float) -> dict:
        """Machine-readable profile for the manifest: total wall, per-stage
        seconds (``totals`` — the cross-run stat target), and a richer
        ``phases`` view (count, mean ms, % of wall)."""
        phases: dict[str, dict] = {}
        for name in sorted(self.totals, key=lambda k: self.totals[k], reverse=True):
            total = self.totals[name]
            cnt = self.counts[name]
            phases[name] = {
                "total_sec": round(total, 3),
                "count": cnt,
                "mean_ms": round(1000.0 * total / cnt, 1) if cnt else 0.0,
                "pct": round(100.0 * total / wall_sec, 1) if wall_sec else 0.0,
            }
        return {
            "wall_sec": round(wall_sec, 2),
            "totals": {k: round(v, 3) for k, v in self.totals.items()},
            "phases": phases,
        }

    def format_lines(self, wall_sec: float) -> list[str]:
        """Human breakdown for stdout (captured into the job log) — slowest
        phase first, with each phase's share of the wall clock."""
        s = self.summary(wall_sec)
        out = [f"profile: wall {s['wall_sec']}s "
               "(phase: total / %wall / mean×count)"]
        for name, p in s["phases"].items():
            out.append(f"  {name:<11} {p['total_sec']:8.2f}s "
                       f"{p['pct']:5.1f}%  {p['mean_ms']:.1f}ms×{p['count']}")
        return out
