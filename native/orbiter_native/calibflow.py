"""Continuous calibration: the operator moves the board; this decides what
each frame is good for, solves in the background as the sets grow, and says
what to push to the server when a solve improved.

**One activity, four measurements.** Each frame that shows the board serves
whichever of these it can:

  * a STILL board showing the solve something new — a new place in the frame,
    a new distance, a new tilt — is a view (per-eye intrinsics), and both eyes
    seeing it at the same instant makes it a pair (the stereo geometry);
  * a still board with the stripe straight across it and a pose is a
    laser-plane frame;
  * a board moving BRISKLY is a readout frame — the rolling shutter's time is
    measured from the motion that ruins a view.

**Raw observations, redone solves.** Every set keeps what was observed —
corners, IDs, stripe pixels, capture instants — never what was derived from
it. Each cycle solves the intrinsics from all views, then the pair, the plane
and the readout from *their* raw sets through the intrinsics just solved: the
plane's points and the readout's poses are recomputed, not accumulated, so
they improve with the camera matrix that places them. Twenty near-duplicate
views are worth less than six different ones, so views are gated on novelty
(`intrinsics.SampleSet`), and the guidance names what is still missing.

**Cycles run on a thread.** `due` says when a cycle is worth starting,
`snapshot` copies the inputs, `run` does the solves (nothing else touches
the flow's state), `finish` adopts the outcome and returns what to save. A
result replaces the previous when it is better — more data at a residual no
worse than a little, or a lower residual — and only then goes to the server,
so a bad early solve is never written over a good stored one.

The pairing of the two eyes' results by camserver's capture clock lives here
too, moved from the panel: it is a decision about frames, not about widgets,
and it wants tests.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .intrinsics import (
    MIN_TILT_SPREAD,
    MIN_VIEWS,
    EyeView,
    PairSample,
    SampleSet,
    SolveResult,
)
from .intrinsics import solve as solve_intrinsics
from .laserplane import LaserPlane, PlaneCollector
from .scan import ScanVolume
from .rolling import MotionCollector, MotionView, Readout, solve_readout
from .stereo import StereoResult
from .stereo import calibrate as solve_stereo

log = logging.getLogger("orbiter_native.calibflow")

#: Two frames count as simultaneous within this much of camserver's capture
#: clock — the one clock that times both cameras, which holds the pair to about
#: 0.05 ms. Well under the ~33 ms frame interval, so it still names one frame.
#: (Measured median gap in practice 3.75 ms: camserver reports the pair as
#: free-running, and a hand-held board keeps moving during that gap.)
PAIR_WINDOW_S = 0.004

#: Recent results kept per eye while looking for a partner frame. Detection
#: takes 5-20 ms and the two threads drift independently; measured over 269
#: attempts the newest left and right results were a median 144 ms apart, so
#: comparing only the newest two captured nothing. This spans about half a
#: second at 30 fps.
PAIR_HISTORY = 16

#: The board must be this still before a view or a plane frame is taken,
#: as median corner movement since the previous detection. Motion blur rounds
#: corners off and quietly biases the solve.
STILL_PX = 1.0
#: ...and this brisk before a frame is worth the readout solve: ~4 px per
#: frame is 120 px/s, a couple of pixels of skew over a 20 ms readout. The
#: solve measures the skew itself and refuses too little.
FAST_PX = 4.0
#: A frame feeds the readout solve from this many corners: the pose has to be
#: sound before its slide across the rows can be measured.
READOUT_MIN_CORNERS = 12

#: Views held at most. calibrateCamera's cost grows with the set and past a
#: hundred-odd genuinely different views nothing improves; the novelty gate
#: keeps the set that size on its own, this is the ceiling.
MAX_VIEWS = 160

#: A cycle starts no sooner than this after the previous one ended, and only
#: when a set grew. Solves are seconds each; the loop paces itself on them.
CYCLE_MIN_S = 4.0

#: A new result replaces a stored one when it has more data and a residual no
#: more than this much worse, or a lower residual outright.
WORSE_TOLERANCE = 1.15

SOLVES = ("intrinsics:left", "intrinsics:right", "stereo", "plane",
          "readout:left", "readout:right")

#: What the error budget assumes when it has not been told better: the
#: subject's distance, the stripe centroid's noise across the stripe (0.6-0.7
#: px measured on this rig's line fits), the focal length's uncertainty for
#: intrinsics that came without one, and a hand's speed for an uncorrected
#: rolling shutter with a typical readout.
ASSUMED_Z_MM = 500.0
ASSUMED_STRIPE_PX = 0.6
ASSUMED_FOCAL_REL = 0.005
ASSUMED_HAND_MM_S = 30.0
ASSUMED_READOUT_S = 0.02


@dataclass(frozen=True)
class ErrorBudget:
    """Expected one-sigma error of a scanned point, in mm, at a working
    distance — what the calibration so far buys, term by term.

    The scan places a stripe centroid on the laser sheet, so its error is
    the centroid's pixel noise carried through the sheet's geometry
    (`stripe_mm`: dZ/dpx = Z² / (f · d) for a sheet `d` from the camera
    containing the optical axis, which is this rig), plus how well the
    sheet itself is known (`sheet_mm`, the plane fit's residual), plus the
    scale the focal length's uncertainty puts on the volume (`scale_mm`),
    plus the rolling shutter while it is uncorrected (`shutter_mm`). The
    terms are independent, so they add in quadrature.
    """

    z_mm: float
    stripe_px: float
    stripe_mm: float
    sheet_mm: float
    scale_mm: float
    shutter_mm: float
    assumed: tuple[str, ...] = ()

    @property
    def total_mm(self) -> float:
        return float(np.sqrt(self.stripe_mm ** 2 + self.sheet_mm ** 2
                             + self.scale_mm ** 2 + self.shutter_mm ** 2))


@dataclass
class Job:
    """One cycle's inputs, copied off the flow so the thread owns them."""

    board: Any
    wh: tuple[int, int]
    views: dict[str, list[EyeView]]
    tilt: dict[str, float]
    pairs: list[PairSample]
    plane_frames: int
    plane: PlaneCollector
    motion: dict[str, list[MotionView]]
    known_k: dict[str, Any]
    laser_active: bool


@dataclass
class Outcome:
    """What a cycle produced: a result or a reason per solve."""

    results: dict[str, Any] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass(frozen=True)
class Saved:
    """What the server holds for one solve, enough to judge a challenger."""

    count: int
    residual: float


def _better(new_count: int, new_res: float, old: Saved | None) -> bool:
    if old is None:
        return True
    if not np.isfinite(old.residual):
        return True
    if new_res < old.residual:
        return True
    return new_count > old.count and new_res <= old.residual * WORSE_TOLERANCE


class CalibrationFlow:
    """The sets, the pairing, the schedule, and the verdicts."""

    def __init__(self, solvers: dict[str, Callable] | None = None) -> None:
        self.samples = SampleSet()
        self.plane = PlaneCollector()
        self.motion = MotionCollector()
        self.board = None
        self.board_spec = None
        #: Best intrinsics known per eye: the server's, then each better solve.
        self.known_k: dict[str, Any] = {}
        self.laser_active = False
        #: Latest result and latest refusal per solve, for the scoreboard.
        self.results: dict[str, Any] = {}
        self.reasons: dict[str, str] = {}
        #: What the server holds, per solve, as far as this flow knows.
        self.saved: dict[str, Saved] = {}
        #: The focal length's one-sigma uncertainty per eye, px, when known.
        self.sigma_f: dict[str, float] = {}
        #: The plane the server holds, until one is solved here.
        self.stored_plane: LaserPlane | None = None
        self.solvers = solvers or {
            "intrinsics": solve_intrinsics, "stereo": solve_stereo,
            "plane": None, "readout": solve_readout}
        self._recent: dict[str, deque] = {"left": deque(maxlen=PAIR_HISTORY),
                                          "right": deque(maxlen=PAIR_HISTORY)}
        self._last_corners: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._moved: dict[str, float | None] = {"left": None, "right": None}
        self._running = False
        self._last_cycle_end = -1e9
        self._seen = self._counts()
        self.lock = threading.Lock()

    # ── configuration ─────────────────────────────────────────────────────

    def set_board(self, spec, board) -> bool:
        """Adopt a board. Returns True if the sets were dropped: views measured
        against a different board must not mix with these."""
        if spec == self.board_spec:
            return False
        had = len(self.samples) or self.plane.frames or self.motion.total
        self.board_spec, self.board = spec, board
        if had:
            self.clear()
        return bool(had)

    def set_known_intrinsics(self, side: str, k, raw: dict | None = None) -> None:
        """Intrinsics the server already holds. `raw` is the stored dict, whose
        `views`/`rms_px` say how strong a challenger has to be."""
        if k is None:
            return
        self.known_k.setdefault(side, k)
        sf = (raw or {}).get("sigma_f")
        if sf is not None and side not in self.sigma_f:
            try:
                self.sigma_f[side] = float(sf)
            except (TypeError, ValueError):
                pass
        key = f"intrinsics:{side}"
        if key not in self.saved:
            n = int((raw or {}).get("views", 0) or 0)
            rms = float((raw or {}).get("rms_px", float("nan")) or float("nan"))
            self.saved[key] = Saved(n, rms)

    def set_stored(self, key: str, count: int, residual: float) -> None:
        """A stereo / plane / readout figure the server holds."""
        self.saved.setdefault(key, Saved(count, residual))

    def clear(self) -> None:
        self.samples.clear()
        self.plane.clear()
        self.motion.clear()
        self.results.clear()
        self.reasons.clear()
        for q in self._recent.values():
            q.clear()
        self._last_corners.clear()

    # ── the live feed ─────────────────────────────────────────────────────

    def offer(self, res, auto: bool = True) -> str | None:
        """Take one eye's result; returns a note when something was banked."""
        moved = self._movement(res)
        board = res.board
        has_board = board is not None and board.corners is not None
        notes = []
        if has_board:
            self._recent[res.side].append(res)
            # Brisk motion feeds the readout solve.
            if (moved is not None and moved >= FAST_PX and board.count >= READOUT_MIN_CORNERS
                    and res.capture_mono is not None):
                self.motion.add(res.side, MotionView(
                    np.array(board.corners, np.float32), np.array(board.ids, np.int32),
                    res.capture_mono, res.wh))
                notes.append(f"readout frame {self.motion.count(res.side)} ({res.side})")
        if self._bank_plane(res, moved):
            notes.append(f"plane frame {self.plane.frames}")
        if auto and self.board is not None:
            n = self._capture(force=False)
            if n:
                notes.append(f"view {n}")
        return "; ".join(notes) if notes else None

    def _movement(self, res) -> float | None:
        """Median corner movement since this eye's previous detection, px, by
        corner ID — the count flickers as marginal corners drop in and out."""
        board = res.board
        if board is None or board.corners is None or board.ids is None:
            self._last_corners[res.side] = None
            self._moved[res.side] = None
            return None
        cur = board.corners.reshape(-1, 2)
        cur_ids = board.ids.ravel()
        prev = self._last_corners.get(res.side)
        self._last_corners[res.side] = (cur_ids, cur)
        moved = None
        if prev is not None:
            prev_ids, prev_pts = prev
            common = np.intersect1d(prev_ids, cur_ids)
            if len(common) >= 4:
                a = prev_pts[np.isin(prev_ids, common)]
                b = cur[np.isin(cur_ids, common)]
                moved = float(np.median(np.linalg.norm(b - a, axis=1)))
        self._moved[res.side] = moved
        return moved

    def _bank_plane(self, res, moved: float | None) -> bool:
        """Left eye, stripe straight across the board, pose known, still."""
        k = self.known_k.get("left")
        line, board = res.laser, res.board
        if (res.side != "left" or k is None or line is None or not line.ok
                or board is None or board.R is None or board.t is None
                or board.corners is None or moved is None or moved > STILL_PX):
            return False
        return bool(self.plane.add_frame(
            line.inlier_points, k, board.R, board.t, line.rms_px,
            corners=board.corners, ids=board.ids, wh=res.wh))

    def _find_pair(self):
        left, right = self._recent["left"], self._recent["right"]
        if not left or not right:
            # Deliberately not "whichever eye has something": the history is
            # cleared after every capture, so the next result would always find
            # the other side empty and be stored one-eyed — and the stereo
            # solve needs the paired ones.
            return None, None
        best = None
        for a in left:
            if a.capture_mono is None:
                continue
            for b in right:
                if b.capture_mono is None:
                    continue
                gap = abs(a.capture_mono - b.capture_mono)
                if best is None or gap < best[0]:
                    best = (gap, a, b)
        if best is None or best[0] > PAIR_WINDOW_S:
            return None, None
        return best[1], best[2]

    def _still(self, side: str) -> bool:
        m = self._moved.get(side)
        return m is not None and m <= STILL_PX

    def capture(self) -> int:
        """Manual capture: whatever is there, still or not. Returns the count."""
        return self._capture(force=True)

    def _capture(self, force: bool) -> int:
        if self.board is None or len(self.samples) >= MAX_VIEWS:
            return 0
        left, right = self._find_pair()
        if left is None or right is None:
            if not force:
                return 0
            left = left or (self._recent["left"][-1] if self._recent["left"] else None)
            right = right or (self._recent["right"][-1] if self._recent["right"] else None)
            if left is None and right is None:
                return 0
        views = {"left": None, "right": None}
        for side, r in (("left", left), ("right", right)):
            if (r is None or r.board is None or r.board.corners is None
                    or r.descriptor is None):
                continue
            if not force and not self._still(side):
                return 0
            views[side] = EyeView(r.board.corners, r.board.ids, r.wh,
                                  r.descriptor, capture_mono=r.capture_mono)
        if views["left"] is None and views["right"] is None:
            return 0
        if not force and not self.samples.is_new(
                views["left"].descriptor if views["left"] else None,
                views["right"].descriptor if views["right"] else None):
            return 0
        self.samples.add(PairSample(left=views["left"], right=views["right"]))
        # Consumed: the same frames must not be captured again while the
        # operator holds the board still.
        for q in self._recent.values():
            q.clear()
        return len(self.samples)

    # ── the schedule ──────────────────────────────────────────────────────

    def _counts(self) -> dict[str, int]:
        return {
            "left": len(self.samples.views("left")),
            "right": len(self.samples.views("right")),
            "pairs": len(self.samples.paired()),
            "plane": self.plane.frames,
            "motion:left": self.motion.count("left"),
            "motion:right": self.motion.count("right"),
        }

    @property
    def running(self) -> bool:
        return self._running

    def request(self) -> None:
        """Make the next `due` true regardless of pacing: the operator asked."""
        self._seen = {}
        self._last_cycle_end = -1e9

    def payload_current(self) -> dict[str, Any] | None:
        """Every current result, in the save shape — the operator's 'save now',
        which does not ask whether it beats what the server holds."""
        payload: dict[str, Any] = {}
        for key, res in self.results.items():
            kind, _, side = key.partition(":")
            if kind == "intrinsics":
                payload.setdefault(side, {})["intrinsics"] = res.as_config()
            elif kind == "stereo":
                payload["_extrinsics"] = res.as_config()
            elif kind == "plane":
                payload["_laser_plane"] = res.as_config()
            elif kind == "readout":
                payload.setdefault(side, {})["readout"] = res.as_config()
            self.saved[key] = Saved(*_measure(key, res))
        return payload or None

    def due(self, now: float) -> bool:
        if self._running or self.board is None:
            return False
        if now - self._last_cycle_end < CYCLE_MIN_S:
            return False
        return self._counts() != self._seen

    def snapshot(self, now: float, wh: tuple[int, int] | None = None) -> Job:
        """Copy the inputs for a cycle and mark it running."""
        self._seen = self._counts()
        self._running = True
        views = {s: list(self.samples.views(s)) for s in ("left", "right")}
        if wh is None:
            for s in ("left", "right"):
                if views[s]:
                    wh = views[s][0].wh
                    break
        return Job(
            board=self.board, wh=wh or (0, 0), views=views,
            tilt={s: self.samples.tilt_spread(s) for s in ("left", "right")},
            pairs=list(self.samples.paired()), plane_frames=self.plane.frames,
            plane=self.plane.copy(),
            motion={s: list(self.motion.views(s)) for s in ("left", "right")},
            known_k=dict(self.known_k), laser_active=self.laser_active,
        )

    def run(self, job: Job) -> Outcome:
        """The solves, in dependency order, on the job's copies only."""
        t0 = time.perf_counter()
        out = Outcome()
        k = dict(job.known_k)
        for side in ("left", "right"):
            key = f"intrinsics:{side}"
            views = job.views[side]
            if len(views) < MIN_VIEWS:
                out.reasons[key] = f"{len(views)} views, need {MIN_VIEWS}"
                continue
            res, why = self.solvers["intrinsics"](views, job.board, tilt_spread=job.tilt[side])
            if res is None:
                out.reasons[key] = why
                continue
            out.results[key] = res
            k[side] = res.intrinsics
        if "left" in k and "right" in k and job.pairs:
            res, why = self.solvers["stereo"](job.pairs, job.board, k["left"], k["right"],
                                              job.pairs[0].left.wh)
            if res is None:
                out.reasons["stereo"] = why
            else:
                out.results["stereo"] = res
        elif not job.pairs:
            out.reasons["stereo"] = "no views where both eyes saw the board at once"
        else:
            out.reasons["stereo"] = "needs intrinsics for both eyes"
        if "left" in k and job.plane_frames:
            plane, why = job.plane.refit(job.board, k["left"], job.wh)
            if plane is None:
                out.reasons["plane"] = why
            else:
                out.results["plane"] = plane
        elif not job.plane_frames:
            out.reasons["plane"] = ("stripe detector is off" if not job.laser_active
                                    else "no stripe across the board yet")
        else:
            out.reasons["plane"] = "needs left-eye intrinsics"
        for side in ("left", "right"):
            key = f"readout:{side}"
            views = job.motion[side]
            if side not in k:
                out.reasons[key] = "needs intrinsics"
                continue
            if not views:
                out.reasons[key] = "no brisk motion seen yet"
                continue
            r, why = self.solvers["readout"](views, job.board, k[side])
            if r is None:
                out.reasons[key] = why
            else:
                out.results[key] = r
        out.seconds = time.perf_counter() - t0
        return out

    def finish(self, out: Outcome, now: float) -> dict[str, Any] | None:
        """Adopt a cycle's outcome. Returns the save payload for what
        improved — in the shape `app._save_intrinsics` sends — or None."""
        self._running = False
        self._last_cycle_end = now
        self.reasons.update(out.reasons)
        for key in out.results:
            self.reasons.pop(key, None)
        payload: dict[str, Any] = {}
        for key, res in out.results.items():
            self.results[key] = res
            count, resid = _measure(key, res)
            if not _better(count, resid, self.saved.get(key)):
                continue
            self.saved[key] = Saved(count, resid)
            kind, _, side = key.partition(":")
            if kind == "intrinsics":
                self.known_k[side] = res.intrinsics
                self.sigma_f[side] = float(res.sigma_f_px)
                payload.setdefault(side, {})["intrinsics"] = res.as_config()
            elif kind == "stereo":
                payload["_extrinsics"] = res.as_config()
            elif kind == "plane":
                payload["_laser_plane"] = res.as_config()
            elif kind == "readout":
                payload.setdefault(side, {})["readout"] = res.as_config()
        return payload or None

    # ── the number ────────────────────────────────────────────────────────

    @property
    def plane_known(self) -> LaserPlane | None:
        return self.results.get("plane") or self.stored_plane

    def readout_known(self, side: str = "left") -> bool:
        key = f"readout:{side}"
        return key in self.results or key in self.saved

    def expected_error(self, z_mm: float | None = None,
                       stripe_px: float | None = None) -> ErrorBudget | None:
        """The budget at `z_mm` for a stripe `stripe_px` wide in noise, or
        None until there is a left camera matrix and a laser sheet to scan
        with. Whatever was not given or measured is assumed, and said."""
        k = self.known_k.get("left")
        plane = self.plane_known
        if k is None or plane is None:
            return None
        assumed = []
        if z_mm is None or not np.isfinite(z_mm) or z_mm <= 0:
            z_mm, _ = ASSUMED_Z_MM, assumed.append(f"{ASSUMED_Z_MM:.0f} mm away")
        if stripe_px is None or not np.isfinite(stripe_px) or stripe_px <= 0:
            stripe_px, _ = ASSUMED_STRIPE_PX, assumed.append(f"stripe {ASSUMED_STRIPE_PX} px")
        f = 0.5 * (float(k.fx) + float(k.fy))
        d = max(float(plane.d), 1e-6)
        stripe_mm = stripe_px * z_mm * z_mm / (f * d)
        sheet_mm = float(plane.rms_mm) if np.isfinite(plane.rms_mm) else 0.0
        sf = self.sigma_f.get("left")
        radius = ScanVolume().radius_mm
        if sf is None or not np.isfinite(sf):
            scale_mm = ASSUMED_FOCAL_REL * radius
            assumed.append(f"focal length ±{ASSUMED_FOCAL_REL * 100:.1f}%")
        else:
            scale_mm = sf / f * radius
        if self.readout_known():
            shutter_mm = 0.0
        else:
            shutter_mm = ASSUMED_HAND_MM_S * ASSUMED_READOUT_S * 0.5
            assumed.append(f"shutter unmeasured, {ASSUMED_HAND_MM_S:.0f} mm/s")
        return ErrorBudget(z_mm=float(z_mm), stripe_px=float(stripe_px),
                           stripe_mm=float(stripe_mm), sheet_mm=sheet_mm,
                           scale_mm=float(scale_mm), shutter_mm=float(shutter_mm),
                           assumed=tuple(assumed))

    # ── words ─────────────────────────────────────────────────────────────

    def scoreboard(self) -> list[str]:
        lines = []
        for side in ("left", "right"):
            key = f"intrinsics:{side}"
            r = self.results.get(key)
            if r is not None:
                lines.append(f"K {side[0].upper()}   rms {r.rms_px:.2f} px / {r.n_views} views"
                             f"{'  saved' if self._is_saved(key, r) else ''}")
            else:
                lines.append(f"K {side[0].upper()}   — {self.reasons.get(key, 'not yet')}")
        s = self.results.get("stereo")
        lines.append(f"pair  baseline {s.baseline_mm:.1f} mm rms {s.rms_px:.2f} px / {s.n_views}"
                     f"{'  saved' if self._is_saved('stereo', s) else ''}"
                     if s is not None else f"pair  — {self.reasons.get('stereo', 'not yet')}")
        p = self.results.get("plane")
        lines.append(f"laser rms {p.rms_mm:.2f} mm / {p.n_frames} poses, {p.n_points} pts"
                     f"{'  saved' if self._is_saved('plane', p) else ''}"
                     if p is not None else f"laser — {self.reasons.get('plane', 'not yet')}")
        for side in ("left", "right"):
            key = f"readout:{side}"
            r = self.results.get(key)
            lines.append(f"T {side[0].upper()}   {r.seconds * 1000:+.2f} ± {r.sigma_s * 1000:.2f} ms"
                         f" / {r.views} frames{'  saved' if self._is_saved(key, r) else ''}"
                         if r is not None else f"T {side[0].upper()}   — {self.reasons.get(key, 'not yet')}")
        return lines

    def _is_saved(self, key: str, res) -> bool:
        saved = self.saved.get(key)
        if saved is None:
            return False
        count, resid = _measure(key, res)
        return saved.count == count and (saved.residual == resid
                                         or (np.isnan(saved.residual) and np.isnan(resid)))

    def advice(self) -> str:
        """What to do with the board next: the weakest link first."""
        c = self._counts()
        for side in ("left", "right"):
            if f"intrinsics:{side}" in self.results:
                continue
            tag = side[0].upper()
            if c[side] < MIN_VIEWS:
                return (f"move the board around for {tag}: hold it still in a new place "
                        f"or at a new tilt for each view ({c[side]}/{MIN_VIEWS})")
            if self.samples.tilt_spread(side) < MIN_TILT_SPREAD:
                return f"tilt the board more for {tag} (spread {self.samples.tilt_spread(side):.1f} / {MIN_TILT_SPREAD:.0f})"
            cov = int(self.samples.coverage(side).sum())
            if cov < 12:
                return f"bring the board to the edges and corners of {tag}'s frame ({cov}/36 cells)"
            return f"a few more still views for {tag}"
        if "stereo" not in self.results:
            return (f"hold the board still where BOTH eyes see it ({c['pairs']} pairs)"
                    if c["pairs"] < MIN_VIEWS else "more pairs, at other tilts and distances")
        if "plane" not in self.results:
            if not self.laser_active:
                return "tick 'laser line' so the stripe is detected"
            return ("bring the laser across the board, hold still, at several tilts "
                    f"({c['plane']} frames)")
        for side in ("left", "right"):
            if f"readout:{side}" not in self.results:
                return ("twist and tilt the board briskly in front of both eyes "
                        f"({c['motion:left']} / {c['motion:right']} frames)")
        return "everything solved — keep going to refine: new places, tilts, a brisk twist now and then"


def _measure(key: str, res) -> tuple[int, float]:
    """(how much data, how good) for a result, for `_better`."""
    kind = key.partition(":")[0]
    if kind == "intrinsics":
        return int(res.n_views), float(res.rms_px)
    if kind == "stereo":
        return int(res.n_views), float(res.rms_px)
    if kind == "plane":
        return int(res.n_frames), float(res.rms_mm)
    if kind == "readout":
        return int(res.views), float(res.sigma_s)
    raise KeyError(key)
