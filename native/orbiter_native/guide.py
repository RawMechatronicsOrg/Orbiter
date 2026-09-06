"""The operator's guide through calibration: which stage comes next, what to
do with the board this instant, and where in the frame to bring it.

The operator holds the scanner in both hands and reads the monitor from
across the bench. So: one stage at a time, in words a glance takes in, with
the place to bring the board drawn on the eye view itself. The stages are
the order the solves depend on each other:

1. the left camera's lens, 2. the right camera's lens — each eye on its own.
   The pair rule keeps the board where both eyes see it, and a lens is
   measured at the frame's own corners and edges, which the other eye does
   not see; `CalibrationFlow.solo` lets a still board that only this eye
   sees become a view;
3. the pair, from the board seen by both eyes at once;
4. the laser sheet, from the stripe across the still board in the left eye;
5. the rolling-shutter readout, optional — the scan averages still batches;
6. a check: with the laser on and the scan running, the right eye's stripe
   must agree with the left eye's triangulation to within a few pixels.
   Hundreds of pixels is what a thin lens set produces, and the only cure
   is to go back and take the views properly.

Nothing here decides anything about the calibration. `CalibrationFlow`
takes the views and solves; the guide reads its counts and its gate and
says what to do. The one thing it sets is `flow.solo`, the eye whose own
stage is running.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibflow import MAX_VIEWS
from .intrinsics import MIN_TILT_SPREAD
from .orient import Orientation, map_point
from .rolling import MIN_VIEWS as READOUT_VIEWS

STAGES = ("left", "right", "pair", "plane", "readout", "check")

#: Views per eye before its lens stage is done. The solve runs from
#: `intrinsics.MIN_VIEWS` (6); on this rig 15–20 views gave the focal length
#: to ±0.5–1.1 % and the scan's veto sat in the hundreds of pixels, while 56
#: views gave ±0.25 %. Thirty, with the coverage below, is the floor.
K_VIEWS = 30
#: Cells of the `GRID`×`GRID` split of the frame the board's centre has
#: visited. Distortion lives at the corners and edges: 24 of 36 leaves only
#: the odd cells of the outer ring unvisited.
K_CELLS = 24
GRID = 6
#: Pairs before the pair stage is done; the solve runs from 6.
PAIR_VIEWS = 20
#: Still poses with the stripe across the board before the sheet is done;
#: the solve runs from `laserplane.MIN_FRAMES` (3).
PLANE_FRAMES = 8
#: Median disagreement between the left eye's triangulation and the right
#: eye's stripe, px, for the geometry to count as confirmed — twice the
#: scan's own veto radius, `ScanParams.confirm_px`.
VETO_OK_PX = 6.0
#: Standard deviation of the pairs' scale (`ViewDescriptor.scale`: the
#: board's size over the frame diagonal) below which every pair so far was
#: taken at one distance. The pair solve is conditioned by depth variety
#: as much as by tilt; at one range the rotation and the translation trade
#: off against each other.
PAIR_SCALE_SPREAD = 0.015

#: Which eyes each stage needs frames from.
_EYES_NEEDED = {"left": ("left",), "right": ("right",), "pair": ("left", "right"),
                "plane": ("left",), "readout": ("left", "right"), "check": ("left", "right")}

_TITLES = {
    "left": "LEFT CAMERA — LENS",
    "right": "RIGHT CAMERA — LENS",
    "pair": "THE PAIR — BOTH CAMERAS",
    "plane": "LASER SHEET — LEFT CAMERA",
    "readout": "ROLLING SHUTTER — OPTIONAL",
    "check": "SCAN CHECK",
}


@dataclass(frozen=True)
class Prompt:
    """One stage's instruction, as the banner and the eye views show it."""

    stage: str
    #: 1-based, of `len(STAGES)`.
    step: int
    title: str
    #: The big line: what to do this instant.
    action: str
    #: The small line: the counts behind it.
    detail: str
    #: "go" — doing the right thing, keep on; "adjust" — change something;
    #: "stop" — a switch, or a re-do, stands in the way; "done" — ready.
    tone: str
    #: The eye view to mark: "left", "right", "both" or None.
    eye: str | None = None
    #: Where to bring the board: (x0, y0, x1, y1) in that eye's ORIGINAL
    #: pixels, or None when the stage has no place to point at.
    target: tuple[float, float, float, float] | None = None
    #: The eye whose own lens stage this is — what `flow.solo` should be.
    solo: str | None = None
    done: bool = False


class Guide:
    """Which stage the operator is on, and the prompt for it.

    Unpinned (the start, and after `restart`), the stage is the first one
    not done, found afresh on every `update`: a stage that stops being done
    — the pair after 'Rig moved', a lens whose re-solve was refused — takes
    the operator back to it. `back` and `next` pin the stage the operator
    chose; a pinned stage still moves forward when it is done, and never
    back.
    """

    def __init__(self) -> None:
        self.index = 0
        self.pinned = False

    @property
    def stage(self) -> str:
        return STAGES[self.index]

    def next(self) -> None:
        self.index = min(self.index + 1, len(STAGES) - 1)
        self.pinned = True

    def back(self) -> None:
        self.index = max(self.index - 1, 0)
        self.pinned = True

    def restart(self) -> None:
        self.index = 0
        self.pinned = False

    def update(self, flow, frames: dict[str, tuple[int, int, Orientation]] | None = None,
               laser_on: bool = False, scanning: bool = False,
               veto_px: float | None = None, kept: int = 0, auto: bool = True,
               offline: set[str] | frozenset[str] = frozenset()) -> Prompt:
        """The prompt for now. `frames` gives each eye's frame size and the
        orientation it is shown in, for the target and its name; `veto_px`
        and `kept` are the scan's last frame, for the check; `auto` is the
        panel's 'calibrate continuously' switch, without which no stage but
        the check can take anything; `offline` names the eyes whose stream
        is down — no board instruction helps those."""
        frames = frames or {}
        if flow.board is None:
            flow.solo = None
            return Prompt(self.stage, self.index + 1, "CALIBRATION",
                          "NO BOARD SPEC FROM THE SERVER — SET THE BOARD IN THE WEB UI",
                          "", "stop")
        done = {s: self._done(s, flow) for s in STAGES}
        if self.pinned:
            if done[self.stage] and self.index < len(STAGES) - 1:
                self.index += 1
        else:
            self.index = next((i for i, s in enumerate(STAGES) if not done[s]), len(STAGES) - 1)
        stage = self.stage
        flow.solo = stage if stage in ("left", "right") else None
        down = [s for s in _EYES_NEEDED[stage] if s in offline]
        if down:
            return self._prompt(stage, " AND ".join(s.upper() for s in down)
                                + " CAMERA OFFLINE — CHECK CAMSERVER",
                                "no frames from that eye; its panel says why", "stop",
                                eye=stage if stage in ("left", "right") else "both",
                                solo=flow.solo)
        if not auto and stage != "check":
            return self._prompt(stage, "TICK 'calibrate continuously' IN THE CALIBRATION PANEL",
                                "the guide only watches; that switch takes the views", "stop",
                                eye=stage if stage in ("left", "right") else "both",
                                solo=flow.solo)
        if stage in ("left", "right"):
            return self._lens(flow, stage, frames.get(stage))
        if stage == "pair":
            return self._pair(flow)
        if stage == "plane":
            return self._plane(flow, laser_on)
        if stage == "readout":
            return self._readout(flow)
        return self._check(flow, laser_on, scanning, veto_px, kept)

    # ── done ──────────────────────────────────────────────────────────────

    @staticmethod
    def _done(stage: str, flow) -> bool:
        if stage in ("left", "right"):
            s = flow.samples
            return (len(s.views(stage)) >= K_VIEWS
                    and int(s.coverage(stage, GRID).sum()) >= K_CELLS
                    and s.tilt_spread(stage) >= MIN_TILT_SPREAD
                    and f"intrinsics:{stage}" in flow.results)
        if stage == "pair":
            return len(flow.samples.paired()) >= PAIR_VIEWS and "stereo" in flow.results
        if stage == "plane":
            return flow.plane.frames >= PLANE_FRAMES and "plane" in flow.results
        if stage == "readout":
            return all(f"readout:{s}" in flow.results for s in ("left", "right"))
        return False   # the check is where the guide ends; it is never skipped

    # ── the stages ────────────────────────────────────────────────────────

    def _prompt(self, stage: str, action: str, detail: str, tone: str, **kw) -> Prompt:
        return Prompt(stage, STAGES.index(stage) + 1,
                      f"STEP {STAGES.index(stage) + 1}/{len(STAGES)} · {_TITLES[stage]}",
                      action, detail, tone, **kw)

    def _lens(self, flow, side: str, frame) -> Prompt:
        key = f"intrinsics:{side}"
        s = flow.samples
        n = len(s.views(side))
        cells = int(s.coverage(side, GRID).sum())
        tilt = s.tilt_spread(side)
        res = flow.results.get(key)
        cam = f"{side.upper()} CAMERA"
        gate, _ = flow.gate()
        tilt_short = tilt < MIN_TILT_SPREAD and n >= 2
        need_tilt = ", TILTED" if tilt_short else ""
        target, place = self._target(flow, side, frame, skip_current=gate == "dup")
        tone, done = "adjust", self._done(side, flow)
        if gate == "board":
            action = f"SHOW THE BOARD TO THE {cam}"
        elif gate in ("settling", "wait"):
            action = "HOLD STILL…"
        elif gate == "moving":
            action = f"HOLD STILL ({flow.moved(side):.0f} px)"
        elif gate == "ceiling":
            action, tone = f"{MAX_VIEWS} VIEWS HELD — PRESS Clear AND START OVER", "stop"
        elif gate == "ok":
            action, tone = f"GOOD — NEXT: {place}{need_tilt}", "go"
        elif n >= K_VIEWS and cells >= K_CELLS and not tilt_short and res is None:
            reason = flow.reasons.get(key)
            action = (f"LENS REFUSED: {reason.upper()} — MORE VIEWS, HELD STILLER" if reason
                      else "HOLD ON — SOLVING THE LENS…")
            tone = "adjust" if reason else "go"
        elif done:
            action, tone = f"LENS DONE — NEXT: {place}", "done"
        elif n >= K_VIEWS and cells >= K_CELLS:
            action = "TILT THE BOARD MORE — HOLD STILL AT EACH TILT"
        else:
            action = f"MOVE THE BOARD {place}{need_tilt}"
        detail = (f"views {n}/{K_VIEWS} · cells {cells}/{K_CELLS} · "
                  f"tilt {tilt:.0f}/{MIN_TILT_SPREAD:.0f}")
        if res is None and flow.reasons.get(key):
            detail += f" · refused: {flow.reasons[key]}"
        if res is not None:
            fx = float(res.intrinsics.K[0, 0])
            sig = res.sigma_f_px / fx * 100.0 if fx > 0 and np.isfinite(res.sigma_f_px) else float("nan")
            detail += f" · f ±{sig:.2f} % · {'saved' if flow.is_saved(key) else 'not saved'}"
        return self._prompt(side, action, detail, tone, eye=side, target=target,
                            solo=side, done=done)

    def _pair(self, flow) -> Prompt:
        paired = flow.samples.paired()
        pairs = len(paired)
        res = flow.results.get("stereo")
        gate, _ = flow.gate()
        tone = "adjust"
        one_range = pairs >= 6 and pair_scale_spread(paired) < PAIR_SCALE_SPREAD
        if gate == "board":
            absent = [s for s in ("left", "right") if flow.where(s) is None]
            action = ("SHOW THE BOARD TO BOTH CAMERAS" if len(absent) == 2
                      else f"THE {absent[0].upper()} CAMERA MUST SEE IT TOO")
        elif gate in ("settling", "wait"):
            action = "HOLD STILL…"
        elif gate == "moving":
            action = "HOLD STILL"
        elif gate in ("gap", "drift"):
            action = "HOLD STILLER — THE EYES ARE OUT OF STEP"
        elif gate == "ceiling":
            action, tone = f"{MAX_VIEWS} VIEWS HELD — PRESS Clear AND START OVER", "stop"
        elif gate == "ok":
            action, tone = ("GOOD — NOW CLOSER OR FARTHER" if one_range
                            else "GOOD — NOW A NEW PLACE, TILT OR DISTANCE"), "go"
        elif pairs >= PAIR_VIEWS and res is None:
            reason = flow.reasons.get("stereo")
            action = (f"PAIR REFUSED: {reason.upper()} — MORE PAIRS" if reason
                      else "HOLD ON — SOLVING THE PAIR…")
            tone = "adjust" if reason else "go"
        elif self._done("pair", flow):
            action, tone = "PAIR DONE — MORE PAIRS ONLY REFINE IT", "done"
        else:
            action = ("CHANGE THE DISTANCE — EVERY PAIR SO FAR IS AT ONE RANGE" if one_range
                      else "NEW PLACE, TILT OR DISTANCE — BOTH CAMERAS ON THE BOARD")
        detail = f"pairs {pairs}/{PAIR_VIEWS}"
        if res is not None:
            detail += (f" · rms {res.rms_px:.2f} px · baseline {res.baseline_mm:.0f} mm · "
                       f"{'saved' if flow.is_saved('stereo') else 'not saved'}")
        return self._prompt("pair", action, detail, tone, eye="both",
                            done=self._done("pair", flow))

    def _plane(self, flow, laser_on: bool) -> Prompt:
        frames, pts = flow.plane.frames, len(flow.plane)
        res = flow.results.get("plane")
        moved = flow.moved("left")
        tone = "adjust"
        if not laser_on:
            action, tone = "TICK 'laser line' IN THE TOOLBAR", "stop"
        elif flow.where("left") is None:
            action = "SHOW THE BOARD TO THE LEFT CAMERA"
        elif not flow.stripe_ok("left"):
            action = "LASER STRIPE STRAIGHT ACROSS THE BOARD"
        elif moved is None or moved > 1.0:
            action = "HOLD STILL"
        elif frames >= PLANE_FRAMES and res is None:
            reason = flow.reasons.get("plane")
            action = (f"SHEET REFUSED: {reason.upper()} — MORE POSES" if reason
                      else "HOLD ON — SOLVING THE SHEET…")
            tone = "adjust" if reason else "go"
        elif self._done("plane", flow):
            action, tone = "SHEET DONE — MORE POSES ONLY REFINE IT", "done"
        else:
            action, tone = "GOOD — HOLD… THEN A NEW TILT OR DISTANCE", "go"
        detail = f"poses {frames}/{PLANE_FRAMES} · {pts} pts"
        if res is not None:
            detail += (f" · rms {res.rms_mm:.2f} mm · d {res.d:.1f} mm · "
                       f"{'saved' if flow.is_saved('plane') else 'not saved'}")
        return self._prompt("plane", action, detail, tone, eye="left",
                            done=self._done("plane", flow))

    def _readout(self, flow) -> Prompt:
        c = {s: flow.motion.count(s) for s in ("left", "right")}
        missing = [s for s in ("left", "right") if f"readout:{s}" not in flow.results]
        which = "BOTH CAMERAS" if len(missing) == 2 else f"THE {missing[0].upper()} CAMERA"
        action = f"TWIST AND TILT THE BOARD BRISKLY IN FRONT OF {which}"
        tone = "adjust"
        if all(c[s] >= READOUT_VIEWS for s in missing):
            action, tone = "HOLD ON — SOLVING THE READOUT…", "go"
        detail = (f"L {c['left']}/{READOUT_VIEWS} · R {c['right']}/{READOUT_VIEWS} frames · "
                  "optional — NEXT skips it")
        return self._prompt("readout", action, detail, tone, eye="both",
                            done=self._done("readout", flow))

    def _check(self, flow, laser_on: bool, scanning: bool, veto_px, kept: int) -> Prompt:
        tone, done = "adjust", False
        if not laser_on:
            action, tone = "TICK 'laser line' IN THE TOOLBAR", "stop"
        elif not scanning:
            action, tone = "TICK 'scanning' AND POINT THE STRIPE AT AN OBJECT", "stop"
        elif veto_px is None:
            action = "WAITING FOR A PAIR WITH THE STRIPE…"
        elif not np.isfinite(veto_px):
            action = "THE RIGHT CAMERA DOES NOT SEE THE STRIPE — TURN THE OBJECT"
        elif veto_px > VETO_OK_PX:
            action, tone = (f"VETO {veto_px:.0f} px — GEOMETRY IS OFF: BACK TO THE PAIR, "
                            "THEN THE SHEET"), "stop"
        elif kept <= 0:
            action = "STRIPE AGREES BUT NOTHING KEPT — OBJECT WITHIN RANGE?"
        else:
            action, tone, done = f"READY — SCAN AWAY (veto {veto_px:.1f} px)", "done", True
        detail = (f"veto ≤ {VETO_OK_PX:g} px means the two eyes agree on the stripe"
                  + (f" · last: veto {veto_px:.1f} px, kept {kept}"
                     if veto_px is not None and np.isfinite(veto_px) else ""))
        return self._prompt("check", action, detail, tone, eye="both", done=done)

    # ── where to bring the board ──────────────────────────────────────────

    @staticmethod
    def _target(flow, side: str, frame, skip_current: bool = False):
        """The least-visited cell of the coverage grid nearest to where the
        board is now — a path across the frame, not a jump to its far corner
        every time — as (rect in original pixels, how to get there in
        words). `skip_current` leaves out the cell the board is in: it is
        there and the view was not new.
        """
        counts = np.zeros((GRID, GRID), int)
        for v in flow.samples.views(side):
            d = v.descriptor
            gx, gy = min(int(d.cx * GRID), GRID - 1), min(int(d.cy * GRID), GRID - 1)
            if gx >= 0 and gy >= 0:
                counts[gy, gx] += 1
        here = flow.where(side) or (0.5, 0.5)
        hx, hy = here[0] * GRID - 0.5, here[1] * GRID - 0.5
        if skip_current:
            cx, cy = min(int(here[0] * GRID), GRID - 1), min(int(here[1] * GRID), GRID - 1)
            counts[cy, cx] = counts.max() + 1
        cand = np.argwhere(counts == counts.min())
        gy, gx = cand[np.argmin(np.hypot(cand[:, 1] - hx, cand[:, 0] - hy))]
        if frame is None:
            return None, "TO THE NEXT CELL"
        w, h, o = frame
        rect = (gx * w / GRID, gy * h / GRID, (gx + 1) * w / GRID, (gy + 1) * h / GRID)
        centre = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        seen = flow.where(side)
        if seen is None:
            return rect, "TO THE " + place_name(*centre, w, h, o)
        return rect, move_words((seen[0] * w, seen[1] * h), centre, w, h, o)


def pair_scale_spread(paired) -> float:
    """Standard deviation of the pairs' scale in the left eye: how much
    distance variety the pair set has. Zero for fewer than two pairs."""
    scales = [p.left.descriptor.scale for p in paired if p.left is not None]
    return float(np.std(scales)) if len(scales) >= 2 else 0.0


def move_words(here: tuple[float, float], there: tuple[float, float],
               w: int, h: int, o: Orientation) -> str:
    """How to move the board from `here` to `there` (both in ORIGINAL pixels
    of a w×h frame) as the operator sees the frame: "A LITTLE UP-LEFT",
    "RIGHT", "FAR DOWN". Distances are in cells of the coverage grid."""
    hx, hy = map_point(*here, w, h, o)
    tx, ty = map_point(*there, w, h, o)
    ow, oh = (h, w) if o.swaps_axes else (w, h)
    dx, dy = (tx - hx) / ow * GRID, (ty - hy) / oh * GRID
    v = "UP" if dy < -0.25 else "DOWN" if dy > 0.25 else ""
    hz = "LEFT" if dx < -0.25 else "RIGHT" if dx > 0.25 else ""
    where = "-".join(s for s in (v, hz) if s)
    if not where:
        return "HOLD IT THERE"
    dist = float(np.hypot(dx, dy))
    return ("A LITTLE " if dist < 1.5 else "FAR " if dist > 3.0 else "") + where


def place_name(x: float, y: float, w: int, h: int, o: Orientation) -> str:
    """Where an ORIGINAL-pixel point of a w×h frame sits in the view the
    operator sees: TOP-LEFT … CENTRE … BOTTOM-RIGHT, by thirds of the
    oriented frame. The operator reads the monitor, not the sensor."""
    ox, oy = map_point(x, y, w, h, o)
    ow, oh = (h, w) if o.swaps_axes else (w, h)
    col = min(max(int(ox / ow * 3), 0), 2)
    row = min(max(int(oy / oh * 3), 0), 2)
    v, hz = ("TOP", "", "BOTTOM")[row], ("LEFT", "", "RIGHT")[col]
    return "-".join(s for s in (v, hz) if s) or "CENTRE"
