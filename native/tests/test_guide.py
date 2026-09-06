"""The operator's guide: the stage follows what is done, the lens stages
take one-eyed views, the target is the least-visited cell nearest the board
named as the monitor shows it, and the check reads the scan's veto."""

from __future__ import annotations

import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from orbiter_native.calibflow import STILL_PX, CalibrationFlow
from orbiter_native.cvcore import BoardSpec, build_board
from orbiter_native.detect import BoardHit
from orbiter_native.guide import (
    GRID,
    K_CELLS,
    K_VIEWS,
    PAIR_SCALE_SPREAD,
    PAIR_VIEWS,
    PLANE_FRAMES,
    STAGES,
    VETO_OK_PX,
    Guide,
    move_words,
    pair_scale_spread,
    place_name,
)
from orbiter_native.intrinsics import MIN_TILT_SPREAD, EyeView, PairSample, ViewDescriptor
from orbiter_native.orient import Orientation

WH = (1280, 720)
SPEC = BoardSpec(8, 8, 36.0, 26.64, cv2.aruco.DICT_5X5_100)


@pytest.fixture(scope="module")
def board():
    return build_board(SPEC)


def _flow(board) -> CalibrationFlow:
    flow = CalibrationFlow()
    flow.set_board(SPEC, board)
    return flow


def _view(cx: float, cy: float, tilt: float = 0.0, scale: float = 0.3) -> EyeView:
    corners = np.zeros((4, 1, 2), np.float32)
    ids = np.arange(4, dtype=np.int32).reshape(-1, 1)
    return EyeView(corners, ids, WH, ViewDescriptor(cx, cy, scale, tilt, -tilt))


def _lens_done(flow: CalibrationFlow, side: str) -> None:
    """Enough one-eyed views, across the frame and at spread tilts, plus a solve."""
    i = 0
    for gy in range(GRID):
        for gx in range(GRID):
            if i >= K_VIEWS:
                break
            v = _view((gx + 0.5) / GRID, (gy + 0.5) / GRID, tilt=20.0 * (i % 3 - 1))
            flow.samples.add(PairSample(**{side: v}))
            i += 1
    flow.results[f"intrinsics:{side}"] = SimpleNamespace(
        intrinsics=SimpleNamespace(K=np.diag([900.0, 900.0, 1.0])), sigma_f_px=2.25, n_views=i)


def _pair_done(flow: CalibrationFlow) -> None:
    for i in range(PAIR_VIEWS):
        flow.samples.add(PairSample(left=_view(0.3 + i * 0.02, 0.5, tilt=i),
                                    right=_view(0.6 + i * 0.02, 0.5, tilt=i)))
    flow.results["stereo"] = SimpleNamespace(rms_px=0.6, n_views=PAIR_VIEWS, baseline_mm=160.0)


def _plane_done(flow: CalibrationFlow) -> None:
    flow.plane._frames = PLANE_FRAMES
    flow.results["plane"] = SimpleNamespace(rms_mm=0.3, n_frames=PLANE_FRAMES, n_points=900, d=70.0)


def _readout_done(flow: CalibrationFlow) -> None:
    for side in ("left", "right"):
        flow.results[f"readout:{side}"] = SimpleNamespace(views=20, sigma_s=0.0002)


def _frame(side: str, t: float, cx: float | None = 0.5, cy: float = 0.5, shift: float = 0.0):
    """One eye's result: the board's corners around (cx, cy) of the frame,
    shifted by `shift` px against the previous frame, or no board at all."""
    if cx is None:
        return SimpleNamespace(side=side, board=None, laser=None, capture_mono=t, wh=WH,
                               descriptor=None)
    base = np.array([[-40, -40], [40, -40], [40, 40], [-40, 40]], np.float32)
    corners = (base + [cx * WH[0] + shift, cy * WH[1]]).reshape(-1, 1, 2)
    ids = np.arange(4, dtype=np.int32).reshape(-1, 1)
    hit = BoardHit(corners=corners, ids=ids)
    return SimpleNamespace(side=side, board=hit, laser=None, capture_mono=t, wh=WH,
                           descriptor=ViewDescriptor(cx, cy, 0.3, 0.0, 0.0))


# ── stages ───────────────────────────────────────────────────────────────

def test_the_stage_is_the_first_one_not_done(board) -> None:
    flow, guide = _flow(board), Guide()
    p = guide.update(flow)
    assert (p.stage, p.step, p.solo, flow.solo) == ("left", 1, "left", None)   # the window copies it
    assert "LEFT CAMERA" in p.title and p.eye == "left"
    _lens_done(flow, "left")
    p = guide.update(flow)
    assert (p.stage, p.solo) == ("right", "right")
    _lens_done(flow, "right")
    p = guide.update(flow)
    assert (p.stage, p.solo, p.eye) == ("pair", None, "both")
    _pair_done(flow)
    assert guide.update(flow).stage == "plane"
    _plane_done(flow)
    assert guide.update(flow).stage == "readout"
    _readout_done(flow)
    assert guide.update(flow).stage == "check"


def test_a_lens_stage_needs_coverage_and_tilt_not_just_a_count(board) -> None:
    flow, guide = _flow(board), Guide()
    for i in range(K_VIEWS):                       # all at the centre, no tilt
        flow.samples.add(PairSample(left=_view(0.5, 0.5)))
    flow.results["intrinsics:left"] = SimpleNamespace(
        intrinsics=SimpleNamespace(K=np.diag([900.0, 900.0, 1.0])), sigma_f_px=2.0, n_views=K_VIEWS)
    p = guide.update(flow)
    assert p.stage == "left" and not p.done
    assert f"views {K_VIEWS}/{K_VIEWS}" in p.detail and "cells 1/" in p.detail
    assert "f ±0.22 %" in p.detail and "not saved" in p.detail


def test_the_check_reads_the_switches_and_the_scans_veto(board) -> None:
    flow, guide = _flow(board), Guide()
    for fill in (lambda: _lens_done(flow, "left"), lambda: _lens_done(flow, "right"),
                 lambda: _pair_done(flow), lambda: _plane_done(flow), lambda: _readout_done(flow)):
        fill()
    p = guide.update(flow, laser_on=False)
    assert p.stage == "check" and p.tone == "stop" and "laser line" in p.action
    p = guide.update(flow, laser_on=True, scanning=False)
    assert p.tone == "stop" and "scanning" in p.action
    p = guide.update(flow, laser_on=True, scanning=True, veto_px=None)
    assert p.tone == "adjust" and "WAITING" in p.action
    p = guide.update(flow, laser_on=True, scanning=True, veto_px=float("nan"))
    assert "RIGHT CAMERA DOES NOT SEE" in p.action
    p = guide.update(flow, laser_on=True, scanning=True, veto_px=550.0, kept=0)
    assert p.tone == "stop" and "550 px" in p.action and "PAIR" in p.action
    p = guide.update(flow, laser_on=True, scanning=True, veto_px=VETO_OK_PX / 3, kept=400)
    assert p.done and p.tone == "done" and p.action.startswith("READY")


def test_without_the_switch_every_stage_but_the_check_asks_for_it(board) -> None:
    flow, guide = _flow(board), Guide()
    p = guide.update(flow, auto=False)
    assert p.tone == "stop" and "calibrate continuously" in p.action
    assert (p.stage, p.eye, p.solo) == ("left", "left", "left")
    guide.index = STAGES.index("check")
    guide.pinned = True
    assert "calibrate continuously" not in guide.update(flow, auto=False).action
    # At the ceiling the switch would not help: Clear is the answer, switch or no switch.
    from orbiter_native.calibflow import MAX_VIEWS
    full = _flow(board)
    for i in range(MAX_VIEWS):
        full.samples.add(PairSample(left=_view(0.5, 0.5, tilt=float(i % 7))))
    full.offer(_frame("left", 1.0), auto=False)
    assert "Clear" in Guide().update(full, auto=False).action


def test_at_the_view_ceiling_the_answer_is_clear_not_next(board) -> None:
    from orbiter_native.calibflow import MAX_VIEWS
    flow, guide = _flow(board), Guide()
    for i in range(MAX_VIEWS):
        flow.samples.add(PairSample(left=_view(0.5, 0.5, tilt=float(i % 7))))
    flow.offer(_frame("left", 1.0), auto=False)
    p = guide.update(flow, {"left": (*WH, Orientation())})
    assert p.tone == "stop" and "Clear" in p.action and str(MAX_VIEWS) in p.action


def test_an_offline_eye_is_named_before_any_board_instruction(board) -> None:
    flow, guide = _flow(board), Guide()
    p = guide.update(flow, offline={"left"})
    assert p.tone == "stop" and p.action.startswith("LEFT CAMERA OFFLINE") and p.eye == "left"
    assert guide.update(flow, offline={"right"}).action.startswith("SHOW THE BOARD")
    _lens_done(flow, "left")
    _lens_done(flow, "right")
    p = guide.update(flow, offline={"left", "right"})
    assert p.stage == "pair" and p.action.startswith("LEFT AND RIGHT CAMERA OFFLINE")
    _pair_done(flow)
    assert guide.update(flow, laser_on=True, offline={"right"}).stage == "plane"
    assert "OFFLINE" not in guide.update(flow, laser_on=True, offline={"right"}).action
    assert "OFFLINE" in guide.update(flow, laser_on=True, offline={"left"}).action


def test_restart_after_rig_moved_lands_on_the_pair_even_when_pinned(board) -> None:
    flow, guide = _flow(board), Guide()
    _lens_done(flow, "left")
    _lens_done(flow, "right")
    _pair_done(flow)
    _plane_done(flow)
    guide.update(flow)
    guide.next()                                                # pinned at the check
    assert guide.stage == "check" and guide.pinned
    flow.rig_moved()
    assert guide.update(flow).stage == "check"                  # pinned: never back by itself
    guide.restart()                                             # what the Rig moved button does
    assert guide.update(flow).stage == "pair"


def test_a_refused_lens_asks_for_stiller_views(board) -> None:
    flow, guide = _flow(board), Guide()
    _lens_done(flow, "left")
    del flow.results["intrinsics:left"]
    flow.reasons["intrinsics:left"] = "reprojection RMS 1.52 px exceeds 1.5 px"
    for t in (1.0, 1.033):                     # held still on a view already taken
        flow.offer(_frame("left", t, 1.5 / GRID, 0.5 / GRID), auto=False)
    p = guide.update(flow, {"left": (*WH, Orientation())})
    assert p.action.startswith("LENS REFUSED: REPROJECTION RMS 1.52 PX") and "STILLER" in p.action
    assert "refused: reprojection RMS 1.52 px" in p.detail


def test_the_pair_stage_asks_for_distance_variety(board) -> None:
    flow, guide = _flow(board), Guide()
    _lens_done(flow, "left")
    _lens_done(flow, "right")
    for i in range(8):                                          # eight pairs, one distance
        flow.samples.add(PairSample(left=_view(0.3 + i * 0.02, 0.5, tilt=i),
                                    right=_view(0.6 + i * 0.02, 0.5, tilt=i)))
    assert pair_scale_spread(flow.samples.paired()) < PAIR_SCALE_SPREAD
    for t in (1.0, 1.033):                                      # still, seen by both, nothing new
        flow.offer(_frame("left", t, 0.3, 0.5), auto=False)
        flow.offer(_frame("right", t + 0.002, 0.6, 0.5), auto=False)
    p = guide.update(flow)
    assert p.stage == "pair" and p.action.startswith("CHANGE THE DISTANCE")
    for i in range(8):                                          # and eight more, near and far
        flow.samples.add(PairSample(left=_view(0.3 + i * 0.02, 0.4, tilt=i, scale=0.2 + 0.02 * i),
                                    right=_view(0.6 + i * 0.02, 0.4, tilt=i, scale=0.2 + 0.02 * i)))
    assert pair_scale_spread(flow.samples.paired()) >= PAIR_SCALE_SPREAD
    assert guide.update(flow).action.startswith("NEW PLACE, TILT OR DISTANCE")


def test_without_a_board_spec_nothing_else_is_asked(board) -> None:
    flow, guide = CalibrationFlow(), Guide()
    p = guide.update(flow)
    assert p.tone == "stop" and "BOARD SPEC" in p.action and flow.solo is None


# ── the lens stage: cues and the target ──────────────────────────────────

def test_the_lens_cues_follow_the_gate(board) -> None:
    flow, guide = _flow(board), Guide()
    frames = {"left": (*WH, Orientation())}
    flow.offer(_frame("left", 1.0, cx=None))
    p = guide.update(flow, frames)
    assert p.action == "SHOW THE BOARD TO THE LEFT CAMERA" and p.tone == "adjust"
    flow.offer(_frame("left", 1.033, 0.5, 0.5))
    assert guide.update(flow, frames).action == "HOLD STILL…"          # settling
    flow.offer(_frame("left", 1.066, 0.5, 0.5, shift=STILL_PX * 6))
    p = guide.update(flow, frames)
    assert p.action.startswith("HOLD STILL (") and p.tone == "adjust"
    flow.offer(_frame("left", 1.099, 0.5, 0.5, shift=STILL_PX * 6), auto=False)
    p = guide.update(flow, frames)
    assert p.action.startswith("GOOD — NEXT: ") and p.tone == "go"     # still and new
    assert p.target is not None and p.eye == "left"


def test_a_still_board_in_one_eye_becomes_a_view_in_its_own_stage(board) -> None:
    flow, guide = _flow(board), Guide()
    frames = {"left": (*WH, Orientation())}
    flow.solo = guide.update(flow, frames).solo                         # as the window does
    for i in range(3):
        flow.offer(_frame("left", 1.0 + i * 0.033, 0.5, 0.5))
    assert len(flow.samples.views("left")) == 1 and len(flow.samples.views("right")) == 0
    assert flow.samples.samples[0].right is None
    # The board still where it was: the next place is the point, not this one.
    p = guide.update(flow, frames)
    assert p.action.startswith("MOVE THE BOARD A LITTLE ")
    x0, y0, x1, y1 = p.target
    assert not (x0 <= WH[0] / 2 < x1 and y0 <= WH[1] / 2 < y1)
    # Without the guide's stage the pair rule holds: one eye alone is not a view.
    other = _flow(board)
    for i in range(3):
        other.offer(_frame("left", 1.0 + i * 0.033, 0.5, 0.5))
    assert len(other.samples) == 0
    assert other.gate()[0] == "board" and "needs both" in other.gate_report()


def test_a_jittery_other_eye_does_not_starve_the_solo_eye(board) -> None:
    """Both eyes see the board; the right one shakes. The left eye's own
    stage still takes its view, one-eyed. When both are still, the pair wins."""
    flow = _flow(board)
    flow.solo = "left"
    for i in range(3):
        t = 1.0 + i * 0.033
        flow.offer(_frame("right", t, 0.5, 0.5, shift=i * STILL_PX * 5))
        flow.offer(_frame("left", t + 0.002, 0.5, 0.5))
    assert len(flow.samples) == 1
    one = flow.samples.samples[0]
    assert one.left is not None and one.right is None
    still = _flow(board)
    still.solo = "left"
    for i in range(3):
        t = 1.0 + i * 0.033
        still.offer(_frame("right", t, 0.5, 0.5))
        still.offer(_frame("left", t + 0.002, 0.5, 0.5))
    assert len(still.samples) == 1 and still.samples.samples[0].both
    # Its gate speaks for the solo eye alone either way.
    assert flow.gate()[0] in ("dup", "wait")


def test_a_partner_whose_stream_stopped_does_not_stall_the_solo_eye(board) -> None:
    """The right eye's last frame held a still board — then nothing more.
    The left eye's stage must go on one-eyed after a couple of frames."""
    flow = _flow(board)
    flow.solo = "left"
    for t in (1.0, 1.033):                                       # a still right board, then silence
        flow.offer(_frame("right", t, 0.5, 0.5))
    n = 0
    for i in range(12):                                          # six places, two frames each
        t = 1.2 + i * 0.033
        flow.offer(_frame("left", t, 0.15 + 0.12 * (i // 2), 0.5))
        n = len(flow.samples)
    assert n == 6
    assert all(s.right is None for s in flow.samples.samples)
    assert flow.gate()[0] in ("dup", "wait")                    # and the gate says so, not "ok"


def test_the_solo_stage_never_banks_duplicates_of_its_own_eye(board) -> None:
    """Left board motionless, right eye visiting new places: pairs would be
    new for the right eye, but the stage is the left lens."""
    flow = _flow(board)
    flow.solo = "left"
    for i in range(8):
        t = 1.0 + i * 0.033
        flow.offer(_frame("right", t, 0.2 + 0.1 * (i // 2), 0.5))
        flow.offer(_frame("left", t + 0.002, 0.5, 0.5))
    assert len(flow.samples.views("left")) == 1
    assert flow.samples.novelty("left", ViewDescriptor(0.5, 0.5, 0.3, 0.0, 0.0)) == 0.0
    assert flow.gate()[0] == "dup" and "left eye" in flow.gate_report()


def test_the_solo_gate_waits_with_a_live_partner_instead_of_saying_ok(board) -> None:
    flow = _flow(board)
    flow.solo = "left"
    for t in (1.0, 1.033, 1.066):
        flow.offer(_frame("right", t, 0.5, 0.5), auto=False)
        if t < 1.05:
            flow.offer(_frame("left", t + 0.002, 0.5, 0.5), auto=False)
    assert flow.gate()[0] == "ok"                                # pair formable, new for the left
    flow.capture()                                               # history cleared, partner fresh
    flow.offer(_frame("left", 1.068, 0.3, 0.5), auto=False)      # off to a new place…
    assert flow.gate()[0] == "moving"
    flow.offer(_frame("left", 1.101, 0.3, 0.5), auto=False)      # …and still there, 35 ms after the right
    assert flow.gate()[0] == "wait"                              # the pair's frame is coming
    flow.offer(_frame("left", 1.3, 0.3, 0.5), auto=False)        # the right eye went quiet
    assert flow.gate()[0] == "ok"                                # so the view is this eye's alone


def test_the_solo_gate_only_asks_its_own_eye(board) -> None:
    flow = _flow(board)
    flow.solo = "right"
    assert flow.gate() == ("board", "no board in the right eye")
    flow.offer(_frame("right", 1.0), auto=False)
    assert flow.gate()[0] == "settling"
    flow.offer(_frame("right", 1.033, shift=3.0), auto=False)
    assert flow.gate()[0] == "moving" and "right" in flow.gate_report()
    flow.offer(_frame("right", 1.066, shift=3.0), auto=False)
    assert flow.gate() == ("ok", "taking a view")
    flow.capture()
    flow.offer(_frame("right", 1.099, shift=3.0), auto=False)
    assert flow.gate()[0] == "dup"
    assert flow.where("right") == pytest.approx((0.5, 0.5)) and flow.where("left") is None


def test_the_target_is_the_least_visited_cell_nearest_the_board(board) -> None:
    flow, guide = _flow(board), Guide()
    for t in (1.0, 1.033):                                              # still, top-left, seen
        flow.offer(_frame("left", t, 0.1, 0.1), auto=False)
    cell = WH[0] / GRID, WH[1] / GRID
    # Every cell visited once but two: one next to the board, one far away.
    for gy in range(GRID):
        for gx in range(GRID):
            if (gx, gy) in ((1, 0), (5, 5)):
                continue
            flow.samples.add(PairSample(left=_view((gx + 0.5) / GRID, (gy + 0.5) / GRID,
                                                   tilt=20.0 * ((gx + gy) % 3 - 1))))
    p = guide.update(flow, {"left": (*WH, Orientation())})
    assert p.target == pytest.approx((cell[0], 0.0, 2 * cell[0], cell[1]))
    assert p.action == "GOOD — NEXT: A LITTLE RIGHT"           # new here: a view is taken
    # Shown turned a quarter clockwise, the move is named as the monitor has it.
    p = guide.update(flow, {"left": (*WH, Orientation(quarter_turns_cw=1))})
    assert p.target == pytest.approx((cell[0], 0.0, 2 * cell[0], cell[1]))
    assert p.action == "GOOD — NEXT: A LITTLE DOWN"
    # No frame yet: a cell is chosen but there is nothing to draw it on.
    p = guide.update(flow, {})
    assert p.target is None and "NEXT CELL" in p.action
    # The far corner, once the near cells are visited, is far.
    flow.samples.add(PairSample(left=_view(1.5 / GRID, 0.5 / GRID)))
    p = guide.update(flow, {"left": (*WH, Orientation())})
    assert p.action == "GOOD — NEXT: FAR DOWN-RIGHT"


@pytest.mark.parametrize("o, name", [
    (Orientation(), "TOP-LEFT"),
    (Orientation(quarter_turns_cw=1), "TOP-RIGHT"),
    (Orientation(quarter_turns_cw=2), "BOTTOM-RIGHT"),
    (Orientation(quarter_turns_cw=3), "BOTTOM-LEFT"),
    (Orientation(flip_h=True), "TOP-RIGHT"),
    (Orientation(flip_v=True), "BOTTOM-LEFT"),
])
def test_place_names_read_the_monitor_not_the_sensor(o, name) -> None:
    assert place_name(10.0, 10.0, *WH, o) == name
    assert place_name(WH[0] / 2, WH[1] / 2, *WH, o) == "CENTRE"
    # And a move from the centre to that corner is named the same way.
    words = move_words((WH[0] / 2, WH[1] / 2), (10.0, 10.0), *WH, o)
    assert words == "FAR " + name.replace("TOP", "UP").replace("BOTTOM", "DOWN")


# ── navigation ───────────────────────────────────────────────────────────

def test_back_and_next_pin_the_stage_and_it_only_moves_forward(board) -> None:
    flow, guide = _flow(board), Guide()
    guide.next()
    assert guide.update(flow).stage == "right" and guide.pinned     # left not done: skipped anyway
    guide.next()
    assert guide.update(flow).stage == "pair"
    guide.back()
    assert guide.update(flow).stage == "right"
    _lens_done(flow, "right")
    assert guide.update(flow).stage == "pair"                       # done: forward on its own
    _lens_done(flow, "left")
    assert guide.update(flow).stage == "pair"                       # never back by itself
    guide.restart()
    assert guide.update(flow).stage == "pair" and not guide.pinned  # first not done
    for _ in range(len(STAGES) + 2):
        guide.next()
    assert guide.stage == "check"
    for _ in range(len(STAGES) + 2):
        guide.back()
    assert guide.stage == "left"


def test_rig_moved_takes_the_guide_back_to_the_pair(board) -> None:
    flow, guide = _flow(board), Guide()
    _lens_done(flow, "left")
    _lens_done(flow, "right")
    _pair_done(flow)
    _plane_done(flow)
    assert guide.update(flow).stage == "readout"
    flow.rig_moved()
    p = guide.update(flow)
    assert p.stage == "pair" and "pairs 0/" in p.detail


def test_the_pair_and_sheet_prompts_name_what_is_missing(board) -> None:
    flow, guide = _flow(board), Guide()
    _lens_done(flow, "left")
    _lens_done(flow, "right")
    flow.offer(_frame("left", 1.0), auto=False)
    p = guide.update(flow)
    assert p.stage == "pair" and p.action == "THE RIGHT CAMERA MUST SEE IT TOO"
    _pair_done(flow)
    p = guide.update(flow, laser_on=False)
    assert p.stage == "plane" and p.tone == "stop" and "laser line" in p.action
    flow.offer(_frame("left", 1.033, cx=None), auto=False)
    p = guide.update(flow, laser_on=True)
    assert p.action == "SHOW THE BOARD TO THE LEFT CAMERA"
    flow.offer(_frame("left", 1.066), auto=False)
    assert guide.update(flow, laser_on=True).action == "LASER STRIPE STRAIGHT ACROSS THE BOARD"
    res = _frame("left", 1.099)
    res.laser = SimpleNamespace(ok=True)
    flow.offer(res, auto=False)
    p = guide.update(flow, laser_on=True)
    assert p.action.startswith("GOOD — HOLD") and p.tone == "go" and p.eye == "left"
    assert f"poses 0/{PLANE_FRAMES}" in p.detail


# ── the widgets ──────────────────────────────────────────────────────────

def _app():
    # No platform override: the GL tests in this suite share the process and
    # need the real window system; a banner is plain widgets either way.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_banner_shows_the_prompt_in_its_tone(board) -> None:
    _app()
    from orbiter_native.guidepanel import GuideBanner

    flow, guide, banner = _flow(board), Guide(), GuideBanner()
    p = guide.update(flow)
    banner.set_prompt(p)
    assert banner.action.text() == p.action and banner.title.text() == p.title
    assert "<u>1 LEFT</u>" in banner.steps.text()
    assert "#4a3406" in banner.styleSheet()                          # amber: adjust
    from orbiter_native.guide import Prompt
    banner.set_prompt(Prompt("check", 6, "STEP 6/6", "READY", "", "done"))
    assert "#0e2c4a" in banner.styleSheet() and "<u>6 CHECK</u>" in banner.steps.text()
    fired = []
    banner.toggled.connect(fired.append)
    banner.enabled.setChecked(False)
    assert fired == [False] and not banner.action.isVisibleTo(banner)
    assert "off" in banner.title.text()
