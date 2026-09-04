"""The continuous calibration flow: what each frame becomes, when a cycle
runs, what a cycle solves, and what goes to the server.

Frames are synthetic eye results — a board projected through a known camera,
with the pose, the descriptor and, where wanted, a straight stripe across the
board — so the flow's decisions can be checked without cameras or widgets.
The solvers are swapped for fakes where the decision under test is about the
flow, and real where it is about the numbers (`refit`, `test_rolling`).
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from orbiter_native.calibflow import (
    ASSUMED_STRIPE_PX,
    ASSUMED_Z_MM,
    CYCLE_MIN_S,
    FAST_PX,
    MAX_VIEWS,
    CalibrationFlow,
    Outcome,
    Saved,
    _better,
)
from orbiter_native.cvcore import BoardSpec, Intrinsics, build_board, estimate_pose
from orbiter_native.detect import BoardHit
from orbiter_native.intrinsics import MIN_VIEWS, SolveResult, describe
from orbiter_native.laser import LaserLine
from orbiter_native.laserplane import LaserPlane, PlaneCollector, points_on_board
from orbiter_native.rolling import Readout
from orbiter_native.stereo import StereoResult

WH = (1280, 720)
K = Intrinsics(fx=900.0, fy=905.0, cx=646.0, cy=358.0, dist=(-0.28, 0.09, 0.001, -0.0015, 0.0))
SPEC = BoardSpec(8, 8, 36.0, 26.64, cv2.aruco.DICT_5X5_100)


@pytest.fixture(scope="module")
def board():
    return build_board(SPEC)


def _result(board, side: str, rvec, tvec_m, t: float, shift_px: float = 0.0,
            stripe: bool = False, rng=None):
    """An eye result: the board at (rvec, tvec) seen by K, corners shifted by
    `shift_px` (motion between frames), and optionally the stripe across it."""
    obj = board.getChessboardCorners().astype(np.float32)
    ids = np.arange(len(obj), dtype=np.int32).reshape(-1, 1)
    img, _ = cv2.projectPoints(obj, np.asarray(rvec, float), np.asarray(tvec_m, float),
                               K.K, K.D)
    img = img.reshape(-1, 2) + [shift_px, 0.0]
    inside = (img[:, 0] >= 0) & (img[:, 0] < WH[0]) & (img[:, 1] >= 0) & (img[:, 1] < WH[1])
    corners = img[inside].reshape(-1, 1, 2).astype(np.float32)
    kept = ids[inside]
    pose = estimate_pose(corners, kept, board, K, None)
    hit = BoardHit(corners=corners, ids=kept, R=None if pose is None else pose[0],
                   t=None if pose is None else pose[1])
    laser = LaserLine()
    if stripe and pose is not None:
        # A straight stripe across the board: the sheet y = 74 mm (camera
        # frame) meeting the board plane, sampled and projected.
        R, tt, _ = pose
        n, p0 = R[:, 2], tt
        xs = np.linspace(-100.0, 100.0, 300)
        # Solve the board plane for z given x and y=74: n·(x, 74, z) = n·p0.
        zs = (n @ p0 - n[0] * xs - n[1] * 74.0) / n[2]
        xyz = np.column_stack([xs, np.full_like(xs, 74.0), zs])
        px, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K.K, K.D)
        px = px.reshape(-1, 2).astype(np.float32)
        laser = LaserLine(points=px, inliers=np.ones(len(px), bool), point=px.mean(axis=0),
                          direction=np.array([1.0, 0.0]), rms_px=0.3, reason=None)
    return SimpleNamespace(side=side, board=hit, laser=laser, capture_mono=t, wh=WH,
                           descriptor=describe(corners, kept, board, WH))


def _flow(board) -> CalibrationFlow:
    flow = CalibrationFlow()
    flow.set_board(SPEC, board)
    flow.laser_active = True
    return flow


# ── what a frame becomes ─────────────────────────────────────────────────

def test_a_still_pair_becomes_a_view_and_a_duplicate_does_not(board) -> None:
    flow = _flow(board)
    pose = ((0.3, -0.2, 0.1), (0.0, 0.0, 0.5))
    # First detections establish stillness; the second of each eye is still.
    for t in (1.000, 1.033):
        flow.offer(_result(board, "left", *pose, t))
        note = flow.offer(_result(board, "right", *pose, t))
    assert note == "view 1", note
    assert len(flow.samples) == 1 and flow.samples.paired()
    for t in (1.066, 1.100):
        flow.offer(_result(board, "left", *pose, t))
        note = flow.offer(_result(board, "right", *pose, t))
    assert note is None and len(flow.samples) == 1          # nothing new to see


def test_a_pair_is_judged_by_the_board_it_saw_not_by_the_clock(board) -> None:
    """The cameras free-run and the offset between the eyes walks: a fixed few
    millisecond window takes nothing for tens of seconds at a stretch. A board
    held still is the same board 12 ms later, and a sliding one is not."""
    flow = _flow(board)
    pose = ((0.2, -0.1, 0.05), (0.0, 0.0, 0.5))
    for i in (0, 1):
        flow.offer(_result(board, "left", *pose, 1.000 + 0.033 * i))
        note = flow.offer(_result(board, "right", *pose, 1.012 + 0.033 * i))
    assert note == "view 1", note                     # 12 ms apart, and still

    sliding = _flow(board)
    for i in range(4):
        sliding.offer(_result(board, "left", *pose, 1.000 + 0.033 * i, shift_px=6.0 * i))
        sliding.offer(_result(board, "right", *pose, 1.012 + 0.033 * i, shift_px=6.0 * i))
    # 6 px a frame is 180 px/s: 2 px of board across the same 12 ms gap.
    assert sliding._find_pair() == (None, None)
    assert len(sliding.samples) == 0


def test_a_moving_board_feeds_the_readout_not_the_views(board) -> None:
    flow = _flow(board)
    pose = ((0.3, -0.2, 0.1), (0.0, 0.0, 0.5))
    notes = []
    for i in range(4):
        notes.append(flow.offer(_result(board, "left", *pose, 1.0 + i / 30,
                                        shift_px=i * (FAST_PX + 2))))
    assert flow.motion.count("left") == 3 and len(flow.samples) == 0
    assert notes[-1] == "readout frame 3 (left)"


def test_the_stripe_across_a_still_board_banks_a_plane_frame(board) -> None:
    flow = _flow(board)
    flow.set_known_intrinsics("left", K)
    pose = ((0.3, -0.2, 0.1), (0.0, 0.0, 0.5))
    flow.offer(_result(board, "left", *pose, 1.0, stripe=True), auto=False)
    note = flow.offer(_result(board, "left", *pose, 1.033, stripe=True), auto=False)
    assert note == "plane frame 1" and flow.plane.raw_frames == 1
    # Same pose again: no new constraint, not banked.
    assert flow.offer(_result(board, "left", *pose, 1.066, stripe=True), auto=False) is None
    # Without intrinsics the stripe cannot be placed, so nothing is banked.
    flow2 = _flow(board)
    for t in (1.0, 1.033):
        flow2.offer(_result(board, "left", *pose, t, stripe=True), auto=False)
    assert flow2.plane.frames == 0


def test_views_stop_at_the_ceiling(board) -> None:
    flow = _flow(board)
    rng = np.random.default_rng(1)
    for i in range(MAX_VIEWS + 20):
        pose = ((rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-1, 1)),
                (rng.uniform(-0.1, 0.1), rng.uniform(-0.06, 0.06), rng.uniform(0.35, 0.8)))
        t = 10.0 + i
        for _ in range(2):
            flow.offer(_result(board, "left", *pose, t))
            flow.offer(_result(board, "right", *pose, t))
            t += 0.033
    assert len(flow.samples) <= MAX_VIEWS


# ── the cycle ────────────────────────────────────────────────────────────

def _fake_solvers(intr_rms=0.3, stereo_rms=0.4, plane_rms=0.3, sigma=0.0002,
                  intr_sigma_f=None):
    # A camera is judged by its focal length's uncertainty, not by the
    # reprojection error it fitted, so the fake has to carry one: left unset
    # it is NaN, and every solve would look like news over an unknown.
    sigma_f = intr_rms * 10.0 if intr_sigma_f is None else intr_sigma_f

    def intrinsics(views, board, tilt_spread=None):
        return SolveResult(K, intr_rms, len(views), WH, tilt_spread=tilt_spread,
                           sigma_f_px=sigma_f), None

    def stereo(pairs, board, kl, kr, wh):
        return StereoResult(R=np.eye(3), T=np.array([-144.0, 0.0, 0.0]), E=np.zeros((3, 3)),
                            F=np.zeros((3, 3)), rms_px=stereo_rms, n_views=len(pairs), wh=wh), None

    def readout(views, board, k):
        return Readout(0.0213, *WH, sigma_s=sigma, skew_px=10.0, rms_px=0.2, views=len(views)), None

    return {"intrinsics": intrinsics, "stereo": stereo, "plane": None, "readout": readout}


class _Plane(PlaneCollector):
    """A collector whose refit is canned, so the cycle test is about the flow."""

    def __init__(self, rms: float = 0.3) -> None:
        super().__init__()
        self.rms = rms

    def copy(self):
        c = _Plane(self.rms)
        c._chunks, c._raw, c._frames = list(self._chunks), list(self._raw), self._frames
        return c

    def refit(self, board, k, wh):
        return LaserPlane(np.array([0.0, 1.0, 0.0]), 74.0, self.rms, 1000, self._frames, wh), None


def _fill(flow: CalibrationFlow, board, n_views: int = MIN_VIEWS, motion: int = 25) -> None:
    rng = np.random.default_rng(3)
    t = 100.0
    for i in range(n_views):
        pose = ((rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-1, 1)),
                (rng.uniform(-0.1, 0.1), rng.uniform(-0.06, 0.06), rng.uniform(0.35, 0.8)))
        for _ in range(2):
            flow.offer(_result(board, "left", *pose, t, stripe=True))
            flow.offer(_result(board, "right", *pose, t))
            t += 0.033
        t += 1.0
    flow.plane._frames = max(flow.plane._frames, 5)
    for i in range(motion):
        for side in ("left", "right"):
            flow.offer(_result(board, side, (0.2, 0.1, 0.0), (0.0, 0.0, 0.5), t,
                               shift_px=i * (FAST_PX + 3)))
        t += 0.033


def test_a_cycle_solves_everything_in_order_and_saves_what_is_new(board) -> None:
    flow = _flow(board)
    flow.solvers = _fake_solvers()
    flow.plane = _Plane()
    _fill(flow, board)
    assert flow.due(now=1000.0)
    job = flow.snapshot(now=1000.0)
    assert flow.running and not flow.due(now=1000.0)
    out = flow.run(job)
    assert set(out.results) == {"intrinsics:left", "intrinsics:right", "stereo", "plane",
                                "readout:left", "readout:right"}, out.reasons
    payload = flow.finish(out, now=1001.0)
    assert set(payload) == {"left", "right", "_extrinsics", "_laser_plane"}
    assert set(payload["left"]) == {"intrinsics", "readout"}
    assert flow.known_k["left"] is out.results["intrinsics:left"].intrinsics
    assert not flow.running
    # The same outcome again is not an improvement, and nothing is saved.
    assert flow.finish(out, now=1002.0) is None
    assert all("saved" in line for line in flow.scoreboard())


def test_a_worse_solve_is_kept_off_the_server(board) -> None:
    flow = _flow(board)
    flow.solvers = _fake_solvers(intr_rms=0.3)
    flow.plane = _Plane()
    _fill(flow, board)
    flow.finish(flow.run(flow.snapshot(now=0.0)), now=1.0)
    # The server now holds rms 0.3 from MIN_VIEWS views. A later solve with the
    # same data but a worse residual must not replace it; one with more data
    # and a residual only a little worse may.
    flow.solvers = _fake_solvers(intr_rms=0.5)
    out = flow.run(flow.snapshot(now=10.0))
    payload = flow.finish(out, now=11.0)
    assert payload is None or "intrinsics" not in payload.get("left", {})
    assert "intrinsics:left" in flow.results          # shown, not saved
    assert not any(line.startswith("K L") and "saved" in line for line in flow.scoreboard())


def test_better_judges_more_data_and_lower_residual() -> None:
    assert _better(10, 0.5, None)
    assert _better(10, 0.4, Saved.first(10, 0.5))         # lower residual
    assert _better(20, 0.55, Saved.first(10, 0.5))        # more data, a little worse
    assert not _better(20, 0.7, Saved.first(10, 0.5))     # more data, much worse
    assert not _better(10, 0.5, Saved.first(10, 0.5))     # the same thing
    assert _better(10, 0.5, Saved.first(10, float("nan")))   # unknown on the server


def test_a_thinner_solve_never_replaces_a_richer_one() -> None:
    """Every residual here is measured on the data that was fitted, so it
    falls as views are taken away. On the rig a 196-view calibration was
    replaced by a 13-view one that reported a smaller error, and every richer
    solve after it was refused for not beating that. Data first, then the
    number."""
    rich = Saved.first(196, 1.13)
    assert not _better(13, 0.67, rich)                # fewer views, better number
    assert _better(196, 1.10, rich)                   # as much data, and better
    assert _better(210, 1.20, rich)                   # more data, a little worse
    # A figure the server never recorded does not open the door either: the
    # view count still holds the bar.
    assert not _better(13, 0.67, Saved.first(196, float("nan")))
    assert _better(200, 0.67, Saved.first(196, float("nan")))


def test_the_tolerance_is_spent_once_and_not_again_every_cycle() -> None:
    """More data at a slightly worse residual is worth saving. Measured
    against the residual in hand it would be worth saving again next cycle,
    and again — 15% at a time walks the calibration away from its best. The
    floor is what the tolerance is measured against, so it cannot."""
    saved = Saved.first(10, 0.50)
    assert _better(20, 0.55, saved)
    saved = saved.adopt(20, 0.55)
    assert saved.residual == 0.55 and saved.floor == 0.50
    assert not _better(30, 0.60, saved)               # 0.60 > 0.50 * 1.15, refused
    assert _better(30, 0.54, saved)                   # inside the floor's tolerance
    assert saved.adopt(40, 0.42).floor == 0.42        # a better solve lowers it
    seeded = Saved.first(10, float("nan"))            # the server said no residual
    assert seeded.adopt(20, 0.9).floor == 0.9


def test_new_intrinsics_retire_what_was_solved_through_the_old(board) -> None:
    """The sheet is coordinates in the left camera's frame. Replace the left
    intrinsics and a sheet refitted through the new ones has to be able to
    take the old one's place, even at a worse residual — otherwise what the
    server holds is a camera from one cycle and a sheet from another, and the
    scan's right-eye veto is the first thing that ever notices."""
    flow = _flow(board)
    flow.solvers = _fake_solvers(intr_rms=0.3)
    flow.plane = _Plane(0.3)
    _fill(flow, board)
    flow.finish(flow.run(flow.snapshot(now=0.0)), now=1.0)
    assert "plane" in flow.saved and "stereo" in flow.saved
    assert flow.plane_known.rms_mm == 0.3

    # Better intrinsics, and a sheet refitted through them that is worse than
    # the one banked under the old ones. It still has to land.
    flow.solvers = _fake_solvers(intr_rms=0.1)
    flow.plane = _Plane(0.9)
    _fill(flow, board, n_views=2, motion=0)
    payload = flow.finish(flow.run(flow.snapshot(now=10.0)), now=11.0)
    assert payload is not None and "_laser_plane" in payload
    assert flow.plane_known.rms_mm == 0.9


def test_a_cycle_that_re_solves_the_same_camera_keeps_its_dependants(board) -> None:
    """Refused is not the same as different: once the intrinsics settle, every
    cycle re-solves the camera already in force and that solve is refused as
    no improvement. What the cycle built on it is still built on the camera
    the rig has, and must not be thrown away with it."""
    flow = _flow(board)
    flow.solvers = _fake_solvers(intr_rms=0.3)
    flow.plane = _Plane(0.5)
    _fill(flow, board)
    flow.finish(flow.run(flow.snapshot(now=0.0)), now=1.0)

    flow.plane = _Plane(0.2)                       # a better sheet, same camera
    _fill(flow, board, n_views=2, motion=0)
    payload = flow.finish(flow.run(flow.snapshot(now=10.0)), now=11.0)
    assert payload is not None and "_laser_plane" in payload
    assert flow.plane_known.rms_mm == 0.2


def test_cycles_are_paced_and_need_new_data(board) -> None:
    flow = _flow(board)
    flow.solvers = _fake_solvers()
    flow.plane = _Plane()
    assert not flow.due(now=0.0)                            # nothing to solve
    _fill(flow, board)
    assert flow.due(now=0.0)
    flow.finish(flow.run(flow.snapshot(now=0.0)), now=1.0)
    assert not flow.due(now=1.0 + CYCLE_MIN_S)               # nothing changed since
    pose = ((0.31, -0.2, 0.1), (0.0, 0.0, 0.5))
    for t in (500.0, 500.033):
        flow.offer(_result(board, "left", *pose, t))
    assert flow.samples.views("left")
    assert not flow.due(now=1.0 + CYCLE_MIN_S / 2)           # too soon
    assert flow.due(now=1.0 + CYCLE_MIN_S)                   # new data, time passed
    flow.request()
    assert flow.due(now=-1e6)                                # the operator asked


def test_advice_names_the_weakest_link(board) -> None:
    flow = _flow(board)
    flow.solvers = _fake_solvers()
    flow.plane = _Plane()
    assert "move the board around for L" in flow.advice()
    _fill(flow, board)
    flow.finish(flow.run(flow.snapshot(now=0.0)), now=1.0)
    assert flow.advice().startswith("everything solved")
    flow.results.pop("plane")
    flow.laser_active = False
    assert "laser line" in flow.advice()
    flow.results.pop("stereo")
    assert "pairs" in flow.advice()


def test_reasons_replace_results_that_stopped_solving(board) -> None:
    flow = _flow(board)
    flow.finish(Outcome(results={}, reasons={"stereo": "needs intrinsics for both eyes"}), now=1.0)
    assert "needs intrinsics" in flow.scoreboard()[2]


# ── the plane through better intrinsics ──────────────────────────────────

def test_plane_refit_redoes_the_points_through_the_given_intrinsics(board) -> None:
    """Frames banked through a wrong K are re-placed through the right one:
    the refit lands on the true sheet, the original fit does not."""
    wrong = Intrinsics(fx=960.0, fy=960.0, cx=640.0, cy=360.0, dist=(0.0,) * 5)
    col = PlaneCollector()
    rng = np.random.default_rng(5)
    for i in range(8):
        pose = ((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-1, 1)),
                (rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), rng.uniform(0.4, 0.7)))
        res = _result(board, "left", *pose, float(i), stripe=True)
        if not res.laser.ok:
            continue
        pose_wrong = estimate_pose(res.board.corners, res.board.ids, board, wrong, None)
        col.add_frame(res.laser.inlier_points, wrong, pose_wrong[0], pose_wrong[1],
                      res.laser.rms_px, corners=res.board.corners, ids=res.board.ids, wh=WH)
    assert col.raw_frames >= 5
    banked, _ = col.fit(WH)
    refit, why = col.refit(board, K, WH)
    assert refit is not None, why
    truth = np.array([0.0, 1.0, 0.0])
    assert abs(refit.normal @ truth) > 0.9999 and abs(refit.d - 74.0) < 0.5, (refit.normal, refit.d)
    assert refit.rms_mm < 0.2
    if banked is not None:
        assert abs(banked.d - 74.0) > abs(refit.d - 74.0)


# ── the expected error ───────────────────────────────────────────────────

def test_expected_error_needs_a_camera_and_a_sheet(board) -> None:
    flow = _flow(board)
    assert flow.expected_error(500.0, 0.5) is None
    flow.set_known_intrinsics("left", K)
    assert flow.expected_error(500.0, 0.5) is None
    flow.stored_plane = LaserPlane(np.array([0.0, 1.0, 0.0]), 74.0, 0.36, 5000, 300, WH)
    b = flow.expected_error(500.0, 0.5)
    assert b is not None
    # The centroid's noise through the sheet: sigma_px * Z^2 / (f * d).
    f = 0.5 * (K.fx + K.fy)
    assert np.isclose(b.stripe_mm, 0.5 * 500.0 ** 2 / (f * 74.0))
    assert b.sheet_mm == 0.36
    # Nothing measured about the focal length or the shutter: assumed, and said.
    assert b.scale_mm > 0 and b.shutter_mm > 0
    assert any("focal" in a for a in b.assumed) and any("shutter" in a for a in b.assumed)
    assert np.isclose(b.total_mm, np.sqrt(b.stripe_mm ** 2 + b.sheet_mm ** 2
                                          + b.scale_mm ** 2 + b.shutter_mm ** 2))


def test_expected_error_falls_as_calibration_fills_in(board) -> None:
    flow = _flow(board)
    flow.set_known_intrinsics("left", K, {"views": 40, "rms_px": 0.3, "sigma_f": 0.9})
    flow.stored_plane = LaserPlane(np.array([0.0, 1.0, 0.0]), 74.0, 0.36, 5000, 300, WH)
    before = flow.expected_error(400.0, 0.5)
    assert not any("focal" in a for a in before.assumed)
    assert np.isclose(before.scale_mm, 0.9 / (0.5 * (K.fx + K.fy)) * 150.0)
    flow.finish(Outcome(results={"readout:left": Readout(0.021, *WH, sigma_s=1e-4,
                                                        views=100)}), now=1.0)
    after = flow.expected_error(400.0, 0.5)
    assert after.shutter_mm == 0.0 and after.total_mm < before.total_mm
    assert not after.assumed


def test_the_number_uses_the_sheet_in_force_not_the_newest_solve(board) -> None:
    """A plane solve the flow refuses belongs on the scoreboard, but not in
    the budget: reading the newest solve is what made the error climb while
    the calibration the server holds stood still."""
    flow = _flow(board)
    flow.set_known_intrinsics("left", K, {"views": 40, "rms_px": 0.3, "sigma_f": 0.9})
    good = LaserPlane(np.array([0.0, 1.0, 0.0]), 74.0, 0.30, 5000, 300, WH)
    assert flow.finish(Outcome(results={"plane": good}), now=1.0) is not None
    assert flow.plane_known is good
    before = flow.expected_error(400.0, 0.5)

    worse = LaserPlane(np.array([0.0, 1.0, 0.0]), 74.0, 0.90, 5000, 310, WH)
    assert flow.finish(Outcome(results={"plane": worse}), now=2.0) is None
    assert flow.results["plane"] is worse                   # shown, marked unsaved
    assert not any(line.startswith("laser") and "saved" in line
                   for line in flow.scoreboard())
    assert flow.plane_known is good                         # but not in force
    assert flow.expected_error(400.0, 0.5).total_mm == before.total_mm


def test_expected_error_assumes_a_distance_and_a_stripe_when_not_told(board) -> None:
    flow = _flow(board)
    flow.set_known_intrinsics("left", K)
    flow.stored_plane = LaserPlane(np.array([0.0, 1.0, 0.0]), 74.0, 0.36, 5000, 300, WH)
    b = flow.expected_error(None, None)
    assert b.z_mm == ASSUMED_Z_MM and b.stripe_px == ASSUMED_STRIPE_PX
    assert any("mm away" in a for a in b.assumed) and any("stripe" in a for a in b.assumed)
    nearer = flow.expected_error(250.0, ASSUMED_STRIPE_PX)
    assert nearer.stripe_mm < b.stripe_mm / 3.5          # quadratic in distance
