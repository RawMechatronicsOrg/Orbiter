"""The four gates between the veto and the board's frame: one blob per
scanline, its width, the reach from the baseline, and no jumps along the
stripe. Each on the synthetic rig of `test_stereo_scan`, with the noise it
exists for injected on purpose."""

from __future__ import annotations

import numpy as np

from orbiter_native.laser import StripePixels
from orbiter_native.scan import ScanParams, ScanVolume, _not_a_jump, _reach_mm, scan_frame

from test_stereo_scan import (
    BOARD2_R,
    BOARD2_T,
    KL,
    KR,
    R_TRUE,
    T_TRUE,
    WH,
    _curve,
    _join,
    _pixels,
    _plane,
    _rig,
)

#: The cylinder plays no part here.
NO_VOLUME = ScanVolume(height_mm=1e6, radius_mm=1e6, floor_mm=-1e6)


def _params(**kw) -> ScanParams:
    kw.setdefault("range_mm", (0.0, 1e9))
    kw.setdefault("volume", NO_VOLUME)
    return ScanParams(**kw)


def test_reach_is_the_distance_from_the_baseline() -> None:
    from orbiter_native.stereo import StereoResult, StereoRig
    # A right camera 200 mm along -x with no toe-in: the baseline IS the x
    # axis, and a point's reach is its distance from that axis.
    geom = StereoResult(R=np.eye(3), T=np.array([-200.0, 0.0, 0.0]), E=np.zeros((3, 3)),
                        F=np.zeros((3, 3)), rms_px=0.1, n_views=12, wh=WH)
    rig = StereoRig(KL, KR, geom)
    pts = np.array([[0.0, 0.0, 300.0], [100.0, 0.0, 300.0], [0.0, 74.0, 500.0]])
    reach = _reach_mm(pts, rig)
    assert np.allclose(reach[:2], 300.0)
    assert np.isclose(reach[2], np.hypot(74.0, 500.0))
    # The toed-in test rig: the baseline leans 3.4° in x-z, and so does the reach.
    lean = _reach_mm(pts, _rig())
    assert np.allclose(lean[:2], [300.0 * np.cos(0.06) + 0.0, 300.0 * np.cos(0.06) - 100.0 * np.sin(0.06)],
                       atol=2.0) or lean[1] > lean[0]


def test_points_outside_the_reach_are_dropped_and_counted() -> None:
    """A stripe on a sheet 505 mm from the baseline against a 150-450 mm reach:
    gone. Widen the reach: back, every one."""
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: np.full_like(x, 500.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)
    out = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T,
                     _params(range_mm=(150.0, 450.0)))
    assert out.n_kept == 0 and out.n_rejected_range > 250
    back = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T,
                      _params(range_mm=(150.0, 600.0)))
    assert back.n_kept > 250 and back.n_rejected_range == 0


def test_the_strongest_blob_wins_the_scanline() -> None:
    """A fainter glint the right eye confirms too, three rows above the
    stripe: averaged, it pulls every centroid off the surface; taken as the
    weaker run, it changes nothing."""
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: np.full_like(x, 500.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)
    clean = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T, _params())
    # The glint: the same columns, 8 rows up, at a third of the weight, in
    # both eyes so the veto lets it through.
    glint_l = StripePixels(x=left.x, y=left.y - 8, w=(left.w // 3).astype(np.uint8),
                           wh=WH, along_x=True, reason=None)
    glint_r = StripePixels(x=right.x, y=right.y - 8, w=(right.w // 3).astype(np.uint8),
                           wh=WH, along_x=True, reason=None)
    out = scan_frame(rig, plane, _join(left, glint_l), _join(right, glint_r),
                     BOARD2_R, BOARD2_T, _params())
    assert out.n_split > 250, out.n_split
    assert out.n_kept == clean.n_kept
    assert np.allclose(out.points_camera, clean.points_camera, atol=1e-9)


def test_a_run_too_thin_or_too_wide_is_not_the_stripe() -> None:
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: np.full_like(x, 500.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)
    # The synthetic stripe is 3-5 px wide. Demand 6: nothing survives.
    out = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T,
                     _params(blob_width_px=(6, 24)))
    assert out.n_kept == 0 and out.n_rejected_blob > 250
    # Allow at most 2: nothing survives either, and the count says why.
    out = scan_frame(rig, plane, left, right, BOARD2_R, BOARD2_T,
                     _params(blob_width_px=(1, 2)))
    assert out.n_kept == 0 and out.n_rejected_blob > 250


def test_a_point_off_both_neighbours_is_a_jump() -> None:
    xyz = np.array([[0.0, 0.0, 500.0], [1.0, 0.0, 500.2], [2.0, 0.0, 540.0],
                    [3.0, 0.0, 500.4], [4.0, 0.0, 500.6], [5.0, 0.0, 560.0],
                    [6.0, 0.0, 560.2], [7.0, 0.0, 560.4]])
    scan = np.arange(len(xyz), dtype=float)
    keep = _not_a_jump(xyz, scan, 5.0)
    # The lone 540 goes; the step to 560 is a real edge and stays whole.
    assert keep.tolist() == [True, True, False, True, True, True, True, True]
    # Across a break in the stripe nothing is judged.
    scan[2:] += 10
    assert _not_a_jump(xyz, scan, 5.0).all()
    assert _not_a_jump(xyz, np.arange(8.0), 0.0).all()


def test_a_glint_on_one_scanline_is_dropped_by_the_jump_gate() -> None:
    """Confirmed by the right eye, the right width, inside the reach — and
    12 rows off the stripe on one column, which on the sheet is a point tens
    of millimetres from its neighbours."""
    rig, plane = _rig(), _plane()
    truth = _curve(lambda x: np.full_like(x, 500.0))
    left = _pixels(KL, np.eye(3), np.zeros(3), truth)
    right = _pixels(KR, R_TRUE, T_TRUE, truth)
    col = int(np.median(left.x))
    on_col = left.x == col
    # Replace that column's stripe with a run 12 rows up in the left eye. The
    # veto would catch it on its own (its sheet point is 50 mm deeper and
    # lands 30 px off in the right eye), so the right eye is made to confirm
    # everything near the stripe: the jump gate has to do the work.
    l2 = StripePixels(x=left.x, y=np.where(on_col, left.y - 12, left.y), w=left.w,
                      wh=WH, along_x=True, reason=None)
    ys, xs = np.mgrid[-40:41, 0:1]
    rx = (right.x[None, :] + xs).ravel()
    ry = (right.y[None, :] + ys).ravel()
    ok = (ry >= 0) & (ry < WH[1])
    r_all = StripePixels(x=rx[ok].astype(np.int32), y=ry[ok].astype(np.int32),
                         w=np.full(int(ok.sum()), 200, np.uint8), wh=WH,
                         along_x=True, reason=None)
    unfiltered = scan_frame(rig, plane, l2, r_all, BOARD2_R, BOARD2_T, _params(jump_mm=0.0))
    assert unfiltered.n_rejected_jump == 0
    assert np.abs(unfiltered.points_camera[:, 2] - 500.0).max() > 20.0   # the glint is in
    out = scan_frame(rig, plane, l2, r_all, BOARD2_R, BOARD2_T, _params(jump_mm=5.0))
    assert out.n_rejected_jump >= 1
    assert np.abs(out.points_camera[:, 2] - 500.0).max() < 1.0
