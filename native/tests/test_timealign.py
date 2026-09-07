"""The right eye brought to the left's instant: corners by id, the stripe
per scanline, and the scan worker handing over the frame to interpolate
against."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from orbiter_native.laser import StripePixels
from orbiter_native.timealign import MAX_BRACKET_S, align_right, interpolate_corners, shift_stripe


def _corners(ids, xy):
    return (np.asarray(xy, np.float32).reshape(-1, 1, 2),
            np.asarray(ids, np.int32).reshape(-1, 1))


def _stripe(cols, ys, wh=(64, 48), along_x=True):
    x = np.array(cols, np.int32)
    y = np.array(ys, np.int32)
    return StripePixels(x=x, y=y, w=np.full(len(x), 200, np.uint8), wh=wh,
                        along_x=along_x, reason=None)


def test_corners_interpolate_per_id_in_time() -> None:
    c0, i0 = _corners([1, 2, 3, 4, 5], [[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]])
    c1, i1 = _corners([5, 4, 3, 2, 9], [[9, 9], [14, 14], [4, 14], [14, 4], [0, 0]])
    corners, ids = interpolate_corners(c0, i0, 1.0, c1, i1, 1.04, 1.01)
    assert ids.ravel().tolist() == [2, 3, 4, 5]                 # ids 1 and 9: one frame only
    assert corners.shape == (4, 1, 2) and corners.dtype == np.float32
    assert np.allclose(corners.reshape(-1, 2)[0], [11, 1])      # id 2, a quarter of the way
    assert interpolate_corners(c0, i0, 1.0, c1, i1, 1.04, 1.05) is None       # outside
    assert interpolate_corners(c0, i0, 1.0, c1, i1, 1.0 + MAX_BRACKET_S + 0.01, 1.02) is None
    c2, i2 = _corners([7, 8, 9], [[0, 0], [1, 1], [2, 2]])
    assert interpolate_corners(c0, i0, 1.0, c2, i2, 1.02, 1.01) is None       # too few in common


def test_stripe_pixels_move_per_scanline_toward_the_instant() -> None:
    base = _stripe(list(range(11)), [20] * 11)                  # column 10 has no counterpart
    other = _stripe(list(range(10)), [24] * 10)
    moved = shift_stripe(base, 1.0, other, 1.033, 1.011)
    assert moved.y.tolist() == [21] * 10 + [20]                 # 4 px × 1/3, rounded; lone stays
    assert moved.x.tolist() == base.x.tolist() and moved.w.tolist() == base.w.tolist()
    # Clipped at the frame edge, never lost.
    clipped = shift_stripe(_stripe(range(10), [45] * 10), 1.0, _stripe(range(10), [60] * 10),
                           1.033, 1.033)
    assert clipped.y.tolist() == [47] * 10
    # No opinion when the frames disagree about the stripe's direction, or at the base instant.
    assert shift_stripe(base, 1.0, _stripe(range(10), [24] * 10, along_x=False), 1.033, 1.01) is base
    assert shift_stripe(base, 1.0, other, 1.033, 1.0) is base


def _inp(t, stripe=None, corners=None, ids=None):
    return SimpleNamespace(capture_mono=t, stripe=stripe, corners=corners, ids=ids)


def test_align_right_brackets_when_it_can_and_says_when_it_cannot() -> None:
    c0, i0 = _corners([1, 2, 3, 4], [[0, 0], [10, 0], [0, 10], [10, 10]])
    c1, i1 = _corners([1, 2, 3, 4], [[2, 0], [12, 0], [2, 10], [12, 10]])
    b = _inp(1.020, _stripe(range(5), [20] * 5), c0, i0)        # 20 ms after the left
    other = _inp(0.987, _stripe(range(5), [16] * 5), c1, i1)    # 13 ms before it
    al = align_right(1.000, b, other)
    assert abs(al.gap_ms - 20.0) < 1e-9 and "brought" in al.note
    f = (1.000 - 0.987) / (1.020 - 0.987)                       # from `other` toward `b`
    assert np.allclose(al.corners.reshape(-1, 2)[0], [2 - 2 * f, 0], atol=1e-5)
    assert al.stripe.y.tolist() == [18] * 5                     # 20 → 16 by the other fraction
    plain = align_right(1.000, b, None)
    assert plain.corners is c0 and plain.stripe is b.stripe and "no frame" in plain.note
    wide = align_right(1.000, b, _inp(0.900, other.stripe, c1, i1))
    assert wide.corners is c0 and "unusable" in wide.note


def test_scan_worker_hands_over_the_bracketing_right_frame() -> None:
    from orbiter_native.scanworker import ScanWorker

    def res(side, t):
        return SimpleNamespace(side=side, capture_mono=t, board=None, stripe=None,
                               wh=(64, 48), bgr=None, jpeg=None,
                               sharpness=float("nan"))

    w = ScanWorker()
    w.set_active(True)
    w.offer(res("right", 1.000))
    w.offer(res("left", 1.010))
    assert w._take_pair() is None                  # the right frame after the left may still come
    w.offer(res("right", 1.033))
    a, b, other, _, _ = w._take_pair()
    assert (a.capture_mono, b.capture_mono, other.capture_mono) == (1.010, 1.000, 1.033)
    # The partner before the left, the far side already in the history.
    w.offer(res("right", 1.066))
    w.offer(res("left", 1.040))
    a, b, other, _, _ = w._take_pair()
    assert (a.capture_mono, b.capture_mono, other.capture_mono) == (1.040, 1.033, 1.066)
    # The partner after the left: the last right consumed brackets from below.
    w.offer(res("left", 1.060))
    a, b, other, _, _ = w._take_pair()
    assert (a.capture_mono, b.capture_mono, other.capture_mono) == (1.060, 1.066, 1.033)
