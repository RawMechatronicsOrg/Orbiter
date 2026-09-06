"""Bringing the right eye to the left eye's instant.

The two cameras free-run. camserver stamps every frame from one kernel clock,
so the gap between the two exposures of a pair is KNOWN — a median of 7-13 ms
on this rig, anywhere up to half a frame — but nothing holds them in phase and
nothing can: these webcams have no trigger pin. Whatever moves in that gap —
the board turned by hand, the subject on it — is seen by the right eye a
little later or a little earlier than by the left, and every comparison
across the two eyes carries the shift: a pose fitted to both is a compromise
between two instants, a veto tests the left's candidate against a stripe
that has moved on.

So the right eye's observations are brought to the left's instant before
anything is compared. A quantity observed in two consecutive right frames
that bracket the left's instant is interpolated linearly in time — at 30
frames/s and hand speeds the motion inside one bracket is straight enough
that the error is a small fraction of a pixel; the readout is not corrected
here (first order). What cannot be bracketed stays as it was, and the caller
is told how far off it is. The left eye is the reference on purpose: its
pixels are what the scan triangulates and its rows are what the readout
correction times.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .laser import StripePixels, stripe_centroids

#: Two right frames further apart than this do not bracket an instant
#: usefully: a hand changes its mind in less, and a straight line across it
#: would invent motion. A frame and a half at 30 frames/s.
MAX_BRACKET_S = 0.05


def interpolate_corners(c0, i0, t0: float, c1, i1, t1: float, t: float):
    """Board corners at instant `t` from the same eye's detections at `t0`
    and `t1`: per id, linear in time. `(corners, ids)` as the detector
    shapes them — (N, 1, 2) float32 and (N, 1) int32 — or None when fewer
    than four ids are in both frames, when the frames do not bracket `t`,
    or when they stand too far apart to bracket anything."""
    if c0 is None or c1 is None or i0 is None or i1 is None:
        return None
    if not (t0 <= t <= t1) or t1 <= t0 or t1 - t0 > MAX_BRACKET_S:
        return None
    common, ia, ib = np.intersect1d(np.asarray(i0).ravel(), np.asarray(i1).ravel(),
                                    return_indices=True)
    if len(common) < 4:
        return None
    a = np.asarray(c0, np.float64).reshape(-1, 2)[ia]
    b = np.asarray(c1, np.float64).reshape(-1, 2)[ib]
    pts = a + (b - a) * ((t - t0) / (t1 - t0))
    return pts.reshape(-1, 1, 2).astype(np.float32), common.reshape(-1, 1).astype(np.int32)


def shift_stripe(base: StripePixels, t_base: float, other: StripePixels, t_other: float,
                 t: float) -> StripePixels:
    """`base`'s stripe pixels moved to instant `t`, by how far the stripe
    travelled on each scanline between `base` and `other`.

    Per scanline the stripe's position is its centroid; the difference
    between the two frames' centroids on the same scanline, scaled by where
    `t` sits between the two instants, moves every pixel of `base` on that
    scanline across it. Scanlines the other frame does not have keep their
    pixels where they were. Nothing moves when the two frames disagree
    about which way the stripe runs, or when either has no pixels.
    """
    if (not base.ok or not other.ok or base.along_x != other.along_x
            or t_other == t_base):
        return base
    f = (t - t_base) / (t_other - t_base)
    if f == 0.0:
        return base
    key_b, acr_b = (base.x, base.y) if base.along_x else (base.y, base.x)
    key_o, acr_o = (other.x, other.y) if other.along_x else (other.y, other.x)
    sb, pb, *_ = stripe_centroids(key_b, acr_b, base.w)
    so, po, *_ = stripe_centroids(key_o, acr_o, other.w)
    if not len(sb) or not len(so):
        return base
    common, ib, io = np.intersect1d(sb.astype(np.int64), so.astype(np.int64),
                                    return_indices=True)
    if not len(common):
        return base
    n = int(key_b.max()) + 1
    delta = np.zeros(n)
    have = np.zeros(n, bool)
    inside = common < n
    delta[common[inside]] = ((po[io] - pb[ib]) * f)[inside]
    have[common[inside]] = True
    k = key_b.astype(np.int64)
    moved = acr_b + np.where(have[k], delta[k], 0.0)
    w, h = base.wh
    limit = (h if base.along_x else w) - 1
    moved = np.clip(np.rint(moved), 0, limit).astype(np.int32)
    return replace(base, y=moved) if base.along_x else replace(base, x=moved)


@dataclass
class Aligned:
    """The right eye's observations at the left's instant, and how they got
    there."""

    stripe: StripePixels | None
    corners: np.ndarray | None
    ids: np.ndarray | None
    #: The raw right frame's instant minus the left's, ms, signed.
    gap_ms: float
    note: str


def align_right(t_left: float, b, other) -> Aligned:
    """The right result `b` (anything with `capture_mono`, `stripe`, `corners`,
    `ids`) brought to `t_left`, interpolating against `other` — the right
    result on the far side of `t_left`, or None. What cannot be bracketed
    stays as it was; the note says what happened either way."""
    gap_ms = (b.capture_mono - t_left) * 1000.0
    if other is None:
        return Aligned(b.stripe, b.corners, b.ids, gap_ms,
                       f"right eye {gap_ms:+.0f} ms off the left, no frame to bracket it")
    t0, t1 = sorted((b.capture_mono, other.capture_mono))
    if not (t0 <= t_left <= t1) or t1 - t0 > MAX_BRACKET_S:
        return Aligned(b.stripe, b.corners, b.ids, gap_ms,
                       f"right eye {gap_ms:+.0f} ms off the left, bracket of "
                       f"{(t1 - t0) * 1000:.0f} ms unusable")
    first, second = (b, other) if b.capture_mono <= other.capture_mono else (other, b)
    fit = interpolate_corners(first.corners, first.ids, first.capture_mono,
                              second.corners, second.ids, second.capture_mono, t_left)
    corners, ids = fit if fit is not None else (b.corners, b.ids)
    stripe = b.stripe
    if b.stripe is not None and other.stripe is not None:
        stripe = shift_stripe(b.stripe, b.capture_mono, other.stripe, other.capture_mono, t_left)
    done = ([] if fit is None else ["corners"]) + ([] if stripe is b.stripe else ["stripe"])
    what = " and ".join(done) if done else "nothing to move"
    return Aligned(stripe, corners, ids, gap_ms,
                   f"right eye {gap_ms:+.0f} ms off the left: {what} brought to its instant "
                   f"(bracket {(t1 - t0) * 1000:.0f} ms)")
