"""Scanning: the laser plane meets each left-eye ray; the right eye confirms.

One sweep of the pipeline, per frame pair:

  1. Each eye reports the pixels it considers stripe, with weights
     (`laser.find_stripe_pixels`) — no shape assumed, no centroid yet.
  2. Every LEFT stripe pixel's ray is intersected with the calibrated laser
     plane. That is a 3D candidate.
  3. The candidate is projected into the RIGHT eye. If the right eye saw no
     stripe there (within a few pixels), the candidate is dropped.
  4. Per scanline, the stripe's position across it is found to a fraction
     of a pixel from the surviving pixels' score profile
     (`laser.stripe_centroids`), and that position's ray meets the plane
     for the point that is kept.
  5. Keep it only if it lies inside a cylinder above the board, expressed in
     the BOARD's frame so the cylinder stays put while the board defines "up".

Between the veto and the board's frame, four gates learned from the noise a
live scan showed — points came rarely and noisy at once:

  * one blob per scanline. A scanline's confirmed pixels can form several
    runs — the stripe and a glint the right eye happened to confirm too —
    and averaging them yields a point that is neither. The strongest run is
    the stripe; the rest are ignored;
  * that blob's width. A lone pixel is noise; a run wider than the stripe
    ever is on this rig is a reflection or a smear;
  * reach. The scanner has a working range: a point must lie between
    `range_mm` of the line through the two camera centres — the baseline.
    Nearer is the rig's own hardware or a hand, farther is the wall; neither
    needs to be recognised to be dropped, and unlike the cylinder this needs
    no board pose to hold;
  * no jumps. Along the stripe consecutive points are a fraction of a
    millimetre apart; one that stands `jump_mm` off BOTH its neighbours while
    they agree with each other is not on the surface they are on. A real
    depth step keeps one neighbour close and is untouched.

The step into the board's frame is where the rolling shutter is paid for.
The sensor reads row by row, so the stripe at the bottom of the frame was
seen a readout later than the corners that gave the pose; with a `Motion`
— the pose's twist from the previous frame, and the sensor's readout time —
each point goes through the pose at its own row's instant (`rolling.py`).
Without one, every point goes through the corners' pose, and moves with the
board by however far it turned during the readout.

**Why the plane, not stereo triangulation.** The first version matched each
left stripe point to the right eye's stripe along its epipolar line and
triangulated. Two things were wrong with it on this rig. The correspondence
was built from per-column centroids, and a column holding the stripe AND a red
wire — 16-24% of columns on the bench, measured — yielded one centroid that was
neither; it still met the right polyline somewhere, and where it did not, the
one-eye rescue put it on the plane unverified. That was the noise the operator
saw. And the plane was the better instrument all along: with the sheet passing
74 mm from the left camera, a 0.5 px centroid gives about 1.2 mm at half a
metre, against 2.4 mm from the 144 mm stereo baseline at its 1.95 px fit.
Checking each PIXEL against the right eye before any centroid is taken removes
the wire from the average instead of averaging it in, and needs no epipolar
search, no segment model and no ambiguity handling.

**What confirms a point.** The right eye does not measure; it vetoes. A stripe
pixel whose plane point projects onto right-eye stripe was seen by both cameras
at a place consistent with the calibrated sheet. A wire, a reflection or a red
switch in the left image lands on the plane at some point too — but that point
projects into the right image where the right eye saw nothing red. Flanks the
right eye cannot see are dropped as well; the subject turns, and they come
round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .laser import BLOB_GAP_PX, StripePixels, stripe_centroids
from .laserplane import LaserPlane, rays
from .rolling import Motion
from .stereo import StereoRig

log = logging.getLogger("orbiter_native.scan")


@dataclass(frozen=True)
class ScanVolume:
    """A cylinder standing on the board, in board coordinates, in millimetres.

    The board frame from `cvcore.estimate_pose` has its origin at the board's
    centre and z pointing out of the printed face, so `height_mm` is "above
    the board" and the cylinder is centred on the board itself. A cylinder
    rather than a box because the board is a disc and the subject stands on
    it: the wall behind the bench sits inside a 200 mm box's corners once the
    board is tilted, and outside a disc the size of the board.
    """

    height_mm: float = 400.0
    #: The board on this rig is 288 mm across, so 144 mm is its own edge.
    radius_mm: float = 150.0
    #: Points below this are the board's own surface — the calibration target,
    #: not the subject. Plane-based points carry about 1 mm of noise at half
    #: a metre; 5 mm keeps the board out without eating the subject's base.
    floor_mm: float = 5.0

    def contains(self, xyz_board: np.ndarray) -> np.ndarray:
        """Boolean mask over (N, 3) points expressed in the board frame."""
        if not len(xyz_board):
            return np.zeros(0, bool)
        x, y, z = xyz_board[:, 0], xyz_board[:, 1], xyz_board[:, 2]
        return ((z >= self.floor_mm) & (z <= self.height_mm)
                & (np.hypot(x, y) <= self.radius_mm))


@dataclass
class ScanParams:
    """Acceptance thresholds for one frame's points."""

    #: How far, in right-eye pixels, a plane point may miss the right eye's
    #: stripe and still count as seen by it. The plane fit is 0.36 mm RMS and
    #: the pair 1.95 px, so 3 px is the calibration's own slack, not a search.
    confirm_px: int = 3
    #: Once any pixel of a stripe blob is confirmed, every pixel of the same
    #: scanline within this many pixels of it counts too. The veto decides
    #: which blobs are real; the centroid must then see the whole blob. The
    #: sheet nearly contains the left eye's rays, so one pixel across the
    #: stripe is ~3.7 mm along the ray and ~2.6 px in the right image: judged
    #: pixel by pixel, a 2 px calibration shift would confirm one flank of the
    #: stripe and not the other, and bias the centroid by a pixel — 2.4 mm.
    blob_px: int = 8
    #: The scanner's reach, mm from the baseline — the line through the two
    #: camera centres. Points nearer or farther are not the subject.
    range_mm: tuple[float, float] = (150.0, 450.0)
    #: Width across the scanline, in pixels, of a run of confirmed pixels
    #: for it to be the stripe: a lone pixel is noise, a smear is a glint.
    blob_width_px: tuple[int, int] = (2, 24)
    #: A point this far from both its neighbours along the stripe, while
    #: those agree with each other, is dropped. 0 disables.
    jump_mm: float = 5.0
    #: Locate the stripe on each scanline by a Gaussian fit of its profile
    #: rather than the centroid — see `laser.stripe_centroids`.
    centroid_fit: bool = True
    #: The veto asks whether the right eye has stripe within `confirm_px` of
    #: where a candidate projects. The pair's calibration puts that
    #: projection a couple of pixels off — the same couple of pixels for
    #: every candidate of the frame, the frame's `veto_px` — and at 2-3 px
    #: of residual half a frame's true points fail a 3 px veto for nothing
    #: they did. With `veto_follow` the projections are first moved by the
    #: frame's median offset, when that offset is small (under
    #: `veto_follow_max_px`: a calibration's slack, not a stale pair) and
    #: the candidates agree on it (median absolute deviation under
    #: `veto_follow_mad_px`: a stripe, not fog); the veto then judges each
    #: candidate against the consensus, which is what it was for. Depth is
    #: untouched — the sheet gives it; the offset stays on the panel.
    veto_follow: bool = True
    veto_follow_max_px: float = 8.0
    veto_follow_mad_px: float = 2.0
    #: Refine each point's depth by the RIGHT eye's own stripe centroid
    #: (`refine_by_right`). The sheet gives depth through one eye across a
    #: baseline of only the laser's offset — 74 mm on this rig — while the
    #: pair's baseline is twice that or more; the right centroid is a second,
    #: independent measurement of the same depth, fused by precision. It is
    #: only as true as the pair's calibration, so it is off above
    #: `stereo_refine_max_rms_px` of pair residual, and a right centroid
    #: farther than `stereo_refine_window_px` from where the sheet put the
    #: point, or asking for more than `stereo_refine_max_shift_mm`, is
    #: something else (a glint, the wrong blob) and is not used.
    stereo_refine: bool = True
    stereo_refine_max_rms_px: float = 1.0
    stereo_refine_window_px: float = 6.0
    stereo_refine_max_shift_mm: float = 10.0
    #: Where a point's colour is read: this many pixels to either side of
    #: the stripe, across it, in the left eye. Under the stripe every
    #: surface is laser-red; beside it, past the halo, it is itself. The
    #: strobed off-frame (BACKLOG) would make this exact; until then 8 px
    #: clears the 4-10 px stripes this rig produces with room to spare.
    colour_offset_px: float = 8.0
    #: Show and export the CONFIDENT cloud rather than every voxel — see
    #: `confident`: a voxel with fewer than `clean_neighbours` other voxels
    #: within a `clean_cell_mm` cell of it is lonely (a glint, a hand), a
    #: voxel seen once where its neighbours were seen `clean_flicker_obs`
    #: times or more is a flicker the passes never confirmed, and what
    #: survives is merged on `clean_merge_mm` voxels, each weighted by how
    #: often it was seen.
    clean: bool = True
    clean_merge_mm: float = 1.0
    clean_cell_mm: float = 2.0
    clean_neighbours: int = 3
    clean_flicker_obs: int = 3
    volume: ScanVolume = field(default_factory=ScanVolume)


@dataclass
class ScanFrame:
    """What one frame pair contributed, and why the rest was dropped."""

    points_board: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), np.float64))
    points_camera: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), np.float64))
    #: The scanline each kept point came from — what lets consecutive
    #: still frames be averaged point by point.
    scanlines: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    #: The left-eye pixel each kept point came from - the scanline and the
    #: sub-pixel position across it - as (x, y) in the full frame. What the
    #: colour is read beside.
    pixels_left: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float64))
    #: RGB per kept point, uint8, read beside the stripe by the scan worker
    #: - or None while no colour image came with the frame.
    colours: np.ndarray | None = None
    #: How much each kept point is worth against others of the same place:
    #: its precision, from the depth it was seen at (`precision_weights`).
    weights: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    #: Scanlines (columns or rows) holding any stripe in the left eye.
    n_scanlines: int = 0
    #: Which of the two `scanlines` counts along: columns when True, rows
    #: when False. Decided per frame from the lit pixels' own extent, so it
    #: can differ between frames — and a scanline id from one is not the
    #: same place as the same id from the other.
    along_x: bool = True
    #: Stripe pixels in the left eye, and how many the right eye confirmed.
    n_pixels: int = 0
    n_confirmed: int = 0
    #: Scanlines none of whose pixels the right eye confirmed.
    n_rejected_unconfirmed: int = 0
    #: Scanlines with no confirmed blob of a valid width.
    n_rejected_blob: int = 0
    #: Scanlines that held more than one confirmed blob (one was taken).
    n_split: int = 0
    #: Points outside the scanner's reach from the baseline.
    n_rejected_range: int = 0
    #: Points standing off both their neighbours along the stripe.
    n_rejected_jump: int = 0
    #: Scanlines whose centroid, put on the sheet, projects farther than
    #: `confirm_px` from the right eye's nearest stripe run on that
    #: scanline. The pixel veto's dilation and the two stripes' widths let
    #: a candidate through from up to ~9 px away — ±7 mm of depth at 400 mm
    #: on this rig (f·B/Z² = 1.3 px/mm); this is the veto done centroid to
    #: centroid, ±2.3 mm.
    n_rejected_offside: int = 0
    n_rejected_volume: int = 0
    #: The stereo refinement (`refine_by_right`): how many kept points the
    #: right eye's centroid took part in, the median correction it made,
    #: mm along the ray, the right eye's mean weight in the fusion, and
    #: why it did nothing when it did nothing.
    n_refined: int = 0
    refine_shift_mm: float = float("nan")
    refine_share: float = float("nan")
    refine_note: str | None = None
    #: How far, in px along the scanline, the right eye's stripe sits from
    #: where the left eye's candidates project into its frame. NaN when the
    #: eyes share no scanline. See `_veto_offset`.
    veto_px: float = float("nan")
    #: The offset the veto was moved by before judging (`veto_follow`),
    #: 0 when it judged the projections as they were, and why not when it
    #: did not follow a measured offset.
    veto_shift_px: float = 0.0
    veto_note: str | None = None
    reason: str | None = None
    #: Which eyes the board pose came from — "left+right", "left" or
    #: "right" — and, with both, how far their independent poses stood apart
    #: (a live check on the pair's calibration) and the joint fit's error.
    pose_source: str = ""
    pose_gap_deg: float = float("nan")
    pose_gap_mm: float = float("nan")
    pose_rms_px: float = float("nan")
    #: How far off the left's instant the right frame was, ms, and what
    #: `timealign` did about it.
    sync_gap_ms: float = float("nan")
    sync_note: str = ""
    #: Rolling shutter: the largest shift the per-row poses made to a kept
    #: point, the board's speed the twist implied, or why none was applied.
    rs_max_mm: float = 0.0
    speed_mm_s: float = 0.0
    spin_deg_s: float = 0.0
    rs_note: str | None = None
    #: How far the pose the points were finally placed through — the median
    #: of the poses around this frame in time (`posesmooth`) — sat from the
    #: frame's own, mm. NaN until the frame was placed.
    pose_smooth_mm: float = float("nan")

    @property
    def n_kept(self) -> int:
        return len(self.points_board)


def _on_plane(k, plane: LaserPlane, pixels: np.ndarray) -> np.ndarray:
    """Where each (N, 2) left-eye pixel's ray meets the laser plane, in the
    left camera's frame. NaN where the ray runs parallel or away."""
    d = rays(pixels, k)
    return plane.intersect_rays(np.zeros_like(d), d)


def _veto_offset(right: StripePixels, uv: np.ndarray) -> float:
    """Median of `_veto_offsets`, NaN when the two eyes share no scanline."""
    off = _veto_offsets(right, uv)
    return float(np.median(off)) if len(off) else float("nan")


def _veto_offsets(right: StripePixels, uv: np.ndarray) -> np.ndarray:
    """Per candidate, the signed distance, px along the scanline, from where
    the left eye puts the stripe in the right frame to where the right eye
    actually has it — for the candidates on a scanline the right eye has
    stripe on; the others are left out.

    The veto asks whether those coincide within `confirm_px`; this says by
    how much they miss, which is what tells a calibration that cannot scan
    from a scene with nothing in it. Intrinsics, sheet and pair geometry that
    disagree put the stripe tens of pixels from where the other eye sees it,
    and then no amount of stripe will ever confirm — while every count on the
    panel reads exactly as it would with the laser switched off. NaN when the
    two eyes share no scanline.
    """
    runs = right_runs(right)
    if runs is None or not len(uv):
        return np.empty(0)
    key = np.rint(uv[:, 0] if right.along_x else uv[:, 1])
    across = uv[:, 1] if right.along_x else uv[:, 0]
    fin = np.isfinite(key) & np.isfinite(across)
    off = np.full(len(uv), np.nan)
    off[fin] = -nearest_run_residual(runs, key[fin].astype(np.int64), across[fin])
    return off[np.isfinite(off)]


def right_runs(right: StripePixels) -> tuple[np.ndarray, np.ndarray] | None:
    """The right eye's stripe as one centroid per RUN — a scanline can hold
    the stripe and a glint, and a candidate is judged against the nearest,
    not the mean of both — as `(key, pos)` sorted by key then pos. None
    without pixels."""
    if not len(right.x):
        return None
    key = (right.x if right.along_x else right.y).astype(np.int64)
    across = (right.y if right.along_x else right.x).astype(np.float64)
    w = np.maximum(right.w.astype(np.float64), 1.0)
    order = np.lexsort((across, key))
    key, across, w = key[order], across[order], w[order]
    new = np.ones(len(key), bool)
    new[1:] = (key[1:] != key[:-1]) | (across[1:] - across[:-1] > BLOB_GAP_PX)
    run = np.cumsum(new) - 1
    nb = int(run[-1]) + 1
    pos = np.bincount(run, weights=w * across, minlength=nb) / np.bincount(run, weights=w, minlength=nb)
    rk = key[new]
    order = np.lexsort((pos, rk))
    return rk[order], pos[order]


#: Runs looked at per scanline when finding the nearest: a scanline holds
#: the stripe and perhaps a glint or two, not more.
_RUNS_PER_LINE = 4


def nearest_run_residual(runs: tuple[np.ndarray, np.ndarray], key: np.ndarray,
                         across: np.ndarray) -> np.ndarray:
    """Per query `(key, across)`, the signed distance from `across` to the
    nearest run's centroid on scanline `key`: run minus query. NaN where
    the scanline has no run."""
    rk, pos = runs
    key = np.asarray(key, np.int64)
    across = np.asarray(across, np.float64)
    start = np.searchsorted(rk, key, side="left")
    best = np.full(len(key), np.nan)
    for i in range(_RUNS_PER_LINE):
        j = np.minimum(start + i, len(rk) - 1)
        hit = (start + i < len(rk)) & (rk[j] == key)
        d = np.where(hit, pos[j] - across, np.nan)
        take = hit & (~np.isfinite(best) | (np.abs(d) < np.abs(best)))
        best = np.where(take, d, best)
    return best


def _whole_blobs(px: StripePixels, confirmed: np.ndarray, reach: int) -> np.ndarray:
    """Extend confirmation from any confirmed pixel to the rest of its blob:
    every stripe pixel of the same scanline within `reach` of it."""
    if not confirmed.any() or reach <= 0:
        return confirmed
    w, h = px.wh
    img = np.zeros((h, w), np.uint8)
    img[px.y[confirmed], px.x[confirmed]] = 255
    # Along the scanline only: (rows, cols) — a column's blob grows down the
    # column, a row's blob along the row. Never sideways into the neighbours.
    shape = (2 * reach + 1, 1) if px.along_x else (1, 2 * reach + 1)
    grown = cv2.dilate(img, np.ones(shape, np.uint8))
    return grown[px.y, px.x] > 0


def _other_centre(rig: StereoRig, eye: str) -> np.ndarray:
    """The OTHER camera's centre in `eye`'s frame: with this eye's origin it
    spans the baseline. X_r = R X_l + T, so the right centre in the left
    frame is -R^T T and the left centre in the right frame is T."""
    R = np.asarray(rig.geom.R, float)
    T = np.asarray(rig.geom.T, float).ravel()
    return -R.T @ T if eye == "left" else T


def _reach_mm(xyz: np.ndarray, rig: StereoRig, eye: str = "left") -> np.ndarray:
    """Distance of each point from the baseline — the line through the two
    camera centres — with the points in `eye`'s frame."""
    if not len(xyz):
        return np.empty(0)
    other = _other_centre(rig, eye)
    norm = float(np.linalg.norm(other))
    if norm < 1e-6:
        return np.linalg.norm(xyz, axis=1)
    u = other / norm
    along = xyz @ u
    return np.linalg.norm(xyz - along[:, None] * u[None, :], axis=1)


#: How far past the reach the row band extends, as a factor: the band is a
#: search window, the reach gate in millimetres does the deciding.
BAND_MARGIN = 1.15


def stripe_rows(plane: LaserPlane, rig: StereoRig, range_mm: tuple[float, float],
                wh: tuple[int, int], eye: str = "left") -> tuple[int, int] | None:
    """The rows of `eye`'s frame where a point of the laser sheet within the
    reach can appear, as `(first, last + 1)`, or None when none can.

    The sheet is fixed in the cameras' frame, and so is the reach, so where
    the stripe CAN be is fixed too — on this rig, with the sheet containing
    the optical axis, rows cy + f·d/450 to cy + f·d/150 for a reach of
    150-450 mm. Searching only there costs nothing the sheet could have
    shown and drops every glint elsewhere before the veto ever sees it.
    Computed by sampling: rays through a grid of pixels meet the sheet,
    the meeting points' reach is measured, and the rows that admit any are
    the band, widened by `BAND_MARGIN`. For the right eye the sheet and
    the baseline are carried into its frame through the extrinsics.
    """
    w, h = wh
    if w <= 0 or h <= 0:
        return None
    if eye == "left":
        k, normal, d = rig.left_k, np.asarray(plane.normal, float), float(plane.d)
    else:
        R = np.asarray(rig.geom.R, float)
        T = np.asarray(rig.geom.T, float).ravel()
        k = rig.right_k
        normal = R @ np.asarray(plane.normal, float)
        d = float(plane.d) + float(normal @ T)
    cols = np.linspace(0.0, w - 1.0, 9)
    rows = np.arange(h, dtype=np.float64)
    grid = np.column_stack([np.tile(cols, h), np.repeat(rows, len(cols))])
    dirs = rays(grid, k)
    denom = dirs @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        s = d / denom
    ahead = np.isfinite(s) & (s > 0)
    pts = dirs * np.where(ahead, s, 0.0)[:, None]
    reach = _reach_mm(pts, rig, eye)
    lo, hi = range_mm
    ok = ahead & (reach >= lo / BAND_MARGIN) & (reach <= hi * BAND_MARGIN)
    row_ok = ok.reshape(h, len(cols)).any(axis=1)
    hit = np.flatnonzero(row_ok)
    if not len(hit):
        return None
    return int(hit[0]), int(hit[-1]) + 1


def _not_a_jump(xyz_cam: np.ndarray, scan: np.ndarray, jump_mm: float) -> np.ndarray:
    """False for a point `jump_mm` off both its neighbours along the stripe
    while those two agree with each other. Neighbours are the points on the
    adjacent scanlines; across a break in the stripe nothing is judged, and
    the two ends are kept."""
    n = len(xyz_cam)
    keep = np.ones(n, bool)
    if n < 3 or jump_mm <= 0:
        return keep
    step = np.linalg.norm(xyz_cam[1:] - xyz_cam[:-1], axis=1)
    adjacent = np.diff(scan) <= 2
    span = np.linalg.norm(xyz_cam[2:] - xyz_cam[:-2], axis=1)
    lone = (adjacent[:-1] & adjacent[1:]
            & (np.minimum(step[:-1], step[1:]) > jump_mm)
            & (span < 2.0 * jump_mm))
    keep[1:-1] = ~lone
    return keep


#: The depth a point's weight is 1 at; nearer points weigh more, farther less.
Z_REF_MM = 300.0


def precision_weights(xyz_cam: np.ndarray) -> np.ndarray:
    """One weight per camera-frame point: how much to trust it against
    others that land in the same place.

    A ray meeting the sheet: a pixel of error across the stripe moves the
    point along the ray by about Z² / (f·d) — twice the depth, four times
    the error. Inverse variance is then (Z_REF / Z)⁴: a point seen at
    150 mm outweighs one seen at 450 mm eighty-one to one, so a close pass
    over a surface overrides what a far pass left there, and the far pass
    still stands wherever nothing closer came. Depth is floored at 50 mm,
    a place nothing is scanned from, so a stray candidate cannot swamp a
    voxel."""
    xyz = np.asarray(xyz_cam, np.float64).reshape(-1, 3)
    z = np.maximum(xyz[:, 2], 50.0)
    return (Z_REF_MM / z) ** 4


def sample_beside(bgr: np.ndarray, pixels: np.ndarray, along_x: bool,
                  offset_px: float, wh: tuple[int, int]) -> np.ndarray:
    """The surface's colour at each stripe point, read BESIDE the stripe.

    Under the stripe every surface is laser-red; a few pixels to either side,
    across the stripe, it is its own colour. `pixels` are (N, 2) left-eye
    positions in the full frame; `bgr` is the eye's colour image as
    published - on the GPU path the half-size copy - so positions are scaled
    to it. Both sides are read, five pixels along the stripe each, and the
    median per channel is the answer: a glint a few pixels wide on one side
    does not tint the point. Returns (N, 3) uint8 RGB.
    """
    n = len(pixels)
    if not n:
        return np.empty((0, 3), np.uint8)
    h, w = bgr.shape[:2]
    sx, sy = w / float(wh[0]), h / float(wh[1])
    p = np.asarray(pixels, np.float64).reshape(-1, 2) * [sx, sy]
    along = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[None, :]
    if along_x:                                  # scanlines are columns: across is y
        off = offset_px * sy
        xs = p[:, 0:1] + along
        X = np.concatenate([xs, xs], axis=1)
        Y = np.concatenate([p[:, 1:2] - off + 0.0 * along, p[:, 1:2] + off + 0.0 * along], axis=1)
    else:                                        # scanlines are rows: across is x
        off = offset_px * sx
        ys = p[:, 1:2] + along
        X = np.concatenate([p[:, 0:1] - off + 0.0 * along, p[:, 0:1] + off + 0.0 * along], axis=1)
        Y = np.concatenate([ys, ys], axis=1)
    xi = np.clip(np.rint(X).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(Y).astype(np.int64), 0, h - 1)
    med = np.median(bgr[yi, xi], axis=1)                       # (N, 3), BGR
    return np.clip(np.rint(med), 0, 255).astype(np.uint8)[:, ::-1]


#: The stripe centroid's own noise per eye, px, as the calibration panel
#: measures it live off the line fits on this rig (0.5-0.7 px). It sets how
#: the two depth measurements are weighed against each other, not whether
#: a point is kept.
STRIPE_SIGMA_PX = 0.6
#: Below this many right-eye pixels per millimetre along the ray the right
#: eye has no depth to offer: its epipolar lines run along the stripe, as
#: they do when the pair's baseline is parallel to the sheet. On this rig
#: the baseline stands across the sheet and a point at 400 mm moves 1.3 px
#: per mm.
MIN_DV_PX_PER_MM = 0.05


@dataclass
class Refined:
    """What `refine_by_right` did to a frame's points."""

    #: The points, camera frame, moved along their rays where refined.
    xyz: np.ndarray
    #: Per point: the right eye's centroid took part.
    used: np.ndarray
    #: Per point: the correction applied, mm along the ray (0 where not used).
    shift_mm: np.ndarray
    #: Per point: the right eye's weight in the fusion (0 where not used).
    share: np.ndarray
    #: Why nothing was refined, when nothing was.
    note: str | None = None


def refine_by_right(rig: StereoRig, plane: LaserPlane, xyz_cam: np.ndarray,
                    pix_left: np.ndarray, left_along_x: bool, right: StripePixels,
                    params: ScanParams,
                    runs: tuple[np.ndarray, np.ndarray] | None = None) -> Refined:
    """Fuse each point's sheet depth with the depth the right eye's stripe
    centroid implies, along the left ray.

    The sheet fixes a point where the left ray meets it; a left-pixel error
    across the stripe moves that point along the sheet by Z²/(f·d) per
    pixel, `d` being the sheet's offset from the left camera. The right eye
    sees the same stripe; where its centroid sits on the scanline the point
    projects to is a second reading of the depth, moving the point along
    the left ray by 1/(∂v/∂h) per right pixel — with `B`, the pair's
    baseline, in place of `d`. One Newton step from the sheet's point gives
    the stereo depth.

    The two are not independent: both carry the same left-pixel error, the
    sheet by Z²/(f·d) of it and the stereo by Z²/(f·B). So the sheet's
    weight `a` in the average is what minimises the variance of
    a·sheet + (1−a)·stereo with that shared term in — not the inverse-
    variance mix of two independent readings, which would keep a third of
    a sheet error the right eye can see past. With equal centroid noise on
    both sides and B = 2d the answer is the stereo depth alone; as the
    pair's residual grows, treated as noise in the right eye, `a` climbs
    back toward the sheet. `a` is kept in [0, 1]: extrapolating past the
    stereo depth to cancel more of the left error is what the algebra
    asks for at wide baselines, and not something a glint should be
    allowed to do. Sensitivities are found numerically, per point, so
    distortion and the rig's actual geometry are in them.
    """
    n = len(xyz_cam)
    none = Refined(xyz_cam, np.zeros(n, bool), np.zeros(n), np.zeros(n))
    if not n:
        return none
    if not params.stereo_refine:
        none.note = "off"
        return none
    rms = float(getattr(rig.geom, "rms_px", float("nan")))
    if not np.isfinite(rms) or rms > params.stereo_refine_max_rms_px:
        none.note = (f"pair rms {rms:.2f} px over {params.stereo_refine_max_rms_px:g}: "
                     "sheet depth only")
        return none
    if not right.ok:
        none.note = "no right stripe"
        return none
    runs = right_runs(right) if runs is None else runs
    if runs is None:
        none.note = "no right centroids"
        return none

    uv = rig.project_right(xyz_cam)
    fin = np.isfinite(uv).all(axis=1)
    scan_axis, across_axis = (0, 1) if right.along_x else (1, 0)
    k = np.full(n, -1, np.int64)
    k[fin] = np.rint(uv[fin, scan_axis]).astype(np.int64)
    # NOT moved by the frame's consensus offset the veto follows: to a pair
    # trusted this far (`stereo_refine_max_rms_px`) a consistent offset is
    # depth, and reading it as slack would be to undo the refinement.
    v_sheet = uv[:, across_axis]
    resid = np.full(n, np.nan)
    resid[fin] = nearest_run_residual(runs, k[fin], v_sheet[fin])
    found = np.isfinite(resid)
    ok = found & (np.abs(resid) <= params.stereo_refine_window_px)

    # How the right's across-coordinate answers a move of 1 mm along the ray.
    ray = xyz_cam / np.linalg.norm(xyz_cam, axis=1, keepdims=True)
    uv_step = rig.project_right(xyz_cam + ray)
    dv = uv_step[:, across_axis] - v_sheet
    sensitive = np.isfinite(dv) & (np.abs(dv) >= MIN_DV_PX_PER_MM)
    ok &= sensitive
    shift = np.zeros(n)
    share = np.zeros(n)
    used = np.zeros(n, bool)
    if not found.any():
        note = "no right centroid on the scanlines the points project to"
    elif not (found & sensitive).any():
        note = "the right eye's stripe runs along its epipolar lines: no depth in it"
    elif not ok.any():
        note = f"no right centroid within {params.stereo_refine_window_px:g} px"
    else:
        note = f"every correction over {params.stereo_refine_max_shift_mm:g} mm: not used"
    if ok.any():
        idx = np.flatnonzero(ok)
        dh = resid[idx] / dv[idx]
        near = np.abs(dh) <= params.stereo_refine_max_shift_mm
        idx, dh = idx[near], dh[near]
        if len(idx):
            # Millimetres of depth per pixel of error: the sheet's, to a left
            # pixel across the stripe, per point; the stereo's, to a pixel of
            # either eye, from the same ∂v/∂h.
            e = np.array([[0.0, 1.0]]) if left_along_x else np.array([[1.0, 0.0]])
            moved = _on_plane(rig.left_k, plane, np.asarray(pix_left)[idx] + e)
            s_sheet = np.maximum(np.linalg.norm(moved - xyz_cam[idx], axis=1), 1e-6)
            s_stereo = 1.0 / np.abs(dv[idx])
            u = s_sheet / s_stereo
            var_left = STRIPE_SIGMA_PX ** 2
            var_right = STRIPE_SIGMA_PX ** 2 + rms ** 2
            a = (var_right - var_left * (u - 1.0)) / (var_left * (u - 1.0) ** 2 + var_right)
            part = 1.0 - np.clip(a, 0.0, 1.0)
            shift[idx] = part * dh
            share[idx] = part
            used[idx] = True
    return Refined(xyz_cam + shift[:, None] * ray, used, shift, share,
                   None if used.any() else note)


def scan_frame(
    rig: StereoRig,
    plane: LaserPlane | None,
    left: StripePixels,
    right: StripePixels,
    board_R: np.ndarray | None,
    board_t: np.ndarray | None,
    params: ScanParams = ScanParams(),
    motion: Motion | None = None,
    rs_note: str | None = None,
) -> ScanFrame:
    """One frame pair's stripe into board-frame points.

    `board_R`/`board_t` are the board's pose in the LEFT camera's frame, as
    `cvcore.estimate_pose` returns it (t in mm), valid at the instant of the
    corners' mean row. `motion`, when known, slides it to every point's own
    row; `rs_note` says why it is not known, for the panel.
    """
    if plane is None:
        return ScanFrame(reason="no laser plane — scanning meets each ray with "
                                "it; calibrate the laser plane first")
    if not left.ok or not right.ok:
        return ScanFrame(reason="both eyes need stripe pixels")
    if board_R is None or board_t is None:
        return ScanFrame(reason="board pose unknown — it defines the scan volume")

    px = np.stack([left.x, left.y], axis=1).astype(np.float64)
    cand = _on_plane(rig.left_k, plane, px)
    ok = np.isfinite(cand).all(axis=1)

    # The veto. A candidate is real only if the right eye saw stripe where
    # the candidate projects.
    confirmed = np.zeros(len(px), bool)
    veto_px = float("nan")
    veto_shift, veto_note = 0.0, None
    if ok.any():
        uv = rig.project_right(cand[ok])
        offsets = _veto_offsets(right, uv)
        veto_px = float(np.median(offsets)) if len(offsets) else float("nan")
        if params.veto_follow and len(offsets):
            mad = float(np.median(np.abs(offsets - veto_px)))
            if abs(veto_px) > params.veto_follow_max_px:
                veto_note = (f"offset {veto_px:+.1f} px is over {params.veto_follow_max_px:g}: "
                             "not a slack to follow — the pair or the sheet is off")
            elif mad > params.veto_follow_mad_px:
                veto_note = (f"candidates disagree on the offset (MAD {mad:.1f} px): "
                             "not followed")
            elif abs(veto_px) >= 0.5:
                # `veto_px` is candidate minus stripe: the stripe sits at
                # minus that from the projections, so that is the move.
                veto_shift = -veto_px
                across = 1 if right.along_x else 0
                uv = uv.copy()
                uv[:, across] += veto_shift
        seen = right.mask(params.confirm_px)
        w, h = right.wh
        fin = np.isfinite(uv).all(axis=1)
        u = np.zeros((len(uv), 2), np.int64)
        u[fin] = np.rint(uv[fin]).astype(np.int64)
        inside = fin & (u[:, 0] >= 0) & (u[:, 0] < w) & (u[:, 1] >= 0) & (u[:, 1] < h)
        hit = np.zeros(len(uv), bool)
        hit[inside] = seen[u[inside, 1], u[inside, 0]] > 0
        confirmed[np.flatnonzero(ok)[hit]] = True
        confirmed = _whole_blobs(left, confirmed, params.blob_px)

    # One sub-pixel centroid per scanline, over its strongest run of
    # confirmed pixels only, and only if that run is stripe-shaped.
    key = left.x if left.along_x else left.y
    across = left.y if left.along_x else left.x
    n_lines = int(len(np.unique(key)))
    scan, pos, n_live, n_split, n_blob = stripe_centroids(
        key[confirmed], across[confirmed], left.w[confirmed], params.blob_width_px,
        fit=params.centroid_fit)
    centroids = (np.stack([scan, pos], axis=1) if left.along_x
                 else np.stack([pos, scan], axis=1))
    n_unconfirmed = n_lines - n_live

    xyz_cam = (_on_plane(rig.left_k, plane, centroids) if len(centroids)
               else np.empty((0, 3)))
    finite = np.isfinite(xyz_cam).all(axis=1)
    xyz_cam, scan = xyz_cam[finite], scan[finite]
    rows = centroids[finite, 1] if len(centroids) else np.empty(0)
    pix = centroids[finite] if len(centroids) else np.empty((0, 2))

    # The veto again, centroid to centroid: the pixel veto's dilation and
    # the stripes' widths admit a candidate from ~9 px away, ±7 mm of
    # depth at 400 mm; the sheet point's projection must sit within
    # `confirm_px` of the right eye's nearest run on that scanline, after
    # the frame's consensus offset. Scanlines the right eye has no run on
    # are not judged here — the pixel veto already saw stripe there.
    runs = right_runs(right)
    n_offside = 0
    if runs is not None and len(xyz_cam):
        uv_c = rig.project_right(xyz_cam)
        fin_c = np.isfinite(uv_c).all(axis=1)
        k_c = np.rint(np.where(fin_c, uv_c[:, 0] if right.along_x else uv_c[:, 1], -1)).astype(np.int64)
        v_c = (uv_c[:, 1] if right.along_x else uv_c[:, 0]) + veto_shift
        resid_c = np.full(len(xyz_cam), np.nan)
        resid_c[fin_c] = nearest_run_residual(runs, k_c[fin_c], v_c[fin_c])
        offside = np.isfinite(resid_c) & (np.abs(resid_c) > params.confirm_px)
        n_offside = int(offside.sum())
        keep = ~offside
        xyz_cam, scan, rows, pix = xyz_cam[keep], scan[keep], rows[keep], pix[keep]

    # The reach: the baseline is the one line both cameras share, and the
    # subject sits a known distance from it whatever the board does.
    lo, hi = params.range_mm
    reach = _reach_mm(xyz_cam, rig)
    in_range = (reach >= lo) & (reach <= hi)
    n_range = int((~in_range).sum())
    xyz_cam, rows, scan, pix = xyz_cam[in_range], rows[in_range], scan[in_range], pix[in_range]

    # No jumps along the stripe.
    smooth = _not_a_jump(xyz_cam, scan, params.jump_mm)
    n_jump = int((~smooth).sum())
    xyz_cam, rows, pix = xyz_cam[smooth], rows[smooth], pix[smooth]

    # The right eye's own centroid, fused into the depth along each ray.
    refined = refine_by_right(rig, plane, xyz_cam, pix, bool(left.along_x), right, params,
                              runs=runs)
    xyz_cam = refined.xyz

    # Into the board's frame: the volume is defined relative to the board, so
    # it stays put when the board moves and "above" keeps meaning above it.
    rs_max = 0.0
    if len(xyz_cam):
        xyz_board = (np.asarray(board_R, float).T
                     @ (xyz_cam.T - np.asarray(board_t, float).reshape(3, 1))).T
        if motion is not None:
            # Each point through the pose at its own row's instant, not the
            # corners'. The shift against the static transform is what the
            # rolling shutter would have cost.
            corrected = motion.to_board(xyz_cam, rows, board_R, board_t)
            rs_max = float(np.linalg.norm(corrected - xyz_board, axis=1).max())
            xyz_board = corrected
    else:
        xyz_board = np.empty((0, 3))
    inside = params.volume.contains(xyz_board)

    return ScanFrame(
        points_board=xyz_board[inside],
        points_camera=xyz_cam[inside],
        pixels_left=pix[inside],
        weights=precision_weights(xyz_cam[inside]),
        scanlines=scan[smooth][inside].astype(np.int64) if len(scan) else np.empty(0, np.int64),
        n_scanlines=n_lines,
        along_x=bool(left.along_x),
        n_pixels=int(len(px)),
        n_confirmed=int(confirmed.sum()),
        n_rejected_unconfirmed=n_unconfirmed,
        n_rejected_blob=n_blob,
        n_split=n_split,
        n_rejected_range=n_range,
        veto_shift_px=veto_shift,
        veto_note=veto_note,
        n_refined=int(refined.used[inside].sum()) if len(refined.used) else 0,
        refine_shift_mm=(float(np.median(np.abs(refined.shift_mm[inside & refined.used])))
                         if (inside & refined.used).any() else float("nan")),
        refine_share=(float(refined.share[inside & refined.used].mean())
                      if (inside & refined.used).any() else float("nan")),
        refine_note=refined.note,
        n_rejected_jump=n_jump,
        n_rejected_offside=n_offside,
        n_rejected_volume=int((~inside).sum()),
        veto_px=veto_px,
        rs_max_mm=rs_max,
        speed_mm_s=0.0 if motion is None else motion.speed_mm_s,
        spin_deg_s=0.0 if motion is None else motion.spin_deg_s,
        rs_note=None if motion is not None else (rs_note or "no motion estimate"),
    )


class PointCloud:
    """Accumulated scan points, in the board's frame, merged on a voxel grid.

    Board-frame rather than camera-frame on purpose: it is the one coordinate
    system that stays fixed while the subject turns on the board, so sweeps
    taken at different times land in the same space.

    Merged rather than appended: a sweep passes the same surface many times,
    and appending every frame's points piles up duplicates that carry their
    own noise each — the cloud looked furry and grew without bound. Each
    voxel (`voxel_mm`, 0.5 by default: well under the point noise, so
    nothing real is lost) holds the running mean of the points that fell in
    it. A surface seen ten times is one point, ten times less noisy; the
    export is one point per voxel; and a per-voxel normal has somewhere to
    live later.

    Bounds are kept running rather than recomputed: the panel shows them after
    every pair, and a min/max over the whole cloud cost 42 ms per pair at a
    million points — on the GUI thread, at the time.
    """

    def __init__(self, voxel_mm: float = 0.5) -> None:
        self.voxel_mm = float(voxel_mm)
        # Voxel key -> row. The rows are preallocated arrays grown geometrically
        # and filled in order, so an add costs the new points, not the cloud:
        # the previous chunk list concatenated everything per pair, 42 ms at a
        # million points, and that sat under the lock the detector threads
        # take to offer frames.
        self._index: dict[int, int] = {}
        # Positions are a WEIGHTED mean: each point counts for its precision
        # (`precision_weights`), so a close pass overrides a far one; `_count`
        # stays the plain number of hits, which is what "seen once" means.
        self._sum = np.empty((0, 3), np.float64)
        self._wsum = np.empty(0, np.float64)
        self._count = np.empty(0, np.int64)
        self._mean = np.empty((0, 3), np.float64)
        # Colour alongside, with a weight sum of its own: a sweep that
        # carried no colour image must not darken a voxel a coloured one lit.
        self._csum = np.empty((0, 3), np.float64)
        self._cwsum = np.empty(0, np.float64)
        self._rgb = np.empty((0, 3), np.uint8)
        self._coloured = False
        self._n = 0
        #: Bumped by every add and clear: what tells a cache built from
        #: `points()` that it is stale.
        self.version = 0
        self._lo = np.full(3, np.inf)
        self._hi = np.full(3, -np.inf)

    def __len__(self) -> int:
        return self._n

    def _keys(self, pts: np.ndarray) -> np.ndarray:
        """One integer per point naming its voxel. 2^21 cells per axis around
        the origin: ±524 m at 0.5 mm, unmasked — the scan volume is a disc."""
        ijk = np.floor(pts / self.voxel_mm).astype(np.int64) + (1 << 20)
        return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]

    def _reserve(self, n_new: int) -> None:
        need = self._n + n_new
        cap = len(self._count)
        if need <= cap:
            return
        cap = max(need, 2 * cap, 4096)
        for name, dtype in (("_sum", np.float64), ("_mean", np.float64),
                            ("_csum", np.float64), ("_rgb", np.uint8)):
            grown = np.zeros((cap, 3), dtype)
            grown[: self._n] = getattr(self, name)[: self._n]
            setattr(self, name, grown)
        for name, dtype in (("_count", np.int64), ("_wsum", np.float64),
                            ("_cwsum", np.float64)):
            grown = np.zeros(cap, dtype)
            grown[: self._n] = getattr(self, name)[: self._n]
            setattr(self, name, grown)

    def add(self, pts: np.ndarray, rgb: np.ndarray | None = None,
            weights: np.ndarray | None = None) -> None:
        """Merge points, their (N, 3) uint8 colours when there are any, and
        their weights (`precision_weights`; 1 each when not given)."""
        if not len(pts):
            return
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        w = (np.ones(len(pts)) if weights is None
             else np.maximum(np.asarray(weights, np.float64).ravel(), 1e-12))
        keys = self._keys(pts)
        uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        # In order of first appearance, so a cloud of distinct points reads
        # back the way it was added.
        order = np.argsort(first, kind="stable")
        uniq = uniq[order]
        inverse = np.argsort(order)[inverse]
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inverse, pts * w[:, None])
        wsums = np.bincount(inverse, weights=w, minlength=len(uniq))
        counts = np.bincount(inverse, minlength=len(uniq))
        rows = np.array([self._index.get(int(k), -1) for k in uniq])
        fresh = rows < 0
        n_new = int(fresh.sum())
        if n_new:
            self._reserve(n_new)
            start = self._n
            for offset, k in enumerate(uniq[fresh]):
                self._index[int(k)] = start + offset
            rows[fresh] = start + np.arange(n_new)
            self._sum[start: start + n_new] = 0.0
            self._wsum[start: start + n_new] = 0.0
            self._count[start: start + n_new] = 0
            self._csum[start: start + n_new] = 0.0
            self._cwsum[start: start + n_new] = 0.0
            self._rgb[start: start + n_new] = 0
            self._n += n_new
        self._sum[rows] += sums
        self._wsum[rows] += wsums
        self._count[rows] += counts
        self._mean[rows] = self._sum[rows] / self._wsum[rows, None]
        if rgb is not None:
            csums = np.zeros((len(uniq), 3))
            np.add.at(csums, inverse, np.asarray(rgb, np.float64).reshape(-1, 3) * w[:, None])
            self._csum[rows] += csums
            self._cwsum[rows] += wsums
            self._rgb[rows] = np.clip(
                np.rint(self._csum[rows] / self._cwsum[rows, None]), 0, 255)
            self._coloured = True
        np.minimum(self._lo, pts.min(axis=0), out=self._lo)
        np.maximum(self._hi, pts.max(axis=0), out=self._hi)
        self.version += 1

    def clear(self) -> None:
        self._index.clear()
        self._n = 0
        self._coloured = False
        self._lo[:] = np.inf
        self._hi[:] = -np.inf
        self.version += 1

    def points(self) -> np.ndarray:
        """One point per voxel: the mean of what fell in it. A view — copy
        before keeping it across an `add`."""
        return self._mean[: self._n]

    def counts(self) -> np.ndarray:
        """How many points fell into each voxel — how often it was seen. A
        view, like `points()`."""
        return self._count[: self._n]

    def weights(self) -> np.ndarray:
        """The precision behind each voxel: the sum of its points' weights."""
        return self._wsum[: self._n]

    def colors(self) -> np.ndarray | None:
        """RGB per voxel, uint8: the mean of what was read beside the stripe
        - or None while no sweep carried colour. A view, like `points()`."""
        return self._rgb[: self._n] if self._coloured else None

    def snapshot(self, max_n: int) -> tuple[np.ndarray, np.ndarray | None]:
        """`decimated`, with the matching colours (None without any)."""
        pts = self.points()
        stride = max(1, -(-len(pts) // max_n))
        rgb = self.colors()
        return pts[::stride].copy(), (None if rgb is None else rgb[::stride].copy())

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._n:
            return None
        return self._lo.copy(), self._hi.copy()

    def decimated(self, max_n: int) -> np.ndarray:
        """About `max_n` points spread over the whole cloud: every k-th voxel,
        so old and new sweeps are represented alike. This is what the eyes
        draw; projecting a million points per frame would cost more than the
        frame."""
        pts = self.points()
        stride = max(1, -(-len(pts) // max_n))
        # A copy: the eyes and the cloud view refresh when the object
        # changes, and the mean array above is updated in place.
        return pts[::stride].copy()

    def write_ply(self, path: str) -> int:
        """Write a binary little-endian PLY, one point per voxel. Returns the
        point count."""
        return write_ply(path, self.points(), self.colors())


#: How many points are put through a `cKDTree` query at a time. A neighbour
#: table is (chunk, k, 3) float64, so a million-point cloud queried whole
#: would ask for gigabytes for nothing; a chunk of this size holds tens of
#: megabytes and the tree is queried once per chunk, not once per point.
KD_CHUNK = 100_000


def normals_pca(points: np.ndarray, k: int = 16) -> np.ndarray:
    """A unit normal per (N, 3) point, from the plane its `k` nearest
    neighbours lie in.

    The neighbourhood's covariance has its smallest eigenvalue across the
    surface and the other two along it, so the matching eigenvector is the
    normal — a plane fit that needs no orientation, no grid and no ordering,
    which is what a scan cloud is. `k` is 16 rather than 6 or 8 because the
    laser cloud carries about a millimetre of noise and a small
    neighbourhood fits the noise instead of the surface.

    The SIGN is arbitrary here — a plane has two normals and PCA cannot
    choose between them. `orient_normals` is what makes them a surface's
    outward normals, and Poisson needs that.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(pts):
        return np.empty((0, 3))
    k = int(min(max(k, 3), len(pts)))
    tree = cKDTree(pts)
    out = np.empty_like(pts)
    for lo in range(0, len(pts), KD_CHUNK):
        hi = min(lo + KD_CHUNK, len(pts))
        _, idx = tree.query(pts[lo:hi], k=k, workers=-1)
        nb = pts[idx.reshape(hi - lo, -1)]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        # `eigh` returns eigenvalues ascending, so column 0 is the direction
        # the neighbourhood spreads least in: across the surface.
        out[lo:hi] = np.linalg.eigh(cov)[1][:, :, 0]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-12)


def orient_normals(normals: np.ndarray, points: np.ndarray,
                   camera_centres: np.ndarray) -> np.ndarray:
    """Flip each normal to face the camera centre nearest its point.

    PCA gives a normal's line, not its direction, and Poisson reconstructs
    the wrong side of the surface — or nothing at all — from normals that
    disagree with their neighbours. A scanned surface was seen from
    somewhere, so "outward" is "toward whichever camera saw it": the
    nearest centre is the best guess available, and on a convex sweep it is
    the right one everywhere. Returns a new array; the input is untouched.
    """
    n = np.asarray(normals, np.float64).reshape(-1, 3).copy()
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    cams = np.asarray(camera_centres, np.float64).reshape(-1, 3)
    if not len(n) or not len(cams):
        return n
    _, j = cKDTree(cams).query(pts, k=1, workers=-1)
    toward = cams[np.atleast_1d(j)] - pts
    flip = np.einsum("ij,ij->i", n, toward) < 0.0
    n[flip] *= -1.0
    return n


def write_ply(path: str, points: np.ndarray, rgb: np.ndarray | None = None,
              normals: np.ndarray | None = None) -> int:
    """Write (N, 3) points, their (N, 3) uint8 colours and their (N, 3)
    normals when given, as a binary little-endian PLY. Returns the point
    count. Binary because an ASCII writer loops in Python: a million points
    took seconds, on the GUI thread, behind the Export button.

    Property order is `x y z nx ny nz red green blue` — COLMAP's own order
    in `fused.ply`, so a file we write and a file we read look alike.
    """
    p = np.ascontiguousarray(np.asarray(points, np.float64).reshape(-1, 3).astype("<f4"))
    props = "property float x\nproperty float y\nproperty float z\n"
    fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    columns = [p[:, 0], p[:, 1], p[:, 2]]
    if normals is not None:
        # float32 like the coordinates: a normal is a direction, and the
        # eighth digit of one has never told anybody anything.
        props += "property float nx\nproperty float ny\nproperty float nz\n"
        nrm = np.asarray(normals, np.float64).reshape(-1, 3).astype("<f4")
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        columns += [nrm[:, 0], nrm[:, 1], nrm[:, 2]]
    if rgb is not None:
        # red/green/blue as uchar: what every viewer, and `read_ply`, expects.
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        col = np.asarray(rgb, np.uint8).reshape(-1, 3)
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        columns += [col[:, 0], col[:, 1], col[:, 2]]
    if len(fields) == 3:
        body = p.tobytes()
    else:
        rec = np.empty(len(p), dtype=fields)
        for (name, _), column in zip(fields, columns):
            rec[name] = column
        body = rec.tobytes()
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(p)}\n" + props + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(body)
    return len(p)


@dataclass
class Confident:
    """The cloud a viewer should trust — merged a step coarser, the lonely
    and the unconfirmed dropped — and how many went."""

    points: np.ndarray
    colours: np.ndarray | None
    #: How many observations stand behind each point.
    obs: np.ndarray
    n_lonely: int = 0
    n_flicker: int = 0


def _cell_keys(pts: np.ndarray, size_mm: float) -> np.ndarray:
    """One integer per point naming its `size_mm` cell — `PointCloud._keys`
    at any size."""
    ijk = np.floor(pts / size_mm).astype(np.int64) + (1 << 20)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def confident(points: np.ndarray, counts: np.ndarray, colours: np.ndarray | None = None,
              merge_mm: float = 1.0, cell_mm: float = 2.0, min_neighbours: int = 3,
              flicker_obs: int = 3, weights: np.ndarray | None = None) -> Confident:
    """Which voxels to trust, merged a step coarser.

    A scan passes the same surface many times, and the voxel grid keeps a
    count of how often each voxel was hit. Two kinds of voxel are not the
    subject:

      * **lonely** — fewer than `min_neighbours` other voxels within the
        3×3×3 block of `cell_mm` cells around it. A surface is never one
        voxel; a glint, a hand, a bounced reflection is.
      * **flicker** — seen once, where the voxels around it were seen
        `flicker_obs` times or more on average. The passes that confirmed
        its neighbours went past it and did not see it again: the point was
        never there. A voxel seen once in a region only swept once is not a
        flicker, just young, and is kept.

    What survives is merged on `merge_mm` cells, each voxel weighted by
    `weights` — its precision (`PointCloud.weights`), or its count when no
    weights are given — so a surface the 0.5 mm grid renders as a fuzz two
    or three voxels thick reads back as one confident point, placed where
    the close passes put it, with its observations added up. All of it is
    bincounts and one sort; a million voxels take a fraction of a second.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    cnt = np.asarray(counts, np.float64).ravel()
    wts = cnt if weights is None else np.maximum(np.asarray(weights, np.float64).ravel(), 1e-12)
    none_rgb = None if colours is None else np.empty((0, 3), np.uint8)
    if not len(pts):
        return Confident(np.empty((0, 3)), none_rgb, np.empty(0, np.int64))

    key = _cell_keys(pts, cell_mm)
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    vox = np.bincount(inv).astype(np.float64)
    obs = np.bincount(inv, weights=cnt)
    nb_vox = np.zeros(len(uniq))
    nb_obs = np.zeros(len(uniq))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                shifted = uniq + ((dx << 42) + (dy << 21) + dz)
                j = np.minimum(np.searchsorted(uniq, shifted), len(uniq) - 1)
                hit = uniq[j] == shifted
                nb_vox += np.where(hit, vox[j], 0.0)
                nb_obs += np.where(hit, obs[j], 0.0)
    others = nb_vox[inv] - 1.0                              # not counting itself
    around = (nb_obs[inv] - cnt) / np.maximum(others, 1.0)  # how often they were seen
    lonely = others < min_neighbours
    flicker = ~lonely & (cnt <= 1.0) & (around >= flicker_obs)
    keep = ~(lonely | flicker)
    n_lonely, n_flicker = int(lonely.sum()), int(flicker.sum())
    if not keep.any():
        return Confident(np.empty((0, 3)), none_rgb, np.empty(0, np.int64), n_lonely, n_flicker)

    p, c, wk = pts[keep], cnt[keep], wts[keep]
    mkey = _cell_keys(p, merge_mm)
    muniq, first, minv = np.unique(mkey, return_index=True, return_inverse=True)
    # First-appearance order, like the grid itself, so a cloud reads back
    # the way it was scanned.
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    minv = rank[minv.ravel()]
    m = len(muniq)
    w = np.bincount(minv, weights=wk, minlength=m)
    obs = np.bincount(minv, weights=c, minlength=m)
    merged = np.column_stack([np.bincount(minv, weights=p[:, i] * wk, minlength=m)
                              for i in range(3)]) / w[:, None]
    rgb = None
    if colours is not None:
        col = np.asarray(colours, np.float64).reshape(-1, 3)[keep]
        rgb = np.column_stack([np.bincount(minv, weights=col[:, i] * wk, minlength=m)
                               for i in range(3)]) / w[:, None]
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return Confident(merged, rgb, np.rint(obs).astype(np.int64), n_lonely, n_flicker)


_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """A PLY's vertices as (N, 3) float64 x/y/z and, when it carries them,
    (N, 3) uint8 colours — `read_ply_full` without the normals, for the
    callers that never wanted them."""
    xyz, rgb, _ = read_ply_full(path)
    return xyz, rgb


def read_ply_full(path: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """A PLY's vertices as (N, 3) float64 x/y/z and, when it carries them,
    (N, 3) uint8 colours and (N, 3) float64 normals. Binary little-endian
    (what `write_ply` writes, and what COLMAP's `fused.ply` is) and ASCII;
    other elements and properties are skipped, not refused.

    **Its scope is point clouds** — `fused.ply` and `merged.ply`. It does
    not read `mesh_texturer`'s output: that PLY carries faces and per-face
    texcoords, and the GLB step reads it with trimesh. Two files, two
    readers, said plainly so nobody widens this one into a mesh parser.
    """
    with open(path, "rb") as f:
        head = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no end_header")
            head += line
            if line.strip() == b"end_header":       # \r\n from a Windows editor is fine
                break
        lines = head.decode("ascii", "replace").split("\n")
        fmt = next((ln.split()[1] for ln in lines if ln.startswith("format")), "")
        elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        for ln in lines:
            words = ln.split()
            if not words:
                continue
            if words[0] == "element":
                elements.append((words[1], int(words[2]), []))
            elif words[0] == "property" and elements:
                if words[1] == "list":
                    elements[-1][2].append((words[4], "list:" + words[2] + ":" + words[3]))
                else:
                    elements[-1][2].append((words[2], words[1]))
        body = f.read()
    xyz = rgb = nrm = None
    offset = 0
    for name, count, props in elements:
        if any(t.startswith("list") for _, t in props):
            if name == "vertex":
                raise ValueError("vertex list properties are not supported")
            break                                   # faces etc. follow; done
        if fmt.startswith("binary_little"):
            dtype = np.dtype([(pn, "<" + _PLY_TYPES[pt]) for pn, pt in props])
            arr = np.frombuffer(body, dtype, count=count, offset=offset)
            offset += count * dtype.itemsize
        elif fmt == "ascii":
            rows = [r for r in body.decode("ascii", "replace").split("\n")[:count] if r.strip()]
            if len(rows) < count:
                raise ValueError(f"{name}: {len(rows)} rows for {count} declared")
            try:
                arr = np.array([[float(v) for v in r.split()] for r in rows])
            except ValueError as exc:
                raise ValueError(f"{name}: bad row — {exc}") from exc
            if arr.ndim != 2 or arr.shape[1] != len(props):
                raise ValueError(f"{name}: rows do not match the {len(props)} properties")
            arr = np.rec.fromarrays(list(arr.T), names=[pn for pn, _ in props])
            body = b"\n".join(body.split(b"\n")[count:])
        else:
            raise ValueError(f"unsupported PLY format {fmt!r}")
        if name == "vertex":
            names = set(arr.dtype.names or ())
            if not {"x", "y", "z"} <= names:
                raise ValueError("vertex element lacks x, y, z")
            xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
            if {"red", "green", "blue"} <= names:
                rgb = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.uint8)
            if {"nx", "ny", "nz"} <= names:
                nrm = np.column_stack([arr["nx"], arr["ny"], arr["nz"]]).astype(np.float64)
    if xyz is None:
        raise ValueError("no vertex element")
    return xyz, rgb, nrm


@dataclass
class MergeParams:
    """The ten thresholds the laser/dense merge is decided by.

    One dataclass so a threshold cannot be passed to one place and not
    another, and so `session.json` can carry the whole set verbatim and a
    run be reproduced from it. The candidate radius is deliberately NOT an
    eleventh field: it is `hypot(support_mm, max_normal_mm)` and would
    otherwise be settable inconsistently with the two numbers that define
    it.
    """

    #: The lateral radius support is counted in — the plane perpendicular
    #: to the local laser normal. It is what decides how wide a hole dense
    #: may fill: a gap narrower than about twice this is supported from
    #: both edges at once, so the laser keeps it.
    support_mm: float = 3.0
    #: Laser neighbours needed within `support_mm` laterally for the laser
    #: to count as having covered the surface under a dense point.
    support_min: int = 5
    #: How far along the normal a supported dense point may sit and still
    #: be a plausible measurement of the same surface. Beyond it the point
    #: hovers, and is a floater rather than hole fill. Also half the height
    #: of the support cylinder the candidate ball circumscribes.
    max_normal_mm: float = 10.0
    #: An unsupported dense point this close to the nearest laser point is
    #: on the rim of a hole — no lateral coverage, but the laser surface
    #: stops right here — and is kept.
    outlier_mm: float = 4.0
    #: The radius the dense cloud's own local density is counted in.
    density_mm: float = 1.5
    #: An unsupported dense point away from any rim is kept when its dense
    #: neighbour count is at least this fraction of the MEASURED median —
    #: a bar relative to what this dense cloud actually is, because an
    #: absolute one sat an order of magnitude below real dense density and
    #: admitted every speck it was written to reject.
    patch_frac: float = 0.3
    #: G1: the largest signed-residual median a gated cell may carry.
    #: 1.5 rather than 1.0 because PatchMatch itself carries a sub-mm
    #: systematic offset, and a limit tighter than the tool's own bias
    #: refuses correct runs.
    agree_mm: float = 1.5
    #: Supported points a cell needs before its median is gated at all.
    band_min: int = 500
    #: G1b: below this many supported points there is nothing to measure a
    #: bias over, and the merge refuses.
    min_supported: int = 2000
    #: The fraction G2 (overlap) and G3 (floaters) warn at. Both are
    #: warnings: dark and specular objects legitimately give the laser very
    #: little coverage, and refusing there would refuse the case dense
    #: exists to rescue.
    warn_frac: float = 0.25

    @property
    def candidate_radius_mm(self) -> float:
        """The smallest ball containing the support cylinder of radius
        `support_mm` and half-height `max_normal_mm` — the query radius,
        ≈ 10.44 mm at the defaults. Wide on purpose: a ball of `support_mm`
        makes the lateral test a no-op, since lateral distance never
        exceeds Euclidean distance."""
        return float(np.hypot(self.support_mm, self.max_normal_mm))


#: Candidates kept per dense point, nearest first. A point sitting in a
#: thicket of laser returns must not drag the whole thicket into its normal
#: estimate, and 64 is far past what `support_min` (5) ever needs.
CANDIDATE_CAP = 64
#: G1's z slabs: equal-HEIGHT quarters of the volume's z range, not count
#: quartiles — a count quartile moves with the cloud, and two runs are then
#: not comparable.
Z_SLABS = 4
#: G4 warns below this keep fraction: dense added nothing. The run is not
#: wrong, it is pointless, and the operator should know before spending
#: another hour.
KEEP_FLOOR = 0.01
#: The eight normal-direction octants, indexed by `(x >= 0) << 2 | (y >= 0)
#: << 1 | (z >= 0)`. The deadband at zero — `sign(x) = +` when `x` is
#: exactly 0 — keeps an axis-aligned face in one octant instead of
#: scattering it across two at the mercy of the last bit.
_OCTANTS = tuple(("+" if x else "-") + ("+" if y else "-") + ("+" if z else "-")
                 for x in (0, 1) for y in (0, 1) for z in (0, 1))
_CELL_KEYS = tuple(f"z{b + 1}:{o}" for b in range(Z_SLABS) for o in _OCTANTS)


@dataclass
class MergeCells:
    """G1's evidence: the signed residual's median per z-slab × normal-octant
    cell, and which cells were too thin to gate."""

    #: z slabs by normal octants.
    grid: tuple[int, int] = (Z_SLABS, 8)
    #: Cells holding at least `band_min` supported points, and therefore
    #: able to refuse the run.
    gated: int = 0
    #: The gated cell with the largest `|median|` — the one a refusal names
    #: first. Its median and count are `medians_mm[worst]` and
    #: `counts[worst]`; None when nothing was gated.
    worst: str | None = None
    #: One entry per GATED cell, keyed `z<band>:<sign pattern of n>`.
    medians_mm: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    #: Cells that held points but fewer than `band_min` of them. They carry
    #: no median and cannot refuse the run however far off they are.
    skipped: list[str] = field(default_factory=list)


@dataclass
class MergeStats:
    """Everything the merge measured, reported whether or not it refused."""

    params: MergeParams
    #: `hypot(support_mm, max_normal_mm)`, so a reader never has to
    #: recompute it.
    candidate_radius_mm: float
    laser_points: int
    #: The POST-CROP dense count — the points inside the `ScanVolume` — and
    #: the denominator of every fraction below.
    dense_points: int
    #: Dense points the laser laterally supports, with a residual inside
    #: `max_normal_mm`: dropped, and the population G1 measures.
    dense_supported: int
    #: Dense points kept as hole fill, by the rim rule or the patch rule.
    dense_kept: int
    #: Dropped as noise: supported but hovering past `max_normal_mm`, or
    #: unsupported, away from any rim and in no coherent dense patch.
    floaters: int
    #: `dense_supported / dense_points` and `dense_kept / dense_points`.
    #: Two DIFFERENT numbers, not one seen from two sides — they differ by
    #: the floater fraction, and the kept set is defined by absent support
    #: rather than by distance.
    agree_frac: float
    keep_frac: float
    cells: MergeCells
    #: The noise figure beside the cells: the 90th percentile of the
    #: supported points' unsigned residual. Reported, never gated on.
    p90_abs_residual_mm: float
    #: The kept set's own median distance to the laser cloud.
    kept_median_mm: float
    #: The density the patch bar was measured against: the median dense
    #: neighbour count within `density_mm` over the supported points.
    median_dense_neighbours: float
    refused: bool = False
    #: Which gate fired: "G1" (registration) or "G1b" (support population).
    refused_by: str | None = None
    #: The refusal, in full, naming the working escape.
    message: str | None = None
    #: G2, G3 and G4, which warn and never refuse.
    warnings: list[str] = field(default_factory=list)


def _grouped(n: int) -> str:
    """`2914501` as `2 914 501` — the refusal messages quote counts in the
    millions, and a wall of digits hides an order of magnitude."""
    return f"{int(n):,}".replace(",", " ")


def _neighbour_counts(pts: np.ndarray, radius: float) -> np.ndarray:
    """How many points of `pts` lie within `radius` of each — itself
    included, which cancels between the per-point count and the median it
    is compared against."""
    if not len(pts):
        return np.empty(0, np.int64)
    tree = cKDTree(pts)
    out = np.empty(len(pts), np.int64)
    for lo in range(0, len(pts), KD_CHUNK):
        hi = min(lo + KD_CHUNK, len(pts))
        out[lo:hi] = tree.query_ball_point(pts[lo:hi], radius, workers=-1,
                                           return_length=True)
    return out


def _support_and_residual(laser: np.ndarray, laser_n: np.ndarray, dense: np.ndarray,
                          params: MergeParams
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per dense point: is the laser under it, how far off the laser surface
    it sits along that surface's normal, how far the nearest laser point is,
    and the local laser normal — as `(supported, residual, nearest, normal)`.

    The candidate ball is `hypot(support_mm, max_normal_mm)` wide, capped at
    the `CANDIDATE_CAP` nearest, which is exactly `query(k=CANDIDATE_CAP,
    distance_upper_bound=r)`. Support is then counted LATERALLY — in the
    plane perpendicular to the mean neighbour normal — whatever the
    neighbours' offset along it. That last clause is the whole rule: it is
    what lets a point 4 mm above covered surface be recognised as sitting
    over covered surface, which no ball of `support_mm` could ever do.
    """
    n = len(dense)
    supported = np.zeros(n, bool)
    resid = np.zeros(n)
    nearest = np.full(n, np.inf)
    normal = np.zeros((n, 3))
    if not len(laser) or not n:
        return supported, resid, nearest, normal
    tree = cKDTree(laser)
    k = min(CANDIDATE_CAP, len(laser))
    for lo in range(0, n, KD_CHUNK):
        hi = min(lo + KD_CHUNK, n)
        p = dense[lo:hi]
        dist, idx = tree.query(p, k=k, distance_upper_bound=params.candidate_radius_mm,
                               workers=-1)
        dist = dist.reshape(len(p), -1)
        idx = idx.reshape(len(p), -1)
        hit = np.isfinite(dist)
        # A miss comes back as `len(laser)`, which would index out of range.
        idx = np.where(hit, idx, 0)
        m = hit.sum(axis=1)
        nearest[lo:hi] = dist[:, 0]              # inf when the ball is empty
        d = np.where(hit, dist, 0.0)
        q = laser[idx]                           # (chunk, k, 3)
        seen = hit[..., None]
        centroid = (q * seen).sum(axis=1) / np.maximum(m, 1)[:, None]
        nsum = (laser_n[idx] * seen).sum(axis=1)
        length = np.linalg.norm(nsum, axis=1, keepdims=True)
        # Opposing normals inside one ball cancel — two faces of a thin
        # wall, say. The nearest neighbour's own normal is then the only
        # honest local frame left.
        nrm = np.where(length > 1e-9, nsum / np.maximum(length, 1e-12), laser_n[idx[:, 0]])
        along = (np.einsum("ckj,cj->ck", q, nrm)
                 - np.einsum("cj,cj->c", p, nrm)[:, None])
        lateral_sq = np.maximum(d ** 2 - along ** 2, 0.0)
        close = hit & (lateral_sq <= params.support_mm ** 2)
        supported[lo:hi] = close.sum(axis=1) >= params.support_min
        resid[lo:hi] = np.where(m > 0, np.einsum("cj,cj->c", p - centroid, nrm), 0.0)
        normal[lo:hi] = nrm
    return supported, resid, nearest, normal


def _cells(z: np.ndarray, normal: np.ndarray, resid: np.ndarray, params: MergeParams,
           volume: ScanVolume) -> tuple[MergeCells, list[str]]:
    """G1's cells over the supported points, and the gated keys whose median
    exceeds `agree_mm`, worst first.

    Cells are z slab × normal octant. The octants are what make the gate
    work on a closed object: normals point outward, so a translation reads
    +δ on one side and −δ on the other, and z slabs alone average the two
    halves to a median of zero — a misregistration the gate would wave
    through. Split by normal direction and the halves land in different
    cells with opposite signs, and both fail.
    """
    out = MergeCells()
    if not len(z):
        return out, []
    step = (volume.height_mm - volume.floor_mm) / Z_SLABS
    band = np.clip(((z - volume.floor_mm) / max(step, 1e-9)).astype(np.int64), 0, Z_SLABS - 1)
    octant = (((normal[:, 0] >= 0).astype(np.int64) << 2)
              | ((normal[:, 1] >= 0).astype(np.int64) << 1)
              | (normal[:, 2] >= 0).astype(np.int64))
    cell = band * 8 + octant
    counts = np.bincount(cell, minlength=len(_CELL_KEYS))
    order = np.argsort(cell, kind="stable")
    start = np.concatenate([[0], np.cumsum(counts)])
    for c in np.flatnonzero(counts):
        key = _CELL_KEYS[c]
        if counts[c] < params.band_min:
            out.skipped.append(key)
            continue
        out.medians_mm[key] = float(np.median(resid[order[start[c]: start[c + 1]]]))
        out.counts[key] = int(counts[c])
    out.gated = len(out.medians_mm)
    if out.medians_mm:
        out.worst = max(out.medians_mm, key=lambda k: abs(out.medians_mm[k]))
    failing = sorted((k for k, v in out.medians_mm.items() if abs(v) > params.agree_mm),
                     key=lambda k: -abs(out.medians_mm[k]))
    return out, failing


#: How many failing cells a G1 refusal lists before saying how many more.
_CELLS_LISTED = 4


def _g1_message(cells: MergeCells, failing: list[str], p90: float,
                params: MergeParams, volume: ScanVolume) -> str:
    """The registration refusal, naming the cells and the working escape."""
    step = (volume.height_mm - volume.floor_mm) / Z_SLABS
    worst = failing[0]
    band = int(worst[1]) - 1
    lo = volume.floor_mm + band * step
    axes = " ".join(s + a for s, a in zip(worst.split(":")[1], "xyz"))
    lines = [
        f"merge refused: the clouds disagree by {cells.medians_mm[worst]:+.2f} mm in cell "
        f"{worst} (z {lo:.0f}-{lo + step:.0f} mm,",
        f"  normals {axes}) — {len(failing)} of {cells.gated} gated cells exceed the "
        f"{params.agree_mm:g} mm limit:",
    ]
    for key in failing[:_CELLS_LISTED]:
        lines.append(f"    {key}  {cells.medians_mm[key]:+.2f} mm over "
                     f"{_grouped(cells.counts[key])} points")
    if len(failing) > _CELLS_LISTED:
        lines[-1] += f"   … and {len(failing) - _CELLS_LISTED} more"
    lines += [
        "  Opposite normal octants disagreeing in SIGN at similar magnitude is a",
        "  translation; a bias growing across the z slabs is a rotation or a scale",
        f"  error. Neither is noise (p90 |residual| {p90:.1f} mm, which gates nothing).",
        "  Check the board pose and which pass the photos came from.",
        "  laser.ply and fused.ply are both intact.",
        "  Fix and re-run with --restart, or run --mode texture-only to mesh and",
        "  texture the laser cloud from what is already on disk.",
    ]
    return "\n".join(lines)


def _g1b_message(dense_supported: int, dense_points: int, params: MergeParams) -> str:
    """The support-population refusal. A uniform offset larger than the
    candidate radius empties every ball and reads exactly like this; a
    4-10 mm one does not, and G1 names its cells instead."""
    r = params.candidate_radius_mm
    return "\n".join([
        "merge refused: no overlap between the laser cloud and the dense cloud",
        f"  — only {_grouped(dense_supported)} dense points have lateral laser support "
        f"(floor {_grouped(params.min_supported)}, of",
        f"  {_grouped(dense_points)} in the volume). Every other point's {r:.1f} mm "
        "candidate ball came",
        "  back empty or too thin, so there is nothing to measure a bias over and",
        "  nothing here can be trusted as hole fill.",
        f"  A uniform offset LARGER than {r:.1f} mm looks exactly like this. A 4-10 mm",
        "  one does not: G1 sees that one and names the cells.",
        "  Same escape: --mode texture-only.",
    ])


def merge_clouds(laser_xyz: np.ndarray, laser_n: np.ndarray, dense_xyz: np.ndarray,
                 dense_n: np.ndarray, params: MergeParams, volume: ScanVolume
                 ) -> tuple[np.ndarray, np.ndarray, MergeStats]:
    """The laser cloud, plus whatever of the dense cloud fills what the laser
    never saw — as `(xyz, normals, MergeStats)`.

    The two clouds are registered by construction: both are placed through
    the same board poses, so no ICP is needed. But registration is not
    accuracy. The laser carries sub-millimetre points and the dense cloud
    1-2 mm ones, ten to a hundred times more of them, so an unweighted
    Poisson over the union would yield the DENSE surface wherever dense
    exists. **The laser wins**, and dense is admitted only where the laser
    has nothing to say. What "nothing to say" means is §2.7's rule, and it
    is the one thing three earlier designs got wrong:

      * a candidate ball of `hypot(support_mm, max_normal_mm)` — the
        smallest ball containing the support cylinder — capped at the
        `CANDIDATE_CAP` nearest;
      * **supported** = at least `support_min` of those neighbours within
        `support_mm` LATERALLY, in the plane perpendicular to the local
        laser normal, whatever their offset along it. Supported means the
        laser covered this surface, and the point is dropped: 0.2 mm off it
        or 4 mm above it, the laser says it better. Past `max_normal_mm`
        along the normal it is dropped as a floater instead, and never
        enters G1's statistics;
      * **unsupported** is a hole. Within `outlier_mm` of the nearest laser
        point it is the hole's rim and is kept; farther out it is kept when
        it sits in a coherent dense patch and dropped as a floater when it
        does not.

    Two gates refuse and three warn, and every number is reported either
    way. G1 gates the SIGNED residual median per z-slab × normal-octant
    cell, so zero-mean noise passes and a bias — a translation, a scale
    error, a rotation — does not. G1b refuses when almost nothing is
    supported, because then there is no bias to measure. On a refusal the
    laser cloud comes back untouched, which is what `--mode texture-only`
    would have produced anyway; the caller reads `stats.refused` and stops.

    `dense_xyz` is cropped to `volume` here — the same volume D1b cropped
    with, so this is a no-op on an already-cropped cloud and makes
    `dense_points` the post-crop count by construction. `volume` also fixes
    G1's four z slabs to equal quarters of `[floor_mm, height_mm]`, never
    the cloud's own extent, so two runs are comparable.
    """
    laser = np.asarray(laser_xyz, np.float64).reshape(-1, 3)
    ln = np.asarray(laser_n, np.float64).reshape(-1, 3)
    dense = np.asarray(dense_xyz, np.float64).reshape(-1, 3)
    dn = np.asarray(dense_n, np.float64).reshape(-1, 3)
    inside = volume.contains(dense)
    dense, dn = dense[inside], dn[inside]
    n_dense = len(dense)

    supported, resid, nearest, normal = _support_and_residual(laser, ln, dense, params)
    # The G1 population: supported AND a plausible measurement of the
    # surface it is supported by. A supported point hovering past
    # `max_normal_mm` is a floater and must not colour the bias.
    measured = supported & (np.abs(resid) <= params.max_normal_mm)
    dense_supported = int(measured.sum())
    floater = supported & ~measured
    p90 = (float(np.percentile(np.abs(resid[measured]), 90)) if dense_supported
           else float("nan"))

    def _stats(n_kept: int, n_float: int, cells: MergeCells, median_nb: float,
               kept_median: float) -> MergeStats:
        return MergeStats(
            params=params, candidate_radius_mm=params.candidate_radius_mm,
            laser_points=len(laser), dense_points=n_dense,
            dense_supported=dense_supported, dense_kept=n_kept, floaters=n_float,
            agree_frac=dense_supported / n_dense if n_dense else 0.0,
            keep_frac=n_kept / n_dense if n_dense else 0.0,
            cells=cells, p90_abs_residual_mm=p90, kept_median_mm=kept_median,
            median_dense_neighbours=median_nb)

    # G1b before the patch rule: the patch bar is a median over the
    # supported set, and there is no such median when that set is empty.
    # Nothing is classified beyond the support test on this path, so
    # `floaters` counts only the hovering points the test itself named and
    # the three-way sum is short by the unjudged remainder — which is the
    # honest report of a merge that stopped before judging them.
    if dense_supported < params.min_supported:
        stats = _stats(0, int(floater.sum()), MergeCells(), float("nan"), float("nan"))
        stats.refused, stats.refused_by = True, "G1b"
        stats.message = _g1b_message(dense_supported, n_dense, params)
        return laser, ln, stats

    counts = _neighbour_counts(dense, params.density_mm)
    median_nb = float(np.median(counts[measured]))
    unsupported = ~supported
    rim = unsupported & (nearest <= params.outlier_mm)
    patch = unsupported & ~rim & (counts >= params.patch_frac * median_nb)
    kept = rim | patch

    kept_median = float("nan")
    if kept.any() and len(laser):
        kept_median = float(np.median(cKDTree(laser).query(dense[kept], k=1,
                                                           workers=-1)[0]))
    cells, failing = _cells(dense[measured][:, 2], normal[measured], resid[measured],
                            params, volume)
    # Every dense point is now judged, so the three-way sum closes:
    # supported + kept + floaters == dense_points.
    stats = _stats(int(kept.sum()), int((floater | (unsupported & ~kept)).sum()),
                   cells, median_nb, kept_median)
    # The three warnings are collected before G1 decides, because they say
    # something about a refused run too — and none of them can refuse it.
    if stats.agree_frac < params.warn_frac:
        stats.warnings.append(
            f"low overlap: {stats.agree_frac:.0%} of the dense cloud has lateral laser "
            f"support (under {params.warn_frac:.0%}) — a dark or specular object gives "
            "the laser little to work with, and the merge is not refused for it")
    floater_frac = stats.floaters / n_dense if n_dense else 0.0
    if floater_frac > params.warn_frac:
        stats.warnings.append(
            f"noisy dense cloud: {floater_frac:.0%} of it is floaters (over "
            f"{params.warn_frac:.0%}) — a quality signal, not a correctness one")
    if stats.keep_frac < KEEP_FLOOR:
        stats.warnings.append(
            f"dense added nothing: it filled {stats.keep_frac:.2%} of itself into the "
            f"laser cloud (under {KEEP_FLOOR:.0%}) — the run is not wrong, it is "
            "pointless")

    if failing:
        stats.refused, stats.refused_by = True, "G1"
        stats.message = _g1_message(cells, failing, p90, params, volume)
        return laser, ln, stats
    return (np.vstack([laser, dense[kept]]), np.vstack([ln, dn[kept]]), stats)


class CloudOverlay:
    """The cloud as the eyes draw it: a decimated, board-frame snapshot.

    Written by the scan thread, read by both detector threads. The array is
    swapped whole and never mutated after publishing, so readers need no lock:
    they see either the previous snapshot or the new one, each consistent.
    """

    def __init__(self) -> None:
        self._pts = np.empty((0, 3), np.float64)
        self._rgb: np.ndarray | None = None

    def publish(self, pts: np.ndarray, rgb: np.ndarray | None = None) -> None:
        # Colours first, points second: a reader that saw the new points
        # and the old colours would index past the end; the other way round
        # it merely draws a stale colour for one frame.
        self._rgb = rgb
        self._pts = pts

    def points(self) -> np.ndarray:
        return self._pts

    def colors(self) -> np.ndarray | None:
        """Matching (M, 3) uint8 colours, or None while the scan has none."""
        rgb = self._rgb
        return rgb if rgb is not None and len(rgb) == len(self._pts) else None
