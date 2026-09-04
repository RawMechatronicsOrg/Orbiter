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

from .laser import StripePixels, stripe_centroids
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
    n_rejected_volume: int = 0
    #: How far, in px along the scanline, the right eye's stripe sits from
    #: where the left eye's candidates project into its frame. NaN when the
    #: eyes share no scanline. See `_veto_offset`.
    veto_px: float = float("nan")
    reason: str | None = None
    #: Rolling shutter: the largest shift the per-row poses made to a kept
    #: point, the board's speed the twist implied, or why none was applied.
    rs_max_mm: float = 0.0
    speed_mm_s: float = 0.0
    spin_deg_s: float = 0.0
    rs_note: str | None = None

    @property
    def n_kept(self) -> int:
        return len(self.points_board)


def _on_plane(k, plane: LaserPlane, pixels: np.ndarray) -> np.ndarray:
    """Where each (N, 2) left-eye pixel's ray meets the laser plane, in the
    left camera's frame. NaN where the ray runs parallel or away."""
    d = rays(pixels, k)
    return plane.intersect_rays(np.zeros_like(d), d)


def _veto_offset(right: StripePixels, uv: np.ndarray) -> float:
    """Median signed distance, px, from where the left eye puts the stripe in
    the right frame to where the right eye actually has it.

    The veto asks whether those coincide within `confirm_px`; this says by
    how much they miss, which is what tells a calibration that cannot scan
    from a scene with nothing in it. Intrinsics, sheet and pair geometry that
    disagree put the stripe tens of pixels from where the other eye sees it,
    and then no amount of stripe will ever confirm — while every count on the
    panel reads exactly as it would with the laser switched off. NaN when the
    two eyes share no scanline.
    """
    if not len(uv) or not len(right.x):
        return float("nan")
    w, h = right.wh
    n = w if right.along_x else h
    key_px = (right.x if right.along_x else right.y).astype(np.int64)
    across_px = (right.y if right.along_x else right.x).astype(np.float64)
    weight = np.maximum(right.w.astype(np.float64), 1.0)
    total = np.bincount(key_px, weights=weight, minlength=n)
    moment = np.bincount(key_px, weights=weight * across_px, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        seen = moment / total                      # NaN on scanlines with none
    key = np.rint(uv[:, 0] if right.along_x else uv[:, 1])
    across = uv[:, 1] if right.along_x else uv[:, 0]
    on = np.isfinite(key) & (key >= 0) & (key < n)
    if not on.any():
        return float("nan")
    off = across[on] - seen[key[on].astype(np.int64)]
    off = off[np.isfinite(off)]
    return float(np.median(off)) if len(off) else float("nan")


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
    if ok.any():
        uv = rig.project_right(cand[ok])
        veto_px = _veto_offset(right, uv)
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

    # The reach: the baseline is the one line both cameras share, and the
    # subject sits a known distance from it whatever the board does.
    lo, hi = params.range_mm
    reach = _reach_mm(xyz_cam, rig)
    in_range = (reach >= lo) & (reach <= hi)
    n_range = int((~in_range).sum())
    xyz_cam, rows, scan = xyz_cam[in_range], rows[in_range], scan[in_range]

    # No jumps along the stripe.
    smooth = _not_a_jump(xyz_cam, scan, params.jump_mm)
    n_jump = int((~smooth).sum())
    xyz_cam, rows = xyz_cam[smooth], rows[smooth]

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
        scanlines=scan[smooth][inside].astype(np.int64) if len(scan) else np.empty(0, np.int64),
        n_scanlines=n_lines,
        along_x=bool(left.along_x),
        n_pixels=int(len(px)),
        n_confirmed=int(confirmed.sum()),
        n_rejected_unconfirmed=n_unconfirmed,
        n_rejected_blob=n_blob,
        n_split=n_split,
        n_rejected_range=n_range,
        n_rejected_jump=n_jump,
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
        self._sum = np.empty((0, 3), np.float64)
        self._count = np.empty(0, np.int64)
        self._mean = np.empty((0, 3), np.float64)
        self._n = 0
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
        for name in ("_sum", "_mean"):
            grown = np.empty((cap, 3), np.float64)
            grown[: self._n] = getattr(self, name)[: self._n]
            setattr(self, name, grown)
        count = np.zeros(cap, np.int64)
        count[: self._n] = self._count[: self._n]
        self._count = count

    def add(self, pts: np.ndarray) -> None:
        if not len(pts):
            return
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        keys = self._keys(pts)
        uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        # In order of first appearance, so a cloud of distinct points reads
        # back the way it was added.
        order = np.argsort(first, kind="stable")
        uniq = uniq[order]
        inverse = np.argsort(order)[inverse]
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inverse, pts)
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
            self._count[start: start + n_new] = 0
            self._n += n_new
        self._sum[rows] += sums
        self._count[rows] += counts
        self._mean[rows] = self._sum[rows] / self._count[rows, None]
        np.minimum(self._lo, pts.min(axis=0), out=self._lo)
        np.maximum(self._hi, pts.max(axis=0), out=self._hi)

    def clear(self) -> None:
        self._index.clear()
        self._n = 0
        self._lo[:] = np.inf
        self._hi[:] = -np.inf

    def points(self) -> np.ndarray:
        """One point per voxel: the mean of what fell in it. A view — copy
        before keeping it across an `add`."""
        return self._mean[: self._n]

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
        point count. Binary because an ASCII writer loops in Python: a million
        points took seconds, on the GUI thread, behind the Export button."""
        p = np.ascontiguousarray(self.points().astype("<f4"))
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(p)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "end_header\n")
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(p.tobytes())
        return len(p)


_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """A PLY's vertices as (N, 3) float64 x/y/z and, when it carries them,
    (N, 3) uint8 colours. Binary little-endian (what `write_ply` writes) and
    ASCII; other elements and properties are skipped, not refused."""
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
    xyz = rgb = None
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
    if xyz is None:
        raise ValueError("no vertex element")
    return xyz, rgb


class CloudOverlay:
    """The cloud as the eyes draw it: a decimated, board-frame snapshot.

    Written by the scan thread, read by both detector threads. The array is
    swapped whole and never mutated after publishing, so readers need no lock:
    they see either the previous snapshot or the new one, each consistent.
    """

    def __init__(self) -> None:
        self._pts = np.empty((0, 3), np.float64)

    def publish(self, pts: np.ndarray) -> None:
        self._pts = pts

    def points(self) -> np.ndarray:
        return self._pts
