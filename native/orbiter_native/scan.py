"""Scanning: the laser plane meets each left-eye ray; the right eye confirms.

One sweep of the pipeline, per frame pair:

  1. Each eye reports the pixels it considers stripe, with weights
     (`laser.find_stripe_pixels`) — no shape assumed, no centroid yet.
  2. Every LEFT stripe pixel's ray is intersected with the calibrated laser
     plane. That is a 3D candidate.
  3. The candidate is projected into the RIGHT eye. If the right eye saw no
     stripe there (within a few pixels), the candidate is dropped.
  4. The surviving pixels are averaged per scanline, weighted by stripe
     intensity — the sub-pixel centroid — and the centroid's ray meets the
     plane for the point that is kept.
  5. Keep it only if it lies inside a cylinder above the board, expressed in
     the BOARD's frame so the cylinder stays put while the board defines "up".

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

from .laser import StripePixels
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
    volume: ScanVolume = field(default_factory=ScanVolume)


@dataclass
class ScanFrame:
    """What one frame pair contributed, and why the rest was dropped."""

    points_board: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), np.float64))
    points_camera: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), np.float64))
    #: Scanlines (columns or rows) holding any stripe in the left eye.
    n_scanlines: int = 0
    #: Stripe pixels in the left eye, and how many the right eye confirmed.
    n_pixels: int = 0
    n_confirmed: int = 0
    #: Scanlines none of whose pixels the right eye confirmed.
    n_rejected_unconfirmed: int = 0
    n_rejected_volume: int = 0
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
    if ok.any():
        uv = rig.project_right(cand[ok])
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

    # One sub-pixel centroid per scanline, over confirmed pixels only.
    key = left.x if left.along_x else left.y
    across = left.y if left.along_x else left.x
    n_lines = int(len(np.unique(key)))
    weight = left.w.astype(np.float64) * confirmed
    length = int(key.max()) + 1
    total = np.bincount(key, weights=weight, minlength=length)
    moment = np.bincount(key, weights=weight * across, minlength=length)
    live = total > 0
    scan = np.flatnonzero(live).astype(np.float64)
    pos = moment[live] / total[live]
    centroids = (np.stack([scan, pos], axis=1) if left.along_x
                 else np.stack([pos, scan], axis=1))
    n_unconfirmed = n_lines - int(live.sum())

    xyz_cam = (_on_plane(rig.left_k, plane, centroids) if len(centroids)
               else np.empty((0, 3)))
    finite = np.isfinite(xyz_cam).all(axis=1)
    xyz_cam = xyz_cam[finite]
    rows = centroids[finite, 1] if len(centroids) else np.empty(0)

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
        n_scanlines=n_lines,
        n_pixels=int(len(px)),
        n_confirmed=int(confirmed.sum()),
        n_rejected_unconfirmed=n_unconfirmed,
        n_rejected_volume=int((~inside).sum()),
        rs_max_mm=rs_max,
        speed_mm_s=0.0 if motion is None else motion.speed_mm_s,
        spin_deg_s=0.0 if motion is None else motion.spin_deg_s,
        rs_note=None if motion is not None else (rs_note or "no motion estimate"),
    )


class PointCloud:
    """Accumulated scan points, in the board's frame.

    Board-frame rather than camera-frame on purpose: it is the one coordinate
    system that stays fixed while the subject turns on the board, so sweeps
    taken at different times land in the same space.

    Bounds are kept running rather than recomputed: the panel shows them after
    every pair, and a min/max over the whole cloud cost 42 ms per pair at a
    million points — on the GUI thread, at the time.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._lo = np.full(3, np.inf)
        self._hi = np.full(3, -np.inf)

    def __len__(self) -> int:
        return self._n

    def add(self, pts: np.ndarray) -> None:
        if len(pts):
            pts = np.asarray(pts, np.float64)
            self._chunks.append(pts)
            self._n += len(pts)
            np.minimum(self._lo, pts.min(axis=0), out=self._lo)
            np.maximum(self._hi, pts.max(axis=0), out=self._hi)

    def clear(self) -> None:
        self._chunks.clear()
        self._n = 0
        self._lo[:] = np.inf
        self._hi[:] = -np.inf

    def points(self) -> np.ndarray:
        if not self._chunks:
            return np.empty((0, 3), np.float64)
        return np.concatenate(self._chunks, axis=0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._n:
            return None
        return self._lo.copy(), self._hi.copy()

    def decimated(self, max_n: int) -> np.ndarray:
        """About `max_n` points spread over the whole cloud.

        Every k-th point of every chunk, so old and new sweeps are represented
        alike. This is what the eyes draw: projecting a million points per
        frame would cost more than the frame.
        """
        if not self._chunks:
            return np.empty((0, 3), np.float64)
        stride = max(1, -(-self._n // max_n))
        if stride == 1:
            return self.points()
        return np.concatenate([c[::stride] for c in self._chunks], axis=0)

    def write_ply(self, path: str) -> int:
        """Write a binary little-endian PLY. Returns the point count.

        Binary because an ASCII writer loops in Python: a million points took
        seconds, on the GUI thread, behind the Export button.
        """
        p = np.ascontiguousarray(self.points().astype("<f4"))
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(p)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "end_header\n")
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(p.tobytes())
        return len(p)


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
