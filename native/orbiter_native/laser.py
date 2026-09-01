"""Laser stripe on the ChArUco board — the input to camera↔laser calibration.

What this produces, per frame and per eye: subpixel points where the laser
crosses the board, and the straight line they fit. That is exactly the payload
the laser-plane solve consumes — back-project each point, intersect it with the
board plane, accumulate across poses, fit the plane the laser sweeps.

Three decisions, each measured on live frames from this rig rather than assumed:

**Colour, not luminance.** The board is black and white, the laser is red. On a
frame from this rig there are 68142 board pixels brighter than gray 170 (the
white squares) and only 31 of them survive a redness threshold of 50. A
luminance threshold does not find the stripe on this board — it finds the white
squares. `redness = r - max(g, b)` has an on-board median of 0 against a stripe
peaking at 109.

**Restricted to the board.** The stripe continues off the board onto the
workbench, and those points are not on the board plane, so they would poison
the solve. The mask is the convex hull of the DETECTED ChArUco corners, which
is deliberately conservative: on a circular board most corners are missing and
the hull covers less than the physical disc, discarding usable signal at the
edges. Fewer, certainly-valid points beat more points of unknown provenance.

**Robust fit, not least squares.** A plane meeting a plane is a line, so the
stripe must be straight in the image and a line fit is both a denoiser and a
validity check. But the per-column centroid wanders where the stripe crosses a
dark square. On a real frame, RANSAC then total least squares gave 0.67 px RMS
against 1.88 px for a plain fit over the same points — 2.8× — and the RMS is
what tells the operator whether this frame is worth keeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class LaserParams:
    """Tuning for `find_laser_line`."""

    #: Minimum `r - max(g, b)` for a pixel to be stripe candidate. Measured on
    #: this rig: on-board p99 of redness is 40 and the stripe peaks near 110,
    #: so 40-60 sits in the gap. Below ~30 the wooden bench starts to qualify.
    redness_min: int = 45
    #: RANSAC inlier band, in pixels either side of the candidate line.
    ransac_tol_px: float = 1.5
    ransac_iters: int = 60
    #: Below this many inliers the frame is not a usable calibration sample —
    #: a couple of stray specular pixels can always be fitted by some line.
    min_inliers: int = 20
    #: A fit worse than this is not a straight stripe; report it as not usable
    #: rather than handing the solve a line through a smear.
    max_rms_px: float = 2.0


@dataclass
class LaserLine:
    """A fitted stripe, in ORIGINAL (pre-orientation) image coordinates."""

    #: All per-scanline centroids considered, (N, 2) as (x, y).
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    #: Boolean mask over `points` — which ones the line fit accepted.
    inliers: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    #: A point on the fitted line (the inlier centroid) and a unit direction.
    point: np.ndarray | None = None
    direction: np.ndarray | None = None
    rms_px: float = float("nan")
    ms: float = 0.0
    #: Why the frame is not usable, or None when it is.
    reason: str | None = "no data"

    @property
    def ok(self) -> bool:
        return self.reason is None

    @property
    def inlier_points(self) -> np.ndarray:
        """The subpixel stripe points the calibration should consume."""
        if self.point is None or self.inliers.size == 0:
            return np.empty((0, 2), np.float32)
        return self.points[self.inliers]

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum()) if self.inliers.size else 0

    @property
    def angle_deg(self) -> float:
        if self.direction is None:
            return float("nan")
        return float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))

    def endpoints(self, length: float = 4000.0) -> tuple[tuple[int, int], tuple[int, int]]:
        """Two far-apart points on the line, for drawing."""
        assert self.point is not None and self.direction is not None
        a = self.point - self.direction * length
        b = self.point + self.direction * length
        return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def board_mask(corners: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Convex hull of the detected ChArUco corners, as a boolean mask.

    Returns None when there are too few corners to bound a region — in which
    case there is no board to restrict to and no calibration sample to take.
    """
    if corners is None or len(corners) < 3:
        return None
    hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.int32))
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def redness(bgr: np.ndarray) -> np.ndarray:
    """How red each pixel is relative to its other channels, 0..255.

    Not the red channel: a white square has a high red channel too. The
    difference against the strongest of the other two is what separates a red
    stripe from anything neutral, however bright.
    """
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    return np.clip(r - np.maximum(b, g), 0, 255).astype(np.uint8)


def _centroids(red: np.ndarray, keep: np.ndarray, along_x: bool) -> np.ndarray:
    """One intensity-weighted centroid per scanline, vectorised over all of them.

    `along_x` scans columns (a roughly horizontal stripe); otherwise rows. The
    axis matters: a near-vertical stripe has many rows sharing one column, and
    scanning columns there would average the whole stripe into one useless
    point per column.
    """
    w = np.where(keep, red.astype(np.float32), 0.0)
    if not along_x:
        w = w.T
    total = w.sum(axis=0)
    live = total > 0
    if not live.any():
        return np.empty((0, 2), np.float32)
    idx = np.arange(w.shape[0], dtype=np.float32)[:, None]
    pos = np.zeros_like(total)
    np.divide((w * idx).sum(axis=0), total, out=pos, where=live)
    scan = np.flatnonzero(live).astype(np.float32)
    across = pos[live]
    return (np.stack([scan, across], 1) if along_x
            else np.stack([across, scan], 1)).astype(np.float32)


def _fit_tls(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Total least squares line through `pts` → (point, unit direction, rms).

    Total least squares rather than a y=ax+b regression because the stripe can
    sit at any angle, and a vertical one has no finite slope.
    """
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    direction = vt[0] / np.linalg.norm(vt[0])
    normal = np.array([-direction[1], direction[0]], np.float64)
    rms = float(np.sqrt((((pts - centre) @ normal) ** 2).mean()))
    return centre, direction, rms


def _ransac(pts: np.ndarray, p: LaserParams, seed: int) -> np.ndarray:
    """Boolean inlier mask for the best two-point consensus line.

    Seeded per call: a detector that silently changed its mind between two runs
    on the same frame would make every downstream measurement unreproducible.
    """
    rng = np.random.default_rng(seed)
    best = np.zeros(len(pts), bool)
    for _ in range(p.ransac_iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        length = float(np.hypot(d[0], d[1]))
        if length < 1e-6:
            continue
        normal = np.array([-d[1], d[0]], np.float64) / length
        inl = np.abs((pts - pts[i]) @ normal) <= p.ransac_tol_px
        if inl.sum() > best.sum():
            best = inl
    return best


def find_laser_line(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    p: LaserParams = LaserParams(),
    seed: int = 0,
) -> LaserLine:
    """Find the laser stripe within `mask` and fit a straight line to it.

    `mask` is normally `board_mask(...)`. Passing None searches the whole frame,
    which is fine for aiming the laser but produces points that are NOT a
    calibration sample — off-board points do not lie on the board plane.
    """
    t0 = time.perf_counter()

    def done(line: LaserLine) -> LaserLine:
        line.ms = (time.perf_counter() - t0) * 1000.0
        return line

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("find_laser_line needs a colour frame; the laser is red")
    if mask is not None and not mask.any():
        return done(LaserLine(reason="board not visible"))

    # Work inside the mask's bounding box only. Computing redness over the full
    # frame costs ~3.1 ms at 720p against ~0.7 ms over a typical board bbox,
    # and every pixel outside is discarded anyway.
    if mask is None:
        y0, x0 = 0, 0
        sub_bgr, sub_keep = bgr, None
    else:
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        sub_bgr = bgr[y0:y1, x0:x1]
        sub_keep = mask[y0:y1, x0:x1]

    red = redness(sub_bgr)
    keep = red >= p.redness_min
    if sub_keep is not None:
        keep &= sub_keep
    if not keep.any():
        return done(LaserLine(reason="no stripe above the redness threshold"))

    # Scan across the stripe's long axis, decided from the lit pixels themselves
    # rather than assumed.
    ys, xs = np.nonzero(keep)
    along_x = (xs.max() - xs.min()) >= (ys.max() - ys.min())

    pts = _centroids(red, keep, along_x)
    if len(pts) < 2:
        return done(LaserLine(points=pts, reason="too few stripe points"))
    pts = pts + np.array([x0, y0], np.float32)          # back to full-frame coords

    inliers = _ransac(pts, p, seed)
    n = int(inliers.sum())
    if n < max(2, p.min_inliers):
        return done(LaserLine(points=pts, inliers=inliers,
                              reason=f"only {n} inliers (need {p.min_inliers})"))

    centre, direction, rms = _fit_tls(pts[inliers])
    line = LaserLine(points=pts, inliers=inliers, point=centre,
                     direction=direction, rms_px=rms)
    if rms > p.max_rms_px:
        line.reason = f"fit RMS {rms:.2f} px — not a straight stripe"
        return done(line)
    line.reason = None
    return done(line)


def draw(bgr: np.ndarray, line: LaserLine, mask_hull: np.ndarray | None = None) -> None:
    """Overlay the fit on an oriented BGR frame, in place.

    Inliers green, rejected points red, the fitted line cyan — so a bad frame
    is recognisable at a glance rather than only in the numbers.
    """
    if mask_hull is not None and len(mask_hull):
        cv2.polylines(bgr, [mask_hull], True, (0, 190, 255), 1)
    if line.points.size:
        pts = np.rint(line.points).astype(np.int32)
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < bgr.shape[1]) & \
             (pts[:, 1] >= 0) & (pts[:, 1] < bgr.shape[0])
        inl = line.inliers if line.inliers.size == len(pts) else np.zeros(len(pts), bool)
        for colour, sel in (((60, 60, 235), ok & ~inl), ((80, 235, 80), ok & inl)):
            bgr[pts[sel, 1], pts[sel, 0]] = colour
    if line.point is not None and line.direction is not None:
        a, b = line.endpoints()
        cv2.line(bgr, a, b, (235, 235, 60), 1, cv2.LINE_AA)
