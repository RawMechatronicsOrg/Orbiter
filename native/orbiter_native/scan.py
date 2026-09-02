"""Scanning: laser points seen by both eyes, triangulated, kept inside a box.

One sweep of the pipeline, per frame pair:

  1. Each eye reports its stripe as per-scanline subpixel centroids, with NO
     shape assumed (`laser.find_laser_points`).
  2. For every stripe point in the LEFT image, its match is where that point's
     epipolar line crosses the right eye's stripe — the actual observed
     polyline, segment by segment.
  3. Triangulate.
  4. Keep it only if it lies inside a box above the board, expressed in the
     BOARD's frame so the box stays put while the board defines "up".

**The stripe is not a straight line and must not be modelled as one.** It is
straight only while it falls on the flat calibration board. On a subject it is
a broken curve that steps at every depth discontinuity, and that shape IS the
measurement. Measured on a real scanning frame from this rig — a drill on the
bench — a straight-line fit kept 126 of 649 stripe points and discarded 523;
those 523 were the object. Hence step 2 intersects the observed polyline rather
than a fitted line, and hence scanning does not mask to the board's outline:
the subject stands above the board, so most of the interesting stripe falls
outside it. Rejecting what is not wanted is the 3D volume filter's job, which
it can do because it works in millimetres.

**What confirms a point.** Because the match is constructed as the crossing of
an epipolar line with an actual observed segment, it is by definition somewhere
the right camera saw laser. Reprojection error cannot add to that — the two
rays meet by construction, so it is ~1e-13 px however wrong a match is. The
real risk here is AMBIGUITY: a polyline can cross one epipolar line more than
once, when the stripe is at two different depths along the same ray. Those
points are genuinely undecidable from geometry alone and are dropped rather
than guessed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .laser import LaserPoints
from .laserplane import LaserPlane
from .stereo import StereoRig

log = logging.getLogger("orbiter_native.scan")


@dataclass(frozen=True)
class ScanVolume:
    """The box, in board coordinates, in millimetres.

    The board frame from `cvcore.estimate_pose` has its origin at the board's
    centre and z pointing out of the printed face, so `height_mm` is "above
    the board" and the x/y extent is centred on the board itself. (OpenCV's
    raw frame has the origin at a corner and z pointing into the board: a box
    like this one in that frame selected the space behind the board, minus
    the far half of the board.)
    """

    height_mm: float = 400.0        # the 40 cm cube
    half_width_mm: float = 200.0
    #: Points below this are the board's own surface — the calibration target,
    #: not the subject.
    floor_mm: float = 3.0

    def contains(self, xyz_board: np.ndarray) -> np.ndarray:
        """Boolean mask over (N, 3) points expressed in the board frame."""
        if not len(xyz_board):
            return np.zeros(0, bool)
        x, y, z = xyz_board[:, 0], xyz_board[:, 1], xyz_board[:, 2]
        return (
            (z >= self.floor_mm) & (z <= self.height_mm)
            & (np.abs(x) <= self.half_width_mm)
            & (np.abs(y) <= self.half_width_mm)
        )


@dataclass
class ScanParams:
    """Acceptance thresholds for one frame's points."""

    #: Two consecutive right-eye centroids are treated as one segment only if
    #: their scan coordinates are this close. The stripe breaks where the
    #: subject does, and interpolating across a break would invent a surface
    #: spanning the gap.
    max_segment_gap_px: float = 4.0
    #: A stripe crossing an epipolar line at too shallow an angle gives an
    #: ill-conditioned intersection: a small error along the line moves the
    #: match a long way. Degrees.
    min_intersection_deg: float = 8.0
    #: Reprojection sanity only. Near-zero by construction — see the module
    #: note on why this cannot be the both-cameras-agree test.
    max_reproj_px: float = 1.5
    #: How far a triangulated point may sit from the laser plane, in mm.
    #:
    #: This is the ONE independent check available. Stereo correspondence is
    #: built by intersecting an epipolar line with the other eye's observation,
    #: so the rays meet by construction whatever the pairing; only the plane
    #: was not assumed by that construction. Ignored when no plane is
    #: calibrated.
    max_plane_mm: float = 2.0
    #: Take points from one eye alone where the other cannot see the stripe, by
    #: meeting its ray with the laser plane. Needs a calibrated plane.
    single_eye: bool = True
    volume: ScanVolume = field(default_factory=ScanVolume)


@dataclass
class ScanFrame:
    """What one frame pair contributed, and why the rest was dropped."""

    points_board: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), np.float64))
    points_camera: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), np.float64))
    reproj_px: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_candidates: int = 0
    #: Left points whose epipolar line never crossed the right stripe.
    n_rejected_nomatch: int = 0
    #: Left points whose line crossed it more than once — undecidable.
    n_rejected_ambiguous: int = 0
    n_rejected_geometry: int = 0
    n_rejected_volume: int = 0
    #: Triangulated points that did not lie on the laser plane.
    n_rejected_plane: int = 0
    #: Points recovered from one eye alone via the plane.
    n_single_eye: int = 0
    reason: str | None = None

    @property
    def n_kept(self) -> int:
        return len(self.points_board)


#: Epipolar lines swept per pass in `_cross_polyline`. 128 rows by ~1700
#: stripe points of float32 stay in cache; measured at 1700×1700: 4.5 ms,
#: against 8.0 ms at 512 rows and 37 ms for the unchunked float64 original.
_CROSS_CHUNK = 128


def _cross_polyline(
    lines: np.ndarray, pts: np.ndarray, along_x: bool, p: ScanParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each epipolar line crosses the observed stripe polyline.

    Returns `(match, n_crossings, angle_deg)`. `match` is NaN where there was
    no crossing; where there were several, the first is returned and
    `n_crossings` says so, so the caller can drop it as ambiguous rather than
    silently pick one.

    Every line against every segment, because a broken polyline can cross any
    line anywhere. The sweep that finds WHICH segment is float32 and chunked:
    only the sign of each vertex's distance to each line matters there, and a
    sign is in doubt only within ~1e-4 px of the line, where either answer puts
    the crossing at the same place. The crossing itself is then solved in
    float64 from the original inputs, for just the segments that matter. A
    vertex lying exactly on a line counts once, with the segment across which
    the sign changes.
    """
    n_lines = len(lines)
    if len(pts) < 2:
        return (np.full((n_lines, 2), np.nan), np.zeros(n_lines, int),
                np.zeros(n_lines))

    a, b = pts[:-1], pts[1:]
    # Only join consecutive centroids that are actually adjacent along the scan
    # axis; a gap means the stripe was interrupted and there is no surface
    # between them to intersect.
    scan_axis = 0 if along_x else 1
    contiguous = np.abs(b[:, scan_axis] - a[:, scan_axis]) <= p.max_segment_gap_px

    hom = np.hstack([pts, np.ones((len(pts), 1))]).astype(np.float32).T   # (3, n_pts)
    lines32 = np.asarray(lines, np.float32)
    n_cross = np.zeros(n_lines, int)
    first = np.zeros(n_lines, int)
    for s in range(0, n_lines, _CROSS_CHUNK):
        dist = lines32[s:s + _CROSS_CHUNK] @ hom          # signed, per vertex
        straddles = np.signbit(dist[:, :-1]) != np.signbit(dist[:, 1:])
        straddles &= contiguous[None, :]
        n_cross[s:s + _CROSS_CHUNK] = straddles.sum(axis=1)
        first[s:s + _CROSS_CHUNK] = np.argmax(straddles, axis=1)

    match = np.full((n_lines, 2), np.nan)
    angle = np.zeros(n_lines)
    rows = np.flatnonzero(n_cross > 0)
    if len(rows):
        seg = first[rows]
        ln = np.asarray(lines, np.float64)[rows]
        da = ln[:, 0] * a[seg, 0] + ln[:, 1] * a[seg, 1] + ln[:, 2]
        db = ln[:, 0] * b[seg, 0] + ln[:, 1] * b[seg, 1] + ln[:, 2]
        denom = da - db
        ok = np.abs(denom) > 1e-12
        t = np.where(ok, da / np.where(ok, denom, 1.0), 0.0)
        match[rows] = a[seg] + t[:, None] * (b[seg] - a[seg])

        # Angle between the epipolar line and the segment it crossed.
        d = b[seg] - a[seg]
        d_norm = np.linalg.norm(d, axis=1)
        l_norm = np.linalg.norm(ln[:, :2], axis=1)
        good = (d_norm > 1e-9) & (l_norm > 1e-9)
        # The line's normal against the segment direction: 90 degrees apart
        # means the segment runs ALONG the line and never usefully meets it.
        cosang = np.zeros(len(rows))
        cosang[good] = np.abs(np.einsum("ij,ij->i", ln[good, :2], d[good])) / (
            l_norm[good] * d_norm[good])
        angle[rows] = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))

    return match, n_cross, angle


def scan_frame(
    rig: StereoRig,
    left: LaserPoints,
    right: LaserPoints,
    board_R: np.ndarray | None,
    board_t: np.ndarray | None,
    params: ScanParams = ScanParams(),
    plane: LaserPlane | None = None,
) -> ScanFrame:
    """Triangulate one frame pair's stripe into board-frame points.

    `board_R`/`board_t` are the board's pose in the LEFT camera's frame, as
    `cvcore.estimate_pose` returns it (t in mm).
    """
    if not left.ok or not right.ok:
        return ScanFrame(reason="both eyes need stripe points")
    if board_R is None or board_t is None:
        return ScanFrame(reason="board pose unknown — it defines the scan volume")

    lp = rig.undistort(left.points.astype(np.float64), "left")
    rp_obs = rig.undistort(right.points.astype(np.float64), "right")

    match, n_cross, angle = _cross_polyline(
        rig.epipolar_lines(lp), rp_obs, right.along_x, params)

    n_none = int((n_cross == 0).sum())
    n_amb = int((n_cross > 1).sum())
    usable = (n_cross == 1) & np.isfinite(match).all(axis=1)
    n_geom = int((usable & (angle < params.min_intersection_deg)).sum())
    usable &= angle >= params.min_intersection_deg

    if not usable.any():
        why = None
        if n_none == len(lp):
            why = "no epipolar crossings — is the right eye seeing the same stripe?"
        elif n_geom and n_geom >= n_amb:
            why = "stripe runs along the epipolar lines — rotate the laser ~90°"
        return ScanFrame(n_candidates=len(lp), n_rejected_nomatch=n_none,
                         n_rejected_ambiguous=n_amb, n_rejected_geometry=n_geom,
                         reason=why)

    lp_u, rp_u = lp[usable], match[usable]
    xyz_cam = rig.triangulate(lp_u, rp_u)
    err = rig.reprojection_error(xyz_cam, lp_u, rp_u)
    sane = np.isfinite(xyz_cam).all(axis=1) & (err <= params.max_reproj_px)
    xyz_cam, err = xyz_cam[sane], err[sane]

    # The independent check. Everything above was consistent by construction.
    n_plane = 0
    if plane is not None and len(xyz_cam):
        on_sheet = np.abs(plane.distance(xyz_cam)) <= params.max_plane_mm
        n_plane = int((~on_sheet).sum())
        xyz_cam, err = xyz_cam[on_sheet], err[on_sheet]

    # Points the other eye could not confirm are not lost when the plane is
    # known: one ray meets one sheet in one place. On a subject this is most
    # of the shadowed flank.
    n_single = 0
    if plane is not None and params.single_eye:
        alone = ~usable
        if alone.any():
            d = lp[alone] - np.array([rig.left_k.cx, rig.left_k.cy])
            d = np.hstack([d / np.array([rig.left_k.fx, rig.left_k.fy]),
                           np.ones((int(alone.sum()), 1))])
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            solo = plane.intersect_rays(np.zeros_like(d), d)
            solo = solo[np.isfinite(solo).all(axis=1)]
            if len(solo):
                n_single = len(solo)
                xyz_cam = np.vstack([xyz_cam, solo])
                err = np.concatenate([err, np.full(len(solo), np.nan)])

    # Into the board's frame: the volume is defined relative to the board, so it
    # stays put when the board moves and "above" keeps meaning above it.
    if len(xyz_cam):
        xyz_board = (board_R.T @ (xyz_cam.T - board_t.reshape(3, 1))).T
    else:
        xyz_board = np.empty((0, 3))

    inside = params.volume.contains(xyz_board)
    n_vol = int((~inside).sum())

    return ScanFrame(
        points_board=xyz_board[inside],
        points_camera=xyz_cam[inside],
        reproj_px=err[inside],
        n_candidates=len(lp),
        n_rejected_nomatch=n_none,
        n_rejected_ambiguous=n_amb,
        n_rejected_geometry=n_geom,
        n_rejected_volume=n_vol,
        n_rejected_plane=n_plane,
        n_single_eye=n_single,
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
