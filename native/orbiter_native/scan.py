"""Scanning: laser points seen by both eyes, triangulated, kept inside a box.

One sweep of the pipeline, per frame pair:

  1. Each eye has already found its laser stripe (`laser.find_laser_line`).
  2. For every stripe point in the LEFT image, its match in the right image is
     where the epipolar line meets the right eye's stripe. Two lines meet in one
     point, so the correspondence is exact — no descriptor, no search window,
     no similarity threshold.
  3. Triangulate.
  4. Keep a point only where the right eye ACTUALLY OBSERVED laser — that is,
     where the intersection lands on one of its detected stripe points and not
     merely on the infinite line fitted through them.

     Reprojection error cannot do this job, and it is worth being explicit
     about why, because it looks like it should. The match in step 2 is
     constructed as a point ON the epipolar line of the left observation, so
     the two rays meet exactly by construction and triangulation reprojects to
     ~1e-13 px no matter where along that line the match landed. Sliding the
     right eye's stripe sideways by 25 px was measured to change nothing: every
     point still "agreed". Support on the observed stripe is the test with
     content in it.
  5. Keep it only if it lies inside a box above the board — expressed in the
     BOARD's frame, not the camera's, so the box stays put while the board is
     what defines "up".

Step 5 is what removes the bench, the operator's hands and the far wall without
any of them needing to be recognised: they are simply not in the volume.

The board must therefore be visible for scanning to work at all. That is a real
constraint, and a deliberate one — the board is what defines where the scanning
volume is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .laser import LaserLine
from .stereo import StereoRig

log = logging.getLogger("orbiter_native.scan")


@dataclass(frozen=True)
class ScanVolume:
    """The box, in board coordinates, in millimetres.

    The board's own frame has z pointing out of its face, so `height_mm` is
    "above the board" and the x/y extent is centred on the board's origin.
    """

    height_mm: float = 400.0        # the 40 cm cube
    half_width_mm: float = 200.0
    #: Points below this are the board's own surface — the stripe lying on the
    #: board itself, which is the calibration target and not the subject.
    floor_mm: float = 3.0

    def contains(self, xyz_board: np.ndarray) -> np.ndarray:
        """Boolean mask over (N, 3) points expressed in the board frame."""
        x, y, z = xyz_board[:, 0], xyz_board[:, 1], xyz_board[:, 2]
        return (
            (z >= self.floor_mm) & (z <= self.height_mm)
            & (np.abs(x) <= self.half_width_mm)
            & (np.abs(y) <= self.half_width_mm)
        )


@dataclass
class ScanParams:
    """Acceptance thresholds for one frame's points."""

    #: How close the epipolar match must land to a point the right eye actually
    #: detected. This is the both-cameras-saw-it test — see the module note on
    #: why reprojection error cannot serve as one here.
    max_support_px: float = 3.0
    #: Reprojection sanity. Near-zero by construction for a well-conditioned
    #: intersection, so this only catches numerical trouble, not mismatches.
    max_reproj_px: float = 1.5
    #: A stripe point whose epipolar line meets the other eye's stripe at too
    #: shallow an angle has an ill-conditioned intersection: a small error
    #: along the line moves the match a long way. Degrees.
    min_intersection_deg: float = 8.0
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
    n_rejected_geometry: int = 0
    #: Matches the right eye did not actually observe laser at, plus any that
    #: failed the reprojection sanity check.
    n_rejected_unconfirmed: int = 0
    n_rejected_volume: int = 0
    reason: str | None = None

    @property
    def n_kept(self) -> int:
        return len(self.points_board)


def _intersect_with_line(lines: np.ndarray, point: np.ndarray,
                         direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intersect epipolar lines (a, b, c) with the right eye's stripe.

    The stripe is given as a point and a unit direction — the fitted line, not
    the raw centroids, because the fit has already rejected the outliers and
    averaged down the per-scanline noise.

    Returns the intersections and the angle between each pair of lines, which
    says how well-conditioned each intersection is.
    """
    # Stripe as a homogeneous line: normal is perpendicular to its direction.
    n = np.array([-direction[1], direction[0]], float)
    c = -float(n @ point)
    stripe = np.array([n[0], n[1], c], float)

    cross = np.cross(lines, stripe)                      # (N, 3) homogeneous
    w = cross[:, 2]
    ok = np.abs(w) > 1e-12
    pts = np.full((len(lines), 2), np.nan)
    pts[ok] = cross[ok, :2] / w[ok, None]

    # Angle between each epipolar line and the stripe.
    ln = lines[:, :2]
    norms = np.linalg.norm(ln, axis=1)
    good = norms > 1e-12
    cosang = np.zeros(len(lines))
    cosang[good] = np.abs(ln[good] @ n) / (norms[good] * np.linalg.norm(n))
    angle = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
    # `angle` above is between the NORMALS, which equals the angle between the
    # lines; 90 degrees means they are parallel and never usefully meet.
    return pts, 90.0 - np.abs(90.0 - angle)


def scan_frame(
    rig: StereoRig,
    left: LaserLine,
    right: LaserLine,
    board_R: np.ndarray | None,
    board_t: np.ndarray | None,
    params: ScanParams = ScanParams(),
) -> ScanFrame:
    """Triangulate one frame pair's stripe into board-frame points.

    `board_R`/`board_t` are the board's pose in the LEFT camera's frame, as
    `calibration.estimate_board_pose_disambiguated` returns it (t in mm).
    """
    if not left.ok or not right.ok:
        return ScanFrame(reason="both eyes need a fitted stripe")
    if board_R is None or board_t is None:
        return ScanFrame(reason="board pose unknown — it defines the scan volume")

    src = left.inlier_points.astype(np.float64)
    if len(src) == 0:
        return ScanFrame(reason="no stripe points in the left eye")

    lp = rig.undistort(src, "left")

    # The right eye's stripe, undistorted the same way so both live in the
    # pinhole geometry triangulation assumes.
    rp_pair = rig.undistort(
        np.stack([right.point - right.direction * 100.0,
                  right.point + right.direction * 100.0]), "right")
    r_dir = rp_pair[1] - rp_pair[0]
    r_norm = float(np.linalg.norm(r_dir))
    if r_norm < 1e-9:
        return ScanFrame(reason="right stripe is degenerate")
    r_dir = r_dir / r_norm

    matches, angles = _intersect_with_line(rig.epipolar_lines(lp), rp_pair[0], r_dir)

    usable = np.isfinite(matches).all(axis=1) & (angles >= params.min_intersection_deg)
    n_geom = int((~usable).sum())
    if not usable.any():
        return ScanFrame(n_candidates=len(src), n_rejected_geometry=n_geom,
                         reason="stripe runs along the epipolar lines — "
                                "no conditioned intersections")

    lp_u, rp_u = lp[usable], matches[usable]

    # Did the right eye actually see laser there? Distance from each epipolar
    # match to the nearest point the right detector really reported.
    observed = rig.undistort(right.inlier_points.astype(np.float64), "right")
    support = np.min(
        np.linalg.norm(rp_u[:, None, :] - observed[None, :, :], axis=2), axis=1)
    seen = support <= params.max_support_px

    xyz_cam = rig.triangulate(lp_u, rp_u)
    err = rig.reprojection_error(xyz_cam, lp_u, rp_u)

    agree = seen & np.isfinite(xyz_cam).all(axis=1) & (err <= params.max_reproj_px)
    n_reproj = int((~agree).sum())
    xyz_cam, err = xyz_cam[agree], err[agree]

    # Into the board's frame: the volume is defined relative to the board, so
    # it stays put when the board moves and "above" keeps meaning above it.
    xyz_board = (board_R.T @ (xyz_cam.T - board_t.reshape(3, 1))).T

    inside = params.volume.contains(xyz_board)
    n_vol = int((~inside).sum())

    return ScanFrame(
        points_board=xyz_board[inside],
        points_camera=xyz_cam[inside],
        reproj_px=err[inside],
        n_candidates=len(src),
        n_rejected_geometry=n_geom,
        n_rejected_unconfirmed=n_reproj,
        n_rejected_volume=n_vol,
    )


class PointCloud:
    """Accumulated scan points, in the board's frame.

    Board-frame rather than camera-frame on purpose: it is the one coordinate
    system that stays fixed while the object turns on the board, so sweeps taken
    at different times land in the same space.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(self, pts: np.ndarray) -> None:
        if len(pts):
            self._chunks.append(np.asarray(pts, np.float64))
            self._n += len(pts)

    def clear(self) -> None:
        self._chunks.clear()
        self._n = 0

    def points(self) -> np.ndarray:
        if not self._chunks:
            return np.empty((0, 3), np.float64)
        return np.concatenate(self._chunks, axis=0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        p = self.points()
        if not len(p):
            return None
        return p.min(axis=0), p.max(axis=0)

    def write_ply(self, path: str) -> int:
        """Write an ASCII PLY. Returns the point count."""
        p = self.points()
        with open(path, "w", encoding="ascii") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(p)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for x, y, z in p:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
        return len(p)
