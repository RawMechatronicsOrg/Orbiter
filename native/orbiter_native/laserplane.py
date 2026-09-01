"""The laser plane in the left camera's frame — calibration and use.

Valid only because the cameras and the laser are rigidly coupled on this rig.
The plane is then fixed in the cameras' frame, the board supplies the pose of
the whole assembly, and one calibration holds for every scan.

**How it is measured.** Where the stripe falls on the board, its 3D position is
already determined: the board's pose gives a known plane in camera coordinates,
and each stripe pixel back-projects to a ray that meets it in exactly one
point. Sweep the board through a range of poses and those points sample the
laser's own plane from many depths and angles; fit a plane through them.
Nothing else is needed — no extra target, no second calibration object.

**What it buys.** Two things, and the second is the larger.

It is the *only* independent check on a triangulated point. Stereo
correspondence along the stripe is built by intersecting an epipolar line with
the other eye's observation, so the two rays meet by construction and the
reprojection error is ~1e-13 px however wrong the pairing was — measured. A
real point must also lie on the laser plane; a mispaired one need not, and
that is the one constraint the pairing did not already assume.

And it lets a SINGLE camera produce points: a ray meets the plane in one place,
so a stripe point visible to one eye alone still yields 3D. With stereo alone a
point is lost whenever either eye cannot see it, which on a real subject means
every shadowed flank and every specular dropout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .cvcore import Intrinsics

log = logging.getLogger("orbiter_native.laserplane")

#: Points further than this from the fitted plane are not on the laser sheet
#: and are dropped before the final fit — a stripe pixel that strayed onto the
#: board's edge, or a specular highlight.
OUTLIER_MM = 3.0

#: A fit worse than this is not a plane. A line laser projects a sheet; if the
#: residual is large, the samples were not all on it.
MAX_RMS_MM = 2.0

MIN_POINTS = 200

#: The stripe lies on a FLAT board during this calibration, so its image is a
#: straight line. A frame whose line fit is worse than this was taken while the
#: board was moving — a smeared centroid and a lagging pose — and its points do
#: not belong on the sheet.
MAX_STRIPE_RMS_PX = 1.0

#: Minimum change in board pose before another frame is banked, in mm of
#: translation. Collecting every frame at 30 fps piles up tens of thousands of
#: points from whatever pose was held longest, which does not constrain the
#: plane any better and does bias it: measured on this rig, 2586 frames yielded
#: 424452 points and a fit that would not converge.
MIN_POSE_STEP_MM = 4.0


@dataclass
class LaserPlane:
    """`n · X = d`, with `n` a unit normal and X in the LEFT camera's frame, mm."""

    normal: np.ndarray
    d: float
    rms_mm: float = float("nan")
    n_points: int = 0
    n_frames: int = 0
    wh: tuple[int, int] = (0, 0)

    def distance(self, xyz: np.ndarray) -> np.ndarray:
        """Signed distance of each (N, 3) point from the plane, in mm."""
        if not len(xyz):
            return np.empty(0)
        return xyz @ self.normal - self.d

    def intersect_rays(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Where each ray meets the plane. NaN where it runs parallel to it.

        This is the single-camera path: one observation, one plane, one point.
        """
        denom = dirs @ self.normal
        out = np.full((len(dirs), 3), np.nan)
        ok = np.abs(denom) > 1e-9
        if ok.any():
            s = (self.d - origins[ok] @ self.normal) / denom[ok]
            # A negative parameter would place the point behind the camera.
            good = s > 0
            idx = np.flatnonzero(ok)[good]
            out[idx] = origins[ok][good] + dirs[ok][good] * s[good, None]
        return out

    def as_config(self) -> dict[str, Any]:
        return {
            "n": [float(v) for v in self.normal],
            "d": float(self.d),
            "rms_mm": float(self.rms_mm),
            "points": int(self.n_points),
            "frames": int(self.n_frames),
            "width": self.wh[0], "height": self.wh[1],
        }


def from_config(cfg: dict[str, Any] | None,
                frame_wh: tuple[int, int] | None) -> LaserPlane | None:
    """Rebuild a stored plane, refusing it at another resolution.

    Same rule as the intrinsics and the pair geometry: this was measured in a
    camera frame defined by a specific camera matrix, which is only valid at
    one frame size.
    """
    if not isinstance(cfg, dict):
        return None
    try:
        wh = (int(cfg["width"]), int(cfg["height"]))
        if frame_wh is not None and tuple(frame_wh) != wh:
            return None
        n = np.asarray(cfg["n"], float).ravel()
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            return None
        return LaserPlane(normal=n / norm, d=float(cfg["d"]) / norm,
                          rms_mm=float(cfg.get("rms_mm", float("nan"))),
                          n_points=int(cfg.get("points", 0)),
                          n_frames=int(cfg.get("frames", 0)), wh=wh)
    except (KeyError, TypeError, ValueError):
        return None


def rays(pixels: np.ndarray, k: Intrinsics) -> np.ndarray:
    """Unit ray directions in the camera frame for (N, 2) DISTORTED pixels."""
    if not len(pixels):
        return np.empty((0, 3))
    norm = cv2.undistortPoints(pixels.reshape(-1, 1, 2).astype(np.float64),
                               k.K, k.D).reshape(-1, 2)
    d = np.hstack([norm, np.ones((len(norm), 1))])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def points_on_board(pixels: np.ndarray, k: Intrinsics,
                    board_R: np.ndarray, board_t: np.ndarray) -> np.ndarray:
    """Where stripe pixels land on the BOARD, in camera coordinates (mm).

    The board's pose makes its plane known, so each ray meets it in exactly one
    point — no triangulation and no second camera involved, which is what makes
    this usable as an independent measurement of the laser.
    """
    if not len(pixels):
        return np.empty((0, 3))
    n = np.asarray(board_R, float)[:, 2]          # board's own z axis
    p0 = np.asarray(board_t, float).ravel()
    d = rays(pixels, k)
    denom = d @ n
    out = np.full((len(d), 3), np.nan)
    ok = np.abs(denom) > 1e-9
    if ok.any():
        s = (p0 @ n) / denom[ok]
        good = s > 0
        idx = np.flatnonzero(ok)[good]
        out[idx] = d[ok][good] * s[good, None]
    return out[np.isfinite(out).all(axis=1)]


def fit(points: np.ndarray, wh: tuple[int, int],
        n_frames: int = 0) -> tuple[LaserPlane | None, str | None]:
    """Fit the laser plane to accumulated board-surface points.

    One outlier pass, for the same reason the intrinsics solve has one: a few
    stray samples — the stripe clipping the board's edge, a specular glint —
    would otherwise tilt a plane that the rest of the data determines well.
    """
    pts = np.asarray(points, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < MIN_POINTS:
        return None, f"only {len(pts)} stripe points on the board (need {MIN_POINTS})"

    def _plane(p):
        centre = p.mean(axis=0)
        # Smallest singular direction is the plane normal.
        normal = np.linalg.svd(p - centre)[2][-1]
        normal = normal / np.linalg.norm(normal)
        return normal, float(normal @ centre)

    normal, d = _plane(pts)
    resid = np.abs(pts @ normal - d)
    # An absolute cut, not a multiple of the median: a contaminated set has a
    # large median and a relative cut then removes nothing, which is exactly
    # how a set of 424452 points failed to converge.
    keep = resid <= OUTLIER_MM
    if keep.sum() >= MIN_POINTS and keep.sum() < len(pts):
        pts = pts[keep]
        normal, d = _plane(pts)

    rms = float(np.sqrt((((pts @ normal) - d) ** 2).mean()))
    if not np.isfinite(rms):
        return None, "plane fit produced a non-finite residual"
    if rms > MAX_RMS_MM:
        return None, (f"plane fit RMS {rms:.2f} mm exceeds {MAX_RMS_MM} mm — "
                      "the samples are not all on one sheet. Collect with the "
                      "board held STILL at each pose, not while moving it.")

    # Point the normal towards the camera so the signed distance has a
    # consistent meaning for every consumer.
    if d < 0:
        normal, d = -normal, -d
    return LaserPlane(normal=normal, d=d, rms_mm=rms, n_points=len(pts),
                      n_frames=n_frames, wh=wh), None


class PlaneCollector:
    """Accumulates stripe-on-board points across frames."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._n = 0
        self._last_t: np.ndarray | None = None
        #: Frames offered but not banked, by reason — so the panel can say what
        #: is wrong rather than just showing a counter that will not move.
        self.skipped: dict[str, int] = {"moving": 0, "same pose": 0}

    def __len__(self) -> int:
        return self._n

    @property
    def frames(self) -> int:
        return self._frames

    def clear(self) -> None:
        self._chunks.clear()
        self._frames = 0
        self._n = 0
        self._last_t = None
        self.skipped = {"moving": 0, "same pose": 0}

    def add_frame(self, pixels: np.ndarray, k: Intrinsics,
                  board_R: np.ndarray, board_t: np.ndarray,
                  stripe_rms_px: float | None = None) -> int:
        """Contribute one frame's stripe. Returns how many points it added.

        Two gates, both learned from a set that would not fit. A frame taken
        while the board was moving carries a smeared stripe and a lagging pose,
        and `stripe_rms_px` reveals it because the stripe must be straight on a
        flat board. And a pose barely different from the last banked one adds
        no constraint while adding weight, so the plane ends up decided by
        wherever the operator paused longest.
        """
        if stripe_rms_px is not None and stripe_rms_px > MAX_STRIPE_RMS_PX:
            self.skipped["moving"] += 1
            return 0
        t = np.asarray(board_t, float).ravel()
        if (self._last_t is not None
                and float(np.linalg.norm(t - self._last_t)) < MIN_POSE_STEP_MM):
            self.skipped["same pose"] += 1
            return 0
        pts = points_on_board(pixels, k, board_R, board_t)
        if not len(pts):
            return 0
        self._last_t = t
        self._chunks.append(pts)
        self._frames += 1
        self._n += len(pts)
        return len(pts)

    def points(self) -> np.ndarray:
        if not self._chunks:
            return np.empty((0, 3))
        return np.concatenate(self._chunks, axis=0)

    def fit(self, wh: tuple[int, int]):
        return fit(self.points(), wh, self._frames)
