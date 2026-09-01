"""Stereo geometry: the two eyes' mutual pose, and triangulation from it.

Step 2 of the calibration. Intrinsics describe each lens on its own; this
describes where the two cameras sit relative to each other, which is what turns
a pair of 2D observations into a 3D point.

**The pairing is why capture stored pairs.** `stereoCalibrate` needs the two
eyes to have seen the board at the SAME instant — a left view from one moment
and a right view from another share no board pose, and feeding them in produces
a rig geometry that never existed. `intrinsics.SampleSet.paired()` is exactly
that subset, matched on camserver's capture clock.

**Intrinsics are fixed during the solve.** They were measured from many more
views than the paired subset, and letting `stereoCalibrate` refit them here
would trade a well-constrained result for a worse one estimated from less data
— and would silently disagree with the intrinsics stored on the server.

**Correspondence along the laser comes from epipolar geometry.** A point in the
left image lies somewhere on a known line in the right image. The laser gives a
second line. Two lines meet in one point, so the match is exact and needs no
descriptor, no search, and no threshold — and the residual of that intersection
is a real confirmation that both cameras are looking at the same illuminated
spot rather than at two unrelated bright things.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .cvcore import Intrinsics
from .intrinsics import MIN_VIEWS, PairSample

log = logging.getLogger("orbiter_native.stereo")

#: Reprojection RMS above this means the pair geometry did not converge on
#: anything usable.
#:
#: Raised from 1.5 px after it refused calibrations from this rig that were in
#: fact usable. Stereo RMS here sits near 1.56 while each eye alone fits at
#: 0.63-0.72, and four candidate explanations for that gap were measured and
#: eliminated: the distortion model (freeing k3 moves the per-eye fit by 0.6%,
#: the rational model no further), holding the intrinsics fixed during the
#: stereo solve (releasing them buys 5%), a rig that shifted mid-capture
#: (splitting the views by capture order gives consistent baselines), and
#: non-simultaneous exposures (the pair's measured capture gap is a median of
#: 0.01 ms, and restricting to pairs under 0.5 ms changes nothing). What
#: remains is physical — board flatness, or rolling shutter in two sensors
#: mounted 178 degrees apart — and neither is fixable from here.
#:
#: So the number is reported in millimetres of depth error rather than used as
#: a pass/fail on its own. A gate this tool cannot justify is worse than a
#: figure the operator can weigh: refusing at exactly this rig's noise floor
#: blocks the work and explains nothing.
MAX_RMS_PX = 2.5


@dataclass
class StereoResult:
    """The right eye's pose in the left eye's frame."""

    R: np.ndarray                   # 3x3, rotation right←left
    T: np.ndarray                   # 3-vector, MILLIMETRES (see `calibrate`)
    E: np.ndarray                   # essential matrix
    F: np.ndarray                   # fundamental matrix
    rms_px: float
    n_views: int
    wh: tuple[int, int]

    def depth_error_mm(self, range_mm: float, fx: float) -> float:
        """Depth uncertainty at `range_mm`, from this fit's reprojection error.

        Triangulated depth error grows with the square of range and shrinks
        with the baseline: dZ = Z^2 / (f * B) per pixel of disparity error. This
        is what the reprojection RMS actually costs, and it is the number worth
        deciding on — 1.5 px over a 140 mm baseline is about 3 mm at half a
        metre, which may be fine or useless depending on the subject.
        """
        if self.baseline_mm <= 0 or fx <= 0:
            return float("nan")
        return (range_mm ** 2) / (fx * self.baseline_mm) * self.rms_px

    @property
    def baseline_mm(self) -> float:
        """Distance between the two optical centres — the measured baseline,
        as opposed to the nominal one somebody typed into the web tab."""
        return float(np.linalg.norm(self.T))

    def as_config(self) -> dict[str, Any]:
        """Payload for `set_stereo_rig`. Carries the resolution for the same
        reason intrinsics do: this geometry is only valid alongside the
        intrinsics it was solved with, which are resolution-specific."""
        return {
            "R": [[float(v) for v in row] for row in self.R],
            "T": [float(v) for v in self.T.ravel()],
            "rms_px": float(self.rms_px),
            "views": int(self.n_views),
            "width": self.wh[0], "height": self.wh[1],
            "baseline_mm": self.baseline_mm,
        }


def _matched(pair: PairSample, board):
    """3D board points plus both eyes' image points for one simultaneous view.

    Only corners BOTH eyes saw are usable: `stereoCalibrate` needs the same
    object point in both image lists, index for index.
    """
    left, right = pair.left, pair.right
    if left is None or right is None:
        return None
    lo, li = board.matchImagePoints(left.corners, left.ids)
    ro, ri = board.matchImagePoints(right.corners, right.ids)
    if lo is None or ro is None:
        return None

    lids = left.ids.ravel()
    rids = right.ids.ravel()
    common = np.intersect1d(lids, rids)
    if len(common) < 6:
        return None
    lsel = np.searchsorted(lids, common, sorter=np.argsort(lids))
    lsel = np.argsort(lids)[lsel]
    rsel = np.searchsorted(rids, common, sorter=np.argsort(rids))
    rsel = np.argsort(rids)[rsel]
    return (lo.reshape(-1, 3)[lsel].astype(np.float32),
            li.reshape(-1, 2)[lsel].astype(np.float32),
            ri.reshape(-1, 2)[rsel].astype(np.float32))


def calibrate(
    pairs: list[PairSample],
    board,
    left_k: Intrinsics,
    right_k: Intrinsics,
    wh: tuple[int, int],
) -> tuple[StereoResult | None, str | None]:
    """Solve the two eyes' mutual pose. `(result, None)` or `(None, reason)`."""
    obj, imgl, imgr = [], [], []
    for p in pairs:
        m = _matched(p, board)
        if m is None:
            continue
        obj.append(m[0])
        imgl.append(m[1])
        imgr.append(m[2])

    if len(obj) < MIN_VIEWS:
        return None, (f"only {len(obj)} views where both eyes saw enough of the "
                      f"board (need {MIN_VIEWS})")

    try:
        rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj, imgl, imgr,
            left_k.K, left_k.D, right_k.K, right_k.D, wh,
            # Intrinsics stay put: they came from many more views than these
            # paired ones, and refitting them here would be a worse estimate
            # that also disagreed with what the server has stored.
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
    except cv2.error as exc:
        return None, f"stereoCalibrate failed: {exc}"

    if not np.isfinite(rms):
        return None, "stereo solve produced a non-finite RMS"
    if rms > MAX_RMS_PX:
        return None, f"stereo reprojection RMS {rms:.2f} px exceeds {MAX_RMS_PX} px"

    # UNITS: the board is built with its lengths in metres, so
    # `matchImagePoints` yields object points in metres and stereoCalibrate
    # returns T in metres too. Everything downstream of here — board poses from
    # `estimate_board_pose`, the scan volume, the point cloud — is in
    # millimetres, so convert once, at the boundary, exactly as the server's
    # `estimate_board_pose` does for its own tvec.
    return StereoResult(R=np.asarray(R, float),
                        T=np.asarray(T, float).ravel() * 1000.0,
                        E=np.asarray(E, float), F=np.asarray(F, float),
                        rms_px=float(rms), n_views=len(obj), wh=wh), None


def result_from_config(cfg: dict[str, Any] | None,
                       frame_wh: tuple[int, int] | None) -> StereoResult | None:
    """Rebuild a stored solve, refusing it at a different resolution.

    Same rule as intrinsics: the geometry was solved alongside a specific pair
    of camera matrices, which are only valid at one frame size.
    """
    if not isinstance(cfg, dict):
        return None
    try:
        wh = (int(cfg["width"]), int(cfg["height"]))
        if frame_wh is not None and tuple(frame_wh) != wh:
            return None
        R = np.asarray(cfg["R"], float).reshape(3, 3)
        T = np.asarray(cfg["T"], float).ravel()
    except (KeyError, TypeError, ValueError):
        return None
    return StereoResult(R=R, T=T, E=np.zeros((3, 3)), F=np.zeros((3, 3)),
                        rms_px=float(cfg.get("rms_px", float("nan"))),
                        n_views=int(cfg.get("views", 0)), wh=wh)


class StereoRig:
    """Projection matrices and the epipolar relation, derived once.

    Built when the calibration changes rather than per frame: these are pure
    functions of the calibration and rebuilding them at 30 Hz would be waste.
    """

    def __init__(self, left_k: Intrinsics, right_k: Intrinsics, geom: StereoResult):
        self.left_k = left_k
        self.right_k = right_k
        self.geom = geom
        # Left camera is the reference frame: P_left = K [I|0], P_right = K [R|T].
        self.P_left = left_k.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P_right = right_k.K @ np.hstack([geom.R, geom.T.reshape(3, 1)])
        # Fundamental matrix from the geometry, so a stored solve (which does
        # not carry F) still yields epipolar lines.
        tx = np.array([[0, -geom.T[2], geom.T[1]],
                       [geom.T[2], 0, -geom.T[0]],
                       [-geom.T[1], geom.T[0], 0]], float)
        E = tx @ geom.R
        self.F = np.linalg.inv(right_k.K).T @ E @ np.linalg.inv(left_k.K)

    def undistort(self, pts: np.ndarray, side: str) -> np.ndarray:
        """Remove lens distortion from (N, 2) points, keeping pixel units.

        Triangulation assumes a pinhole camera; feeding it raw pixels from a
        lens with k1 = -0.28 would bend every ray.
        """
        k = self.left_k if side == "left" else self.right_k
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64),
                                  k.K, k.D, P=k.K)
        return out.reshape(-1, 2)

    def epipolar_lines(self, left_pts: np.ndarray) -> np.ndarray:
        """For each undistorted left point, its line (a, b, c) in the right image."""
        h = np.hstack([left_pts, np.ones((len(left_pts), 1))])
        return (self.F @ h.T).T

    def triangulate(self, left_pts: np.ndarray, right_pts: np.ndarray) -> np.ndarray:
        """(N, 3) points in the LEFT camera's frame, in mm."""
        x = cv2.triangulatePoints(self.P_left, self.P_right,
                                  left_pts.T.astype(np.float64),
                                  right_pts.T.astype(np.float64))
        w = x[3]
        safe = np.abs(w) > 1e-12
        out = np.full((len(left_pts), 3), np.nan)
        out[safe] = (x[:3, safe] / w[safe]).T
        return out

    def reprojection_error(self, xyz: np.ndarray, left_pts: np.ndarray,
                           right_pts: np.ndarray) -> np.ndarray:
        """Per-point reprojection error in pixels, worst of the two views.

        This is what "confirmed by both cameras" means numerically: a point
        that reprojects close to BOTH observations is one thing seen twice; a
        large error means the two images were showing different things and the
        triangulation split the difference.
        """
        h = np.hstack([xyz, np.ones((len(xyz), 1))])
        err = np.full(len(xyz), np.inf)
        for P, obs in ((self.P_left, left_pts), (self.P_right, right_pts)):
            proj = (P @ h.T).T
            w = proj[:, 2]
            ok = np.abs(w) > 1e-12
            d = np.full(len(xyz), np.inf)
            d[ok] = np.linalg.norm(proj[ok, :2] / w[ok, None] - obs[ok], axis=1)
            err = np.where(np.isinf(err), d, np.maximum(err, d))
        return err
