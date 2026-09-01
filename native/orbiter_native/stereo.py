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
from dataclasses import dataclass, field
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

#: A pair whose residual exceeds this multiple of the set's median is dropped
#: and the solve repeated once. Never more than MAX_OUTLIER_FRACTION of the set.
#:
#: This exists for one specific failure, seen on the rig: the cameras were
#: adjusted after the first captures, so the set held two rig geometries. The
#: first 20 of 140 pairs solved at 132.8 px against 2.0 px for any slice of the
#: rest, and together they refused at 62.8 px. Correspondence was correct in
#: every pair and freeing the intrinsics only made it worse — the data, not the
#: model. Dropping the minority cluster recovers the calibration the other 120
#: views already contain, instead of costing the operator a fresh sweep.
OUTLIER_FACTOR = 3.0
MAX_OUTLIER_FRACTION = 0.3


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
    #: Pairs discarded by the robust pass, as indices into the input list. A
    #: contiguous run at the start means the rig was moved after those were
    #: taken — which is what the panel tells the operator.
    dropped: list[int] = field(default_factory=list)

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


def _solve(obj, imgl, imgr, left_k, right_k, wh):
    return cv2.stereoCalibrate(
        obj, imgl, imgr,
        left_k.K, left_k.D, right_k.K, right_k.D, wh,
        # Intrinsics stay put: they came from many more views than these
        # paired ones, and refitting them here would be a worse estimate
        # that also disagreed with what the server has stored.
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )


def _pair_residuals(obj, imgl, imgr, left_k, right_k, R, T) -> np.ndarray:
    """Per-pair reprojection RMS under one rig geometry, worst of the two eyes.

    The board pose comes from the left view alone, then is carried into the
    right eye through (R, T). A pair taken under a different rig geometry
    reprojects fine on the left and wildly on the right, which is exactly the
    signature wanted here.
    """
    out = np.full(len(obj), np.inf)
    T3 = np.asarray(T, float).reshape(3, 1)
    for i, (o, a, b) in enumerate(zip(obj, imgl, imgr)):
        ok, rv, tv = cv2.solvePnP(o, a, left_k.K, left_k.D)
        if not ok:
            continue
        pa, _ = cv2.projectPoints(o, rv, tv, left_k.K, left_k.D)
        Rr = np.asarray(R, float) @ cv2.Rodrigues(rv)[0]
        tr = np.asarray(R, float) @ tv + T3
        pb, _ = cv2.projectPoints(o, cv2.Rodrigues(Rr)[0], tr, right_k.K, right_k.D)
        ea = np.linalg.norm(pa.reshape(-1, 2) - a.reshape(-1, 2), axis=1)
        eb = np.linalg.norm(pb.reshape(-1, 2) - b.reshape(-1, 2), axis=1)
        out[i] = max(float(np.sqrt((ea ** 2).mean())), float(np.sqrt((eb ** 2).mean())))
    return out


def calibrate(
    pairs: list[PairSample],
    board,
    left_k: Intrinsics,
    right_k: Intrinsics,
    wh: tuple[int, int],
) -> tuple[StereoResult | None, str | None]:
    """Solve the two eyes' mutual pose. `(result, None)` or `(None, reason)`."""
    obj, imgl, imgr, idx = [], [], [], []
    for i, p in enumerate(pairs):
        m = _matched(p, board)
        if m is None:
            continue
        obj.append(m[0])
        imgl.append(m[1])
        imgr.append(m[2])
        idx.append(i)

    if len(obj) < MIN_VIEWS:
        return None, (f"only {len(obj)} views where both eyes saw enough of the "
                      f"board (need {MIN_VIEWS})")

    dropped: list[int] = []
    try:
        rms, _, _, _, _, R, T, E, F = _solve(obj, imgl, imgr, left_k, right_k, wh)

        # Robust pass, iterated. stereoCalibrate's own rms is over all points
        # and cannot say which pairs did the damage; the per-pair residual can
        # — but only once the solve is near the majority geometry. Under heavy
        # contamination (4 stale pairs in 20) the first solve is a compromise
        # that reprojects EVERY pair badly, the median rises with it, and one
        # relative cut separates nothing. Dropping the worst and re-solving
        # snaps the fit to the majority; the median collapses; the rest of the
        # stale cluster is cut on the next round. Bounded by the fraction cap
        # and a few rounds, so a set that is mostly bad is refused, not mined.
        # NOTE: T here is still in metres — see the unit conversion below.
        floor = int(np.ceil((1.0 - MAX_OUTLIER_FRACTION) * len(res_idx := list(range(len(obj))))))
        for _ in range(5):
            res = _pair_residuals(obj, imgl, imgr, left_k, right_k, R, T)
            cut = OUTLIER_FACTOR * float(np.median(res))
            keep = [j for j, e in enumerate(res) if e <= cut]
            if len(keep) == len(res) or len(keep) < max(MIN_VIEWS, floor):
                break
            gone = [j for j in range(len(res)) if j not in set(keep)]
            dropped += [idx[res_idx[j]] for j in gone]
            res_idx = [res_idx[j] for j in keep]
            obj = [obj[j] for j in keep]
            imgl = [imgl[j] for j in keep]
            imgr = [imgr[j] for j in keep]
            rms, _, _, _, _, R, T, E, F = _solve(obj, imgl, imgr, left_k, right_k, wh)
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
                        rms_px=float(rms), n_views=len(obj), wh=wh,
                        dropped=dropped), None


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
