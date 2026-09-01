"""Per-eye intrinsics from ChArUco views — capture, diversity, solve.

The operator moves the board; this module decides which of those views are
worth keeping and solves the camera matrix from them.

**Why diversity is the whole problem.** `calibrateCamera` will happily return a
confident answer from twenty views of the board sitting in the same place at
the same angle, and that answer will be wrong. Focal length and board distance
trade off against each other in a fronto-parallel view — only tilt separates
them — and distortion is only observable where the board reaches the edges of
the frame. So a set is judged on three axes, none of which needs intrinsics to
measure:

  * where the board sits in the frame (its corners' centroid),
  * how big it is (scale — near versus far),
  * how tilted it is, read off the perspective terms of the board→image
    homography, which is exactly zero for a fronto-parallel view.

A new view is kept only when it is far enough from every view already held in
that space. Twenty near-duplicates are worth less than six genuinely different
ones, and the count alone would hide that.

**Views are captured in pairs.** Intrinsics are solved per eye and would not
need it, but `stereoCalibrate` needs the two eyes to have seen the board at the
same instant, and recapturing a whole sweep later to get that would be a waste
of the operator's time. A pair is stored whenever at least one eye sees the
board; the intrinsics solve uses whichever side is present, and the stereo
solve later uses the subset where both are.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cvcore import Intrinsics

log = logging.getLogger("orbiter_native.intrinsics")

#: A view with fewer matched corners than this tells the solve very little and
#: mostly adds noise.
MIN_CORNERS = 8

#: calibrateCamera needs enough views to constrain the model; below this the
#: solve is over-parameterised and its RMS is meaninglessly small.
MIN_VIEWS = 6

#: Reprojection RMS above this means the solve did not converge on anything
#: trustworthy — a bad set, a wrong board spec, or motion-blurred corners.
#:
#: A LOW rms proves almost nothing on its own. See MIN_TILT_SPREAD.
MAX_RMS_PX = 1.5

#: Minimum `SampleSet.tilt_spread` for a solve to be trusted.
#:
#: This gate exists because reprojection RMS cannot detect the failure it
#: guards against. Measured on synthetic views of this board with 0.15 px
#: corner noise, sweeping how much the board was allowed to tilt:
#:
#:     tilt spread   RMS px   focal-length error
#:          5.27      0.204        +108 px
#:          5.45      0.204        +114 px
#:          6.04      0.204         +31 px
#:          8.12      0.205          +8 px
#:         12.09      0.205          +3 px
#:         16.06      0.206          +1 px
#:
#: The RMS is flat to three decimal places across a focal length that is wrong
#: by 12%. With no noise at all a fronto-parallel set solves to RMS 0.0000 px
#: and a focal length 27% low. Focal length and board distance are not
#: separable in a head-on view; only tilt separates them, and only the spread
#: of tilt across the set reveals whether it was there.
MIN_TILT_SPREAD = 8.0

#: A view whose reprojection error exceeds this multiple of the set's median is
#: dropped and the solve repeated once.
#:
#: Measured on 98 live views from this rig: the per-view median was 0.73 px but
#: the worst reached 3.25 px, and those few dragged the overall rms to 0.91.
#: Refitting the best half gave 0.54 px while moving the focal length by 1% and
#: the principal point by 0.1% — so the bad views were adding error without
#: adding information. Dropping them reports an honest figure instead of one
#: inflated by frames the operator could not have known were blurred.
OUTLIER_FACTOR = 2.0

#: Never discard more than this fraction, however bad the set looks. Beyond it
#: the problem is the whole set, and silently calibrating from a third of it
#: would hide that.
MAX_OUTLIER_FRACTION = 0.3


#: Divisor that brings the tilt terms onto the same 0..1 footing as position
#: and scale for the novelty distance. Measured: a stationary board's tilt
#: terms wander with a spread around 0.2-0.5, while a proper sweep spans about
#: 15, so 20 puts noise near 0.02 and real variety near 0.75.
_TILT_UNITS = 20.0


@dataclass(frozen=True)
class ViewDescriptor:
    """Where/how the board sits, in units that need no intrinsics.

    All components are normalised so a single distance threshold is meaningful
    across them.
    """

    cx: float           # corner centroid, 0..1 across the frame
    cy: float
    scale: float        # sqrt(bbox area) / frame diagonal
    tilt_x: float       # homography perspective terms, scaled
    tilt_y: float

    def as_array(self) -> np.ndarray:
        """The descriptor as a vector with COMPARABLE axes.

        Position and scale are already 0..1; the tilt terms are kept in their
        own larger units elsewhere (MIN_TILT_SPREAD was measured in them) so
        they are divided here instead. Without this the novelty distance is
        dominated by tilt, and its measurement noise alone clears any sane
        threshold: a board sitting still on the bench was seen to accumulate
        seven "new" views in twelve seconds, all of the same pose.
        """
        return np.array([self.cx, self.cy, self.scale,
                         self.tilt_x / _TILT_UNITS, self.tilt_y / _TILT_UNITS],
                        np.float64)


def describe(corners: np.ndarray, ids: np.ndarray, board,
             wh: tuple[int, int]) -> ViewDescriptor | None:
    """Describe one detection. None when it is too sparse to characterise."""
    if corners is None or ids is None or len(corners) < 4:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < 4:
        return None
    pts = img.reshape(-1, 2).astype(np.float64)
    w, h = wh
    diag = float(np.hypot(w, h))

    centroid = pts.mean(axis=0)
    span = pts.max(axis=0) - pts.min(axis=0)

    # Tilt without intrinsics: the board is planar, so a homography maps its
    # own coordinates to the image exactly. Its bottom row is the perspective
    # part — identically zero when the board faces the camera square on, and
    # growing with tilt. Scaled by the board's image size so the number means
    # "how tilted", not "how close".
    tilt_x = tilt_y = 0.0
    plane = obj.reshape(-1, 3)[:, :2].astype(np.float64)
    if len(plane) >= 4:
        H, _ = cv2.findHomography(plane, pts, 0)
        if H is not None and abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
            extent = float(max(np.ptp(plane, axis=0).max(), 1e-9))
            tilt_x = float(H[2, 0] * extent * 100.0)
            tilt_y = float(H[2, 1] * extent * 100.0)

    return ViewDescriptor(
        cx=float(centroid[0] / w),
        cy=float(centroid[1] / h),
        scale=float(np.sqrt(max(span[0] * span[1], 0.0)) / diag),
        tilt_x=tilt_x,
        tilt_y=tilt_y,
    )


@dataclass
class EyeView:
    """One eye's detection within a captured pair."""

    corners: np.ndarray
    ids: np.ndarray
    wh: tuple[int, int]
    descriptor: ViewDescriptor
    #: camserver's capture instant for this frame. Kept so the gap between the
    #: two eyes of a pair stays measurable after the fact — a stereo residual
    #: that correlates with it means the board moved between the two exposures,
    #: which no amount of calibration can fix and which is otherwise invisible.
    capture_mono: float | None = None


@dataclass
class PairSample:
    """What both eyes saw at one instant. Either side may be missing."""

    left: EyeView | None = None
    right: EyeView | None = None

    def side(self, name: str) -> EyeView | None:
        return self.left if name == "left" else self.right

    @property
    def both(self) -> bool:
        return self.left is not None and self.right is not None


class SampleSet:
    """Captured views plus the rule for what counts as a new one."""

    #: Minimum distance in descriptor space before a view is considered new.
    #: Tuned so that nudging the board a few centimetres does not qualify but
    #: moving it to another part of the frame, or tilting it, does.
    novelty_threshold: float = 0.12

    def __init__(self) -> None:
        self.samples: list[PairSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def clear(self) -> None:
        self.samples.clear()

    def views(self, side: str) -> list[EyeView]:
        out = [s.side(side) for s in self.samples]
        return [v for v in out if v is not None]

    def paired(self) -> list[PairSample]:
        """Samples where both eyes saw the board — the stereo solve's input."""
        return [s for s in self.samples if s.both]

    def pair_gaps_ms(self) -> list[float]:
        """How far apart, in ms, the two exposures of each pair were.

        camserver reports the pair as free-running rather than held, so this is
        not always ~0. With a hand-held board it bounds how much the board
        could have moved between the two views, which sets a floor on the
        stereo residual that no calibration can remove.
        """
        out = []
        for p in self.paired():
            a, b = p.left.capture_mono, p.right.capture_mono
            if a is not None and b is not None:
                out.append(abs(a - b) * 1000.0)
        return out

    def novelty(self, side: str, d: ViewDescriptor | None) -> float:
        """Distance from `d` to the nearest held view of that eye.

        Large when this view shows the solve something it has not seen.
        Infinite for the first sample, zero for a duplicate.
        """
        if d is None:
            return 0.0
        held = self.views(side)
        if not held:
            return float("inf")
        a = d.as_array()
        return float(min(np.linalg.norm(a - v.descriptor.as_array()) for v in held))

    def is_new(self, left: ViewDescriptor | None, right: ViewDescriptor | None) -> bool:
        """True when EITHER eye is seeing something meaningfully new.

        Either, not both: the eyes look at the board from different angles, and
        a view that is fresh for one of them is still worth the pair.
        """
        return max(self.novelty("left", left), self.novelty("right", right)) \
            >= self.novelty_threshold

    def add(self, sample: PairSample) -> None:
        self.samples.append(sample)

    # ── persistence ───────────────────────────────────────────────────────
    #
    # Captured views outlive the process on purpose. A board sweep is minutes
    # of the operator's time, and losing it to a restart means sweeping again;
    # worse, `stereoCalibrate` needs the SIMULTANEOUS views, so a session that
    # solved intrinsics and then restarted could not solve the pair at all
    # without redoing everything.

    def save(self, path: str | os.PathLike) -> int:
        """Write the captured views to `path` (npz). Returns the count."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, np.ndarray] = {}
        for i, sample in enumerate(self.samples):
            for side in ("left", "right"):
                v = sample.side(side)
                if v is None:
                    continue
                blob[f"{i}_{side}_corners"] = v.corners
                blob[f"{i}_{side}_ids"] = v.ids
                blob[f"{i}_{side}_wh"] = np.asarray(v.wh, np.int32)
                blob[f"{i}_{side}_t"] = np.asarray(
                    [float("nan") if v.capture_mono is None else v.capture_mono],
                    np.float64)
        blob["count"] = np.asarray([len(self.samples)], np.int32)
        # Written to a temp file and renamed, so an interrupted save cannot
        # leave a half-written set where a complete one used to be. The handle
        # is passed rather than the path because `savez_compressed` appends
        # `.npz` to any name that lacks it — which would write beside the temp
        # name instead of to it.
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **blob)
        os.replace(tmp, p)
        return len(self.samples)

    def load(self, path: str | os.PathLike, board) -> int:
        """Replace the held views with those in `path`. Returns the count.

        Descriptors are recomputed from the corners rather than stored: they
        are derived data, and recomputing them keeps a loaded set honest if the
        descriptor definition ever changes.
        """
        p = Path(path)
        if not p.exists():
            return 0
        try:
            with np.load(p) as blob:
                n = int(blob["count"][0])
                loaded: list[PairSample] = []
                for i in range(n):
                    sides: dict[str, EyeView | None] = {"left": None, "right": None}
                    for side in ("left", "right"):
                        key = f"{i}_{side}_corners"
                        if key not in blob:
                            continue
                        corners = blob[key]
                        ids = blob[f"{i}_{side}_ids"]
                        wh = tuple(int(v) for v in blob[f"{i}_{side}_wh"])
                        d = describe(corners, ids, board, wh)
                        if d is not None:
                            tkey = f"{i}_{side}_t"
                            t = float(blob[tkey][0]) if tkey in blob else float("nan")
                            sides[side] = EyeView(
                                corners, ids, wh, d,
                                capture_mono=None if np.isnan(t) else t)
                    if sides["left"] is not None or sides["right"] is not None:
                        loaded.append(PairSample(**sides))
        except (OSError, KeyError, ValueError) as exc:
            log.warning("could not load captured views from %s: %s", p, exc)
            return 0
        self.samples = loaded
        return len(loaded)

    def coverage(self, side: str, grid: int = 6) -> np.ndarray:
        """Which cells of a `grid`x`grid` split of the frame the board has
        visited, as a boolean map.

        Distortion is only measurable where the board actually went, so the
        empty cells are the instruction: put the board there next.
        """
        out = np.zeros((grid, grid), bool)
        for v in self.views(side):
            gx = min(int(v.descriptor.cx * grid), grid - 1)
            gy = min(int(v.descriptor.cy * grid), grid - 1)
            if gx >= 0 and gy >= 0:
                out[gy, gx] = True
        return out

    def tilt_spread(self, side: str) -> float:
        """How much tilt variety the set has.

        Near zero means every view is fronto-parallel — the case where focal
        length and distance are not separable and the solve is degenerate no
        matter how many views were taken.
        """
        held = self.views(side)
        if len(held) < 2:
            return 0.0
        t = np.array([[v.descriptor.tilt_x, v.descriptor.tilt_y] for v in held])
        return float(np.linalg.norm(t.std(axis=0)))


@dataclass
class SolveResult:
    """A finished per-eye solve."""

    intrinsics: Intrinsics
    rms_px: float
    n_views: int
    wh: tuple[int, int]
    #: Tilt variety of the set that produced this. Provenance: a solve from a
    #: barely-tilted set is suspect however good its RMS looks.
    tilt_spread: float = float("nan")
    #: Per-view reprojection RMS — the outlier view is visible here even when
    #: the overall figure looks acceptable.
    per_view_rms: list[float] = field(default_factory=list)
    #: Views the robust pass discarded before the final fit.
    dropped: int = 0

    def as_config(self) -> dict:
        """The payload `set_stereo_rig` stores on an eye.

        `width`/`height` travel with the numbers because a camera matrix is
        only valid at the resolution it was solved at.
        """
        i = self.intrinsics
        return {
            "fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy,
            "dist": list(i.dist),
            "width": self.wh[0], "height": self.wh[1],
            "rms_px": self.rms_px,
            "views": self.n_views,
        }


def solve(
    views: list[EyeView], board, tilt_spread: float | None = None
) -> tuple[SolveResult | None, str | None]:
    """Solve one eye's intrinsics. Returns `(result, None)` or `(None, reason)`.

    `tilt_spread` is `SampleSet.tilt_spread` for these views. Pass it: without
    it the degeneracy check cannot run, and the check is the only thing
    standing between the operator and a confident, badly wrong focal length.

    The distortion model frees k1, k2, p1 and p2 but fixes k3. k3 is only
    identifiable from views that push the board right into the frame corners,
    and left free on a modest set it absorbs error from everywhere else and
    destabilises the focal length. Tangential terms are kept free because these
    are inexpensive sensors where a slightly tilted element is real — the
    server's phone-lens solve fixes them, but that was a different lens.
    """
    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    for v in views:
        obj, img = board.matchImagePoints(v.corners, v.ids)
        if obj is None or len(obj) < MIN_CORNERS:
            continue
        obj_pts.append(obj.astype(np.float32))
        img_pts.append(img.astype(np.float32))

    if len(obj_pts) < MIN_VIEWS:
        return None, f"only {len(obj_pts)} usable views (need {MIN_VIEWS})"

    if tilt_spread is not None and tilt_spread < MIN_TILT_SPREAD:
        return None, (
            f"views are too flat-on (tilt spread {tilt_spread:.1f}, "
            f"need {MIN_TILT_SPREAD:.0f}) — tilt the board more between shots"
        )

    sizes = {v.wh for v in views}
    if len(sizes) != 1:
        # calibrateCamera takes one image size for all views, and mixing them
        # would silently produce a matrix valid for neither.
        return None, f"views span several frame sizes: {sorted(sizes)}"
    wh = views[0].wh

    flags = cv2.CALIB_FIX_K3
    try:
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, wh, None, None, flags=flags,
        )
        per_view = _per_view_rms(obj_pts, img_pts, rvecs, tvecs, k, dist)

        # One robust pass: drop the views the fit cannot explain and refit. A
        # blurred frame adds error without adding information, and the operator
        # had no way to see it at capture time.
        dropped = 0
        median = float(np.median(per_view))
        cut = OUTLIER_FACTOR * median
        keep = [j for j, e in enumerate(per_view) if e <= cut]
        if (len(keep) >= MIN_VIEWS
                and len(keep) >= (1.0 - MAX_OUTLIER_FRACTION) * len(per_view)
                and len(keep) < len(per_view)):
            dropped = len(per_view) - len(keep)
            obj_pts = [obj_pts[j] for j in keep]
            img_pts = [img_pts[j] for j in keep]
            rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_pts, img_pts, wh, None, None, flags=flags,
            )
            per_view = _per_view_rms(obj_pts, img_pts, rvecs, tvecs, k, dist)
    except cv2.error as exc:
        return None, f"solve failed: {exc}"

    if not np.isfinite(rms):
        return None, "solve produced a non-finite RMS"
    if rms > MAX_RMS_PX:
        return None, f"reprojection RMS {rms:.2f} px exceeds {MAX_RMS_PX} px"

    return SolveResult(
        intrinsics=Intrinsics(
            fx=float(k[0, 0]), fy=float(k[1, 1]),
            cx=float(k[0, 2]), cy=float(k[1, 2]),
            dist=tuple(float(x) for x in dist.ravel()[:5]),
        ),
        rms_px=float(rms),
        n_views=len(obj_pts),
        wh=wh,
        tilt_spread=float("nan") if tilt_spread is None else float(tilt_spread),
        per_view_rms=per_view,
        dropped=dropped,
    ), None


def _per_view_rms(obj_pts, img_pts, rvecs, tvecs, k, dist) -> list[float]:
    """Reprojection RMS for each view, in pixels."""
    out = []
    for o, i, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, k, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)
        out.append(float(np.sqrt((err ** 2).mean())))
    return out
