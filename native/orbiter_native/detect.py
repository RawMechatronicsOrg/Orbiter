"""Per-frame detectors: ChArUco corners (reused) and the laser line (new).

ChArUco detection is NOT reimplemented here. `orbiter_server.calibration`
already owns it, including the flat-board pose ambiguity work, and those
functions turned out to be pure — they take a board and intrinsics as
arguments and never touch the global model — so they are imported and called
directly. Duplicating that numerics would be the one genuinely expensive
mistake available in this module.

The laser detector IS new. The repository has no laser code: README lists
"Laser-stripe scanner + live triangulator" among what this kit deliberately
excludes, and the only mentions left in `calibration.py` are historical
comments about a laser-era rig. It is written vectorised — a per-column
intensity-weighted centroid over the whole frame at once. Measured ~3.4 ms on
a 1080p frame on this machine; the same thing as a Python loop over 1920
columns would be orders of magnitude slower and is the one place in this
pipeline where "just write a loop" actually would have cost real time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .cvcore import build_board, charuco_detect, estimate_pose


@dataclass
class BoardHit:
    """ChArUco detection on one frame, in ORIGINAL (pre-orientation) pixels."""

    corners: np.ndarray | None = None       # (N, 1, 2) float32
    ids: np.ndarray | None = None           # (N, 1) int32
    #: Board→camera pose, only when per-eye intrinsics were supplied. See the
    #: module note in `panel.py`: the model's intrinsics belong to the phone,
    #: not to this pair, so this stays None until the pair is calibrated.
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    ms: float = 0.0

    @property
    def count(self) -> int:
        return 0 if self.corners is None else int(len(self.corners))

    def coverage(self, w: int, h: int) -> float:
        """Fraction of the frame area spanned by the detected corners' bbox.

        A calibration sweep wants corners spread across the frame, not clustered
        in the middle; this is the cheap proxy the overlay shows for that.
        """
        if self.corners is None or self.count < 4:
            return 0.0
        pts = self.corners.reshape(-1, 2)
        span = pts.max(axis=0) - pts.min(axis=0)
        return float((span[0] * span[1]) / max(w * h, 1))


@dataclass
class LaserHit:
    """Per-column subpixel laser centroid, in ORIGINAL pixels."""

    #: Column indices that carried enough signal.
    xs: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    #: Subpixel row centroid for each of those columns.
    ys: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    ms: float = 0.0

    @property
    def count(self) -> int:
        return int(self.xs.size)


class BoardDetector:
    """Holds the built `cv2.aruco.CharucoBoard` so it is not rebuilt per frame.

    Rebuilding the board object every frame would be pure waste; the spec only
    changes when the operator edits the board params in the web UI, which
    `set_spec` handles.
    """

    def __init__(self, spec=None) -> None:
        self._spec = None
        self._board = None
        self.set_spec(spec)

    def set_spec(self, spec) -> None:
        """Swap the board spec. No-op when unchanged, so it is safe to call on
        every config poll."""
        if spec is None or spec == self._spec:
            if spec is None:
                self._spec, self._board = None, None
            return
        self._spec = spec
        self._board = build_board(spec)

    @property
    def ready(self) -> bool:
        return self._board is not None

    def detect(self, gray: np.ndarray, intrinsics=None) -> BoardHit:
        """Detect on a GRAYSCALE frame. Pose only when intrinsics are given."""
        if self._board is None:
            return BoardHit()
        t0 = time.perf_counter()
        corners, ids = charuco_detect(gray, self._board)
        hit = BoardHit(corners=corners, ids=ids)
        if corners is not None and intrinsics is not None:
            pose = estimate_pose(corners, ids, self._board, intrinsics)
            if pose is not None:
                hit.R, hit.t = pose
        hit.ms = (time.perf_counter() - t0) * 1000.0
        return hit


@dataclass(frozen=True)
class LaserParams:
    """Tuning for `find_laser`.

    `min_intensity` is an absolute 0..255 floor on the brightest pixel in a
    column — a column whose peak is below it is treated as carrying no line at
    all.

    The default is deliberately high. A line laser saturates the sensor where
    it lands, so the signal to look for is "near 255", not "brighter than the
    room". Measured on a lit workbench frame from this rig with no laser
    powered: a floor of 60 reports a line in all 1280 columns, 180 still
    reports 974, and only 245 falls to 9. A low floor does not find a faint
    laser — it finds the scene, and reports a confident subpixel centroid for
    it. Lower this only against frames where the laser is actually on.
    """

    min_intensity: int = 230
    #: Only rows within this fraction of the column's peak contribute to the
    #: centroid. Rejects the broad dim halo around a bright stripe, which would
    #: otherwise drag the centroid toward the image centre.
    rel_threshold: float = 0.5


def find_laser(gray: np.ndarray, p: LaserParams = LaserParams()) -> LaserHit:
    """Subpixel laser line: one intensity-weighted row centroid per column.

    Fully vectorised — every column is solved in the same handful of numpy
    operations, with no Python-level loop over columns.
    """
    t0 = time.perf_counter()
    if gray.ndim != 2:
        raise ValueError("find_laser expects a single-channel image")

    img = gray.astype(np.float32, copy=False)
    peak = img.max(axis=0)                                  # brightest pixel per column
    live = peak >= p.min_intensity
    if not live.any():
        return LaserHit(ms=(time.perf_counter() - t0) * 1000.0)

    # Zero everything that is not close to that column's own peak, so the
    # centroid is computed over the stripe alone.
    floor = np.maximum(peak * p.rel_threshold, float(p.min_intensity))
    w = np.where(img >= floor, img, 0.0)

    total = w.sum(axis=0)
    rows = np.arange(img.shape[0], dtype=np.float32)[:, None]
    # A column can pass the peak test yet still sum to zero if its only bright
    # pixel sits exactly at the floor; guard the division rather than filtering
    # twice.
    live &= total > 0
    if not live.any():
        return LaserHit(ms=(time.perf_counter() - t0) * 1000.0)

    centroid = np.zeros_like(total)
    np.divide((w * rows).sum(axis=0), total, out=centroid, where=live)

    xs = np.flatnonzero(live).astype(np.int32)
    return LaserHit(
        xs=xs,
        ys=centroid[xs].astype(np.float32),
        ms=(time.perf_counter() - t0) * 1000.0,
    )


def draw_board(bgr: np.ndarray, hit: BoardHit) -> None:
    """Draw detected corners onto an oriented BGR frame, in place."""
    if hit.corners is None:
        return
    cv2.aruco.drawDetectedCornersCharuco(bgr, hit.corners, hit.ids, (0, 235, 120))


def draw_laser(bgr: np.ndarray, hit: LaserHit) -> None:
    """Draw the laser centroid as single pixels, in place.

    One pixel per column, not a polyline: the line can be broken where the
    stripe leaves the object, and joining across a gap would draw a segment
    through empty space that no measurement supports.
    """
    if hit.count == 0:
        return
    ys = np.rint(hit.ys).astype(np.int32)
    ok = (ys >= 0) & (ys < bgr.shape[0])
    bgr[ys[ok], hit.xs[ok]] = (80, 80, 255)
