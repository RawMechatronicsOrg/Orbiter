"""ChArUco detection per frame.

NOT reimplemented here. `orbiter_server.calibration` already owns it, including
the flat-board pose ambiguity work, and those functions turned out to be pure —
they take a board and intrinsics as arguments and never touch the global model
— so they are imported and called directly. Duplicating that numerics would be
the one genuinely expensive mistake available in this module: a second copy
would drift from the one the calibration actually runs, and the drift would be
invisible until a solve came out wrong.

The laser stripe lives in `laser.py`, not here: it needs colour rather than
luminance and is fitted against the board region, so it consumes this module's
output instead of sitting beside it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .cvcore import build_board, charuco_detect, estimate_pose


@dataclass
class BoardHit:
    """ChArUco detection on one frame, in ORIGINAL (pre-orientation) pixels."""

    corners: np.ndarray | None = None       # (N, 1, 2) float32
    ids: np.ndarray | None = None           # (N, 1) int32
    #: Board→camera pose, only when per-eye intrinsics were supplied. Board
    #: frame: origin at the board's centre, z out of the printed face toward
    #: the camera — `cvcore.estimate_pose` explains why it is not OpenCV's. The
    #: model's intrinsics belong to the phone, not to this pair, so this stays
    #: None until the pair itself is calibrated — see `cvcore.intrinsics_from_eye`.
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    #: Geodesic angle between the two planar-PnP candidates. Large means the
    #: view was genuinely ambiguous and the temporal prior did the deciding.
    ambiguity_deg: float = 0.0
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


class BoardDetector:
    """Holds the built `cv2.aruco.CharucoBoard` so it is not rebuilt per frame.

    Rebuilding the board object every frame would be pure waste; the spec only
    changes when the operator edits the board params in the web UI, which
    `set_spec` handles.
    """

    def __init__(self, spec=None) -> None:
        self._spec = None
        self._board = None
        #: Last accepted rotation, used as the prior that breaks the flat-board
        #: pose ambiguity on the next frame.
        self._last_R = None
        self.set_spec(spec)

    def set_spec(self, spec) -> None:
        """Swap the board spec. No-op when unchanged, so it is safe to call on
        every config poll."""
        if spec is None:
            self._spec, self._board = None, None
            return
        if spec == self._spec:
            return
        self._spec = spec
        self._board = build_board(spec)
        self._last_R = None

    @property
    def ready(self) -> bool:
        return self._board is not None

    @property
    def board(self):
        return self._board

    def detect(self, gray: np.ndarray, intrinsics=None) -> BoardHit:
        """Detect on a GRAYSCALE frame. Pose only when intrinsics are given."""
        if self._board is None:
            return BoardHit()
        t0 = time.perf_counter()
        corners, ids = charuco_detect(gray, self._board)
        hit = BoardHit(corners=corners, ids=ids)
        if corners is not None and intrinsics is not None:
            pose = estimate_pose(corners, ids, self._board, intrinsics,
                                 self._last_R)
            if pose is not None:
                hit.R, hit.t, hit.ambiguity_deg = pose
                self._last_R = hit.R
        elif corners is None:
            # The board left the view; the old rotation is no longer a prior
            # for whatever comes back, and reusing it could lock in the wrong
            # twin for the rest of the session.
            self._last_R = None
        hit.ms = (time.perf_counter() - t0) * 1000.0
        return hit


def draw_board(bgr: np.ndarray, hit: BoardHit) -> None:
    """Draw detected corners onto an oriented BGR frame, in place."""
    if hit.corners is None:
        return
    cv2.aruco.drawDetectedCornersCharuco(bgr, hit.corners, hit.ids, (0, 235, 120))
