"""Per-eye image orientation — the native half of a contract shared with the web.

THE ORDER IS PART OF THE CONTRACT: an eye is **flipped first, then rotated**.
`flip_h` / `flip_v` act in the sensor's own frame and `quarter_turns_cw` is
applied to the already-flipped image.

The same order is written down in two other places and all three must agree:

  * `server/orbiter_server/commands.py::_cmd_set_stereo_rig` — the authority,
  * `ui/src/viewer/StereoView.tsx::eyeTransform` — the settings preview, which
    emits `rotate(...) scaleX(...) scaleY(...)` (CSS composes right-to-left,
    so that reads flip-then-rotate).

If they drift, the operator lines the rig up against a preview the solver never
sees, and it surfaces later as an inexplicable calibration error rather than as
an obvious mirror. `test_orient.py` pins the equivalence.

Why `flip`/`rotate` rather than a `remap` map: for 90° multiples and mirrors
these are exact pixel permutations with no interpolation, so nothing is
resampled and no corner is softened. Once per-eye undistort and stereo
rectification exist, the orientation folds into that single `cv2.remap` map
instead (measured ~2.1 ms for 1080p on this machine) and becomes free —
`fold_into_map` below is where that will live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Orientation:
    """One eye's orientation policy. Mirrors `orbiter_model._DEFAULT_EYE`."""

    quarter_turns_cw: int = 0
    flip_h: bool = False
    flip_v: bool = False

    @classmethod
    def from_eye(cls, eye: dict[str, Any] | None) -> "Orientation":
        """Build from a `stereo_rig.left` / `.right` dict as the server sends it.

        Tolerant of missing or malformed keys — a config that lost a field
        should orient the image wrongly at worst, never crash a worker thread.
        """
        eye = eye or {}
        try:
            turns = int(eye.get("quarter_turns_cw", 0)) % 4
        except (TypeError, ValueError):
            turns = 0
        return cls(
            quarter_turns_cw=turns,
            flip_h=bool(eye.get("flip_h", False)),
            flip_v=bool(eye.get("flip_v", False)),
        )

    @property
    def is_identity(self) -> bool:
        return not (self.quarter_turns_cw or self.flip_h or self.flip_v)

    @property
    def swaps_axes(self) -> bool:
        """True when the output is transposed relative to the input (90°/270°)."""
        return self.quarter_turns_cw % 2 == 1


#: 90° steps → the exact cv2 rotation code. Index is `quarter_turns_cw`.
_ROTATE_CODE = {
    1: cv2.ROTATE_90_CLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply(img: np.ndarray, o: Orientation) -> np.ndarray:
    """Orient one frame. Flips first, then the rotation.

    Returns the input array itself when the orientation is identity — callers
    must not mutate the result in place.
    """
    if o.is_identity:
        return img
    if o.flip_h and o.flip_v:
        img = cv2.flip(img, -1)      # both axes in one pass
    elif o.flip_h:
        img = cv2.flip(img, 1)       # around the vertical axis → mirror L/R
    elif o.flip_v:
        img = cv2.flip(img, 0)       # around the horizontal axis
    code = _ROTATE_CODE.get(o.quarter_turns_cw)
    if code is not None:
        img = cv2.rotate(img, code)
    return img


def map_point(x: float, y: float, w: int, h: int, o: Orientation) -> tuple[float, float]:
    """Map a point from ORIGINAL image coordinates into oriented ones.

    `w`, `h` are the original frame's dimensions. Needed whenever something is
    detected on the original pixels but drawn over the oriented view — and the
    reverse is needed to report a detection in sensor coordinates, which is what
    a calibration consumer wants.
    """
    if o.flip_h:
        x = (w - 1) - x
    if o.flip_v:
        y = (h - 1) - y
    turns = o.quarter_turns_cw % 4
    for _ in range(turns):
        # One 90° clockwise step on a w×h image: (x, y) → (h-1-y, x),
        # and the frame's width and height trade places.
        x, y = (h - 1) - y, x
        w, h = h, w
    return x, y


def fold_into_map(
    map_x: np.ndarray, map_y: np.ndarray, o: Orientation
) -> tuple[np.ndarray, np.ndarray]:
    """Fold this orientation into an existing `cv2.remap` sampling map.

    Not used yet — there is no undistort/rectify map for the pair until per-eye
    intrinsics exist. It is here so that when that map arrives the orientation
    joins it instead of staying a second pass over the pixels, and so the fold
    happens in the one module that owns the ordering.

    A remap map is *sampling* coordinates: output pixel (r, c) is read from
    input (map_y[r, c], map_x[r, c]). Orienting the OUTPUT is therefore the same
    permutation applied to the maps themselves.
    """
    return apply(map_x, o), apply(map_y, o)
