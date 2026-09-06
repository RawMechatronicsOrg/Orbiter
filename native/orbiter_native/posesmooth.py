"""The board's pose smoothed over time, because a hand cannot jump.

The scan places every frame's points through that frame's own board pose.
A pose comes from whichever corners the detector found, and at the frame's
edge, in a glint, behind the stripe, a few corners come and go from one
frame to the next; each change moves the pose by a fraction of a degree
and a millimetre or two — with the scanner standing perfectly still. Every
such frame then lays the same surface down a little to one side, and a
still scan grows a fuzz of ghosts around itself.

Motion is not like that: the scanner is a hand, and a hand moves smoothly
at a few tens of millimetres a second — a millimetre a frame. So a pose is
taken as the MEDIAN of the poses around it in time, a centred window of
`WINDOW` frames: a lone jump is outvoted by its neighbours, noise averages
down, and a steady sweep passes through unchanged, because the median of
a straight run is its middle. The price is `WINDOW // 2` frames of delay
before a frame's points join the cloud — a tenth of a second, which the
cloud does not mind.

The window does not reach across a gap in time: after `MAX_GAP_S` without
a frame the pending frames are placed with what they have, and the next
frame starts a window of its own.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

#: Frames in the window, odd: as many after the frame as before it.
WINDOW = 7
#: Frames farther apart than this are not neighbours.
MAX_GAP_S = 0.5


def median_pose(poses: list[tuple[np.ndarray, np.ndarray]],
                centre: int) -> tuple[np.ndarray, np.ndarray]:
    """The component-wise median of the translations and, for the
    rotations, the median rotation vector taken relative to the pose at
    `centre` — small vectors, where a per-axis median means something —
    put back onto it."""
    ts = np.array([np.asarray(t, np.float64).ravel() for _, t in poses])
    t = np.median(ts, axis=0)
    r_ref = np.asarray(poses[centre][0], np.float64)
    vecs = np.array([cv2.Rodrigues(r_ref.T @ np.asarray(r, np.float64))[0].ravel()
                     for r, _ in poses])
    r = r_ref @ cv2.Rodrigues(np.median(vecs, axis=0))[0]
    return r, t


class PoseSmoother:
    """Feed frames in with their poses; get them back, `WINDOW // 2` frames
    later, with the median pose of the frames around them."""

    def __init__(self, window: int = WINDOW, max_gap_s: float = MAX_GAP_S) -> None:
        if window < 1 or window % 2 == 0:
            raise ValueError("the window must be odd, so a frame sits in its middle")
        self.window = window
        self.max_gap_s = max_gap_s
        self._buf: deque = deque(maxlen=window)
        #: Items pushed so far, and how many of them have been handed back.
        self._seen = 0
        self._emitted = 0

    @property
    def pending(self) -> int:
        """Frames held back, waiting for their neighbours."""
        return self._seen - self._emitted

    def push(self, item: Any, R: np.ndarray, t: np.ndarray, when: float) -> list[tuple]:
        """Add a frame; returns `[(item, R, t), ...]` for every frame whose
        pose is now decided — the one at the window's centre, and, after a
        gap in time, every frame that was still waiting."""
        out: list[tuple] = []
        if self._buf and when - self._buf[-1][3] > self.max_gap_s:
            out += self.flush()
        self._buf.append((item, np.asarray(R, np.float64), np.asarray(t, np.float64).ravel(), when))
        self._seen += 1
        if len(self._buf) == self.window:
            out += self._emit(self.window // 2)
        return out

    def flush(self) -> list[tuple]:
        """Hand back every frame still waiting, each with the median of
        the frames around it as far as they go."""
        out: list[tuple] = []
        first = len(self._buf) - self.pending
        for i in range(max(first, 0), len(self._buf)):
            out += self._emit(i)
        self._buf.clear()
        self._seen = self._emitted = 0
        return out

    def _emit(self, index: int) -> list[tuple]:
        number = self._seen - len(self._buf) + index
        if number < self._emitted:
            return []
        poses = [(r, t) for _, r, t, _ in self._buf]
        r, t = median_pose(poses, index)
        self._emitted = number + 1
        return [(self._buf[index][0], r, t)]
