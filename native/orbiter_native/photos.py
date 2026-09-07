"""Photographs taken beside the scan: where they live, which ones are worth
keeping, and who writes them to disk.

Photogrammetry wants photographs with known poses. The scanner already solves
one board pose per frame pair and smooths it over time, so every photograph it
takes can be handed out with the pose the scan itself trusted — which is why
this app never asks COLMAP to recover poses at all. What is left is the
bookkeeping: a session directory, a decision about which frames are worth a
file, a manifest that survives a crash, and a sidecar carrying the stripe
pixels, because the laser has to be masked out **offline**, hours after the
frame itself is gone and scalar counts cannot rebuild a mask.

Three constraints shaped everything here.

**The detector threads must not wait on disk.** A photograph is 300 KB of JPEG
plus a sidecar; writing it takes milliseconds, and the scan thread has a 33 ms
budget it shares with two eyes. So the decision is a pure function that runs on
the scan thread and the writing happens on a thread of its own, behind a
bounded queue that is never blocked on: a full queue drops its OLDEST entry and
counts it. Losing the oldest photograph of a burst is a far smaller loss than
stalling both cameras.

**This module imports nothing from `scanworker`.** `scanworker` imports the
scan, the smoother and the workers; a photo module underneath it would close a
cycle. The price is the two stillness constants below, duplicated rather than
imported — and `scanworker`'s `_still` is pairwise, over a batch, which is not
the window the policy wants anyway.

**The panel asks for the session's size on every tick.** A `du` over a
directory with thousands of files is not something to do in a Qt timer, so
`bytes_on_disk` is walked ONCE when the session opens, incremented by the
writer with every file it writes, and re-walked only when something outside
this module (a reconstruct) has been writing too.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .laser import StripePixels
from .scan import ScanVolume

log = logging.getLogger("orbiter_native.photos")

#: What `session.json` says it is, so a reader can refuse a file it does not
#: understand instead of guessing at missing keys.
SCHEMA = "orbiter.native.photo_session.v1"

#: Sessions live here unless the environment says otherwise. One directory per
#: session, named for the local instant it started.
SESSIONS_ENV = "ORBITER_SESSIONS_DIR"
DEFAULT_SESSIONS_ROOT = Path.home() / ".orbiter-native" / "sessions"

#: The two eyes, in the order every manifest and every directory listing uses.
SIDES = ("left", "right")

#: Every pose in this session is the board's, in one camera's frame, in
#: millimetres — `cvcore.estimate_pose`'s centred, face-out board frame.
POSE_CONVENTION = "board->camera, world=board"
FRAME_NOTE = ("board — origin at the board centre, +z out of the printed face, "
              "y up; millimetres")
#: Photographs are the camera's own bytes. The per-eye display transforms are
#: applied by the viewer and by nothing else, and the intrinsics were solved
#: in the raw sensor frame, so a rotated copy would not match its own K.
EYE_NOTE = ("raw sensor frame; quarter_turns_cw/flip are display-only and "
            "NOT applied here")

#: Two board poses this close together are one pose: the rig is standing
#: still. Duplicated from `scanworker.py:83-84` (`STILL_MM` / `STILL_DEG`)
#: rather than imported, because this module must stay underneath
#: `scanworker` in the import graph.
STILL_MM = 0.5
STILL_DEG = 0.1
#: Poses looked at when asking whether the rig is still. At 30 fps this is a
#: sixth of a second — long enough that a hand drifting slowly fails it,
#: short enough that a deliberate pause passes it almost at once.
STILL_HISTORY = 5

#: Board corners across both eyes below which a pose is too loose to hang a
#: photograph on — `scanworker.py:100-104`, same number, same reason.
MIN_POSE_CORNERS = 12
#: How far the eyes' independent board poses may stand apart and still be one
#: pose seen twice — `scanworker.py:91-98`.
POSE_GAP_MAX_DEG = 2.0
POSE_GAP_MAX_MM = 15.0

#: A photograph is novel when it is not BOTH near an existing one in position
#: and near it in viewing direction. 20 mm at the working distance is about
#: 4° of parallax, which is the least a second view is worth having.
NOVELTY_MM = 20.0
NOVELTY_DEG = 8.0
#: No side takes two photographs closer together than this. The novelty gate
#: would usually catch them anyway; this bounds the cost of a hand that
#: wobbles across the threshold.
MIN_INTERVAL_S = 0.5

#: A frame is sharp enough when it reaches this fraction of the running median
#: of what the run has been offering. Relative, not absolute: the Laplacian
#: variance depends on the subject's own texture, so no fixed number travels.
SHARPNESS_FRAC = 0.7
#: Offers the median is taken over, and how many must have been seen before
#: the gate means anything. Under ten samples the median is the noise.
SHARPNESS_WINDOW = 30
SHARPNESS_MIN_SAMPLES = 10

#: Photographs the writer may fall behind by. Thirty-two is about ten seconds
#: of a fast pass; past it the oldest is dropped, because a stalled disk must
#: never become a stalled camera.
QUEUE_DEPTH = 32
#: How often the writer thread looks up to see whether it has been stopped.
_POLL_S = 0.1


def sessions_root() -> Path:
    """Where sessions are kept: `$ORBITER_SESSIONS_DIR` when it is set and
    not empty, otherwise `~/.orbiter-native/sessions`."""
    named = os.environ.get(SESSIONS_ENV, "").strip()
    return Path(named) if named else DEFAULT_SESSIONS_ROOT


def photo_name(side: str, n: int) -> str:
    """`left_0001.jpg`. This basename is also the `NAME` COLMAP is given and
    the stem of every mask file, so it is formatted in exactly one place."""
    return f"{side}_{n:04d}.jpg"


def stripe_name(side: str, n: int) -> str:
    """`left_0001.npz` — the sidecar beside `left_0001.jpg`."""
    return f"{side}_{n:04d}.npz"


def _quat_wxyz(R: np.ndarray) -> list[float]:
    """A rotation as a Hamilton quaternion, scalar first — the order COLMAP's
    `images.txt` uses, so the manifest and the model agree without a swap."""
    q = Rotation.from_matrix(np.asarray(R, float)).as_quat(scalar_first=True)
    return [float(v) for v in q]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two rotations, in degrees."""
    cos = (np.trace(np.asarray(a, float).T @ np.asarray(b, float)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def camera_centre(R: np.ndarray, t_mm: np.ndarray) -> np.ndarray:
    """Where the camera is, in the board's frame. The pose maps board points
    into the camera (`x_cam = R x_board + t`), so the centre is `-Rᵀt`."""
    R = np.asarray(R, float)
    return -R.T @ np.asarray(t_mm, float).ravel()


def view_direction(R: np.ndarray) -> np.ndarray:
    """Where the camera is looking, in the board's frame: its own +z axis
    carried across, which is the third row of `R`."""
    return np.asarray(R, float)[2].copy()


# ── the rig, as the session records it ───────────────────────────────────


@dataclass(frozen=True)
class BoardSnapshot:
    """The ChArUco board the poses are expressed against."""

    squares_x: int = 0
    squares_y: int = 0
    square_mm: float = 0.0
    marker_mm: float = 0.0
    #: The dictionary's name as the server calls it, not its OpenCV int: the
    #: manifest is read by people and by a converter, neither of which wants
    #: to look up `DICT_5X5_1000` from `7`.
    dictionary: str = ""


@dataclass(frozen=True)
class EyeSnapshot:
    """One eye's identity and the intrinsics its photographs were taken at."""

    camera_id: str = ""
    wh: tuple[int, int] = (0, 0)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    #: k1, k2, p1, p2, k3 — OpenCV's order, as the server stores them.
    dist: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    rms_px: float = float("nan")


@dataclass(frozen=True)
class Extrinsics:
    """The pair's geometry: `x_right = R x_left + t`, millimetres."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rms_px: float = float("nan")


@dataclass(frozen=True)
class RigSnapshot:
    """Everything about the rig a reconstruction needs, frozen at the instant
    the session opened.

    Frozen on purpose: the operator can re-solve the pair while photographs
    are still being taken, and a session whose intrinsics changed halfway
    through is one nobody can reconstruct. What is written is what was true
    when the first photograph was.
    """

    board: BoardSnapshot | None = None
    volume: ScanVolume = ScanVolume()
    left: EyeSnapshot | None = None
    right: EyeSnapshot | None = None
    extrinsics: Extrinsics | None = None

    def to_dict(self) -> dict[str, Any]:
        """The `board`/`frame`/`volume`/`eyes`/`extrinsics` block of
        `session.json`."""
        eyes: dict[str, Any] = {}
        for side in SIDES:
            eye = getattr(self, side)
            if eye is None:
                continue
            eyes[side] = {
                "camera_id": eye.camera_id,
                "wh": [int(eye.wh[0]), int(eye.wh[1])],
                "intrinsics": {"fx": eye.fx, "fy": eye.fy, "cx": eye.cx, "cy": eye.cy,
                               "dist": [float(d) for d in eye.dist],
                               "rms_px": eye.rms_px},
                "note": EYE_NOTE,
            }
        ext = self.extrinsics
        return {
            "board": None if self.board is None else asdict(self.board),
            "frame": FRAME_NOTE,
            "volume": {"height_mm": self.volume.height_mm,
                       "radius_mm": self.volume.radius_mm,
                       "floor_mm": self.volume.floor_mm},
            "eyes": eyes,
            "extrinsics": None if ext is None else {
                "R": np.asarray(ext.R, float).reshape(3, 3).tolist(),
                "T_mm": np.asarray(ext.t_mm, float).ravel().tolist(),
                "rms_px": ext.rms_px,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RigSnapshot:
        """Rebuild a snapshot from what `to_dict` wrote. Missing blocks come
        back as None rather than as an error: a session opened before the pair
        was solved has no extrinsics, and that is a fact about it, not a
        corrupt file."""
        board = data.get("board")
        vol = data.get("volume") or {}
        ext = data.get("extrinsics")
        eyes = data.get("eyes") or {}
        return cls(
            board=None if not isinstance(board, dict) else BoardSnapshot(
                squares_x=int(board.get("squares_x", 0)),
                squares_y=int(board.get("squares_y", 0)),
                square_mm=float(board.get("square_mm", 0.0)),
                marker_mm=float(board.get("marker_mm", 0.0)),
                dictionary=str(board.get("dictionary", "")),
            ),
            volume=ScanVolume(height_mm=float(vol.get("height_mm", 400.0)),
                              radius_mm=float(vol.get("radius_mm", 150.0)),
                              floor_mm=float(vol.get("floor_mm", 5.0))),
            left=_eye_from_dict(eyes.get("left")),
            right=_eye_from_dict(eyes.get("right")),
            extrinsics=None if not isinstance(ext, dict) else Extrinsics(
                R=np.asarray(ext.get("R"), float).reshape(3, 3),
                t_mm=np.asarray(ext.get("T_mm"), float).ravel(),
                rms_px=float(ext.get("rms_px", float("nan"))),
            ),
        )


def _eye_from_dict(raw: Any) -> EyeSnapshot | None:
    if not isinstance(raw, dict):
        return None
    k = raw.get("intrinsics") or {}
    wh = raw.get("wh") or (0, 0)
    return EyeSnapshot(
        camera_id=str(raw.get("camera_id", "")),
        wh=(int(wh[0]), int(wh[1])),
        fx=float(k.get("fx", 0.0)), fy=float(k.get("fy", 0.0)),
        cx=float(k.get("cx", 0.0)), cy=float(k.get("cy", 0.0)),
        dist=tuple(float(d) for d in (k.get("dist") or ())),
        rms_px=float(k.get("rms_px", float("nan"))),
    )


# ── one pair, and one photograph out of it ───────────────────────────────


@dataclass
class EyePhoto:
    """One eye's half of a candidate: its pixels, its instant, its pose.

    The two eyes are kept apart all the way to the manifest because almost
    nothing about them is shared. Each has its own bytes, its own capture
    instant, its own frame size and its own pose — the left's is the smoothed
    pose itself, the right's is that pose carried across the pair
    (`stereo.compose_right_pose`). Writing one pose to both photographs would
    be wrong by the baseline, which is 144 mm on this rig.
    """

    camera_id: str = ""
    #: The camera's own JPEG, untouched. None when this eye was not retaining
    #: bytes, which is what the policy refuses on.
    jpeg: bytes | None = None
    wh: tuple[int, int] = (0, 0)
    #: This eye's OWN capture instant on camserver's clock, before pairing
    #: rewrote it to the left's.
    capture_mono: float = 0.0
    #: The board's pose in THIS camera's frame, millimetres.
    R: np.ndarray | None = None
    t_mm: np.ndarray | None = None
    sharpness: float = float("nan")
    #: The pixels this eye called stripe, kept for the offline mask builder.
    stripe: StripePixels | None = None
    #: True when these pixels were interpolated to the other eye's instant by
    #: `align_right` rather than detected at this eye's own. The mask builder
    #: needs to know which it is holding.
    stripe_shifted: bool = False


@dataclass
class PhotoCandidate:
    """One frame pair offered as a photograph, before anyone decides.

    Built on the scan thread from the pair the scan just fused, and consumed
    by `CapturePolicy` and then — for whichever sides survive — by the writer.
    """

    left: EyePhoto = field(default_factory=EyePhoto)
    right: EyePhoto = field(default_factory=EyePhoto)
    #: The instant the POSE holds for, which is the left eye's. The right
    #: photograph can be up to `PAIR_WINDOW_S` (20 ms) from it; recorded so
    #: the asymmetry can be audited rather than discovered.
    pair_capture_mono: float = 0.0
    #: Where the pose came from: "left+right", "left" or "right".
    pose_source: str = ""
    pose_rms_px: float = float("nan")
    #: How far the eyes' independent poses stood apart. NaN with one eye.
    pose_gap_deg: float = float("nan")
    pose_gap_mm: float = float("nan")
    #: Board corners across both eyes the pose was fitted through.
    pose_corners: int = 0
    #: How far the smoother moved this frame's pose from its own solve.
    pose_smooth_mm: float = float("nan")
    #: Bumped on every "photo pass" toggle, so the passes can be told apart
    #: when the operator has switched the laser off between them.
    pass_id: int = 0
    laser_on: bool = True
    #: The frame's kept points in the BOARD frame, filled in after `_place`
    #: has run. Empty in photo-pass mode, where there is no frame at all.
    kept_xyz_board: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.float32))

    def eye(self, side: str) -> EyePhoto:
        if side not in SIDES:
            raise ValueError(f"no such eye: {side!r}")
        return self.left if side == "left" else self.right

    def record(self, side: str) -> PhotoRecord:
        """This candidate's one side, as the writer wants it."""
        eye = self.eye(side)
        if eye.jpeg is None or eye.R is None or eye.t_mm is None:
            raise ValueError(f"the {side} eye has no photograph to write")
        return PhotoRecord(
            side=side, camera_id=eye.camera_id, jpeg=eye.jpeg, wh=eye.wh,
            capture_mono=eye.capture_mono, pair_capture_mono=self.pair_capture_mono,
            R=np.asarray(eye.R, float).reshape(3, 3),
            t_mm=np.asarray(eye.t_mm, float).ravel(),
            pose_source=self.pose_source,
            # The right eye's pose is the left's composed across the pair, and
            # the left's is the smoother's own output (§2.4).
            pose_composed=(side == "right"),
            pose_rms_px=self.pose_rms_px,
            pose_gap_deg=self.pose_gap_deg, pose_gap_mm=self.pose_gap_mm,
            pose_corners=self.pose_corners, pose_smooth_mm=self.pose_smooth_mm,
            sharpness=eye.sharpness, pass_id=self.pass_id, laser_on=self.laser_on,
            stripe=eye.stripe, stripe_shifted=eye.stripe_shifted,
            kept_xyz_board=np.asarray(self.kept_xyz_board, np.float32).reshape(-1, 3),
        )


@dataclass
class PhotoRecord:
    """One photograph on its way to disk: the bytes, the pose, the sidecar's
    contents and everything the manifest line says about them."""

    side: str
    camera_id: str
    jpeg: bytes
    wh: tuple[int, int]
    capture_mono: float
    pair_capture_mono: float
    R: np.ndarray
    t_mm: np.ndarray
    pose_source: str
    pose_composed: bool
    pose_rms_px: float
    pose_gap_deg: float
    pose_gap_mm: float
    pose_corners: int
    pose_smooth_mm: float
    sharpness: float
    pass_id: int
    laser_on: bool
    stripe: StripePixels | None
    stripe_shifted: bool
    kept_xyz_board: np.ndarray

    def manifest(self, n: int) -> dict[str, Any]:
        """One `photos.jsonl` object. `n` is this side's running number, which
        the writer assigns — so the numbers count what actually reached the
        disk, not what was offered."""
        return {
            "n": n,
            "side": self.side,
            "camera_id": self.camera_id,
            "file": f"photos/{photo_name(self.side, n)}",
            "stripe": f"stripe/{stripe_name(self.side, n)}",
            "wh": [int(self.wh[0]), int(self.wh[1])],
            "capture_mono": float(self.capture_mono),
            "pair_capture_mono": float(self.pair_capture_mono),
            "pose": {"q_wxyz": _quat_wxyz(self.R),
                     "t_mm": [float(v) for v in np.asarray(self.t_mm, float).ravel()],
                     "convention": POSE_CONVENTION},
            "pose_frame": self.side,
            "pose_source": self.pose_source,
            "pose_composed": bool(self.pose_composed),
            "pose_rms_px": float(self.pose_rms_px),
            "pose_gap_deg": float(self.pose_gap_deg),
            "pose_gap_mm": float(self.pose_gap_mm),
            "pose_corners": int(self.pose_corners),
            "pose_smooth_mm": float(self.pose_smooth_mm),
            "sharpness": float(self.sharpness),
            "pass_id": int(self.pass_id),
            "laser_on": bool(self.laser_on),
            "stripe_pixels": 0 if self.stripe is None else int(self.stripe.count),
            "stripe_shifted": bool(self.stripe_shifted),
            "kept_points": int(len(self.kept_xyz_board)),
        }

    def sidecar(self) -> dict[str, Any]:
        """What goes into `stripe/<side>_<n>.npz`: the `StripePixels` arrays
        exactly as `laser.py` holds them, plus the frame's kept points. The
        mask is built offline, when the frame itself is long gone, and a
        pixel count cannot be turned back into a mask."""
        s = self.stripe
        empty_i = np.empty(0, np.int32)
        empty_u = np.empty(0, np.uint8)
        return {
            "x": empty_i if s is None else np.asarray(s.x, np.int32),
            "y": empty_i if s is None else np.asarray(s.y, np.int32),
            "w": empty_u if s is None else np.asarray(s.w, np.uint8),
            "r": empty_u if s is None else np.asarray(s.r, np.uint8),
            "kept_xyz_board": np.asarray(self.kept_xyz_board, np.float32).reshape(-1, 3),
            "wh": np.asarray(self.wh, np.int32),
            "along_x": np.asarray(True if s is None else s.along_x, bool),
            "pass_id": np.asarray(self.pass_id, np.int32),
            "stripe_shifted": np.asarray(self.stripe_shifted, bool),
        }


# ── the decision ─────────────────────────────────────────────────────────


def _spread(window: Sequence[tuple[float, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """How far the poses in `window` stand from the newest of them, in
    millimetres and degrees — the largest of each."""
    _, R_now, t_now = window[-1]
    t_now = np.asarray(t_now, float).ravel()
    mm = max(float(np.linalg.norm(np.asarray(t, float).ravel() - t_now))
             for _, _, t in window)
    deg = max(_angle_deg(R_now, R) for _, R, _ in window)
    return mm, deg


@dataclass(frozen=True)
class CapturePolicy:
    """Whether one side of one candidate is worth a file.

    A pure decision: everything it looks at is passed in, nothing is
    remembered between calls, and it imports nothing from the scanner. That
    is what lets the whole gate be tested without a camera, a board or a Qt
    event loop — and it is also why the stillness constants are duplicated at
    the top of this module rather than imported from `scanworker`.

    **Why stillness is the first requirement, and not a sharpness heuristic.**
    These are rolling-shutter sensors: a photograph is exposed row by row over
    the sensor's whole readout, so its rows were taken at different instants
    and, if the rig was moving, from different poses. COLMAP is handed ONE
    pose per image and will treat every row as if it held. The 0.5 mm / 0.1°
    gate is the assumption that makes that true — at those numbers the spread
    across a readout is far under a pixel — not a way of avoiding blur. Blur
    has its own gate, further down.

    The rest is economy. A viewpoint already photographed adds nothing to a
    bundle that is not being adjusted; a frame the run itself would call soft
    is a frame whose features will not match; and half a second is the floor
    on how often one side may spend a file.
    """

    #: Stillness, over the last `still_history` poses. See `STILL_MM`.
    still_mm: float = STILL_MM
    still_deg: float = STILL_DEG
    still_history: int = STILL_HISTORY
    #: Pose quality: a one-eyed pose needs this many corners, and a two-eyed
    #: one needs the eyes to agree.
    min_corners: int = MIN_POSE_CORNERS
    max_gap_deg: float = POSE_GAP_MAX_DEG
    max_gap_mm: float = POSE_GAP_MAX_MM
    #: Novelty against the photographs already kept on this side.
    novelty_mm: float = NOVELTY_MM
    novelty_deg: float = NOVELTY_DEG
    min_interval_s: float = MIN_INTERVAL_S
    #: Sharpness, relative to the run's own running median.
    sharpness_frac: float = SHARPNESS_FRAC
    sharpness_window: int = SHARPNESS_WINDOW
    sharpness_min_samples: int = SHARPNESS_MIN_SAMPLES

    def decide(self, cand: PhotoCandidate, side: str,
               poses: Sequence[tuple[float, np.ndarray, np.ndarray]],
               kept: Sequence[tuple[float, np.ndarray, np.ndarray]],
               offers: Sequence[float]) -> str | None:
        """None when this side is worth a file, otherwise the reason it is
        not — phrased for the panel, not for a log grep.

        `poses` is the recent board→left pose history as `(capture_mono, R,
        t_mm)`, oldest first, with this candidate's own pose last. `kept` is
        the same shape for the photographs already written on THIS side, so
        one list answers both the interval and the novelty. `offers` is the
        sharpness of the recent offers, newest last — the run's own scale.
        """
        eye = cand.eye(side)
        if eye.jpeg is None or eye.R is None or eye.t_mm is None:
            return f"the {side} eye kept no pixels"

        if cand.pose_source != "left+right" and cand.pose_corners < self.min_corners:
            return (f"the pose came from the {cand.pose_source or 'unknown'} eye alone "
                    f"with {cand.pose_corners} corners, {self.min_corners} needed")
        # The gaps are NaN when only one eye saw the board — nothing to
        # compare, and a NaN comparison would answer False anyway. Said
        # explicitly so the gate reads as the deliberate pass it is.
        if cand.pose_gap_deg == cand.pose_gap_deg and (
                cand.pose_gap_deg > self.max_gap_deg or cand.pose_gap_mm > self.max_gap_mm):
            return (f"the eyes' board poses differ by {cand.pose_gap_deg:.1f}° / "
                    f"{cand.pose_gap_mm:.0f} mm")

        window = list(poses)[-self.still_history:]
        if len(window) < self.still_history:
            return (f"only {len(window)} poses so far — stillness is judged over "
                    f"{self.still_history}")
        mm, deg = _spread(window)
        if mm > self.still_mm or deg > self.still_deg:
            return (f"moving: {mm:.2f} mm / {deg:.2f}° across the last "
                    f"{self.still_history} poses")

        if kept:
            since = eye.capture_mono - float(kept[-1][0])
            if since < self.min_interval_s:
                return (f"{since:.2f} s since the last {side} photograph, "
                        f"{self.min_interval_s:.1f} s apart is the floor")
            near = self._nearest(eye, kept)
            if near is not None:
                return (f"{near[0]:.0f} mm and {near[1]:.1f}° from a photograph "
                        f"already kept — the same viewpoint")

        recent = list(offers)[-self.sharpness_window:]
        if len(recent) >= self.sharpness_min_samples:
            floor = self.sharpness_frac * float(np.median(recent))
            if not (eye.sharpness >= floor):
                return (f"sharpness {eye.sharpness:.0f} against a running median of "
                        f"{float(np.median(recent)):.0f}")
        return None

    def _nearest(self, eye: EyePhoto,
                 kept: Sequence[tuple[float, np.ndarray, np.ndarray]]
                 ) -> tuple[float, float] | None:
        """The kept photograph this one repeats — near in position AND near in
        direction — as `(mm, degrees)`, or None when every one of them differs
        in at least one of the two."""
        centre = camera_centre(eye.R, eye.t_mm)
        look = view_direction(eye.R)
        for _, R_k, t_k in kept:
            mm = float(np.linalg.norm(centre - camera_centre(R_k, t_k)))
            look_k = view_direction(R_k)
            cos = float(np.dot(look, look_k)
                        / (np.linalg.norm(look) * np.linalg.norm(look_k)))
            deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            if mm < self.novelty_mm and deg < self.novelty_deg:
                return mm, deg
        return None


# ── the session on disk ──────────────────────────────────────────────────


class PhotoSession:
    """One capture session: a directory, a manifest, and the counters the
    panel shows.

    The directory is named for the local instant it opened, because that is
    how an operator finds it again a week later; `started_utc` inside the
    manifest is the same instant without the ambiguity.

    Counters are touched by the writer thread and read by the GUI thread, so
    they live under a lock — a small one, held for a few assignments and
    never across a write.
    """

    def __init__(self, root: Path | None = None,
                 rig: RigSnapshot | None = None,
                 policy: CapturePolicy | None = None,
                 started: datetime | None = None) -> None:
        started = (started or datetime.now()).astimezone()
        self.root = Path(root) if root is not None else sessions_root()
        self.rig = rig or RigSnapshot()
        self.policy = policy or CapturePolicy()
        self.started = started
        self.session_id = started.strftime("%Y%m%d-%H%M%S")
        self.path = self._make_dir()
        self.photos_dir = self.path / "photos"
        self.stripe_dir = self.path / "stripe"
        self.photos_dir.mkdir(exist_ok=True)
        self.stripe_dir.mkdir(exist_ok=True)
        self.manifest_path = self.path / "photos.jsonl"
        self.json_path = self.path / "session.json"

        self._lock = threading.Lock()
        self._counts = {side: 0 for side in SIDES}
        self._pass_id = 0
        self.json_path.write_text(json.dumps(self.to_dict(), indent=2),
                                  encoding="utf-8")
        # One walk, here and nowhere near a Qt timer: from now on the writer
        # adds what it writes and the number stays true without touching the
        # filesystem again.
        self._bytes = self._walk()

    def _make_dir(self) -> Path:
        """The session's own directory. A second session inside the same
        second gets a suffix rather than sharing — two sessions interleaved in
        one manifest is not something a reconstruction can untangle."""
        path = self.root / self.session_id
        for suffix in range(1, 100):
            if not path.exists():
                break
            path = self.root / f"{self.session_id}-{suffix}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── the manifest ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """`session.json` as it is written at the session's start. Every
        threshold that decided anything goes in, so a run can be told apart
        from another run by reading one file."""
        return {
            "schema": SCHEMA,
            "session_id": self.session_id,
            "started_utc": self.started.astimezone(timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
            **self.rig.to_dict(),
            "params": {"capture": asdict(self.policy)},
        }

    # ── counters ─────────────────────────────────────────────────────────

    @property
    def counts(self) -> dict[str, int]:
        """Photographs written per side."""
        with self._lock:
            return dict(self._counts)

    @property
    def pass_id(self) -> int:
        with self._lock:
            return self._pass_id

    def next_pass(self) -> int:
        """Start a new pass — the operator has just flipped the laser switch,
        and photographs taken either side of that are not the same evidence."""
        with self._lock:
            self._pass_id += 1
            return self._pass_id

    def next_index(self, side: str) -> int:
        """This side's next running number, claimed. Called by the writer, so
        the numbers count files on disk rather than offers made."""
        if side not in SIDES:
            raise ValueError(f"no such eye: {side!r}")
        with self._lock:
            self._counts[side] += 1
            return self._counts[side]

    @property
    def bytes_on_disk(self) -> int:
        """What this session occupies. Counted, not walked — the panel asks
        for it on every tick."""
        with self._lock:
            return self._bytes

    def wrote(self, n_bytes: int) -> None:
        """The writer's report of what it just put on disk."""
        with self._lock:
            self._bytes += int(n_bytes)

    def rewalk(self) -> int:
        """Walk the directory again and take the answer as the truth. For
        after a reconstruct, which writes gigabytes this module never sees."""
        total = self._walk()
        with self._lock:
            self._bytes = total
            return self._bytes

    def _walk(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())


# ── the writer thread ────────────────────────────────────────────────────


class PhotoWriter:
    """Writes photographs off the scan thread, and never makes it wait.

    The queue is bounded and is never blocked on. When it is full the OLDEST
    entry is dropped and counted: the scan thread's job is to keep the
    cameras paired at 30 Hz, and a disk that has fallen ten seconds behind
    must not be able to stop it. The count is shown in the panel, so a run
    that is losing photographs says so instead of looking healthy.

    A photograph, its sidecar and its manifest line are written as one unit,
    in that order, so a crash can leave a file the manifest does not mention —
    which a reconstruction ignores — and never a manifest line without its
    file, which it would try to open.
    """

    def __init__(self, session: PhotoSession, depth: int = QUEUE_DEPTH) -> None:
        self.session = session
        self._q: queue.Queue[PhotoRecord] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._dropped = 0
        self._written = 0

    @property
    def dropped(self) -> int:
        """Photographs the queue threw away to stay ahead of the scan."""
        with self._lock:
            return self._dropped

    @property
    def written(self) -> int:
        with self._lock:
            return self._written

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="photo-write",
                                        daemon=True)
        self._thread.start()

    def put_nowait(self, rec: PhotoRecord) -> None:
        """Hand a photograph over. Returns at once, always — this is called
        from the scan thread."""
        try:
            self._q.put_nowait(rec)
            return
        except queue.Full:
            pass
        try:
            self._q.get_nowait()
        except queue.Empty:      # the writer emptied it between the two calls
            pass
        else:
            with self._lock:
                self._dropped += 1
        try:
            self._q.put_nowait(rec)
        except queue.Full:       # it filled up again; this one goes instead
            with self._lock:
                self._dropped += 1

    def stop(self, timeout: float = 5.0) -> None:
        """Drain what is queued, then join. Called from `closeEvent`, where
        losing the last few photographs of a session would be gratuitous."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                log.warning("the photo writer did not finish within %.1f s", timeout)
                return
        while True:                     # nothing is running now; finish by hand
            try:
                self._write(self._q.get_nowait())
            except queue.Empty:
                return

    def _run(self) -> None:
        while True:
            try:
                rec = self._q.get(timeout=_POLL_S)
            except queue.Empty:
                if self._stop.is_set():
                    return              # stopped AND drained
                continue
            try:
                self._write(rec)
            except OSError:
                log.exception("could not write a %s photograph", rec.side)

    def _write(self, rec: PhotoRecord) -> None:
        n = self.session.next_index(rec.side)
        jpeg_path = self.session.photos_dir / photo_name(rec.side, n)
        npz_path = self.session.stripe_dir / stripe_name(rec.side, n)
        # Verbatim: the intrinsics were solved against these exact pixels, and
        # a re-encode would put COLMAP on a different image than the model.
        jpeg_path.write_bytes(rec.jpeg)
        # Uncompressed on purpose — this runs per photograph, and the arrays
        # are a few tens of kilobytes of already-incompressible coordinates.
        with npz_path.open("wb") as fh:
            np.savez(fh, **rec.sidecar())
        line = (json.dumps(rec.manifest(n), separators=(",", ":")) + "\n").encode("utf-8")
        # Binary append, so Windows does not turn the newline into CRLF —
        # which would make the manifest disagree with its own byte count and
        # give every line a stray carriage return.
        with self.session.manifest_path.open("ab") as fh:
            fh.write(line)
        self.session.wrote(jpeg_path.stat().st_size + npz_path.stat().st_size + len(line))
        with self._lock:
            self._written += 1
