"""Scanning off the GUI thread: pair the eyes' results, triangulate, accumulate.

Why a thread of its own. The first version ran `scan_frame` in the GUI thread,
in the slot that received each eye's result. Measured on this rig at 1080p,
that slot was doing 32 ms of triangulation per pair, 8.6 ms of frame
conversion per eye at 50 results a second, and a min/max over the whole cloud
per pair (42 ms at a million points) — more than a second of work per second.
Qt's queued signals do not drop, so the backlog grew without bound and the
window fell further behind the cameras the longer it ran. Detection already
runs off the GUI thread; this puts scanning there too, and leaves the GUI
thread nothing but painting the newest frame and reading a few counters.

Pairing. Each eye's detector skips frames on its own, so "the newest left and
the newest right" are rarely from the same instant. Both eyes' recent results
are kept and paired by camserver's capture clock — the one clock that timed
both sensors — oldest first, each result used once. On this rig that finds a
partner for about twice as many frames as newest-against-newest did.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import RigConfig
from .laser import StripePixels
from .laserplane import LaserPlane, from_config as plane_from_config
from .photos import (
    SHARPNESS_WINDOW,
    SIDES,
    EyePhoto,
    PhotoCandidate,
    PhotoSession,
    PhotoWriter,
)
from .posesmooth import PoseSmoother
from .rolling import Motion, Readout
from .scan import (
    CloudOverlay,
    Confident,
    PointCloud,
    ScanFrame,
    ScanParams,
    confident,
    sample_beside,
    scan_frame,
    stripe_rows,
    write_ply,
)
from .stereo import (
    StereoRig,
    compose_left_pose,
    compose_right_pose,
    result_from_config,
)
from .timealign import align_right
from .worker import EyeResult, Latest
from .cvcore import build_board, estimate_pose, refine_pose_pair
from dataclasses import replace
from scipy.spatial.transform import Rotation

log = logging.getLogger("orbiter_native.scanworker")

#: Two frames count as simultaneous within this much of camserver's capture
#: clock. Wider than calibration's 4 ms: the subject turns slowly by hand.
#: 10 ms found partners for 26 of 58 left frames on this rig against 11 at
#: 4 ms — and the reason it was not 58 is that the left camera ran at 20 fps
#: under auto-exposure while the right ran at 30, so every other left frame
#: had its nearest right frame 16.7 ms away. 20 ms takes those too. What it
#: costs is the subject's motion over that gap: 0.5 mm at 30 mm/s, about a
#: pixel in the right eye against a 3 px confirmation slack.
PAIR_WINDOW_S = 0.020

#: Results kept per eye while waiting for a partner. At 30 fps this is half a
#: second — far more than the two streams ever drift apart.
_HISTORY = 16

#: Cloud points handed to the eyes for drawing, at most. Projecting them costs
#: about 1 ms per 30k, per eye, per frame.
OVERLAY_MAX = 40000

#: Two left poses further apart than this say nothing about the motion
#: inside one readout: a hand changes its mind in less. No correction then.
MAX_TWIST_GAP_S = 0.2

#: While the board holds still — its pose within this of the batch's first
#: — consecutive pairs see the same surface, and their points are averaged
#: per scanline before they join the cloud: noise falls by the square root
#: of the batch. A hand-held board jitters far less than this; the pose
#: itself is repeatable to a tenth of a millimetre.
STILL_MM = 0.5
STILL_DEG = 0.1
STILL_BATCH = 5

#: The right eye keeps scoring the stripe for this long after the left eye
#: last had a board pose — a moment's loss of the board is not a reason to
#: miss the stripe the scan will want on the next frame.
POSE_RECENT_S = 0.5

#: The two eyes' own board poses may differ by this much and still be one
#: pose seen twice — five times what the pair's residual accounts for at
#: the working distance. Past it the pair's geometry describes another rig
#: (the cameras were re-aimed and the pair not redone), every point would
#: land where that other rig would have put it, and the frame is not
#: scanned; the panel says why.
POSE_GAP_MAX_DEG = 2.0
POSE_GAP_MAX_MM = 15.0
#: Board corners across both eyes below which a pose is too loose to place
#: points by: at the working distance a pose from a handful of corners
#: wanders by millimetres from frame to frame.
MIN_POSE_CORNERS = 12
#: An eye's pose is taken from the corners it has had in EVERY one of its
#: last `STEADY_FRAMES` frames, when that leaves `MIN_STEADY_CORNERS` or
#: more. The detector's full ChArUco pass, every `detect.redetect_every`
#: (10) frames, is when marginal corners come and go, and a pose solved
#: through a changing set moves although nothing did; the same corners
#: every frame give the same pose every frame. Longer than the pass
#: period, so a set survives one.
STEADY_FRAMES = 12
MIN_STEADY_CORNERS = 12

#: Smoothed board poses kept for the photo policy's stillness window. Longer
#: than `photos.STILL_HISTORY` so a policy configured with a wider window
#: still finds one, and short enough to stay a handful of frames of memory.
PHOTO_POSES = 16
#: A pause longer than this ends the stillness window rather than spanning it:
#: nobody watched the rig across the gap, so the poses either side of it are
#: not consecutive looks at a rig standing still. The smoother draws its own
#: windows at the same number (`posesmooth.MAX_GAP_S`) and for the same reason.
PHOTO_GAP_S = 0.5


def _still(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    """Two board poses within STILL_MM and STILL_DEG of each other."""
    (R0, t0), (R1, t1) = a, b
    if float(np.linalg.norm(np.asarray(t1, float).ravel() - np.asarray(t0, float).ravel())) > STILL_MM:
        return False
    cos = (np.trace(np.asarray(R0, float).T @ np.asarray(R1, float)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))) <= STILL_DEG


def average_still(frames: list[ScanFrame]) -> tuple[np.ndarray, np.ndarray | None]:
    """One point per scanline from a batch of still frames: the per-axis
    TRIMMED mean of the frames' points on that scanline — the lowest and the
    highest dropped once there are four or more — kept when at least half
    the frames had one. Trimmed, not plain: a glint that passed every gate in
    one frame of five is 40 mm off, and a plain mean would move the point
    8 mm toward it; as the extreme it is dropped instead, and the remaining
    three average with most of the mean's noise reduction (a median of five
    keeps only 70% of it). A scanline seen once in five is a flicker, not a
    surface.

    Returns `(points, colours, weights)`. The colours and the precision
    weights go through the same trimmed mean, column by column, when every
    frame of the batch carried them; a batch with a colourless frame in it
    gives None for the colours — half a colour is worse than none — and a
    frame without weights (an older ScanFrame) makes every weight 1.
    """
    coloured = all(f.colours is not None for f in frames)
    weighted = all(len(f.weights) == len(f.points_board) for f in frames)

    def one_weight(f: ScanFrame) -> np.ndarray:
        return f.weights if weighted else np.ones(len(f.points_board))

    if len(frames) == 1:
        f = frames[0]
        return f.points_board, (f.colours if coloured else None), one_weight(f)
    keys = np.concatenate([f.scanlines for f in frames])
    pts = np.concatenate([f.points_board for f in frames])
    pts = np.concatenate([pts, np.concatenate([one_weight(f) for f in frames])[:, None]], axis=1)
    if coloured:
        rgb = np.concatenate([f.colours for f in frames]).astype(np.float64)
        pts = np.concatenate([pts, rgb], axis=1)
    if not len(keys):
        return (np.empty((0, 3)), (np.empty((0, 3), np.uint8) if coloured else None),
                np.empty(0))
    order = np.argsort(keys, kind="stable")
    keys, pts = keys[order], pts[order]
    uniq, start, counts = np.unique(keys, return_index=True, return_counts=True)
    rank = np.arange(len(keys)) - np.repeat(start, counts)
    width = int(counts.max())
    table = np.full((len(uniq), width, pts.shape[1]), np.nan)
    table[np.repeat(np.arange(len(uniq)), counts), rank] = pts
    enough = counts >= -(-len(frames) // 2)
    table, counts = table[enough], counts[enough]
    table = np.sort(table, axis=1)                        # NaN sorts last, per axis
    ranks = np.arange(width)[None, :]
    trim = (counts >= 4)[:, None]
    keep = np.where(trim, (ranks >= 1) & (ranks <= counts[:, None] - 2), ranks < counts[:, None])
    weights = keep.astype(np.float64)[:, :, None]
    out = np.nansum(table * weights, axis=1) / weights.sum(axis=1)
    xyz, w = out[:, :3], out[:, 3]
    if not coloured:
        return xyz, None, w
    return xyz, np.clip(np.rint(out[:, 4:]), 0, 255).astype(np.uint8), w


@dataclass
class ScanInput:
    """What scanning needs from one eye's result — without its 6 MB frame."""

    capture_mono: float
    stripe: StripePixels | None
    board_R: np.ndarray | None
    board_t: np.ndarray | None
    wh: tuple[int, int]
    #: Mean row of the corners the pose came from: the instant, within the
    #: frame's readout, that the pose holds for. NaN without a board.
    pose_row: float = float("nan")
    #: The eye's colour image as it was published — on the GPU path the
    #: half-size copy the view draws, 1.5 MB, and a reference to it, not a
    #: copy — kept for the left eye only, so each kept point can be given
    #: the colour beside its stripe pixel.
    bgr: np.ndarray | None = None
    #: The board corners this pose came from, as the detector handed them
    #: over — what a pose fitted to both eyes at once is fitted through.
    corners: np.ndarray | None = None
    ids: np.ndarray | None = None
    #: This eye's own JPEG bytes for this frame, when photo capture is armed
    #: — a reference, never a copy. Kept for BOTH eyes, unlike `bgr`: a
    #: right-eye photo is a photo in its own right, taken from its own place
    #: on the rig, and it can only be made of the right eye's pixels.
    jpeg: bytes | None = None
    #: This frame's focus measure, NaN while photo capture is disarmed.
    sharpness: float = float("nan")


@dataclass
class PoseFix:
    """The board's pose for one pair, in the LEFT camera's frame, and where
    it came from."""

    R: np.ndarray
    t: np.ndarray
    #: "left+right" (both eyes, fitted jointly), "left" or "right".
    source: str
    #: With both eyes: how far their independent poses stood apart. A live
    #: check on the pair's calibration — the veto is the other one.
    gap_deg: float = float("nan")
    gap_mm: float = float("nan")
    #: The joint fit's reprojection error over both images, px.
    rms_px: float = float("nan")


def fuse_pose(a: ScanInput, b: ScanInput, rig: StereoRig, board=None) -> PoseFix | None:
    """One board pose from the two eyes' results: the left's, the right's
    carried into the left frame through the pair, or — when both saw the
    board — one pose fitted to both images' corners at once (the mean of the
    two when the corners are not there to fit through). None when neither
    eye saw the board. This is what lets the rig be turned any way round the
    subject: the scan carries on while either camera sees the board."""
    have_l = a.board_R is not None and a.board_t is not None
    have_r = b.board_R is not None and b.board_t is not None
    if not have_l and not have_r:
        return None
    if have_r:
        R_r, t_r = compose_left_pose(b.board_R, b.board_t, rig.geom)
    if not have_l:
        return PoseFix(R_r, t_r, "right")
    R_l, t_l = np.asarray(a.board_R, float), np.asarray(a.board_t, float).ravel()
    if not have_r:
        return PoseFix(R_l, t_l, "left")
    cos = (np.trace(R_l.T @ R_r) - 1.0) / 2.0
    gap_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    gap_mm = float(np.linalg.norm(t_l - t_r))
    fit = None
    if board is not None:
        fit = refine_pose_pair(board, rig.geom, rig.left_k, rig.right_k,
                               (a.corners, a.ids), (b.corners, b.ids), R_l, t_l)
    if fit is None:
        R = Rotation.from_matrix(np.stack([R_l, R_r])).mean().as_matrix()
        return PoseFix(R, (t_l + t_r) / 2.0, "left+right", gap_deg, gap_mm)
    R, t, rms = fit
    return PoseFix(R, t, "left+right", gap_deg, gap_mm, rms)


@dataclass
class ScanStatus:
    """For the panel: the cloud so far, and what the last pair did."""

    n_points: int
    bounds: tuple[np.ndarray, np.ndarray] | None
    pairs: int
    #: Left results offered while scanning — with `pairs`, the pairing rate.
    offered_left: int = 0
    #: Pairs held in the current still batch, waiting to be averaged.
    batched: int = 0
    frame: ScanFrame | None = None
    #: A blocking condition — no calibration, no board, laser off.
    note: str | None = None
    #: The confident cloud (see `scan.confident`): how many points it has,
    #: and how many voxels were dropped as lonely or as flickers. -1 while
    #: cleaning is off.
    n_confident: int = -1
    n_lonely: int = 0
    n_flicker: int = 0
    #: Photographs written this session, per side, and how many the writer
    #: threw away because the disk fell behind the cameras.
    photos_left: int = 0
    photos_right: int = 0
    photos_dropped: int = 0
    #: The session's directory name, the pass the operator is on, and what
    #: the session occupies on disk — all empty or zero without a session.
    session_id: str = ""
    pass_id: int = 0
    session_bytes: int = 0


class ScanWorker:
    """Owns the scan thread, the cloud, and the overlay snapshot."""

    def __init__(self) -> None:
        # Guards the history, the batch and the configuration — taken by
        # the detector threads on every result, so nothing slow runs under it.
        self._lock = threading.Lock()
        # The cloud has its own: merging and decimating are the slow parts,
        # and the eyes must not wait on them. Order, where both are held:
        # `_lock` first, never the other way round.
        self._cloud_lock = threading.Lock()
        self._hist: dict[str, deque[ScanInput]] = {
            "left": deque(maxlen=_HISTORY), "right": deque(maxlen=_HISTORY)}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._cfg: RigConfig | None = None
        self._params = ScanParams()
        self._pairs = 0
        self._offered_left = 0
        # Projection geometry, rebuilt only when the calibration or the frame
        # size changes. Touched by the scan thread alone.
        self._geom_key = None
        self._geom: tuple = (None, None, None, None)      # rig, plane, readout, board
        #: Per eye, the rows the sheet can appear in for the current reach, or
        #: None for the whole frame; the window pushes them to the workers.
        self.stripe_rows: dict[str, tuple[int, int] | None] = {"left": None, "right": None}
        # The previous pair's pose, with its capture instant and the left
        # corners' mean row: with the current one it gives the board's twist,
        # which is what slides the pose to each stripe row's instant.
        self._prev: tuple[PoseFix, float, float] | None = None
        #: When either eye last offered a board pose.
        self._pose_at = -1e9
        #: The right result the last pair used: it brackets the next left
        #: instant from below when no older right is left in the history.
        self._last_right: ScanInput | None = None
        # Consecutive frames of a still board, averaged before they join the
        # cloud; the pose they are still against is the first one's.
        self._batch: list[ScanFrame] = []
        self._batch_pose: tuple[np.ndarray, np.ndarray] | None = None
        #: Frames wait here for their neighbours in time, and are placed
        #: through the median pose of the window — see `posesmooth`.
        self._smooth = PoseSmoother()
        #: Per eye, the corner ids of its last frames, for `_steady`.
        self._ids_seen: dict[str, deque] = {"left": deque(maxlen=STEADY_FRAMES),
                                            "right": deque(maxlen=STEADY_FRAMES)}

        # ── photographs ──────────────────────────────────────────────────
        # Guards the photo bookkeeping below, and nothing else. Deliberately
        # not `_lock`: a decision walks every photograph already kept on a
        # side, and `_lock` is the one both detector threads take on every
        # result. This one is never held while `_lock` is — the decision
        # happens after that block is left — so the two cannot deadlock.
        self._photo_lock = threading.Lock()
        self._session: PhotoSession | None = None
        self._writer: PhotoWriter | None = None
        #: The decision gate. Candidates are built, judged and enqueued only
        #: while it is on, so an unarmed scan costs what it always did.
        self._armed = False
        #: Photo-pass mode: the laser is switched off, so a pair carries no
        #: stripe and exists only to be photographed.
        self._photo_pass = False
        #: The rig the pending candidates were paired through. `_process` has
        #: one to hand; `set_active` and `clear` flush candidates it built and
        #: have none, and a right photograph's pose cannot be composed without
        #: it. One reference, assigned whole, read by whoever flushes.
        self._photo_rig: StereoRig | None = None
        #: The smoothed board poses the recent candidates were judged
        #: against, oldest first — what stillness is measured over.
        self._pose_hist: deque[tuple[float, np.ndarray, np.ndarray]] = deque(
            maxlen=PHOTO_POSES)
        #: Per eye, the photographs already written as `(capture_mono, R, t)`
        #: — one list answers both the minimum interval and the novelty test.
        self._kept: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {
            side: [] for side in SIDES}
        #: Per eye, the sharpness of the recent offers. The blur gate is
        #: relative to the run's own scale, so the run has to keep one.
        self._offers: dict[str, deque[float]] = {
            side: deque(maxlen=SHARPNESS_WINDOW) for side in SIDES}

        self.cloud = PointCloud()
        self.overlay = CloudOverlay()
        #: Newest counters; the GUI takes them on its own clock.
        self.status = Latest()
        # The confident cloud, built from the grid when it changed and the
        # last build has had time to pay for itself: (grid version, the
        # cleaning parameters, the result, when, how long it took).
        self._clean: tuple[int, tuple, Confident, float, float] | None = None

    # ── the confident cloud ───────────────────────────────────────────────

    @staticmethod
    def _clean_key(p: ScanParams) -> tuple:
        return (p.clean_merge_mm, p.clean_cell_mm, p.clean_neighbours, p.clean_flicker_obs)

    def _confident(self, params: ScanParams, force: bool = False) -> Confident | None:
        """The confident cloud for the grid as it stands, or None with
        cleaning off. Rebuilt when the grid or the parameters changed, and
        no more often than three times its own cost — a million voxels take
        a few hundred milliseconds, and the scan thread has frames to pair.
        `force` rebuilds regardless: an export wants the grid as it is."""
        if not params.clean:
            return None
        key = self._clean_key(params)
        now = time.monotonic()
        with self._cloud_lock:
            version = self.cloud.version
            cached = self._clean
            if cached is not None and cached[0] == version and cached[1] == key:
                return cached[2]
            if (cached is not None and cached[1] == key and not force
                    and now - cached[3] < 3.0 * cached[4]):
                return cached[2]                            # stale, but paid for
            pts = self.cloud.points().copy()
            cnt = self.cloud.counts().copy()
            wts = self.cloud.weights().copy()
            rgb = self.cloud.colors()
            rgb = None if rgb is None else rgb.copy()
        out = confident(pts, cnt, rgb, params.clean_merge_mm, params.clean_cell_mm,
                        params.clean_neighbours, params.clean_flicker_obs, weights=wts)
        with self._cloud_lock:
            self._clean = (version, key, out, now, time.monotonic() - now)
        return out

    def _snapshot(self, params: ScanParams) -> tuple[np.ndarray, np.ndarray | None]:
        """What the eyes and the cloud view draw: the confident cloud when
        cleaning is on, every voxel otherwise, decimated to OVERLAY_MAX."""
        clean = self._confident(params)
        if clean is None:
            with self._cloud_lock:
                return self.cloud.snapshot(OVERLAY_MAX)
        stride = max(1, -(-len(clean.points) // OVERLAY_MAX))
        rgb = clean.colours
        return clean.points[::stride].copy(), (None if rgb is None else rgb[::stride].copy())

    # ── configuration (GUI thread) ────────────────────────────────────────

    def set_config(self, cfg: RigConfig) -> None:
        with self._lock:
            self._cfg = cfg

    def set_params(self, params: ScanParams) -> None:
        with self._lock:
            self._params = params

    def set_session(self, session: PhotoSession | None,
                    writer: PhotoWriter | None) -> None:
        """The session photographs go into, and the thread that writes them.

        The seam the window wires up: nothing here opens, starts or stops
        either of them. A new session has photographed nothing yet, and the
        poses and sharpnesses of the last one describe another run, so the
        bookkeeping starts again with it.
        """
        with self._photo_lock:
            self._session = session
            self._writer = writer
            self._pose_hist.clear()
            for side in SIDES:
                self._kept[side].clear()
                self._offers[side].clear()

    def arm_photos(self, on: bool) -> None:
        """Build candidates, consult the policy and enqueue what it accepts —
        or, off, do none of those three. This is the scan worker's own gate;
        retaining the JPEG bytes it decides on is `EyeWorker`'s
        `set_photo_capture`, which is a different switch on the same
        checkbox."""
        self._armed = bool(on)

    def set_photo_pass(self, on: bool) -> None:
        """Photo-pass mode: the laser has been switched off by hand, so pairs
        carry no stripe and exist only to be photographed. Pairing runs on
        this alone, with no scan underneath it, and every toggle starts a new
        pass — photographs taken either side of that switch are not evidence
        of the same configuration."""
        with self._lock:
            if bool(on) == self._photo_pass:
                return
            self._photo_pass = bool(on)
        # Outside that lock on purpose: the session has a lock of its own,
        # and `_lock` is the one both detector threads take.
        session = self._session
        if session is not None:
            session.next_pass()

    def set_active(self, on: bool) -> None:
        snap = None
        emitted: list[tuple] = []
        with self._lock:
            self._active = on
            spilled = []
            if not on:
                for q in self._hist.values():
                    q.clear()
                self._prev = None
                self._last_right = None
                for q in self._ids_seen.values():
                    q.clear()
                # Frames still waiting for neighbours are placed with what
                # they have: the session is over, no neighbour is coming.
                for item, r_s, t_s in self._smooth.flush():
                    f, m, own_t, _cand = item
                    self._place(f, m, own_t, r_s, t_s, self._params)
                    spilled += self._bank(f, r_s, t_s)
                    emitted.append((item, r_s, t_s))
                spilled += self._take_batch()
                # The next session starts its own batch: kept, this pose
                # would be what the first frame of that session is judged
                # still against.
                self._batch_pose = None
        # Collected under the lock, decided outside it — the rule every flush
        # site here obeys.
        self._photos(emitted, self._photo_rig)
        if self._merge(spilled):
            snap = self._snapshot(self._params)
        if snap is not None:
            # The batch's points joined the cloud; the eyes and the panel
            # must see them even though no pair follows.
            self.overlay.publish(*snap)
            self._publish(None, None)

    def clear(self) -> None:
        emitted: list[tuple] = []
        with self._lock:
            self._batch.clear()
            self._batch_pose = None
            # The cloud is going, but the photographs are not: pending frames
            # ride a photo candidate each, and flushing them into the bin
            # would silently lose every photograph the operator had earned in
            # the last tenth of a second. Their points are not placed — the
            # cloud they would join is being emptied in the same call.
            for item, r_s, t_s in self._smooth.flush():
                emitted.append((item, r_s, t_s))
            self._pairs = 0
            self._offered_left = 0
        self._photos(emitted, self._photo_rig)
        with self._cloud_lock:
            self.cloud.clear()
            self._clean = None
        self.overlay.publish(*self._snapshot(self._params))
        self._publish(None, None)

    def export(self, path: str) -> int:
        """The cloud as shown: confident when cleaning is on, every voxel
        otherwise."""
        with self._lock:
            params = self._params
        clean = self._confident(params, force=True)
        if clean is None:
            with self._cloud_lock:
                return self.cloud.write_ply(path)
        return write_ply(path, clean.points, clean.colours)

    # ── input (detector threads) ──────────────────────────────────────────

    def offer(self, res: EyeResult) -> None:
        """Take one eye's result. Cheap: it runs on the detector thread."""
        if res.capture_mono is None:
            return
        board = res.board
        row = float("nan")
        if board is not None and board.corners is not None and board.R is not None:
            row = float(np.asarray(board.corners).reshape(-1, 2)[:, 1].mean())
        item = ScanInput(res.capture_mono, res.stripe,
                         None if board is None else board.R,
                         None if board is None else board.t, res.wh, pose_row=row,
                         bgr=res.bgr if res.side == "left" else None,
                         corners=None if board is None else board.corners,
                         ids=None if board is None else board.ids,
                         jpeg=res.jpeg, sharpness=res.sharpness)
        with self._lock:
            # A photo pass pairs the eyes with no scan underneath it: the
            # laser is off, there is nothing to triangulate, and the pairs
            # exist so the photographs can carry a pose.
            if not (self._active or self._photo_pass):
                return
            self._hist[res.side].append(item)
            if res.side == "left":
                self._offered_left += 1
            if item.board_R is not None:
                self._pose_at = time.monotonic()
        self._wake.set()

    def pose_recent(self) -> bool:
        """For the right eye's worker: has the left eye had a board pose
        within `LEFT_POSE_RECENT_S`? Lock-free on purpose — a float read."""
        return time.monotonic() - self._pose_at <= POSE_RECENT_S

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="scan", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── scan thread ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.25):
                continue
            self._wake.clear()
            while not self._stop.is_set():
                pair = self._take_pair()
                if pair is None:
                    break
                try:
                    self._process(*pair)
                except Exception:                              # noqa: BLE001
                    # One odd frame must not end scanning for the session.
                    log.exception("scan frame raised; continuing")

    def _take_pair(self):
        """The oldest left result that has a partner, with that partner and
        the right result on the far side of the left's instant — what
        `timealign` interpolates against — or None for that when there is
        none.

        The pair is removed, along with everything older on either side —
        those could only have paired with results already gone; the far-side
        right stays, it is the next pair's partner. A left without a partner
        is kept waiting while one could still arrive, that is while the newest
        right is not yet later than the left plus the window; once it is, that
        left has been passed over and is skipped. A left whose partner came
        BEFORE it also waits for the right frame after it — one frame at most,
        and only while no newer left says time has moved on — because that
        frame is what brings the partner to the left's instant.
        """
        with self._lock:
            left, right = self._hist["left"], self._hist["right"]
            for i, a in enumerate(left):
                best = None
                for j, b in enumerate(right):
                    gap = abs(a.capture_mono - b.capture_mono)
                    if gap <= PAIR_WINDOW_S and (best is None or gap < best[0]):
                        best = (gap, j)
                if best is not None:
                    j = best[1]
                    b = right[j]
                    if b.capture_mono >= a.capture_mono:
                        other = right[j - 1] if j >= 1 else self._last_right
                    else:
                        other = right[j + 1] if j + 1 < len(right) else None
                        if (other is None
                                and left[-1].capture_mono - a.capture_mono <= PAIR_WINDOW_S):
                            return None  # the frame that would bracket it may still come
                    for _ in range(i + 1):
                        left.popleft()
                    for _ in range(j + 1):
                        right.popleft()
                    self._last_right = b
                    return a, b, other, self._cfg, self._params
                if not right or right[-1].capture_mono <= a.capture_mono + PAIR_WINDOW_S:
                    return None          # its partner may still be on the way
            return None

    def _process(self, a: ScanInput, b: ScanInput, other: ScanInput | None,
                 cfg: RigConfig | None, params: ScanParams) -> None:
        # Read once, at the top: both are toggled from the GUI thread, and one
        # pair must be paired, posed and photographed in a single mode.
        photo_pass, armed = self._photo_pass, self._armed
        rig, plane, readout, board = self._geometry(cfg, a.wh, b.wh)
        if rig is None:
            self._publish(None, "no pair calibration for this frame size — "
                                "solve intrinsics and the stereo pair first")
            return
        # The right eye's OWN instant and its own stripe, before the alignment
        # below rewrites the one to the left's instant and moves the other to
        # it. A right photograph is taken from its own place on the rig at its
        # own moment, and the manifest records both (§2.4).
        right_capture_mono = b.capture_mono
        right_stripe = b.stripe
        # The right eye at the left's instant: corners and stripe interpolated
        # against the right frame on the far side, and the right's own pose
        # re-solved through the moved corners, so every comparison below —
        # the joint pose, the veto — is between two views of one moment.
        aligned = align_right(a.capture_mono, b, other)
        if aligned.corners is not b.corners:
            pose = (estimate_pose(aligned.corners, aligned.ids, board, rig.right_k, b.board_R)
                    if board is not None else None)
            b = replace(b, stripe=aligned.stripe, corners=aligned.corners, ids=aligned.ids,
                        capture_mono=a.capture_mono,
                        board_R=pose[0] if pose is not None else b.board_R,
                        board_t=pose[1] if pose is not None else b.board_t)
        elif aligned.stripe is not b.stripe:
            b = replace(b, stripe=aligned.stripe)
        if board is not None:
            a = self._steady("left", a, rig.left_k, board)
            b = self._steady("right", b, rig.right_k, board)
        fix = fuse_pose(a, b, rig, board)
        if fix is None:
            self._prev = None
            self._publish(None, "board not visible to either eye — it is what defines "
                                "the scan volume")
            return
        if fix.gap_deg == fix.gap_deg and (fix.gap_deg > POSE_GAP_MAX_DEG
                                           or fix.gap_mm > POSE_GAP_MAX_MM):
            self._prev = None
            self._publish(None, (f"the eyes' board poses differ by {fix.gap_deg:.1f}° / "
                                 f"{fix.gap_mm:.0f} mm: the pair does not describe this rig — "
                                 "Rig moved, then the PAIR and LASER stages"))
            return
        corners = sum(len(x.corners) for x in (a, b) if x.corners is not None)
        if corners < MIN_POSE_CORNERS:
            self._publish(None, (f"{corners} board corners across both eyes: too few for a "
                                 f"steady pose, {MIN_POSE_CORNERS} needed"))
            return
        # The stripe guard sits below the pose block on purpose: a photo pass
        # has the laser switched off and wants the pose, not the stripe. In
        # production it is unreachable either way — `worker.py:388-389` always
        # constructs a `StripePixels` — so `_photo_pass`, and not the absence
        # of a stripe, is what tells the two modes apart. It stays as the
        # defensive no-op it has always been.
        if not photo_pass and (a.stripe is None or b.stripe is None):
            self._publish(None, "laser detector is off")
            return
        motion, note = self._motion(a, fix, readout)
        self._prev = (fix, a.capture_mono, a.pose_row)
        # A photo pass scans nothing: no stripe means no `ScanFrame`, and
        # building an empty one would bank empty frames and publish a scan
        # that is not happening.
        frame = None
        if not photo_pass:
            frame = scan_frame(rig, plane, a.stripe, b.stripe, fix.R, fix.t, params,
                               motion=motion, rs_note=note)
            frame.pose_source, frame.pose_rms_px = fix.source, fix.rms_px
            frame.pose_gap_deg, frame.pose_gap_mm = fix.gap_deg, fix.gap_mm
            frame.sync_gap_ms, frame.sync_note = aligned.gap_ms, aligned.note
            if frame.n_kept and a.bgr is not None:
                frame.colours = sample_beside(a.bgr, frame.pixels_left, frame.along_x,
                                              params.colour_offset_px, a.wh)
        # Everything about the photograph except its pose, which the smoother
        # has not decided yet, and its kept points, which `_place` has not
        # produced yet. Both are filled in below, after the lock.
        cand = None
        if armed:
            cand = self._candidate(a, b, fix, corners, right_capture_mono,
                                   b.stripe is not right_stripe, photo_pass)
            # The flush sites have no rig of their own, and a candidate they
            # hand over still needs one to carry its pose across the pair.
            self._photo_rig = rig
        snap = None
        placed = None
        emitted: list[tuple] = []
        with self._lock:
            self._pairs += 1
            # The frame waits for its neighbours in time; what comes back is
            # the frame from the middle of the window, with the median pose
            # of the window to place its points through.
            spilled = []
            for item, r_s, t_s in self._smooth.push(
                    (frame, motion, np.asarray(fix.t, float).ravel(), cand),
                    fix.R, fix.t, a.capture_mono):
                f, m, own_t, _cand = item
                self._place(f, m, own_t, r_s, t_s, params)
                spilled += self._bank(f, r_s, t_s)
                if f is not None:
                    placed = f
                emitted.append((item, r_s, t_s))
        # Deliberately outside that lock: `offer` takes it on both detector
        # threads, and a voxel merge — or a photo decision, which walks every
        # photograph already kept on a side — under it stalls both eyes every
        # pair.
        self._photos(emitted, rig)
        if self._merge(spilled):
            snap = self._snapshot(params)
        if snap is not None:
            self.overlay.publish(*snap)
        self._publish(placed if placed is not None else frame, None)

    def _candidate(self, a: ScanInput, b: ScanInput, fix: PoseFix, corners: int,
                   right_capture_mono: float, right_shifted: bool,
                   photo_pass: bool) -> PhotoCandidate:
        """This pair offered as a photograph.

        Both eyes' bytes, because a right-eye photograph is a photograph in
        its own right, taken from its own place on the rig; both eyes' own
        capture instants, because pairing rewrote the right one to the left's
        and the manifest has to be able to show the difference; and both
        eyes' stripe pixels, because the laser is masked out offline, when
        the frames themselves are long gone.
        """
        session = self._session
        return PhotoCandidate(
            left=EyePhoto(camera_id=self._camera_id("left"), jpeg=a.jpeg, wh=a.wh,
                          capture_mono=a.capture_mono, sharpness=a.sharpness,
                          stripe=a.stripe),
            right=EyePhoto(camera_id=self._camera_id("right"), jpeg=b.jpeg, wh=b.wh,
                           capture_mono=right_capture_mono, sharpness=b.sharpness,
                           stripe=b.stripe, stripe_shifted=right_shifted),
            pair_capture_mono=a.capture_mono,
            pose_source=fix.source, pose_rms_px=fix.rms_px,
            pose_gap_deg=fix.gap_deg, pose_gap_mm=fix.gap_mm,
            pose_corners=corners,
            pass_id=0 if session is None else session.pass_id,
            # The laser is off for the length of a photo pass, which is the
            # whole point of one — §2.7 groups the photographs by it.
            laser_on=not photo_pass)

    def _camera_id(self, side: str) -> str:
        """What the session calls this eye, or "" before there is a session."""
        session = self._session
        eye = None if session is None else getattr(session.rig, side)
        return "" if eye is None else eye.camera_id

    def _photos(self, emitted: list[tuple], rig: StereoRig | None) -> None:
        """Judge the emitted candidates, and hand what the policy accepts to
        the writer.

        Called from all three flush sites and, at every one of them, only
        after `_lock` has been released. That is the whole point of the
        method: a decision walks every photograph already kept on a side and
        ends at the writer's queue, and `_lock` is the one both detector
        threads take on every result — the same reason the voxel merge above
        is outside it.

        The poses arrive here rather than at the candidate because a
        photograph is placed through the SMOOTHED pose, and the smoother only
        decides a frame's pose once its neighbours in time have arrived. The
        left photograph takes that pose as it is; the right one takes it
        carried across the pair, because the two cameras stand 200 mm apart
        and one pose written to both would be wrong by the baseline.
        """
        writer = self._writer
        if writer is None or not emitted:
            return
        with self._photo_lock:
            session = self._session
            if session is None:
                return
            policy = session.policy
            for (f, _motion, own_t, cand), r_s, t_s in emitted:
                if cand is None:
                    continue
                R_s = np.asarray(r_s, float)
                t_s = np.asarray(t_s, float).ravel()
                cand.pose_smooth_mm = float(np.linalg.norm(
                    t_s - np.asarray(own_t, float).ravel()))
                # `_place` has run and cropped the frame to the scan volume,
                # so these are the points that actually joined the cloud. A
                # photo pass has no frame, and so no points at all.
                cand.kept_xyz_board = (
                    np.zeros((0, 3), np.float32) if f is None
                    else np.asarray(f.points_board, np.float32).reshape(-1, 3))
                cand.left.R, cand.left.t_mm = R_s, t_s
                if rig is not None:
                    cand.right.R, cand.right.t_mm = compose_right_pose(R_s, t_s, rig.geom)
                if (self._pose_hist
                        and cand.pair_capture_mono - self._pose_hist[-1][0] > PHOTO_GAP_S):
                    self._pose_hist.clear()
                self._pose_hist.append((cand.pair_capture_mono, R_s, t_s))
                poses = list(self._pose_hist)
                for side in SIDES:
                    eye = cand.eye(side)
                    if eye.jpeg is None or eye.R is None or eye.t_mm is None:
                        continue
                    if eye.sharpness == eye.sharpness:          # not NaN: disarmed
                        self._offers[side].append(float(eye.sharpness))
                    if policy.decide(cand, side, poses, self._kept[side],
                                     self._offers[side]) is not None:
                        continue
                    writer.put_nowait(cand.record(side))
                    self._kept[side].append((eye.capture_mono, eye.R, eye.t_mm))

    def _steady(self, side: str, x: ScanInput, k, board) -> ScanInput:
        """This eye's input with its pose re-solved through the corners it
        has had in every one of its last `STEADY_FRAMES` frames — the same
        corners every frame, so the pose does not move when a marginal one
        comes or goes. The input as it was when the steady set is too
        small, or when nothing was dropped anyway."""
        seen = self._ids_seen[side]
        if x.corners is None or x.ids is None or not len(x.ids):
            return x
        ids = np.asarray(x.ids).ravel()
        seen.append(set(int(i) for i in ids))
        if len(seen) < 2:
            return x
        steady = set.intersection(*seen)
        if len(steady) < MIN_STEADY_CORNERS or len(steady) == len(ids):
            return x
        keep = np.isin(ids, list(steady))
        pose = estimate_pose(x.corners[keep], x.ids[keep], board, k, x.board_R)
        if pose is None:
            return x
        return replace(x, corners=x.corners[keep], ids=x.ids[keep],
                       board_R=pose[0], board_t=pose[1])

    @staticmethod
    def _place(frame: ScanFrame | None, motion: Motion | None, own_t: np.ndarray,
               R: np.ndarray, t: np.ndarray, params: ScanParams) -> None:
        """Put a frame's points into the board frame through a pose other
        than its own — the smoothed one — and keep what the volume holds.
        The rolling-shutter twist, when the frame had one, is applied on
        top of the new pose the same way it was on the old.

        A photo pass emits no frame: nothing was scanned, so there is nothing
        to place, and the candidate riding the same payload is placed by
        `_photos` instead."""
        if frame is None:
            return
        frame.pose_smooth_mm = float(np.linalg.norm(np.asarray(t, float).ravel() - own_t))
        if not frame.n_kept:
            return
        xyz_cam = frame.points_camera
        if motion is not None:
            xyz = motion.to_board(xyz_cam, frame.pixels_left[:, 1], R, t)
        else:
            xyz = (np.asarray(R, float).T @ (xyz_cam.T - np.asarray(t, float).reshape(3, 1))).T
        inside = params.volume.contains(xyz)
        frame.points_board = xyz[inside]
        frame.points_camera = xyz_cam[inside]
        frame.pixels_left = frame.pixels_left[inside]
        frame.weights = frame.weights[inside]
        frame.scanlines = frame.scanlines[inside]
        if frame.colours is not None:
            frame.colours = frame.colours[inside]
        frame.n_rejected_volume += int((~inside).sum())

    # ── still batches (under the lock) ────────────────────────────────────

    def _bank(self, frame: ScanFrame | None, R, t) -> list[ScanFrame]:
        """Sort a frame into the still batch. Returns whatever the batch gave
        up, for `_merge` to average and add.

        The merge is not done here on purpose: this runs under the lock the
        detector threads take to hand their results over, and a voxel merge
        under that lock stalls both eyes.

        A photo pass banks nothing, because it scanned nothing.
        """
        if frame is None:
            return []
        pose = (np.asarray(R, float), np.asarray(t, float).ravel())
        # A scanline id counts along columns in one frame and along rows in
        # another — `along_x` is decided per frame — so a batch that spans a
        # flip would average columns into rows. It ends at the flip instead.
        turned = bool(self._batch) and self._batch[-1].along_x != frame.along_x
        spilled: list[ScanFrame] = []
        if turned or self._batch_pose is None or not _still(self._batch_pose, (R, t)):
            spilled += self._take_batch()
            self._batch_pose = pose
        if frame.n_kept:
            self._batch.append(frame)
        if len(self._batch) >= STILL_BATCH:
            spilled += self._take_batch()
            self._batch_pose = pose
        return spilled

    def _take_batch(self) -> list[ScanFrame]:
        """Empty the batch, under the caller's lock."""
        frames, self._batch = self._batch, []
        return frames

    def _merge(self, frames: list[ScanFrame]) -> bool:
        """Average what a batch gave up and add it to the cloud. Returns True
        when the cloud changed. Takes the cloud's own lock, never the
        offer lock."""
        if not frames:
            return False
        pts, rgb, w = average_still(frames)
        if not len(pts):
            return False
        with self._cloud_lock:
            self.cloud.add(pts, rgb, w)
        return True

    def _motion(self, a: ScanInput, fix: PoseFix, readout: Readout | None):
        """The board's twist into this frame from the previous pair's pose, or
        why there is none. The instant a pose holds for is the left corners'
        mean row — the readout is the left eye's — so a pose the left eye did
        not see the board for is not slid. Touched by the scan thread alone."""
        if readout is None:
            return None, "no readout time for this frame size — measure it in CALIBRATION"
        prev = self._prev
        if prev is None or not np.isfinite(prev[2]):
            return None, "waiting for a second board pose"
        prev_fix, prev_capture, prev_row = prev
        gap = a.capture_mono - prev_capture
        if not 0 < gap <= MAX_TWIST_GAP_S:
            return None, f"previous pose {gap * 1000:.0f} ms ago — too long to infer the motion"
        if not np.isfinite(a.pose_row):
            return None, "the left eye has no corners to time the pose by"
        motion = Motion.between(prev_fix.R, prev_fix.t, prev_capture, prev_row,
                                fix.R, fix.t, a.capture_mono, a.pose_row, readout)
        return motion, None if motion is not None else "poses out of order"

    def _geometry(self, cfg: RigConfig | None, wh: tuple[int, int],
                  right_wh: tuple[int, int] | None = None):
        """Projection geometry for this calibration and frame size, cached.

        Keyed by the calibration's content rather than the config object: the
        poll hands over a fresh object every two seconds with the same numbers
        in it, and an object id can be reused after the old one is freed.
        """
        if cfg is None:
            return None, None, None, None
        right_wh = wh if right_wh is None else right_wh
        with self._lock:
            reach = tuple(self._params.range_mm)
        key = (wh, right_wh,
               cfg.left.intrinsics_raw if cfg.left else None,
               cfg.right.intrinsics_raw if cfg.right else None,
               cfg.extrinsics_raw, cfg.laser_plane_raw,
               cfg.left.readout_raw if cfg.left else None, reach, cfg.board)
        if key != self._geom_key:
            # The key goes in last: a raise below would otherwise leave a
            # half-built cache marked as current for this calibration.
            rig = None
            if cfg.left is not None and cfg.right is not None:
                kl = cfg.left.intrinsics_for(wh)
                kr = cfg.right.intrinsics_for(right_wh)
                geom = result_from_config(cfg.extrinsics_raw, wh)
                if kl is not None and kr is not None and geom is not None:
                    rig = StereoRig(kl, kr, geom)
            plane = plane_from_config(cfg.laser_plane_raw, wh)
            board = build_board(cfg.board) if cfg.board is not None else None
            self._geom = (rig, plane,
                          Readout.from_config(cfg.left.readout_raw if cfg.left else None, wh),
                          board)
            self.stripe_rows = {
                side: (stripe_rows(plane, rig, reach, size, side)
                       if rig is not None and plane is not None else None)
                for side, size in (("left", wh), ("right", right_wh))}
            self._geom_key = key
        return self._geom

    def _publish(self, frame: ScanFrame | None, note: str | None) -> None:
        # The session's counters are read outside both locks: the session has
        # a lock of its own, and nesting it inside `_lock` would put the GUI's
        # tick behind the two detector threads.
        session, writer = self._session, self._writer
        counts = {} if session is None else session.counts
        with self._cloud_lock:
            n, bounds = len(self.cloud), self.cloud.bounds()
            clean = self._clean[2] if self._clean is not None else None
        with self._lock:
            st = ScanStatus(n, bounds, self._pairs,
                            self._offered_left, len(self._batch), frame, note)
            if self._params.clean and clean is not None:
                st.n_confident = len(clean.points)
                st.n_lonely, st.n_flicker = clean.n_lonely, clean.n_flicker
        st.photos_left = counts.get("left", 0)
        st.photos_right = counts.get("right", 0)
        st.photos_dropped = 0 if writer is None else writer.dropped
        if session is not None:
            st.session_id = session.session_id
            st.pass_id = session.pass_id
            st.session_bytes = session.bytes_on_disk
        self.status.put(st)
