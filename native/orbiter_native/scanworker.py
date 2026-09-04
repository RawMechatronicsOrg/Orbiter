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
from .rolling import Motion, Readout
from .scan import CloudOverlay, PointCloud, ScanFrame, ScanParams, scan_frame, stripe_rows
from .stereo import StereoRig, result_from_config
from .worker import EyeResult, Latest

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
LEFT_POSE_RECENT_S = 0.5


def _still(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    """Two board poses within STILL_MM and STILL_DEG of each other."""
    (R0, t0), (R1, t1) = a, b
    if float(np.linalg.norm(np.asarray(t1, float).ravel() - np.asarray(t0, float).ravel())) > STILL_MM:
        return False
    cos = (np.trace(np.asarray(R0, float).T @ np.asarray(R1, float)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))) <= STILL_DEG


def average_still(frames: list[ScanFrame]) -> np.ndarray:
    """One point per scanline from a batch of still frames: the per-axis
    TRIMMED mean of the frames' points on that scanline — the lowest and the
    highest dropped once there are four or more — kept when at least half
    the frames had one. Trimmed, not plain: a glint that passed every gate in
    one frame of five is 40 mm off, and a plain mean would move the point
    8 mm toward it; as the extreme it is dropped instead, and the remaining
    three average with most of the mean's noise reduction (a median of five
    keeps only 70% of it). A scanline seen once in five is a flicker, not a
    surface."""
    if len(frames) == 1:
        return frames[0].points_board
    keys = np.concatenate([f.scanlines for f in frames])
    pts = np.concatenate([f.points_board for f in frames])
    if not len(keys):
        return np.empty((0, 3))
    order = np.argsort(keys, kind="stable")
    keys, pts = keys[order], pts[order]
    uniq, start, counts = np.unique(keys, return_index=True, return_counts=True)
    rank = np.arange(len(keys)) - np.repeat(start, counts)
    width = int(counts.max())
    table = np.full((len(uniq), width, 3), np.nan)
    table[np.repeat(np.arange(len(uniq)), counts), rank] = pts
    enough = counts >= -(-len(frames) // 2)
    table, counts = table[enough], counts[enough]
    table = np.sort(table, axis=1)                        # NaN sorts last, per axis
    ranks = np.arange(width)[None, :]
    trim = (counts >= 4)[:, None]
    keep = np.where(trim, (ranks >= 1) & (ranks <= counts[:, None] - 2), ranks < counts[:, None])
    weights = keep.astype(np.float64)[:, :, None]
    return np.nansum(table * weights, axis=1) / weights.sum(axis=1)


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
        self._geom: tuple[StereoRig | None, LaserPlane | None, Readout | None] = (
            None, None, None)
        #: Per eye, the rows the sheet can appear in for the current reach, or
        #: None for the whole frame; the window pushes them to the workers.
        self.stripe_rows: dict[str, tuple[int, int] | None] = {"left": None, "right": None}
        # The previous left result that carried a pose: with the current one
        # it gives the board's twist, which is what slides the pose to each
        # stripe row's instant.
        self._prev_left: ScanInput | None = None
        self._left_pose_at = -1e9
        # Consecutive frames of a still board, averaged before they join the
        # cloud; the pose they are still against is the first one's.
        self._batch: list[ScanFrame] = []
        self._batch_pose: tuple[np.ndarray, np.ndarray] | None = None

        self.cloud = PointCloud()
        self.overlay = CloudOverlay()
        #: Newest counters; the GUI takes them on its own clock.
        self.status = Latest()

    # ── configuration (GUI thread) ────────────────────────────────────────

    def set_config(self, cfg: RigConfig) -> None:
        with self._lock:
            self._cfg = cfg

    def set_params(self, params: ScanParams) -> None:
        with self._lock:
            self._params = params

    def set_active(self, on: bool) -> None:
        snap = None
        with self._lock:
            self._active = on
            spilled = []
            if not on:
                for q in self._hist.values():
                    q.clear()
                self._prev_left = None
                # The next session starts its own batch: kept, this pose
                # would be what the first frame of that session is judged
                # still against.
                self._batch_pose = None
                spilled = self._take_batch()
        if self._merge(spilled):
            with self._cloud_lock:
                snap = self.cloud.decimated(OVERLAY_MAX)
        if snap is not None:
            # The batch's points joined the cloud; the eyes and the panel
            # must see them even though no pair follows.
            self.overlay.publish(snap)
            self._publish(None, None)

    def clear(self) -> None:
        with self._lock:
            self._batch.clear()
            self._batch_pose = None
            self._pairs = 0
            self._offered_left = 0
        with self._cloud_lock:
            self.cloud.clear()
            snap = self.cloud.decimated(OVERLAY_MAX)
        self.overlay.publish(snap)
        self._publish(None, None)

    def export(self, path: str) -> int:
        with self._cloud_lock:
            return self.cloud.write_ply(path)

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
                         None if board is None else board.t, res.wh, pose_row=row)
        with self._lock:
            if not self._active:
                return
            self._hist[res.side].append(item)
            if res.side == "left":
                self._offered_left += 1
                if item.board_R is not None:
                    self._left_pose_at = time.monotonic()
        self._wake.set()

    def left_pose_recent(self) -> bool:
        """For the right eye's worker: has the left eye had a board pose
        within `LEFT_POSE_RECENT_S`? Lock-free on purpose — a float read."""
        return time.monotonic() - self._left_pose_at <= LEFT_POSE_RECENT_S

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
        """The oldest left result that has a partner, with that partner.

        Both are removed, along with everything older on either side — those
        could only have paired with results already gone. A left without a
        partner is kept waiting while one could still arrive, that is while the
        newest right is not yet later than the left plus the window; once it
        is, that left has been passed over and is skipped.
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
                    b = right[best[1]]
                    for _ in range(i + 1):
                        left.popleft()
                    for _ in range(best[1] + 1):
                        right.popleft()
                    return a, b, self._cfg, self._params
                if not right or right[-1].capture_mono <= a.capture_mono + PAIR_WINDOW_S:
                    return None          # its partner may still be on the way
            return None

    def _process(self, a: ScanInput, b: ScanInput, cfg: RigConfig | None,
                 params: ScanParams) -> None:
        rig, plane, readout = self._geometry(cfg, a.wh, b.wh)
        if rig is None:
            self._publish(None, "no pair calibration for this frame size — "
                                "solve intrinsics and the stereo pair first")
            return
        if a.stripe is None or b.stripe is None:
            self._publish(None, "laser detector is off")
            return
        if a.board_R is None or a.board_t is None:
            self._prev_left = None
            self._publish(None, "board not visible — it is what defines the scan volume")
            return
        motion, note = self._motion(a, readout)
        self._prev_left = a
        frame = scan_frame(rig, plane, a.stripe, b.stripe, a.board_R, a.board_t, params,
                           motion=motion, rs_note=note)
        snap = None
        with self._lock:
            self._pairs += 1
            spilled = self._bank(frame, a.board_R, a.board_t)
        # Deliberately outside that lock: `offer` takes it on both detector
        # threads, and a voxel merge under it stalls both eyes every pair.
        if self._merge(spilled):
            with self._cloud_lock:
                snap = self.cloud.decimated(OVERLAY_MAX)
        if snap is not None:
            self.overlay.publish(snap)
        self._publish(frame, None)

    # ── still batches (under the lock) ────────────────────────────────────

    def _bank(self, frame: ScanFrame, R, t) -> list[ScanFrame]:
        """Sort a frame into the still batch. Returns whatever the batch gave
        up, for `_merge` to average and add.

        The merge is not done here on purpose: this runs under the lock the
        detector threads take to hand their results over, and a voxel merge
        under that lock stalls both eyes.
        """
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
        pts = average_still(frames)
        if not len(pts):
            return False
        with self._cloud_lock:
            self.cloud.add(pts)
        return True

    def _motion(self, a: ScanInput, readout: Readout | None):
        """The board's twist into this frame from the previous left pose, or
        why there is none. Touched by the scan thread alone."""
        if readout is None:
            return None, "no readout time for this frame size — measure it in CALIBRATION"
        prev = self._prev_left
        if prev is None or prev.board_R is None or not np.isfinite(prev.pose_row):
            return None, "waiting for a second board pose"
        gap = a.capture_mono - prev.capture_mono
        if not 0 < gap <= MAX_TWIST_GAP_S:
            return None, f"previous pose {gap * 1000:.0f} ms ago — too long to infer the motion"
        if not np.isfinite(a.pose_row):
            return None, "no corners to time the pose by"
        motion = Motion.between(prev.board_R, prev.board_t, prev.capture_mono, prev.pose_row,
                                a.board_R, a.board_t, a.capture_mono, a.pose_row, readout)
        return motion, None if motion is not None else "poses out of order"

    def _geometry(self, cfg: RigConfig | None, wh: tuple[int, int],
                  right_wh: tuple[int, int] | None = None):
        """Projection geometry for this calibration and frame size, cached.

        Keyed by the calibration's content rather than the config object: the
        poll hands over a fresh object every two seconds with the same numbers
        in it, and an object id can be reused after the old one is freed.
        """
        if cfg is None:
            return None, None, None
        right_wh = wh if right_wh is None else right_wh
        with self._lock:
            reach = tuple(self._params.range_mm)
        key = (wh, right_wh,
               cfg.left.intrinsics_raw if cfg.left else None,
               cfg.right.intrinsics_raw if cfg.right else None,
               cfg.extrinsics_raw, cfg.laser_plane_raw,
               cfg.left.readout_raw if cfg.left else None, reach)
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
            self._geom = (rig, plane,
                          Readout.from_config(cfg.left.readout_raw if cfg.left else None, wh))
            self.stripe_rows = {
                side: (stripe_rows(plane, rig, reach, size, side)
                       if rig is not None and plane is not None else None)
                for side, size in (("left", wh), ("right", right_wh))}
            self._geom_key = key
        return self._geom

    def _publish(self, frame: ScanFrame | None, note: str | None) -> None:
        with self._cloud_lock:
            n, bounds = len(self.cloud), self.cloud.bounds()
        with self._lock:
            st = ScanStatus(n, bounds, self._pairs,
                            self._offered_left, len(self._batch), frame, note)
        self.status.put(st)
