"""The gate between a paired frame and a photograph on disk.

`ScanWorker` is driven directly here — `offer`, `_take_pair`, `_process`, no
thread — the way `test_stereo_scan` drives it, so every decision can be
watched without a camera, a board or a Qt event loop. The rig, the sheet and
the stripe are `test_stereo_scan`'s synthetic ones, where every number is
chosen rather than measured.

Two things about the setup are deliberate. The geometry is handed straight to
the worker instead of being built from a `RigConfig`: these tests are about
`_process`, not about the calibration cache. And the board object is `None`,
which stops `fuse_pose` re-fitting the pose through the corners — the pose the
worker then uses is exactly the pose the test put in, which is what lets a
smoothed pose be told apart from an unsmoothed one.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orbiter_native import scanworker
from orbiter_native.detect import BoardHit
from orbiter_native.laser import StripePixels
from orbiter_native.photos import (
    CapturePolicy,
    EyeSnapshot,
    PhotoSession,
    PhotoWriter,
    RigSnapshot,
)
from orbiter_native.scan import ScanParams, ScanVolume
from orbiter_native.scanworker import ScanWorker
from orbiter_native.stereo import compose_right_pose
from orbiter_native.timealign import shift_stripe

from test_stereo_scan import (       # noqa: F401 — `board` is a fixture
    KL,
    KR,
    R_TRUE,
    T_TRUE,
    WH,
    _board_in_both_eyes,
    _curve,
    _eye_result,
    _pixels,
    _plane,
    _project,
    _rig,
    board,
)

#: The cylinder and the reach play no part here: this file is about the photo
#: gate, and both have their own tests next door.
WIDE = ScanParams(range_mm=(0.0, 1e9), stereo_refine=False,
                  volume=ScanVolume(height_mm=1e6, radius_mm=1e6, floor_mm=-1e6))

#: How far apart consecutive pairs are offered, and how far behind the left
#: eye the right one is — a plausible 30 fps with the two sensors free-running.
STEP_S = 0.033
EYE_GAP_S = 0.004

#: The first pair that can become a photograph. The smoother holds the first
#: three back (`posesmooth.WINDOW // 2`) and the policy judges stillness over
#: five poses, so the fifth emission — the eighth pair, index 7 — is the
#: earliest one the gate can accept.
FIRST_PHOTO = 7
#: Pairs a test drives to get past it with a frame to spare.
PAIRS = 12

#: A stripe on the sheet as the two eyes report it. A scan-mode pair needs
#: one in both eyes to get past the laser guard at all; the photo-pass tests
#: hand None instead, which is the whole point of that mode.
_TRUTH = _curve(lambda x: 500.0 + 30.0 * np.sin(x / 25.0))
STRIPE_L = _pixels(KL, np.eye(3), np.zeros(3), _TRUTH)
STRIPE_R = _pixels(KR, R_TRUE, T_TRUE, _TRUTH)


# ── the rig, the scene and the driving ───────────────────────────────────


def _session(tmp_path, policy: CapturePolicy | None = None) -> PhotoSession:
    return PhotoSession(tmp_path, policy=policy, rig=RigSnapshot(
        left=EyeSnapshot(camera_id="cam2", wh=WH),
        right=EyeSnapshot(camera_id="cam4", wh=WH)))


def _worker(session=None, writer=None, *, params=WIDE) -> ScanWorker:
    """A worker wired to the synthetic rig, armed and ready to be driven."""
    rig, plane = _rig(), _plane()
    sw = ScanWorker()
    sw._geometry = lambda cfg, wh, right_wh=None: (rig, plane, None, None)
    sw.set_params(params)
    sw.set_session(session, writer)
    sw.arm_photos(True)
    return sw


def _result(side: str, when: float, *, R, t_mm, corners, ids, stripe=None,
            jpeg: bytes = b"jpeg", sharpness: float = 100.0):
    """One eye's detection as its worker publishes it. `bgr` is None on
    purpose: no colour image came with these frames, and the colour block is
    not what is under test."""
    return replace(_eye_result(side, when), bgr=None, wh=WH, stripe=stripe,
                   jpeg=jpeg, sharpness=sharpness,
                   board=BoardHit(corners=corners, ids=ids, R=R, t=t_mm))


def _pair(sw: ScanWorker, i: int, *, R, t, li, ri, ids, right_pose=None,
          stripe_l=STRIPE_L, stripe_r=STRIPE_R, sharpness=100.0, jpeg=b"jpeg",
          start: float = 1000.0) -> None:
    """Offer pair `i` and process everything it completes."""
    if right_pose is None:
        right_pose = (None, None) if R is None else compose_right_pose(R, t, _rig().geom)
    when = start + i * STEP_S
    sw.offer(_result("left", when, R=R, t_mm=t, corners=li, ids=ids,
                     stripe=stripe_l, sharpness=sharpness, jpeg=jpeg))
    sw.offer(_result("right", when + EYE_GAP_S, R=right_pose[0], t_mm=right_pose[1],
                     corners=ri, ids=ids, stripe=stripe_r, sharpness=sharpness,
                     jpeg=jpeg))
    while (pair := sw._take_pair()) is not None:
        sw._process(*pair)


def _still_run(sw: ScanWorker, board, n: int = PAIRS, **kw) -> tuple:
    """`n` pairs of a board standing perfectly still. Returns the corners,
    the ids and the pose every one of them was taken at."""
    li, ri, ids, R, t = _board_in_both_eyes(board)
    for i in range(n):
        _pair(sw, i, R=R, t=t, li=li, ri=ri, ids=ids, **kw)
    return li, ri, ids, R, t


def _manifest(session: PhotoSession) -> list[dict]:
    if not session.manifest_path.exists():
        return []
    return [json.loads(line) for line in
            session.manifest_path.read_text(encoding="utf-8").splitlines()]


def _one(session: PhotoSession, side: str) -> dict:
    """The one photograph written on this side, insisted upon."""
    lines = [m for m in _manifest(session) if m["side"] == side]
    assert len(lines) == 1, lines
    return lines[0]


def _pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """The manifest's pose back as a rotation matrix and a translation."""
    q = record["pose"]["q_wxyz"]
    R = Rotation.from_quat(np.asarray(q, float), scalar_first=True).as_matrix()
    return R, np.asarray(record["pose"]["t_mm"], float)


# ── the tests ────────────────────────────────────────────────────────────


def test_photo_pass_records_without_a_stripe(tmp_path, board, monkeypatch) -> None:
    """The laser is off and the board is visible: photographs accrue, the
    cloud does not, and nothing is scanned at all."""
    def never_scan(*a, **kw):
        raise AssertionError("a photo pass must not scan a frame")

    monkeypatch.setattr(scanworker, "scan_frame", never_scan)
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    # No `set_active`: a photo pass pairs the eyes on its own.
    sw.set_photo_pass(True)
    _still_run(sw, board, stripe_l=None, stripe_r=None)
    sw.set_active(False)
    writer.stop()

    assert session.counts == {"left": 1, "right": 1}
    assert len(sw.cloud) == 0
    # Every photograph of a pass is labelled with it, so the reconstruction
    # can tell the laser-lit frames from the clean ones.
    assert session.pass_id == 1
    for side in ("left", "right"):
        record = _one(session, side)
        assert record["laser_on"] is False
        assert record["pass_id"] == 1
        assert record["stripe_pixels"] == 0
        assert record["kept_points"] == 0
        assert (session.photos_dir / f"{side}_0001.jpg").read_bytes() == b"jpeg"


def test_photo_uses_the_smoothed_pose(tmp_path, board) -> None:
    """One frame's own pose is a third of a millimetre off its neighbours'.
    The photograph carries the pose the smoother decided — the one its points
    would have been placed through — and records how far that moved it."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    li, ri, ids, R, t = _board_in_both_eyes(board)
    for i in range(PAIRS):
        # Well inside the 0.5 mm stillness gate, and far outside the noise
        # the smoother's own median leaves behind.
        own = t + [0.0, 0.0, 0.3] if i == FIRST_PHOTO else t
        _pair(sw, i, R=R, t=own, li=li, ri=ri, ids=ids)
    sw.set_active(False)
    writer.stop()

    record = _one(session, "left")
    R_photo, t_photo = _pose(record)
    assert np.allclose(R_photo, R, atol=1e-9)
    assert np.allclose(t_photo, t, atol=1e-9)                     # the median
    assert not np.allclose(t_photo, t + [0.0, 0.0, 0.3])          # not its own
    assert record["pose_smooth_mm"] == pytest.approx(0.3, abs=1e-6)
    assert record["pose_composed"] is False
    assert record["pose_source"] == "left+right"


def test_right_photo_pose_is_composed_through_the_pair(tmp_path, board) -> None:
    """The right photograph's pose is the smoothed pose carried across the
    pair, not the smoothed pose itself: a board point projected through it
    and the right eye's own K lands where the rig says the right eye sees it.
    Composing beats letting the right eye solve its own ChArUco — it is the
    noisier eye, and its own solve would forfeit both the joint fit and the
    smoother."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    _still_run(sw, board)
    sw.set_active(False)
    writer.stop()

    rig = _rig()
    R_l, t_l = _pose(_one(session, "left"))
    R_r, t_r = _pose(_one(session, "right"))
    assert _one(session, "right")["pose_composed"] is True
    assert _one(session, "right")["pose_frame"] == "right"
    # A point on the subject, in the board frame.
    p = np.array([[12.0, -30.0, 45.0], [-40.0, 5.0, 120.0]])
    want = rig.project_right(p @ R_l.T + t_l)
    got = _project(KR, np.eye(3), np.zeros(3), p @ R_r.T + t_r)
    assert np.allclose(got, want, atol=1e-6)
    # And the baseline really is in there: the two poses are 200 mm apart.
    assert np.linalg.norm(t_r - t_l) > 100.0


def test_right_photo_records_its_own_capture_mono(tmp_path, board) -> None:
    """Pairing rewrites the right input's instant to the left's. The
    photograph keeps the right eye's own, and the pair's instant beside it,
    so the asymmetry can be audited rather than discovered."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    _still_run(sw, board)
    sw.set_active(False)
    writer.stop()

    right, left = _one(session, "right"), _one(session, "left")
    assert left["capture_mono"] == left["pair_capture_mono"]
    off = right["capture_mono"] - right["pair_capture_mono"]
    assert off == pytest.approx(EYE_GAP_S, abs=1e-9)
    assert 0.0 < off <= scanworker.PAIR_WINDOW_S


def test_sidecar_carries_the_frames_kept_points_in_the_board_frame(
        tmp_path, board) -> None:
    """A scan-mode photograph's sidecar holds exactly the points `_place`
    kept for that frame, in the board frame, as float32. A photo-pass
    photograph's holds an empty (0, 3), because there was no frame at all."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    placed: list = []
    original = ScanWorker._place

    def spy(frame, motion, own_t, R, t, params):
        original(frame, motion, own_t, R, t, params)
        placed.append(frame)

    sw._place = spy
    _still_run(sw, board)
    sw.set_active(False)
    writer.stop()

    # The fifth emission is the first photograph the gate can accept.
    frame = placed[4]
    assert frame.n_kept > 250
    for side in ("left", "right"):
        with np.load(session.stripe_dir / f"{side}_0001.npz") as npz:
            kept = npz["kept_xyz_board"]
        assert kept.dtype == np.float32
        assert kept.shape == (len(frame.points_board), 3)
        assert np.allclose(kept, frame.points_board.astype(np.float32))
        assert _one(session, side)["kept_points"] == len(kept)

    # The same run with the laser off: no frame, and so no points.
    other = _session(tmp_path / "pass")
    other_writer = PhotoWriter(other)
    other_writer.start()
    sw2 = _worker(other, other_writer)
    sw2.set_photo_pass(True)
    _still_run(sw2, board)
    sw2.set_active(False)
    other_writer.stop()
    with np.load(other.stripe_dir / "left_0001.npz") as npz:
        assert npz["kept_xyz_board"].shape == (0, 3)
        assert npz["kept_xyz_board"].dtype == np.float32


def test_candidate_carries_both_stripes_and_marks_the_right_one_shifted(
        tmp_path, board) -> None:
    """The left sidecar holds the pixels the left eye detected. The right one
    holds the pixels `align_right` brought to the left's instant — which is a
    different set whenever the stripe is moving — and says so, because the
    mask builder runs offline and cannot tell by looking."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    left, base = STRIPE_L, STRIPE_R

    def drifted(i: int) -> StripePixels:
        """The right eye's stripe walking down the frame, so bringing it to
        the left's instant actually has something to move."""
        return replace(base, y=(base.y + 8 * i).astype(np.int32))

    li, ri, ids, R, t = _board_in_both_eyes(board)
    for i in range(PAIRS):
        _pair(sw, i, R=R, t=t, li=li, ri=ri, ids=ids,
              stripe_l=left, stripe_r=drifted(i))
    sw.set_active(False)
    writer.stop()

    with np.load(session.stripe_dir / "left_0001.npz") as npz:
        assert not bool(npz["stripe_shifted"])
        assert np.array_equal(npz["x"], left.x) and np.array_equal(npz["y"], left.y)

    # What `align_right` had to work with for the accepted pair: this frame's
    # right stripe, the previous one, and the left's instant between them.
    k = FIRST_PHOTO
    want = shift_stripe(drifted(k), 1000.0 + k * STEP_S + EYE_GAP_S,
                        drifted(k - 1), 1000.0 + (k - 1) * STEP_S + EYE_GAP_S,
                        1000.0 + k * STEP_S)
    assert not np.array_equal(want.y, drifted(k).y)          # it really moved
    with np.load(session.stripe_dir / "right_0001.npz") as npz:
        assert bool(npz["stripe_shifted"])
        assert np.array_equal(npz["x"], want.x)
        assert np.array_equal(npz["y"], want.y)
    assert _one(session, "right")["stripe_shifted"] is True
    assert _one(session, "right")["stripe_pixels"] == want.count


def test_no_photo_without_a_pose(tmp_path, board) -> None:
    """Neither eye sees the board: there is no pose to hang a photograph on,
    so there is no photograph."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    for i in range(PAIRS):
        _pair(sw, i, R=None, t=None, li=None, ri=None, ids=None)
    sw.set_active(False)
    writer.stop()
    # The pairs did reach `_process`; it is the pose that was missing.
    assert "board not visible" in (sw.status.take().note or "")
    assert writer.written == 0 and session.counts == {"left": 0, "right": 0}
    assert _manifest(session) == []


def test_no_photo_when_the_eyes_disagree(tmp_path, board) -> None:
    """The eyes' own board poses stand three degrees apart: the pair does not
    describe this rig, the frame is not scanned, and it is not photographed
    either."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    li, ri, ids, R, t = _board_in_both_eyes(board)
    off = cv2.Rodrigues(np.array([0.0, np.radians(3.0), 0.0]))[0] @ R
    for i in range(PAIRS):
        _pair(sw, i, R=R, t=t, li=li, ri=ri, ids=ids,
              right_pose=compose_right_pose(off, t, _rig().geom))
    sw.set_active(False)
    writer.stop()
    assert "differ by 3.0" in (sw.status.take().note or "")
    assert writer.written == 0 and _manifest(session) == []


class _WatchedLock:
    """A lock that says whether it is held, so a test can assert that nothing
    slow happens underneath it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = False
        #: How often it was taken — so a test can tell "nothing happened
        #: under the lock" apart from "the lock was never taken".
        self.entries = 0

    def __enter__(self) -> _WatchedLock:
        self._lock.acquire()
        self.held = True
        self.entries += 1
        return self

    def __exit__(self, *exc) -> bool:
        self.held = False
        self._lock.release()
        return False

    def acquire(self, *a, **kw):
        got = self._lock.acquire(*a, **kw)
        self.held = self.held or bool(got)
        return got

    def release(self) -> None:
        self.held = False
        self._lock.release()


class _Watchful:
    """The policy and the writer at once: it answers like both and records
    whether the scan lock was held when it was asked."""

    def __init__(self, lock: _WatchedLock) -> None:
        self.lock = lock
        self.policy = CapturePolicy()
        self.under_lock: list[str] = []
        self.decisions = 0
        self.records: list = []
        self.dropped = 0

    def decide(self, *a, **kw):
        self.decisions += 1
        if self.lock.held:
            self.under_lock.append("decide")
        return self.policy.decide(*a, **kw)

    def put_nowait(self, rec) -> None:
        if self.lock.held:
            self.under_lock.append("put_nowait")
        self.records.append(rec)


def test_decision_is_not_taken_under_the_scan_lock(tmp_path, board) -> None:
    """`offer` takes `_lock` on both detector threads forty times a second.
    A decision walks every photograph already kept on a side and ends at the
    writer's queue, so none of the three flush sites — `_process`,
    `set_active` and `clear` — may hold the lock while it happens. A fourth
    site that forgot fails here rather than shipping."""
    session = _session(tmp_path)
    watch = _Watchful(_WatchedLock())
    session.policy = watch
    sw = _worker(session, watch)
    sw._lock = watch.lock
    sw.set_active(True)

    _still_run(sw, board)                                  # `_process`
    assert watch.decisions > 0 and watch.records            # not vacuous
    assert watch.lock.entries > 0                           # and instrumented
    seen = watch.decisions
    sw.clear()                                              # `clear`
    assert watch.decisions > seen, "clear() dropped the pending candidates"

    seen = watch.decisions
    _still_run(sw, board, n=4, start=2000.0)
    sw.set_active(False)                                    # `set_active`
    assert watch.decisions > seen
    assert watch.under_lock == []


def test_clear_drains_pending_candidates_instead_of_dropping_them(
        tmp_path, board) -> None:
    """The smoother holds three frames back at any moment. `clear` empties
    the cloud, but the photographs riding those frames were already earned —
    flushing them into the bin loses them silently.

    The novelty and interval gates are opened here on purpose: they would
    refuse the three pending candidates for reasons that have nothing to do
    with whether `clear` offered them, and the point is that it offered them.
    """
    session = _session(tmp_path, CapturePolicy(min_interval_s=0.0, novelty_mm=0.0,
                                               novelty_deg=0.0))
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    _still_run(sw, board)
    before = session.counts["left"]
    assert before > 0                                 # the run itself worked
    assert sw._smooth.pending == 3
    sw.clear()
    writer.stop()

    assert sw._smooth.pending == 0
    assert len(sw.cloud) == 0
    # Exactly the three the smoother was holding, each judged and written.
    assert session.counts["left"] == before + 3
    assert session.counts["right"] == before + 3
    assert len(_manifest(session)) == 2 * (before + 3)


def test_stopping_a_scan_with_frames_pending_does_not_raise(tmp_path, board) -> None:
    """The smoother's payload is a four-tuple now; `set_active`'s unpack site
    has to know that as well as `_process`'s does."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    _still_run(sw, board, n=5)
    assert sw._smooth.pending == 5                    # nothing emitted yet
    sw.set_active(False)
    writer.stop()
    assert sw._smooth.pending == 0
    assert len(sw.cloud) > 250                        # placed and banked


def test_scanning_still_works_unchanged(tmp_path, board) -> None:
    """Disarmed, the worker builds no candidate, consults no policy and
    writes nothing — a scan runs exactly as it did before photographs
    existed, and the cloud is the proof."""
    session = _session(tmp_path)
    watch = _Watchful(_WatchedLock())
    session.policy = watch
    sw = _worker(session, watch)
    sw.arm_photos(False)
    sw.set_active(True)
    _still_run(sw, board)
    sw.set_active(False)

    assert watch.decisions == 0 and watch.records == []
    assert len(sw.cloud) > 250
    status = sw.status.take()
    assert (status.photos_left, status.photos_right, status.photos_dropped) == (0, 0, 0)
    assert status.session_id == session.session_id and status.pass_id == 0


def test_status_carries_the_sessions_counters(tmp_path, board) -> None:
    """The panel reads the session off the status object, so the counts, the
    pass and the size on disk have to be on it."""
    session = _session(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    sw = _worker(session, writer)
    sw.set_active(True)
    _still_run(sw, board)
    sw.set_active(False)
    writer.stop()
    sw._publish(None, None)

    status = sw.status.take()
    assert (status.photos_left, status.photos_right) == (1, 1)
    assert status.photos_dropped == 0
    assert status.session_id == session.session_id
    assert status.session_bytes == session.bytes_on_disk > 0
