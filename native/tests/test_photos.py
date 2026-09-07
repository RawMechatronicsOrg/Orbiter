"""The photograph side of a scan: what the policy will spend a file on, what
the writer puts on disk, and what a session says about itself.

Everything here runs against `tmp_path`. The real session root is under the
operator's home, and a test that wrote there would leave sessions behind that
look exactly like real ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from orbiter_native.colmapio import quat_wxyz
from orbiter_native.laser import StripePixels
from orbiter_native.photos import (
    QUEUE_DEPTH,
    SCHEMA,
    SESSIONS_ENV,
    STILL_HISTORY,
    BoardSnapshot,
    CapturePolicy,
    EyeSnapshot,
    Extrinsics,
    EyePhoto,
    PhotoCandidate,
    PhotoSession,
    PhotoWriter,
    RigSnapshot,
    camera_centre,
    sessions_root,
)
from orbiter_native.scan import ScanVolume


def _pose_at(centre_mm, deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A board→camera pose putting the camera at `centre_mm` in the board
    frame, turned `deg` about the board's y axis. `t = -R c`, so the centre
    comes back out of `camera_centre` unchanged."""
    R = Rotation.from_euler("y", deg, degrees=True).as_matrix()
    return R, -R @ np.asarray(centre_mm, float)


def _stripe(n: int = 7, wh: tuple[int, int] = (1920, 1080)) -> StripePixels:
    return StripePixels(x=np.arange(n, dtype=np.int32) + 100,
                        y=np.arange(n, dtype=np.int32) * 3 + 40,
                        w=np.full(n, 200, np.uint8),
                        r=np.full(n, 253, np.uint8),
                        wh=wh, along_x=False, ms=1.2, reason=None)


def _eye(side: str, pose=None, *, jpeg: bytes = b"jpeg", sharpness: float = 100.0,
         capture_mono: float = 1000.0, shifted: bool = False) -> EyePhoto:
    R, t = pose if pose is not None else _pose_at([0.0, 0.0, -500.0])
    return EyePhoto(camera_id="cam2" if side == "left" else "cam4", jpeg=jpeg,
                    wh=(1920, 1080), capture_mono=capture_mono, R=R, t_mm=t,
                    sharpness=sharpness, stripe=_stripe(), stripe_shifted=shifted)


def _candidate(pose=None, *, mono: float = 1000.0, sharpness: float = 100.0,
               corners: int = 24, source: str = "left+right",
               gap_deg: float = 0.2, gap_mm: float = 1.0,
               kept: np.ndarray | None = None) -> PhotoCandidate:
    """A pair that passes every gate the arguments do not spoil."""
    R, t = pose if pose is not None else _pose_at([0.0, 0.0, -500.0])
    return PhotoCandidate(
        left=_eye("left", (R, t), sharpness=sharpness, capture_mono=mono),
        right=_eye("right", (R, t), sharpness=sharpness, capture_mono=mono - 0.012,
                   shifted=True),
        pair_capture_mono=mono, pose_source=source, pose_rms_px=0.31,
        pose_gap_deg=gap_deg, pose_gap_mm=gap_mm, pose_corners=corners,
        pose_smooth_mm=0.21, pass_id=0, laser_on=True,
        kept_xyz_board=(np.zeros((0, 3), np.float32) if kept is None else kept))


def _still(pose=None, mono: float = 1000.0,
           n: int = STILL_HISTORY) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """A pose history of a rig that is not moving."""
    R, t = pose if pose is not None else _pose_at([0.0, 0.0, -500.0])
    return [(mono - 0.033 * (n - 1 - k), R, t) for k in range(n)]


# ── the policy ───────────────────────────────────────────────────────────


def test_policy_rejects_while_moving() -> None:
    policy = CapturePolicy()
    cand = _candidate()
    # A hand sliding 2 mm across the window: every frame is sharp, the pose is
    # good, and one pose still cannot describe a rolling-shutter readout.
    moving = [(1000.0 - 0.033 * k, np.eye(3), np.array([0.0, 0.0, 500.0 + 0.5 * k]))
              for k in range(STILL_HISTORY - 1, -1, -1)]
    why = policy.decide(cand, "left", moving, [], [])
    assert why is not None and "moving" in why
    assert policy.decide(cand, "left", _still(), [], []) is None


def test_policy_needs_a_full_window_before_it_calls_anything_still() -> None:
    policy = CapturePolicy()
    short = _still(n=STILL_HISTORY - 1)
    why = policy.decide(_candidate(), "left", short, [], [])
    assert why is not None and "stillness" in why


def test_policy_rejects_a_repeat_viewpoint() -> None:
    policy = CapturePolicy()
    kept_pose = _pose_at([0.0, 0.0, -500.0])
    # 5 mm along the view axis and not a degree of turn: the same photograph.
    cand = _candidate(_pose_at([0.0, 0.0, -505.0]))
    kept = [(990.0, kept_pose[0], kept_pose[1])]            # ten seconds ago
    why = policy.decide(cand, "left", _still(), kept, [])
    assert why is not None and "same viewpoint" in why


def test_policy_accepts_a_new_angle() -> None:
    policy = CapturePolicy()
    kept_pose = _pose_at([0.0, 0.0, -500.0])
    kept = [(990.0, kept_pose[0], kept_pose[1])]
    # Round the subject: a quarter of a metre away and 30° round.
    far = _candidate(_pose_at([250.0, 0.0, -433.0], 30.0))
    assert policy.decide(far, "left", _still(), kept, []) is None
    # And the OR is a real OR: from almost the same place, looking 30° off.
    turned = _candidate(_pose_at([0.0, 0.0, -505.0], 30.0))
    assert float(np.linalg.norm(camera_centre(*_pose_at([0.0, 0.0, -505.0], 30.0))
                                - np.array([0.0, 0.0, -500.0]))) < policy.novelty_mm
    assert policy.decide(turned, "left", _still(), kept, []) is None


def test_policy_min_interval() -> None:
    policy = CapturePolicy()
    kept_pose = _pose_at([250.0, 0.0, -433.0], 30.0)        # nowhere near
    cand = _candidate()
    recent = [(1000.0 - 0.2, kept_pose[0], kept_pose[1])]
    why = policy.decide(cand, "left", _still(), recent, [])
    assert why is not None and "since the last left photograph" in why
    # The same photograph, a second later, is taken.
    older = [(1000.0 - 1.0, kept_pose[0], kept_pose[1])]
    assert policy.decide(cand, "left", _still(), older, []) is None


def test_policy_rejects_a_blurred_frame() -> None:
    policy = CapturePolicy()
    offers = [100.0] * 12
    blurred = _candidate(sharpness=50.0)                    # under 0.7 × 100
    why = policy.decide(blurred, "left", _still(), [], offers)
    assert why is not None and "sharpness" in why
    assert policy.decide(_candidate(sharpness=75.0), "left", _still(), [], offers) is None


def test_policy_ignores_sharpness_until_ten_samples() -> None:
    policy = CapturePolicy()
    hopeless = _candidate(sharpness=1.0)
    # Nine offers say nothing about what this run's frames look like.
    assert policy.decide(hopeless, "left", _still(), [], [100.0] * 9) is None
    why = policy.decide(hopeless, "left", _still(), [], [100.0] * 10)
    assert why is not None and "sharpness" in why


def test_policy_wants_a_pose_it_can_stand_behind() -> None:
    policy = CapturePolicy()
    thin = _candidate(source="left", corners=6, gap_deg=float("nan"),
                      gap_mm=float("nan"))
    why = policy.decide(thin, "left", _still(), [], [])
    assert why is not None and "corners" in why
    # One eye with corners enough is fine — the gaps are NaN and pass.
    ok = _candidate(source="left", corners=20, gap_deg=float("nan"),
                    gap_mm=float("nan"))
    assert policy.decide(ok, "left", _still(), [], []) is None
    # Two eyes that disagree are not.
    apart = _candidate(gap_deg=3.5, gap_mm=40.0)
    why = policy.decide(apart, "left", _still(), [], [])
    assert why is not None and "differ by" in why


def test_policy_refuses_a_side_that_kept_no_pixels() -> None:
    cand = _candidate()
    cand.right.jpeg = None
    why = CapturePolicy().decide(cand, "right", _still(), [], [])
    assert why is not None and "no pixels" in why
    assert CapturePolicy().decide(cand, "left", _still(), [], []) is None


def test_policy_imports_nothing_from_scanworker() -> None:
    """The circularity is real — `scanworker` imports the scan and the
    workers — and it is why the stillness constants are duplicated in
    `photos`. Asserted in a subprocess because another test in this session
    may already have imported the very modules being looked for."""
    code = ("import sys, orbiter_native.photos; "
            "print(sorted(m for m in sys.modules if m in "
            "('orbiter_native.scanworker', 'orbiter_native.worker') "
            "or m.split('.')[0] == 'PySide6'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"


# ── the session ──────────────────────────────────────────────────────────


def _rig() -> RigSnapshot:
    return RigSnapshot(
        board=BoardSnapshot(squares_x=12, squares_y=9, square_mm=24.0,
                            marker_mm=18.0, dictionary="DICT_5X5_1000"),
        volume=ScanVolume(height_mm=380.0, radius_mm=140.0, floor_mm=6.0),
        left=EyeSnapshot("cam2", (1920, 1080), 1500.1, 1499.7, 962.3, 541.8,
                         (-0.31, 0.12, 0.0004, -0.0002, 0.0), 0.81),
        right=EyeSnapshot("cam4", (1920, 1080), 1498.4, 1497.9, 958.1, 538.2,
                          (-0.30, 0.11, 0.0003, -0.0001, 0.0), 0.84),
        extrinsics=Extrinsics(R=Rotation.from_euler("y", 3.4, degrees=True).as_matrix(),
                              t_mm=np.array([-144.0, 0.7, 8.2]), rms_px=0.89))


def test_session_root_honours_the_env_var(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(SESSIONS_ENV, str(tmp_path))
    assert sessions_root() == tmp_path
    session = PhotoSession()
    assert session.path.parent == tmp_path
    assert session.photos_dir.is_dir() and session.stripe_dir.is_dir()
    monkeypatch.setenv(SESSIONS_ENV, "   ")
    assert sessions_root() != tmp_path                       # blank is not a path


def test_session_json_round_trips_the_rig_snapshot(tmp_path) -> None:
    rig = _rig()
    session = PhotoSession(tmp_path, rig=rig)
    data = json.loads(session.json_path.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA and data["session_id"] == session.session_id
    assert data["started_utc"].endswith("Z")

    back = RigSnapshot.from_dict(data)
    assert back.board == rig.board
    assert back.volume == rig.volume
    assert back.left == rig.left and back.right == rig.right
    assert np.allclose(back.extrinsics.R, rig.extrinsics.R)
    assert np.allclose(back.extrinsics.t_mm, rig.extrinsics.t_mm)
    assert back.extrinsics.rms_px == rig.extrinsics.rms_px
    # The thresholds that decided what was photographed travel with it, so a
    # run can be told apart from another run by reading one file.
    assert data["params"]["capture"]["novelty_mm"] == CapturePolicy().novelty_mm
    assert data["params"]["capture"]["still_mm"] == CapturePolicy().still_mm


def test_two_sessions_in_one_second_do_not_share_a_directory(tmp_path) -> None:
    first = PhotoSession(tmp_path)
    second = PhotoSession(tmp_path, started=first.started)
    assert first.path != second.path
    # And each is named for the directory it actually got: the id travels into
    # session.json, onto the panel and through every ScanStatus, and one that
    # named the directory the session did NOT get would send a reconstruction
    # to the other session's files.
    assert first.session_id == first.path.name
    assert second.session_id == second.path.name
    assert first.session_id != second.session_id
    for session in (first, second):
        data = json.loads(session.json_path.read_text(encoding="utf-8"))
        assert data["session_id"] == session.path.name


def test_a_new_pass_is_a_new_number(tmp_path) -> None:
    session = PhotoSession(tmp_path)
    assert session.pass_id == 0
    assert session.next_pass() == 1 and session.pass_id == 1


# ── the writer ───────────────────────────────────────────────────────────


def test_writer_writes_bytes_verbatim_and_one_jsonl_line_each(tmp_path) -> None:
    session = PhotoSession(tmp_path, rig=_rig())
    writer = PhotoWriter(session)
    writer.start()
    for k in range(3):
        cand = _candidate(mono=1000.0 + k)
        cand.left.jpeg = b"\xff\xd8left-%d\xff\xd9" % k
        cand.right.jpeg = b"\xff\xd8right-%d\xff\xd9" % k
        writer.put_nowait(cand.record("left"))
        writer.put_nowait(cand.record("right"))
    writer.stop()

    assert session.counts == {"left": 3, "right": 3}
    for k in range(3):
        assert (session.photos_dir / f"left_{k + 1:04d}.jpg").read_bytes() \
            == b"\xff\xd8left-%d\xff\xd9" % k
        assert (session.photos_dir / f"right_{k + 1:04d}.jpg").read_bytes() \
            == b"\xff\xd8right-%d\xff\xd9" % k

    lines = session.manifest_path.read_bytes().decode("utf-8").splitlines()
    assert len(lines) == 6
    rows = [json.loads(line) for line in lines]
    assert [r["side"] for r in rows] == ["left", "right"] * 3
    assert [r["n"] for r in rows] == [1, 1, 2, 2, 3, 3]
    left0, right0 = rows[0], rows[1]
    assert left0["file"] == "photos/left_0001.jpg"
    assert left0["stripe"] == "stripe/left_0001.npz"
    assert left0["camera_id"] == "cam2" and right0["camera_id"] == "cam4"
    assert left0["pose"]["convention"] == "board->camera, world=board"
    assert len(left0["pose"]["q_wxyz"]) == 4
    assert left0["pose_frame"] == "left" and right0["pose_frame"] == "right"
    # The right photograph carries its OWN instant beside the pose's.
    assert left0["pose_composed"] is False and right0["pose_composed"] is True
    assert left0["capture_mono"] == left0["pair_capture_mono"]
    assert right0["capture_mono"] != right0["pair_capture_mono"]
    assert abs(right0["capture_mono"] - right0["pair_capture_mono"]) <= 0.020
    assert right0["stripe_shifted"] is True and left0["stripe_shifted"] is False
    assert left0["stripe_pixels"] == 7 and left0["kept_points"] == 0
    assert left0["pass_id"] == 0 and left0["laser_on"] is True


def test_stripe_sidecar_round_trips(tmp_path) -> None:
    session = PhotoSession(tmp_path)
    writer = PhotoWriter(session)
    kept = np.array([[1.5, -2.5, 30.0], [4.0, 5.0, 31.25]], np.float32)
    cand = _candidate(kept=kept)
    writer.put_nowait(cand.record("right"))
    writer.stop()                                   # never started: drains by hand

    with np.load(session.stripe_dir / "right_0001.npz") as z:
        stripe = _stripe()
        assert z["x"].dtype == np.int32 and np.array_equal(z["x"], stripe.x)
        assert z["y"].dtype == np.int32 and np.array_equal(z["y"], stripe.y)
        assert z["w"].dtype == np.uint8 and np.array_equal(z["w"], stripe.w)
        assert z["r"].dtype == np.uint8 and np.array_equal(z["r"], stripe.r)
        assert z["kept_xyz_board"].dtype == np.float32
        assert z["kept_xyz_board"].shape == (2, 3)
        assert np.array_equal(z["kept_xyz_board"], kept)
        assert list(z["wh"]) == [1920, 1080]
        assert bool(z["along_x"]) is False           # this stripe runs down columns
        assert int(z["pass_id"]) == 0
        assert bool(z["stripe_shifted"]) is True


def test_a_photo_pass_sidecar_carries_no_points(tmp_path) -> None:
    """Photo-pass mode builds no `ScanFrame` at all, so there are no kept
    points to carry — and the sidecar says so with a shape, not a null."""
    session = PhotoSession(tmp_path)
    writer = PhotoWriter(session)
    writer.put_nowait(_candidate().record("left"))
    writer.stop()
    with np.load(session.stripe_dir / "left_0001.npz") as z:
        assert z["kept_xyz_board"].shape == (0, 3)
        assert z["kept_xyz_board"].dtype == np.float32


def test_writer_never_blocks_and_drops_oldest_when_full(tmp_path) -> None:
    session = PhotoSession(tmp_path)
    writer = PhotoWriter(session)                   # deliberately not started
    extra = 8
    started = time.perf_counter()
    for k in range(QUEUE_DEPTH + extra):
        cand = _candidate()
        cand.left.jpeg = b"jpeg-%03d" % k
        writer.put_nowait(cand.record("left"))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5                            # a stalled disk is not a stall
    assert writer.dropped == extra

    writer.start()
    writer.stop()
    assert writer.written == QUEUE_DEPTH
    # What survived is the newest QUEUE_DEPTH offers: the oldest went first.
    kept = [(session.photos_dir / f"left_{n:04d}.jpg").read_bytes()
            for n in range(1, QUEUE_DEPTH + 1)]
    assert kept == [b"jpeg-%03d" % k for k in range(extra, QUEUE_DEPTH + extra)]


def test_bytes_on_disk_tracks_what_the_writer_wrote(tmp_path) -> None:
    session = PhotoSession(tmp_path)
    at_open = session.bytes_on_disk
    assert at_open == session.json_path.stat().st_size     # one walk, one file

    writer = PhotoWriter(session)
    writer.start()
    for k in range(5):
        cand = _candidate(mono=1000.0 + k, kept=np.full((3, 3), 1.0, np.float32))
        writer.put_nowait(cand.record("left"))
        writer.put_nowait(cand.record("right"))
    writer.stop()

    walked = sum(p.stat().st_size for p in session.path.rglob("*") if p.is_file())
    assert session.bytes_on_disk == walked
    assert session.bytes_on_disk > at_open
    assert session.rewalk() == walked


def test_a_write_that_fails_does_not_escape_the_hand_drain(tmp_path, monkeypatch) -> None:
    """`stop` finishes the queue by hand, from `closeEvent`.

    The thread's loop has always caught what a write raises; the drain did
    not, so a disk that filled up on the last photograph of a session took
    every thread shutdown after it — the eye workers, the scan worker, the
    exposure keeper — out through the same exception.
    """
    session = PhotoSession(tmp_path)
    writer = PhotoWriter(session)                   # never started: drains by hand
    real_write, refused = writer._write, []

    def write(rec) -> None:
        if not refused:
            refused.append(rec.side)
            raise OSError(28, "No space left on device")
        real_write(rec)

    monkeypatch.setattr(writer, "_write", write)
    writer.put_nowait(_candidate(mono=1000.0).record("left"))
    writer.put_nowait(_candidate(mono=1001.0).record("right"))
    writer.stop()                                   # must return, not raise

    assert refused == ["left"]
    assert writer.failed == 1
    # The one behind it went to disk all the same.
    assert session.counts == {"left": 0, "right": 1}
    assert (session.photos_dir / "right_0001.jpg").is_file()


def test_a_record_that_raises_does_not_kill_the_writer_thread(tmp_path) -> None:
    """A pose that is not a rotation matrix — one NaN is enough — raises
    `ValueError` out of `Rotation.from_matrix`, not `OSError`.

    Caught narrowly, that ended the thread; and `start` refuses to raise a
    second one, so every photograph after it was queued and then dropped in a
    session that went on looking healthy. One photograph is worth losing here,
    the rest of the pass is not.
    """
    session = PhotoSession(tmp_path)
    writer = PhotoWriter(session)
    writer.start()
    bad = _candidate().record("left")
    bad.R = np.zeros((3, 3))                        # no quaternion comes out of this
    writer.put_nowait(bad)
    writer.put_nowait(_candidate(mono=1001.0).record("right"))

    for _ in range(500):
        if writer.failed and writer.written:
            break
        time.sleep(0.01)
    assert writer._thread is not None and writer._thread.is_alive()
    writer.stop()

    assert writer.failed == 1 and writer.written == 1
    assert session.counts["right"] == 1
    assert [r["side"] for r in _rows(session)] == ["right"]


def _no_constants(name: str):
    raise AssertionError(f"the manifest is not JSON: it carries a bare {name}")


def _rows(session: PhotoSession) -> list[dict]:
    """Every manifest line, parsed by a reader that refuses JavaScript's
    `NaN` and `Infinity` — which is what a reader outside Python is."""
    if not session.manifest_path.exists():
        return []
    return [json.loads(line, parse_constant=_no_constants)
            for line in session.manifest_path.read_text(encoding="utf-8").splitlines()]


def test_a_pose_gap_that_is_not_a_number_is_written_as_null(tmp_path) -> None:
    """A pose one eye solved alone has no gap between two eyes to report, and
    carries NaN. `json.dumps` writes that as a bare `NaN`, which is not JSON
    and which every reader that is not Python's refuses — including the ones
    that would read this manifest to build a model."""
    session = PhotoSession(tmp_path)
    writer = PhotoWriter(session)
    cand = _candidate(source="left", corners=20, gap_deg=float("nan"),
                      gap_mm=float("nan"))
    cand.pose_rms_px = float("nan")
    writer.put_nowait(cand.record("left"))
    writer.stop()

    assert "NaN" not in session.manifest_path.read_text(encoding="utf-8")
    row = _rows(session)[0]
    assert row["pose_gap_deg"] is None and row["pose_gap_mm"] is None
    assert row["pose_rms_px"] is None
    # The numbers that ARE numbers are untouched, nulls or no nulls.
    assert row["pose_smooth_mm"] == 0.21
    assert row["pose_corners"] == 20 and row["pose_source"] == "left"


def test_the_manifest_quaternion_is_the_one_colmap_is_written_with(tmp_path) -> None:
    """One function in this app turns a rotation into a scalar-first
    quaternion, and it is `colmapio.quat_wxyz` — the one that canonicalises
    the sign and has a golden test. A second copy here would be a second thing
    to get wrong, and the two would disagree for exactly half the rotations."""
    session = PhotoSession(tmp_path)
    writer = PhotoWriter(session)
    # Past a half turn, where q and -q are both the rotation and only one of
    # them is what gets written.
    pose = _pose_at([0.0, 0.0, -500.0], 200.0)
    writer.put_nowait(_candidate(pose).record("left"))
    writer.stop()

    q = _rows(session)[0]["pose"]["q_wxyz"]
    assert q == list(quat_wxyz(pose[0]))
    assert q[0] >= 0.0
