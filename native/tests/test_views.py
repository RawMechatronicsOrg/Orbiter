"""Choosing the photographs, seeding the model, and tracking the seeds into it.

Everything here is synthetic and therefore checkable: the subject is a box of
known size standing on the board, its normals are the outward normals of the
face each point sits on, and every camera is placed by hand at a known point
looking at a known one. That is the only way to test a visibility rule — with
a real cloud and a real pose there is nothing to compare the answer against.

The box is 220 mm across and 100 mm tall with its base at z = 20, which puts
it inside the scan volume and, from 250 mm away through
`test_stereo_scan`'s intrinsics, fills about three quarters of the frame at a
median depth near 180 mm. Every default gate in `SelectParams` passes on it
with room to spare, so a test that sees a photograph dropped is watching the
rule it is about and not the fixture's margins.

One deliberate avoidance: no camera is placed at an azimuth that is a multiple
of 15 degrees. Those sit exactly on a view-bucket boundary, where which side a
direction falls on is decided by the last bit of an `arctan2`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from orbiter_native.photos import (
    BoardSnapshot,
    EyePhoto,
    EyeSnapshot,
    Extrinsics,
    PhotoCandidate,
    PhotoSession,
    PhotoWriter,
    RigSnapshot,
    camera_centre,
    view_direction,
)
from orbiter_native.scan import ScanVolume
from orbiter_native.stereo import compose_right_pose
from orbiter_native.views import (
    CAMERA_ID,
    PhotoMeta,
    SelectParams,
    SessionInfo,
    buckets,
    image_list_text,
    load_session,
    observation_counts,
    seed_points,
    select,
    texture_set,
    tracks,
    write_image_list,
)

from test_stereo_scan import KL, KR, R_TRUE, T_TRUE, WH
from test_stereo_scan import _rig as _stereo_rig

#: The subject's centre, and what every camera looks at.
TARGET = np.array([0.0, 0.0, 70.0])
#: Where a camera stands to see it well: 250 mm out, 20 degrees above it. Both
#: numbers are checked by `test_the_fixture_clears_every_default_gate`, so a
#: change to either is caught there rather than in whichever test it breaks.
GOOD_MM = 250.0
GOOD_ELEV = 20.0
#: Far enough that the subject is a speck and the depth gate refuses it too.
FAR_MM = 900.0


# ── the fixture: a box on the board, and cameras around it ───────────────


def _box(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """A box standing on the board — 220 mm across, 100 mm tall, base at
    z = 20 — sampled on its four walls and its top, with the outward normal of
    the face each point lies on.

    A box rather than a sphere because its faces are flat: which points face a
    given camera is three dot products the reader can do in their head, and
    the front-facing rule is what half of these tests are about. The base is
    not sampled — it sits on the board and no camera ever sees it.
    """
    across = np.linspace(-110.0, 110.0, n)
    up = np.linspace(20.0, 120.0, n)
    wall_a, wall_z = np.meshgrid(across, up, indexing="ij")
    points, normals = [], []
    for sign in (1.0, -1.0):
        points.append(np.stack([np.full(wall_a.size, sign * 110.0),
                                wall_a.ravel(), wall_z.ravel()], axis=1))
        normals.append(np.tile([sign, 0.0, 0.0], (wall_a.size, 1)))
        points.append(np.stack([wall_a.ravel(),
                                np.full(wall_a.size, sign * 110.0),
                                wall_z.ravel()], axis=1))
        normals.append(np.tile([0.0, sign, 0.0], (wall_a.size, 1)))
    top_x, top_y = np.meshgrid(across, across, indexing="ij")
    points.append(np.stack([top_x.ravel(), top_y.ravel(),
                            np.full(top_x.size, 120.0)], axis=1))
    normals.append(np.tile([0.0, 0.0, 1.0], (top_x.size, 1)))
    return (np.concatenate(points).astype(float),
            np.concatenate(normals).astype(float))


BOX_XYZ, BOX_N = _box()


def _look_at(centre_mm, target_mm=TARGET) -> tuple[np.ndarray, np.ndarray]:
    """A board→camera pose for a camera standing at `centre_mm` and pointing
    at `target_mm`, both in the board frame.

    The rows of `R` are the camera's own axes in board coordinates, so its
    third row is where the camera looks — which is what `view_direction`
    reads — and `t = -R c` puts `camera_centre` back at `centre_mm`.
    """
    centre = np.asarray(centre_mm, float)
    forward = np.asarray(target_mm, float) - centre
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    R = np.stack([right, np.cross(forward, right), forward])
    return R, -R @ centre


def _stand(az_deg: float, radius: float = GOOD_MM,
           elev_deg: float = GOOD_ELEV) -> np.ndarray:
    """A camera's place: `radius` from the subject, at `az_deg` around it and
    `elev_deg` above it."""
    az, elev = np.radians(az_deg), np.radians(elev_deg)
    return TARGET + radius * np.array([np.cos(elev) * np.cos(az),
                                       np.cos(elev) * np.sin(az),
                                       np.sin(elev)])


_n = [0]


def _meta(R: np.ndarray, t_mm: np.ndarray, *, side: str = "left",
          sharpness: float = 100.0, laser_on: bool = True,
          pass_id: int = 0) -> PhotoMeta:
    """One manifest row's worth of photograph, at the pose given."""
    _n[0] += 1
    n = _n[0]
    return PhotoMeta(
        n=n, side=side, camera_id="cam2" if side == "left" else "cam4",
        file=f"photos/{side}_{n:04d}.jpg", name=f"{side}_{n:04d}.jpg",
        stripe=f"stripe/{side}_{n:04d}.npz", wh=WH,
        capture_mono=1000.0 + n, pair_capture_mono=1000.0 + n,
        R=np.asarray(R, float), t_mm=np.asarray(t_mm, float),
        pose_source="left+right", pose_composed=(side == "right"),
        pose_rms_px=0.31, pose_gap_deg=0.2, pose_gap_mm=1.0, pose_corners=24,
        pose_smooth_mm=0.21, sharpness=sharpness, pass_id=pass_id,
        laser_on=laser_on, stripe_pixels=612 if laser_on else 0,
        stripe_shifted=(side == "right"), kept_points=0)


def _at(az_deg: float, **kw) -> PhotoMeta:
    """A photograph taken from `az_deg` around the subject, looking at it."""
    R, t = _look_at(_stand(az_deg, kw.pop("radius", GOOD_MM),
                           kw.pop("elev_deg", GOOD_ELEV)))
    return _meta(R, t, **kw)


def _arc(az_from: float, az_to: float, count: int, **kw) -> list[PhotoMeta]:
    """`count` photographs spread evenly around an arc of the orbit."""
    return [_at(float(az), **kw)
            for az in np.linspace(az_from, az_to, count)]


def _eye(k, camera_id: str) -> EyeSnapshot:
    return EyeSnapshot(camera_id=camera_id, wh=WH, fx=k.fx, fy=k.fy,
                       cx=k.cx, cy=k.cy, dist=(0.0,) * 5, rms_px=0.81)


def _rig() -> RigSnapshot:
    """The rig as a session records it. The volume's radius is 200 mm rather
    than the live 150 so the 220 mm box stands inside it — a fixture whose
    subject pokes out of its own scan volume would be describing a scan
    nobody could have taken."""
    return RigSnapshot(
        board=BoardSnapshot(8, 8, 36.0, 26.64, "DICT_5X5_100"),
        volume=ScanVolume(height_mm=400.0, radius_mm=200.0, floor_mm=5.0),
        left=_eye(KL, "cam2"), right=_eye(KR, "cam4"),
        extrinsics=Extrinsics(R=R_TRUE, t_mm=T_TRUE, rms_px=0.89))


def _session(session_id: str = "20260906-213500") -> SessionInfo:
    return SessionInfo(session_id=session_id, path=Path(session_id),
                       started_utc="2026-09-06T18:35:00Z", rig=_rig())


def _select(photos: list[PhotoMeta], params: SelectParams | None = None,
            session: SessionInfo | None = None):
    session = session or _session()
    return session, select(session, photos, BOX_XYZ, BOX_N, params)


# ── the fixture itself ───────────────────────────────────────────────────


def test_the_fixture_clears_every_default_gate() -> None:
    """The box seen from the standard place passes all three gates with
    margin, so every later drop is the rule and not the geometry."""
    _, sel = _select(_arc(7.0, 340.0, 8))
    assert len(sel.accepted) == 8
    for v in sel.accepted:
        assert v.n_visible > 2000
        assert v.covered > 0.5
        assert 150.0 < v.median_depth_mm < 250.0


# ── selection ────────────────────────────────────────────────────────────


def test_selection_drops_a_photo_looking_away() -> None:
    """A camera in the right place, turned around: every point of the cloud is
    behind it, so it sees none of them."""
    centre = _stand(37.0)
    good = _meta(*_look_at(centre))
    away = _meta(*_look_at(centre, 2.0 * centre - TARGET))
    _, sel = _select([good, away])

    (kept,) = sel.accepted
    assert kept.photo is good
    (lost,) = sel.dropped
    assert lost.photo is away
    assert lost.n_visible == 0
    assert "0 seed points" in lost.dropped_because


def test_selection_prefers_a_clean_photo_over_a_laser_photo_at_the_same_pose() -> None:
    """Where a laser photograph and a clean one stand in the same place the
    clean one survives — even when the laser one is the sharper of the two,
    because a stripe in the atlas costs more than a little blur."""
    R, t = _look_at(_stand(37.0))
    laser = _meta(R, t, sharpness=400.0, laser_on=True)
    clean = _meta(R, t, sharpness=100.0, laser_on=False, pass_id=1)
    _, sel = _select([laser, clean])

    (kept,) = sel.accepted
    assert kept.photo is clean
    (lost,) = sel.dropped
    assert lost.photo is laser
    assert clean.name in lost.dropped_because


def test_selection_deduplicates_near_identical_poses() -> None:
    """Two photographs that are both within 8 degrees and within 20 mm of each
    other are one photograph. Either difference alone is a new view."""
    first = _at(37.0, sharpness=400.0)
    # 4 mm along the orbit and a fraction of a degree round it: the same view.
    # Blurrier, so which of the two the greedy pass reaches first is decided
    # rather than left to a third decimal place of coverage.
    twin = _at(37.0 + np.degrees(4.0 / GOOD_MM), sharpness=100.0)
    # Far enough round to clear the distance rule on its own.
    apart = _at(37.0 + np.degrees(40.0 / GOOD_MM), sharpness=200.0)
    _, sel = _select([first, twin, apart])

    assert [v.photo for v in sel.accepted] == [first, apart]
    (lost,) = sel.dropped
    assert lost.photo is twin
    assert "the same view as" in lost.dropped_because
    assert first.name in lost.dropped_because


def test_selection_respects_the_cap() -> None:
    """The cap is a hard stop, and the photographs it stops are named as
    stopped rather than dropped for some geometric reason they did not have."""
    assert SelectParams().cap == 150             # the shipped number
    photos = _arc(7.0, 340.0, 12)
    _, sel = _select(photos, SelectParams(cap=5))

    assert len(sel.accepted) == 5
    assert len(sel.dropped) == 7
    assert all("full at 5 photographs" in v.dropped_because for v in sel.dropped)


def test_every_dropped_photo_has_a_reason() -> None:
    """Four different ways to lose a photograph, and a sentence for each — a
    photograph that vanished without one is the failure this asserts against."""
    centre = _stand(53.0)
    # Sharpness fixes the greedy order, so which photograph meets which rule
    # is decided here rather than by a third decimal place of coverage: the
    # sharpest is accepted, its twin is a duplicate, the next two fill the cap
    # and the last one arrives to a full selection.
    photos = [
        _at(7.0, sharpness=500.0),
        _at(7.2, sharpness=400.0),                              # a duplicate
        _at(60.0, sharpness=300.0),
        _at(113.0, sharpness=200.0),
        _at(166.0, sharpness=100.0),                            # past the cap
        _meta(*_look_at(centre, 2.0 * centre - TARGET)),        # looking away
        _at(203.0, radius=FAR_MM),                              # too far
    ]
    _, sel = _select(photos, SelectParams(cap=3))

    assert len(sel.views) == len(sel.accepted) + len(sel.dropped)
    assert len(sel.accepted) == 3
    assert all(v.dropped_because for v in sel.dropped)
    assert all(v.dropped_because is None for v in sel.accepted)
    reasons = " | ".join(v.dropped_because for v in sel.dropped)
    assert "seed points" in reasons
    assert "median depth" in reasons or "of the frame" in reasons
    assert "the same view as" in reasons
    assert "full at 3 photographs" in reasons


def test_selection_counts_the_sides_and_the_clean_pass() -> None:
    """`session.json`'s `selected` block, straight off the selection."""
    photos = [*_arc(7.0, 160.0, 3, side="left"),
              *_arc(203.0, 340.0, 3, side="right", laser_on=False, pass_id=1),
              _at(7.0, radius=FAR_MM)]
    _, sel = _select(photos)
    assert sel.counts == {"left": 3, "right": 3, "clean": 3, "dropped": 1}
    assert sel.names == [v.photo.name for v in sel.accepted]
    assert [v.image_id for v in sel.accepted] == [1, 2, 3, 4, 5, 6]


# ── view buckets ─────────────────────────────────────────────────────────


def test_buckets_measure_the_selection_not_the_object() -> None:
    """Buckets bin what was SELECTED. Photographs the gates threw away are not
    buckets nobody covered — they are buckets nobody usefully photographed,
    and counting them would fail the clean gate for a reason no clean pass
    could ever fix."""
    near = _arc(7.0, 160.0, 4)
    far = _arc(203.0, 340.0, 4, radius=FAR_MM)
    session, strict = _select([*near, *far])
    _, loose = _select([*near, *far], SelectParams(
        min_visible=0, min_covered=0.0, depth_mm=(0.0, 1.0e9)), session=session)

    assert len(strict.accepted) == 4 and len(loose.accepted) == 8
    tight, wide = set(buckets(session, strict)), set(buckets(session, loose))
    assert tight < wide                       # the far arc's buckets are only
    assert len(wide - tight) == 4             # in the selection that kept it

    # And every accepted photograph is in exactly one bucket, once.
    listed = sorted(i for ids in buckets(session, strict).values() for i in ids)
    assert listed == [v.image_id for v in strict.accepted]


# ── the texture set ──────────────────────────────────────────────────────


def test_texture_set_contains_only_clean_photos_when_coverage_is_good() -> None:
    clean = _arc(7.0, 340.0, 16, laser_on=False, pass_id=1)
    session, sel = _select(clean)
    names, source, warning = texture_set(session, sel, "texture-only")

    assert source == "clean" and warning is None
    assert names == [v.photo.name for v in sel.clean] == sel.names


def test_texture_set_falls_back_below_twelve_clean_photos() -> None:
    """Six clean photographs cover every bucket the selection covers — and are
    still too few. This is the count gate on its own."""
    session, sel = _select(_arc(7.0, 340.0, 6, laser_on=False, pass_id=1))
    names, source, warning = texture_set(session, sel, "texture-only")

    assert source == "raw"
    assert names == sel.names
    assert "only 6 clean photos" in warning
    assert "100% bucket coverage" in warning


def test_texture_set_falls_back_when_clean_photos_miss_a_region() -> None:
    """Thirty clean photographs, all in one half of the orbit: the count gate
    passes easily and the coverage gate refuses. This is the case a bare count
    could not see, and the warning has to name where the hole is."""
    session, sel = _select([*_arc(7.5, 172.5, 30, laser_on=False, pass_id=1),
                            *_arc(187.5, 352.5, 12)])
    names, source, warning = texture_set(session, sel, "texture-only")

    assert len(sel.clean) == 30 >= SelectParams().clean_min
    assert source == "raw" and names == sel.names
    assert "only 30 clean photos" in warning

    covered = buckets(session, sel)
    clean_ids = {v.image_id for v in sel.clean}
    missing = {key for key, ids in covered.items()
               if not clean_ids.intersection(ids)}
    assert missing, "the laser half must cover buckets the clean half does not"

    # The warning names them by azimuth range, and those ranges are exactly
    # the uncovered bins — not a rounded-up arc that sweeps in covered ones.
    named: set[int] = set()
    for lo, hi in re.findall(r"az (-?[\d.]+)-(-?[\d.]+)", warning):
        named.update(range(int(float(lo) // 15.0), int(float(hi) // 15.0)))
    assert named == {az for az, _ in missing}


def test_texture_set_source_depends_on_the_mode() -> None:
    """The same failing clean set is `raw` in texture-only and `inpainted` in
    dense — the dense workspace is undistorted from inpainted pixels, and the
    label has to say so. A passing set is `clean` in both."""
    session, thin = _select(_arc(7.0, 340.0, 6, laser_on=False, pass_id=1))
    assert texture_set(session, thin, "texture-only")[1] == "raw"
    assert texture_set(session, thin, "dense")[1] == "inpainted"

    session, full = _select(_arc(7.0, 340.0, 16, laser_on=False, pass_id=1))
    assert texture_set(session, full, "texture-only")[1] == "clean"
    assert texture_set(session, full, "dense")[1] == "clean"


def test_image_list_is_one_name_per_line_matching_images_txt(tmp_path) -> None:
    session, sel = _select(_arc(7.0, 340.0, 6, laser_on=False, pass_id=1))
    names, _, _ = texture_set(session, sel, "texture-only")
    path = tmp_path / "texture_images.txt"
    write_image_list(path, names)

    raw = path.read_bytes()
    assert b"\r" not in raw                       # LF on every host
    assert raw.decode("utf-8").splitlines() == names
    assert image_list_text(names) == raw.decode("utf-8")


# ── seeds ────────────────────────────────────────────────────────────────


def test_seed_points_are_capped_and_subsampled_uniformly() -> None:
    """400 000 points come back as at most 30 000, spread through the cloud's
    volume rather than through its ordering — a prefix or a stride would leave
    whole regions of the object with no depth range at all."""
    rng = np.random.default_rng(4)
    xyz = rng.uniform(-50.0, 50.0, (400_000, 3))
    # Normals stand in for any per-point payload: whatever rows survive, the
    # payload's rows must be the same ones.
    out, rgb, normals = seed_points(xyz, None, xyz.copy(), cap=30_000)

    assert len(out) <= 30_000
    assert rgb.shape == out.shape and normals.shape == out.shape
    assert np.array_equal(normals, out)
    assert (rgb == 128).all()                     # an uncoloured cloud

    counts = np.bincount(
        ((out[:, 0] >= 0) * 4 + (out[:, 1] >= 0) * 2 + (out[:, 2] >= 0)),
        minlength=8)
    assert counts.min() >= 0.8 * counts.max()     # uniform, within 20 %


def test_seed_points_leave_a_small_cloud_alone() -> None:
    xyz, _ = _box(n=8)
    out, rgb, normals = seed_points(xyz, None, None, cap=30_000)
    assert np.array_equal(out, xyz)
    assert (normals == 0.0).all()                 # given none, reports none


# ── tracks ───────────────────────────────────────────────────────────────


def _tracked(n_photos: int = 8, cap: int = 400):
    session, sel = _select(_arc(7.0, 340.0, n_photos))
    seeds, _, normals = seed_points(BOX_XYZ, None, BOX_N, cap=cap)
    points3d, points2d = tracks(sel, seeds, normals)
    return sel, seeds, points3d, points2d


def test_tracks_have_length_two_or_more_and_index_valid_points2d() -> None:
    """A point seen by one image constrains nothing, and a track entry that
    indexes past the end of an image's POINTS2D aborts the dense stage."""
    sel, _, points3d, points2d = _tracked()
    assert points3d and set(points2d) == {v.image_id for v in sel.accepted}

    for point in points3d:
        assert len(point.track) >= 2
        for image_id, idx in point.track:
            seen_by = points2d[image_id]
            assert 0 <= idx < len(seen_by)
            assert seen_by[idx][2] == point.point_id


def test_points2d_and_tracks_are_generated_from_one_list() -> None:
    """Every observation is named by exactly one track entry and vice versa,
    so no POINT3D_ID is ever -1 and no index can shift. Building one side and
    filtering the other is how a converter ends up with dangling entries."""
    _, _, points3d, points2d = _tracked()

    observations = sum(len(seen) for seen in points2d.values())
    assert observations == sum(len(p.track) for p in points3d)
    assert observations > 0
    assert all(pid > 0 for seen in points2d.values() for _, _, pid in seen)

    named = {(image_id, idx) for p in points3d for image_id, idx in p.track}
    assert named == {(image_id, idx) for image_id, seen in points2d.items()
                     for idx in range(len(seen))}

    ids = [p.point_id for p in points3d]
    assert ids == list(range(1, len(points3d) + 1))


def test_tracks_use_each_eyes_own_camera_and_stay_inside_its_frame() -> None:
    """Both eyes are tracked, each through its own intrinsics, and every
    observation lands inside the image it belongs to."""
    session, sel = _select([*_arc(7.0, 160.0, 4, side="left"),
                            *_arc(203.0, 340.0, 4, side="right")])
    seeds, _, normals = seed_points(BOX_XYZ, None, BOX_N, cap=400)
    _, points2d = tracks(sel, seeds, normals)

    by_id = sel.by_image_id()
    assert {by_id[i].photo.colmap_camera_id for i in points2d} == {
        CAMERA_ID["left"], CAMERA_ID["right"]}
    for image_id, seen in points2d.items():
        w, h = by_id[image_id].photo.wh
        assert seen
        for x, y, _ in seen:
            assert 0.0 <= x < w and 0.0 <= y < h


def test_tracks_report_per_image_observation_counts() -> None:
    """The number C2b's depth-range fallback reads, and the number that says
    whether the 30 k seed cap was generous enough for this object."""
    sel, _, _, points2d = _tracked()
    counts = observation_counts(points2d)
    assert set(counts) == {v.image_id for v in sel.accepted}
    assert all(n > 0 for n in counts.values())
    assert counts == {i: len(seen) for i, seen in points2d.items()}


# ── the session, read back ───────────────────────────────────────────────


#: Where the pairs in a written session stand. The left eye is aimed half a
#: baseline to the side of the subject, because `test_stereo_scan`'s pair is
#: 200 mm wide next to a 220 mm box: aimed straight at it, the right eye gets
#: the subject at the edge of its frame and the coverage gate refuses it.
PAIR_AZ = (7.0, 53.0, 127.0)
HALF_BASELINE_MM = 100.0


def _pair_pose(az_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The LEFT eye's pose for a pair standing at `az_deg`, aimed so the
    subject falls between the two frames."""
    centre = _stand(az_deg)
    R, _ = _look_at(centre)
    return _look_at(centre, TARGET - HALF_BASELINE_MM * R[0])


def _written_session(tmp_path) -> PhotoSession:
    """A real session on disk, written the way the app writes one — including
    the right eye's pose, which is the left's composed across the pair and not
    the left's copied over (that would be wrong by the whole baseline)."""
    geom = _stereo_rig().geom
    session = PhotoSession(tmp_path, rig=_rig())
    writer = PhotoWriter(session)
    writer.start()
    for k, az in enumerate(PAIR_AZ):
        R, t = _pair_pose(az)
        R_r, t_r = compose_right_pose(R, t, geom)
        cand = PhotoCandidate(
            left=EyePhoto(camera_id="cam2", jpeg=b"\xff\xd8left\xff\xd9",
                          wh=WH, capture_mono=1000.0 + k, R=R, t_mm=t,
                          sharpness=100.0 + k),
            right=EyePhoto(camera_id="cam4", jpeg=b"\xff\xd8right\xff\xd9",
                           wh=WH, capture_mono=1000.0 + k - 0.012, R=R_r,
                           t_mm=t_r, sharpness=90.0 + k,
                           stripe_shifted=True),
            pair_capture_mono=1000.0 + k, pose_source="left+right",
            pose_rms_px=0.31, pose_gap_deg=0.2, pose_gap_mm=1.0,
            pose_corners=24, pose_smooth_mm=0.21, pass_id=0, laser_on=True)
        writer.put_nowait(cand.record("left"))
        writer.put_nowait(cand.record("right"))
    writer.stop()
    return session


def test_load_session_round_trips_what_the_writer_wrote(tmp_path) -> None:
    """`load_session` reads back exactly what `PhotoSession` and `PhotoWriter`
    put on disk — the rig, and one `PhotoMeta` per manifest line with its pose
    turned back into a matrix."""
    written = _written_session(tmp_path)
    info, photos = load_session(written.path)

    assert info.session_id == written.session_id
    assert info.path == written.path
    assert info.board == written.rig.board
    assert info.volume == written.rig.volume
    assert info.eye("left") == written.rig.left
    assert info.eye("right") == written.rig.right
    assert np.allclose(info.extrinsics.R, R_TRUE)
    assert np.allclose(info.extrinsics.t_mm, T_TRUE)
    assert info.params["capture"]["novelty_mm"] == written.policy.novelty_mm

    rows = [json.loads(line) for line
            in written.manifest_path.read_bytes().decode("utf-8").splitlines()]
    assert len(photos) == len(rows) == 6
    assert [p.name for p in photos] == ["left_0001.jpg", "right_0001.jpg",
                                        "left_0002.jpg", "right_0002.jpg",
                                        "left_0003.jpg", "right_0003.jpg"]
    for photo, row in zip(photos, rows):
        assert photo.side == row["side"] and photo.n == row["n"]
        assert photo.file == row["file"] and photo.stripe == row["stripe"]
        assert photo.wh == tuple(row["wh"])
        assert photo.laser_on == row["laser_on"]
        assert photo.pose_composed == (photo.side == "right")
        assert photo.colmap_camera_id == CAMERA_ID[photo.side]
        assert np.allclose(photo.centre_mm, camera_centre(photo.R, photo.t_mm))
        assert np.allclose(photo.direction, view_direction(photo.R))

    # The pose that went in is the pose that came out, both eyes — and the
    # right eye's is the composed one, 200 mm from the left's.
    geom = _stereo_rig().geom
    for k, az in enumerate(PAIR_AZ):
        R, t = _pair_pose(az)
        R_r, t_r = compose_right_pose(R, t, geom)
        assert np.allclose(photos[2 * k].R, R, atol=1e-9)
        assert np.allclose(photos[2 * k].t_mm, t, atol=1e-9)
        assert np.allclose(photos[2 * k + 1].R, R_r, atol=1e-9)
        assert np.allclose(photos[2 * k + 1].t_mm, t_r, atol=1e-9)
        assert np.linalg.norm(photos[2 * k].centre_mm
                              - photos[2 * k + 1].centre_mm) > 20.0


def test_a_loaded_session_selects_and_names_its_own_cameras(tmp_path) -> None:
    """The whole path in one line: a session off disk, a selection made from
    it, and the two `EyeCamera`s `write_cameras` is handed — each carrying the
    raw sizes of its own selected photographs, which is what lets that writer
    refuse a photograph taken at a resolution its intrinsics never saw."""
    info, photos = load_session(_written_session(tmp_path).path)
    sel = select(info, photos, BOX_XYZ, BOX_N, SelectParams())

    assert len(sel.accepted) == 6            # three places, two eyes each
    left, right = info.cameras(sel)
    assert (left.camera_id, left.side) == (1, "left")
    assert (right.camera_id, right.side) == (2, "right")
    assert left.photo_wh == right.photo_wh == (WH,)
    assert (left.width, left.height) == WH
    assert left.fx == KL.fx and right.fx == KR.fx
