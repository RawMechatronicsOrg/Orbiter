"""The `patch-match.cfg` rewrite, and the depth range that is only ever a
fallback.

Both of these are about a file somebody else wrote. `image_undistorter` writes
`dense/stereo/patch-match.cfg` with one `__auto__, 20` block per image it
registered, and that list — not our manifest, not our selection — is the
authority on what `patch_match_stereo` can be asked to load. Naming a frame
COLMAP did not register aborts the stage, so the tests here spend most of
their assertions on names: which survive, which are dropped, and which are
never allowed to appear.

The geometry comes from `test_views`, so the box, the intrinsics and the
selection rules are the same ones every other view test uses. The cameras
stand on a **level** orbit — elevation 0 — for one reason: with the whole ring
at the same height the angle between two viewing directions is exactly the
difference in their azimuths, so a bucket a test means to fill can be filled
by arithmetic the reader can do in their head. `_ring` asserts that identity
rather than assuming it. The radius is 300 mm rather than `test_views`' 250
because a level camera at 250 mm sees the near wall at 140 mm, under the
selection's depth gate — the fixture would then be testing the gate.

Two spacings matter and are chosen, not stumbled into: every pair of cameras
is at least 5.5 degrees apart, which is 29 mm at this radius and therefore
clear of the 20 mm half of the novelty rule, so nothing is deduplicated behind
a test's back; and where a test needs a candidate under the 8 degree source
floor it is placed 6.5 degrees from the reference and no nearer than 5.5
degrees to anything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbiter_native.scan import ScanVolume
from orbiter_native.views import (
    AUTO_SOURCES,
    CfgParams,
    SelectParams,
    depth_range_fallback,
    plan_patch_match_cfg,
    seed_points,
    tracks,
)

from test_views import BOX_N, BOX_XYZ, _at, _select

#: The reference photograph's azimuth. Off a multiple of 15 degrees for the
#: same reason `test_views` avoids those — nothing here should sit on a
#: boundary of anything.
REF_AZ = 3.0
#: How far the level orbit stands from the subject. See the module docstring:
#: at 250 mm a level camera falls out of the depth gate.
RING_MM = 300.0


# ── the fixture: a level orbit, where angle is azimuth ───────────────────


def _angle(a, b) -> float:
    """The angle between two photographs' viewing directions, degrees."""
    cos = float(np.clip(np.dot(a.direction, b.direction), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _ring(offsets: list[float]):
    """The reference photograph, then one per offset in degrees around the
    level orbit — and the proof that each offset *is* that photograph's angle
    from the reference, which is what every bucket assertion below rests on."""
    photos = [_at(REF_AZ + float(offset), radius=RING_MM, elev_deg=0.0)
              for offset in (0.0, *offsets)]
    for offset, photo in zip(offsets, photos[1:]):
        assert _angle(photos[0], photo) == pytest.approx(abs(offset), abs=1e-6)
    return photos


def _selected(offsets: list[float], params: SelectParams | None = None):
    """`(photos, selection)` for a ring, with the fixture's own health
    checked here so a later failure is the rule under test and not a
    photograph that quietly lost its place."""
    photos = _ring(offsets)
    _, sel = _select(photos, params)
    assert len(sel.accepted) == len(photos)
    return photos, sel


def _cfg(names: list[str]) -> str:
    """A cfg as `image_undistorter` writes one: a name, its `__auto__` line,
    and on to the next."""
    return "".join(f"{name}\n{AUTO_SOURCES}\n" for name in names)


def _entries(cfg_text: str) -> dict[str, str]:
    """`{image name: source line}`, pairing the way COLMAP's own reader does
    — blank and comment lines skipped."""
    rows = [line.strip() for line in cfg_text.splitlines()
            if line.strip() and not line.strip().startswith("#")]
    assert len(rows) % 2 == 0, "a cfg with a dangling name is not a cfg"
    return dict(zip(rows[0::2], rows[1::2]))


def _sources(cfg_text: str, name: str) -> list[str]:
    return [part.strip() for part in _entries(cfg_text)[name].split(",")]


def _between(angles: list[float], lo: float, hi: float) -> list[float]:
    """The angles that fall in one bucket — half-open `[lo, hi)`, the way the
    buckets themselves are, so a candidate at exactly 15 degrees is counted
    once and in the same place both here and in `CfgParams`."""
    return [deg for deg in angles if lo <= deg < hi]


# ── what is kept, and what is dropped ────────────────────────────────────


def test_cfg_preserves_unknown_images_and_only_rewrites_sources() -> None:
    """The undistorter's image lines come through verbatim and in order, and
    an image we hold no pose for keeps its own `__auto__, 20` byte for byte.

    The blank line and the comment are there because COLMAP's reader skips
    both before it pairs the rest: if ours did not, they would shift every
    pair after them and the file would name sources under the wrong image.
    """
    photos, sel = _selected([12.0, 21.0, 34.0, -18.0, -33.0])
    stranger = "left_9999.jpg"          # registered, but not ours
    written = _cfg(sel.names[:3]) + "\n# hand-edited\n" \
        + _cfg([stranger, *sel.names[3:]])

    report = plan_patch_match_cfg(written, sel)
    out = report.text.splitlines()
    was = written.splitlines()

    assert len(out) == len(was)
    assert list(_entries(report.text)) == list(_entries(written))
    # Every line that moved was an `__auto__` line, and the line above each
    # of them is one of ours: nothing else in the file was touched.
    changed = [i for i, (now, before) in enumerate(zip(out, was))
               if now != before]
    assert {was[i] for i in changed} == {AUTO_SOURCES}
    assert [was[i - 1] for i in changed] == sel.names
    assert _entries(report.text)[stranger] == AUTO_SOURCES
    assert "# hand-edited" in out
    assert report.kept_auto == (stranger,)
    assert set(report.rewritten) == set(sel.names)
    for name in sel.names:
        assert _entries(report.text)[name] != AUTO_SOURCES


def test_patch_match_cfg_drops_a_selection_entry_missing_from_the_undistorter_cfg() -> None:
    """A photograph we selected that COLMAP did not register is dropped, not
    added — naming it would abort `patch_match_stereo` before it computes a
    single depth map. The report says which, so the run can complain."""
    photos, sel = _selected([11.0, 19.0, 28.0, -17.0, -37.0])
    registered = sel.names[:-1]
    lost = sel.names[-1]

    report = plan_patch_match_cfg(_cfg(registered), sel)

    assert report.missing == (lost,)
    assert lost not in report.text
    assert list(_entries(report.text)) == registered


def test_sources_never_name_an_image_outside_the_cfg() -> None:
    """Every source of every image is one of the images the cfg registered.
    This is the assertion the whole rewrite exists to satisfy, so it is made
    over the file rather than over one line of it."""
    photos, sel = _selected([9.0, 14.5, -9.5, 20.0, 26.0, -16.0, -23.0,
                             32.0, 38.0, -31.0])
    registered = sel.names[:6]          # the undistorter registered six

    text = plan_patch_match_cfg(_cfg(registered), sel).text

    entries = _entries(text)
    assert list(entries) == registered
    for name in registered:
        assert set(_sources(text, name)) <= set(registered) - {name}


# ── the buckets ──────────────────────────────────────────────────────────


def test_sources_span_a_range_of_angles_not_just_the_nearest() -> None:
    """Eight sources, filling all three buckets — 2 from 8-15 degrees, 3 from
    15-30, 3 from 30-45 — so at least three of them are past 20 degrees.

    Nearest-first would have taken eight neighbours inside 31 degrees, which
    on this rig is the minimum baseline the novelty gate allows and a depth
    sigma worse than the laser cloud dense is meant to improve on. The last
    assertion is that the answer is *not* that set.
    """
    offsets = [9.0, 14.5, -9.5, 20.0, 26.0, -16.0, -23.0,
               32.0, 38.0, 44.0, -31.0, -40.0]
    photos, sel = _selected(offsets)
    angle_of = {p.name: _angle(photos[0], p) for p in photos[1:]}

    picked = _sources(plan_patch_match_cfg(_cfg(sel.names), sel).text,
                      photos[0].name)

    assert len(picked) == 8 == CfgParams().n_sources
    assert len(picked) == len(set(picked))
    angles = sorted(angle_of[name] for name in picked)
    assert sum(1 for deg in angles if deg > 20.0) >= 3
    assert _between(angles, 8.0, 15.0) == pytest.approx([9.0, 9.5])
    assert _between(angles, 15.0, 30.0) == pytest.approx([16.0, 20.0, 23.0])
    assert _between(angles, 30.0, 45.0) == pytest.approx([31.0, 32.0, 38.0])

    nearest = sorted(angle_of, key=lambda name: angle_of[name])[:8]
    assert set(picked) != set(nearest)


def test_a_short_bucket_is_filled_from_the_nearest_remaining() -> None:
    """Nothing stands between 30 and 45 degrees of the reference, so that
    bucket's three places go to the nearest candidates still unused — 26, 52
    and 60 degrees — and not to the 70 degree one behind them.

    The 6.5 degree photograph is the other half of the rule: the fill is
    hungry, but the 8 degree floor is a floor, and a source at 6.5 degrees is
    the degenerate pair the buckets exist to refuse.
    """
    offsets = [6.5, -9.5, 14.5, -16.0, 20.0, -23.0, 26.0, 52.0, 60.0, 70.0]
    photos, sel = _selected(offsets)
    by_offset = {offset: photo.name
                 for offset, photo in zip(offsets, photos[1:])}

    picked = _sources(plan_patch_match_cfg(_cfg(sel.names), sel).text,
                      photos[0].name)

    assert len(picked) == 8
    angles = sorted(_angle(photos[0], p) for p in photos[1:]
                    if p.name in picked)
    assert not _between(angles, 30.0, 45.0)     # the bucket really is empty
    assert by_offset[6.5] not in picked                        # the floor holds
    assert by_offset[70.0] not in picked                       # not the nearest
    for offset in (26.0, 52.0, 60.0):
        assert by_offset[offset] in picked


# ── the depth range, which is a fallback and says so ─────────────────────


def _rim_points(volume: ScanVolume, n: int = 2048) -> np.ndarray:
    """The two rim circles of the scan volume — where a linear function of
    position takes its extremes over a cylinder."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = volume.radius_mm * np.stack([np.cos(t), np.sin(t)], axis=1)
    return np.concatenate([
        np.column_stack([ring, np.full(n, volume.floor_mm)]),
        np.column_stack([ring, np.full(n, volume.height_mm)])])


def test_depth_range_fallback_is_none_when_tracks_are_healthy() -> None:
    """Real tracks over the box fixture give every image thousands of
    observations, so nothing is passed and COLMAP keeps its own per-image
    ranges — the better answer, and the default.

    The other two thirds of this are the boundary: one thin image in ten is
    exactly 10 %, which is not *more* than 10 %, and two is.
    """
    photos, sel = _selected([12.0, 24.0, 37.0, -14.0, -29.0, -41.0,
                             80.0, 140.0, -100.0])
    volume = sel.session.volume
    seeds, rgb, normals = seed_points(BOX_XYZ, None, BOX_N)
    _, points2d = tracks(sel, seeds, normals, rgb)
    counts = {i: len(seen) for i, seen in points2d.items()}

    assert len(sel.accepted) == 10
    assert min(counts.values()) > 50
    assert depth_range_fallback(sel, counts, volume) is None

    ids = [v.image_id for v in sel.accepted]
    one_thin = {i: (49 if i == ids[0] else 500) for i in ids}
    assert depth_range_fallback(sel, one_thin, volume) is None
    two_thin = {i: (49 if i in ids[:2] else 500) for i in ids}
    assert depth_range_fallback(sel, two_thin, volume) is not None


def test_depth_range_fallback_covers_the_volume_from_every_selected_camera_padded() -> None:
    """The range reaches the whole cylinder from every camera that was
    selected, and then 20 % of its own span further at each end.

    The cylinder's extremes along an axis sit on its two rims, so a fine
    sample of those two circles is the exact bound to compare against — the
    padding is checked as an identity, not as "roughly wider".
    """
    photos, sel = _selected([17.0, 33.0, -21.0, -39.0, 90.0, -95.0])
    volume = sel.session.volume
    counts = {v.image_id: 3 for v in sel.accepted}       # every image is thin

    got = depth_range_fallback(sel, counts, volume)

    assert got is not None
    rim = _rim_points(volume)
    near, far = float("inf"), float("-inf")
    for v in sel.accepted:
        depths = (rim - v.photo.centre_mm) @ v.photo.direction
        assert got.depth_min_mm < depths.min()
        assert got.depth_max_mm > depths.max()
        near, far = min(near, depths.min()), max(far, depths.max())

    span = far - near
    assert got.depth_min_mm == pytest.approx(near - 0.2 * span, abs=0.01)
    assert got.depth_max_mm == pytest.approx(far + 0.2 * span, abs=0.01)
    assert got.depth_min_mm > 0.0


def test_depth_range_warning_names_the_seed_cap() -> None:
    """Thin tracks usually mean the seed cap is too low for this object, not
    that the range wants forcing — so the warning names the cap first, reads
    it off the selection that was actually made, and admits that the range it
    offers is unverified."""
    photos, sel = _selected([16.0, 31.0, -22.0, -38.0])
    volume = sel.session.volume
    counts = {v.image_id: 0 for v in sel.accepted}

    got = depth_range_fallback(sel, counts, volume)

    assert got is not None
    assert "30,000-point seed cap is the first thing to try" in got.warning
    assert "100% of the selected images (5 of 5)" in got.warning
    assert "fewer than 50 seed points" in got.warning
    assert f"{got.depth_min_mm:.0f}-{got.depth_max_mm:.0f} mm" in got.warning
    assert "unverified" in got.warning

    _, thrifty = _select(photos, SelectParams(seed_cap=12_345))
    other = depth_range_fallback(
        thrifty, {v.image_id: 0 for v in thrifty.accepted}, volume)
    assert other is not None
    assert "12,345-point seed cap" in other.warning
