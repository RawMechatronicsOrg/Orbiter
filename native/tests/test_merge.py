"""The merge: the laser keeps what it covered, dense fills only the rest.

Three earlier designs failed on one case — a dense point 4 mm above a piece
of surface the laser fully covered — because a Euclidean nearest-neighbour
distance cannot tell *the laser covered this and the point floats above it*
from *the laser never saw this region*. The rule under test answers that
with lateral support inside a ball wide enough for the lateral test to do
work, and the first test here is that case, with the negative that made
revision 4 fail asserted alongside.

The gates are the other half. G1 gates the SIGNED residual median per
z-slab × normal-octant cell, so zero-mean noise passes and a bias does not;
the octants are what stop a closed object from cancelling its own
misregistration. G1b refuses when there is no supported population to
measure a bias over at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from orbiter_native.scan import MergeParams, ScanVolume, merge_clouds

#: The volume the crop used, passed explicitly to every call: G1's four z
#: slabs are equal quarters of `[floor_mm, height_mm]`, so a fixture that
#: leaned on a default would silently change which slab its points land in.
VOLUME = ScanVolume(height_mm=400.0, radius_mm=150.0, floor_mm=5.0)
PARAMS = MergeParams()
#: Every laser fixture is sampled 1.0 mm apart, because that is what the
#: confident cloud is — it is merged onto 1.0 mm voxels (`clean_merge_mm`).
#: At any other spacing "5 neighbours within 3.0 mm laterally" would mean
#: something else, so the sampling term is in the tests rather than assumed.
STEP_MM = 1.0


def _plane(half: float = 30.0, z: float = 50.0,
           shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A laser plane with +z normals. `shift` moves the grid off the
    integers, which is how a slot of an odd width gets its two edges."""
    g = np.arange(-half, half + STEP_MM / 2, STEP_MM) + shift
    x, y = np.meshgrid(g, g)
    pts = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])
    return pts, np.repeat([[0.0, 0.0, 1.0]], len(pts), 0)


def _cylinder(radius: float, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """A cylinder shell standing on the board, OUTWARD radial normals, both
    the circumference and the axis sampled `STEP_MM` apart.

    The closed object a z-band-only gate cannot judge: outward normals mean
    a translation reads +δ on one side and −δ on the other, and a band that
    holds both halves has a median of zero.
    """
    n_ring = max(int(round(2.0 * np.pi * radius / STEP_MM)), 8)
    phi = np.arange(n_ring) * (2.0 * np.pi / n_ring)
    u = np.column_stack([np.cos(phi), np.sin(phi), np.zeros(n_ring)])
    zs = np.arange(z0, z1 + STEP_MM / 2, STEP_MM)
    pts = np.vstack([np.column_stack([u[:, 0] * radius, u[:, 1] * radius,
                                      np.full(n_ring, z)]) for z in zs])
    return pts, np.tile(u, (len(zs), 1))


def test_merge_drops_a_dense_point_above_covered_laser_surface() -> None:
    laser, ln = _plane()
    hover = np.array([[0.0, 0.0, 54.0]])                # 4 mm above covered surface
    # The negative three revisions tripped over: a ball of `support_mm`
    # alone comes back EMPTY here, so a Euclidean rule calls this point
    # uncovered and keeps it as hole fill.
    assert cKDTree(laser).query_ball_point(hover[0], PARAMS.support_mm) == []
    dense = np.vstack([laser, hover])
    dn = np.repeat([[0.0, 0.0, 1.0]], len(dense), 0)
    xyz, nrm, st = merge_clouds(laser, ln, dense, dn, PARAMS, VOLUME)
    assert not st.refused
    assert st.dense_supported == len(dense) and st.dense_kept == 0 and st.floaters == 0
    assert len(xyz) == len(laser) == len(nrm)           # nothing of dense survived
    # And the residual the rule reads for such a point: lift the whole cloud
    # by the same 4 mm and its cell median is +4.0 — signed, positive away
    # from the laser surface along that surface's normal.
    _, _, up = merge_clouds(laser, ln, laser + [0.0, 0.0, 4.0], ln, PARAMS, VOLUME)
    assert up.cells.medians_mm["z1:+++"] == pytest.approx(4.0, abs=1e-9)


def test_merge_drops_a_hovering_point_beyond_max_normal_as_a_floater() -> None:
    laser, ln = _plane()
    hover = np.array([[0.0, 0.0, 60.2]])                # 10.2 mm up: past 10.0
    dense = np.vstack([laser, hover])
    dn = np.repeat([[0.0, 0.0, 1.0]], len(dense), 0)
    _, _, st = merge_clouds(laser, ln, dense, dn, PARAMS, VOLUME)
    assert st.floaters == 1 and st.dense_kept == 0
    # Supported, dropped — but as a floater, so it never enters G1's
    # statistics and cannot colour the bias.
    assert st.dense_supported == len(laser)


def test_merge_keeps_a_dense_patch_in_a_laser_hole() -> None:
    laser, ln = _plane()
    hole = np.hypot(laser[:, 0], laser[:, 1]) < 7.0     # 14 mm across
    dense, dn = _plane()
    xyz, _, st = merge_clouds(laser[~hole], ln[~hole], dense, dn, PARAMS, VOLUME)
    assert not st.refused and st.dense_kept > 0
    kept = xyz[int((~hole).sum()):]
    # The hole's CENTRE: 7 mm from either edge, so it is past `outlier_mm`
    # and the rim rule cannot reach it — the density rule is what keeps it.
    assert cKDTree(kept).query([0.0, 0.0, 50.0], k=1)[0] < 1e-9
    # Its rim, within `support_mm` of the edge, is laterally supported by
    # that edge and is dropped: the laser owns what it covered.
    assert cKDTree(kept).query([6.5, 0.0, 50.0], k=1)[0] > 1e-9


def test_merge_does_not_fill_a_narrow_slot() -> None:
    # A half-integer grid so the surviving edges stand exactly 3 mm apart.
    laser, ln = _plane(shift=0.5)
    slot = np.abs(laser[:, 0]) < 1.5
    dense, dn = _plane(shift=0.5)
    xyz, _, st = merge_clouds(laser[~slot], ln[~slot], dense, dn, PARAMS, VOLUME)
    # A gap narrower than 2 × `support_mm` is supported from both edges at
    # once, so the laser keeps it. That is deliberate, and it is the price
    # §2.7 states for the lateral rule.
    assert not st.refused and st.dense_kept == 0
    assert len(xyz) == int((~slot).sum())


def test_merge_refuses_a_translation_offset() -> None:
    laser, ln = _plane()
    xyz, _, st = merge_clouds(laser, ln, laser + [0.0, 0.0, 2.5], ln, PARAMS, VOLUME)
    assert st.refused and st.refused_by == "G1"
    assert st.dense_supported == len(laser)             # every point is supported
    assert st.cells.medians_mm["z1:+++"] == pytest.approx(2.5, abs=1e-9)
    # A refusal hands back the laser cloud untouched — what --mode
    # texture-only would have produced anyway.
    assert len(xyz) == len(laser)


def test_merge_does_not_refuse_pure_noise_of_2_mm_sigma() -> None:
    laser, ln = _plane()
    rng = np.random.default_rng(7)
    dense = laser + np.column_stack([np.zeros(len(laser)), np.zeros(len(laser)),
                                     rng.normal(0.0, 2.0, len(laser))])
    _, _, st = merge_clouds(laser, ln, dense, ln, PARAMS, VOLUME)
    # This is the test that proves G1 measures bias, not noise.
    assert not st.refused and st.refused_by is None
    assert abs(st.cells.medians_mm["z1:+++"]) < 0.2
    # 2 σ × 1.645, the untruncated distribution now that nothing clips the
    # tail below `max_normal_mm`: reported, and never gated on.
    assert st.p90_abs_residual_mm == pytest.approx(3.29, abs=0.3)


def test_merge_refuses_a_translation_on_a_closed_object() -> None:
    volume = ScanVolume(height_mm=100.0, radius_mm=60.0, floor_mm=5.0)
    laser, ln = _cylinder(28.0, 6.0, 99.0)
    _, _, st = merge_clouds(laser, ln, laser + [2.5, 0.0, 0.0], ln, PARAMS, volume)
    assert st.dense_supported >= 16_000                 # every cell clears band_min
    assert st.refused and st.refused_by == "G1"
    assert not st.cells.skipped
    # 2.5 × cos 45°: the octant median is damped on a curved surface, which
    # is why the fixture offsets by 2.5 mm and not by 1.5.
    plus = [v for k, v in st.cells.medians_mm.items() if k.split(":")[1][0] == "+"]
    minus = [v for k, v in st.cells.medians_mm.items() if k.split(":")[1][0] == "-"]
    assert len(plus) == len(minus) == 8
    assert all(v == pytest.approx(1.77, abs=0.15) for v in plus)
    assert all(v == pytest.approx(-1.77, abs=0.15) for v in minus)
    # And the fact the octants exist for: the z-band medians alone are ≈ 0,
    # so a band-only gate would have waved this through.
    span = (volume.height_mm - volume.floor_mm) / 4.0
    band = np.clip(((laser[:, 2] - volume.floor_mm) / span).astype(np.int64), 0, 3)
    for b in range(4):
        assert abs(np.median(2.5 * ln[band == b, 0])) < 0.05


def test_merge_refuses_a_rotational_misregistration() -> None:
    laser, ln = _cylinder(28.0, 6.0, 399.0)
    a = np.deg2rad(0.5)                                 # about a base axis
    rot = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(a), -np.sin(a)],
                    [0.0, np.sin(a), np.cos(a)]])
    _, _, st = merge_clouds(laser, ln, laser @ rot.T, ln @ rot.T, PARAMS, VOLUME)
    assert st.refused and st.refused_by == "G1"
    base = [abs(v) for k, v in st.cells.medians_mm.items() if k.startswith("z1:")]
    top = [abs(v) for k, v in st.cells.medians_mm.items() if k.startswith("z4:")]
    # A bias growing across the z slabs is a rotation. The whole-cloud
    # median would have passed; the top slab does not.
    assert max(base) < PARAMS.agree_mm
    assert min(top) > PARAMS.agree_mm


def test_merge_refuses_when_nothing_overlaps() -> None:
    laser, ln = _plane()
    # Further than the 10.44 mm candidate radius: every ball comes back
    # empty, so there is nothing to measure a bias over.
    _, _, st = merge_clouds(laser, ln, laser + [0.0, 0.0, 12.0], ln, PARAMS, VOLUME)
    assert st.refused and st.refused_by == "G1b"
    assert st.dense_supported == 0
    assert "no overlap between the laser cloud and the dense cloud" in st.message


def test_a_six_mm_uniform_offset_is_refused_by_g1_not_g1b() -> None:
    laser, ln = _plane()
    # The boundary the two messages divide: at 6 mm the balls are still
    # populated, so the diagnosis is G1's and it names the cells.
    _, _, st = merge_clouds(laser, ln, laser + [0.0, 0.0, 6.0], ln, PARAMS, VOLUME)
    assert st.refused and st.refused_by == "G1"
    assert st.dense_supported >= PARAMS.min_supported
    assert st.cells.medians_mm["z1:+++"] == pytest.approx(6.0, abs=1e-9)


def test_merge_warns_on_low_overlap_instead_of_refusing() -> None:
    laser, ln = _plane(half=27.5)                       # 3 136 laser points
    on_it, on_n = _plane(half=27.5)
    away, away_n = _plane(half=95.0, z=200.0)           # never seen by the laser
    dense, dn = np.vstack([on_it, away]), np.vstack([on_n, away_n])
    _, _, st = merge_clouds(laser, ln, dense, dn, PARAMS, VOLUME)
    # A dark or specular object gives the laser little coverage by nature,
    # and refusing here would refuse the case dense exists to rescue.
    assert not st.refused and st.refused_by is None
    assert st.dense_supported >= PARAMS.min_supported
    assert st.agree_frac == pytest.approx(0.08, abs=0.01)
    assert any("overlap" in w for w in st.warnings)


def test_merge_reports_the_kept_sets_own_distance_and_all_fractions() -> None:
    laser, ln = _plane()
    hole = np.hypot(laser[:, 0], laser[:, 1]) < 7.0
    dense, dn = _plane()
    _, _, st = merge_clouds(laser[~hole], ln[~hole], dense, dn, PARAMS, VOLUME)
    assert st.dense_points == len(dense)                # the post-crop count
    assert st.dense_supported + st.dense_kept + st.floaters == st.dense_points
    assert st.laser_points == int((~hole).sum())
    assert st.candidate_radius_mm == pytest.approx(np.hypot(3.0, 10.0))
    assert st.agree_frac == pytest.approx(st.dense_supported / st.dense_points)
    assert st.keep_frac == pytest.approx(st.dense_kept / st.dense_points)
    assert np.isfinite(st.kept_median_mm) and st.kept_median_mm > 0.0
    assert st.median_dense_neighbours > 0.0
    assert np.isfinite(st.p90_abs_residual_mm)
    assert st.cells.grid == (4, 8)
    assert st.cells.gated == len(st.cells.medians_mm) == len(st.cells.counts) == 1
    assert st.cells.worst in st.cells.medians_mm and st.cells.skipped == []
    assert st.params is PARAMS


def test_cells_below_band_min_are_skipped_not_gated() -> None:
    low, low_n = _plane()                               # 3 721 points, z 50
    high, high_n = _plane(half=8.5, z=380.0)            # 324 points, the top slab
    laser, ln = np.vstack([low, high]), np.vstack([low_n, high_n])
    dense = np.vstack([low, high + [0.0, 0.0, 5.0]])    # the top slab is 5 mm out
    _, _, st = merge_clouds(laser, ln, dense, ln, PARAMS, VOLUME)
    assert st.cells.counts["z1:+++"] == len(low)
    assert st.cells.skipped == ["z4:+++"]
    assert "z4:+++" not in st.cells.medians_mm
    # 5 mm out and it still cannot refuse the run: too few points to say so.
    assert not st.refused


def test_merge_refusal_names_mode_texture_only_not_to_poisson_mesher() -> None:
    laser, ln = _plane()
    _, _, g1 = merge_clouds(laser, ln, laser + [0.0, 0.0, 2.5], ln, PARAMS, VOLUME)
    _, _, g1b = merge_clouds(laser, ln, laser + [0.0, 0.0, 12.0], ln, PARAMS, VOLUME)
    for st in (g1, g1b):
        # In dense mode `merge` precedes `poisson_mesher`, so --to
        # poisson_mesher runs the merge again and refuses again. The wrong
        # escape is worse than no escape.
        assert st.refused and "--mode texture-only" in st.message
        assert "poisson_mesher" not in st.message
