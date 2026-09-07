"""The confident cloud: merged a step coarser, the lonely and the unconfirmed
dropped, what is young kept."""

from __future__ import annotations

import numpy as np

from orbiter_native.scan import PointCloud, confident, read_ply, write_ply


def _surface(n: int = 30, seen: int = 1):
    """A line of voxels 0.5 mm apart along x, each seen `seen` times."""
    pts = np.column_stack([np.arange(n) * 0.5, np.zeros(n), np.zeros(n)])
    return pts, np.full(n, seen, np.int64)


def test_a_lonely_voxel_goes_and_a_surface_stays() -> None:
    pts, cnt = _surface()
    pts = np.vstack([pts, [[100.0, 100.0, 100.0]]])         # one glint far away
    cnt = np.append(cnt, 5)                                 # even a bright one
    out = confident(pts, cnt, merge_mm=0.5)
    assert out.n_lonely == 1 and out.n_flicker == 0
    assert len(out.points) == 30 and out.points[:, 0].max() < 15.0
    assert (out.obs == 1).all()


def test_a_flicker_among_confirmed_neighbours_goes_but_a_young_voxel_stays() -> None:
    pts, cnt = _surface(seen=4)                             # swept four times
    pts = np.vstack([pts, [[7.25, 0.7, 0.0]]])              # seen once, beside them
    cnt = np.append(cnt, 1)
    out = confident(pts, cnt, merge_mm=0.5)
    assert out.n_flicker == 1 and out.n_lonely == 0 and len(out.points) == 30
    # The same voxel in a region swept only once is simply young: kept.
    pts1, cnt1 = _surface(seen=1)
    pts1 = np.vstack([pts1, [[7.25, 0.7, 0.0]]])
    cnt1 = np.append(cnt1, 1)
    out1 = confident(pts1, cnt1, merge_mm=0.5)
    assert out1.n_flicker == 0 and len(out1.points) == 31


def test_survivors_merge_weighted_by_how_often_they_were_seen() -> None:
    # Two voxels of one 1 mm cell, seen 1 and 3 times, with colours.
    base, cnt = _surface(seen=2)
    pts = np.vstack([base, [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    cnt = np.concatenate([cnt, [1, 3]])
    rgb = np.zeros((len(pts), 3), np.uint8)
    rgb[-2], rgb[-1] = (0, 0, 0), (200, 200, 200)
    out = confident(pts, cnt, rgb, merge_mm=1.0)
    # Cell [0, 1): the surface voxels at 0.0 and 0.5 (2 obs each) plus the two.
    first = out.points[0]
    x = (0.0 * 2 + 0.5 * 2 + 0.2 * 1 + 0.4 * 3) / 8.0
    assert abs(first[0] - x) < 1e-9 and out.obs[0] == 8
    assert out.colours[0].tolist() == [75, 75, 75]          # 600 / 8
    assert out.colours.dtype == np.uint8 and len(out.colours) == len(out.points)


def test_empty_and_all_dropped_come_back_empty() -> None:
    out = confident(np.empty((0, 3)), np.empty(0))
    assert len(out.points) == 0 and out.colours is None
    out = confident(np.array([[0.0, 0.0, 0.0]]), np.array([1]), np.zeros((1, 3), np.uint8))
    assert len(out.points) == 0 and out.n_lonely == 1 and out.colours.shape == (0, 3)


def test_a_close_pass_overrides_a_far_one_in_the_grid() -> None:
    from orbiter_native.scan import ScanFrame, precision_weights
    from orbiter_native.scanworker import average_still
    # Weights fall as the fourth power of depth: 1 at 300 mm, 16 at 150 mm,
    # (2/3)^4 at 450 mm — 150 mm against 450 mm is 81:1.
    w = precision_weights(np.array([[0, 0, 150.0], [0, 0, 300.0], [0, 0, 450.0]]))
    assert np.allclose(w / w[1], [16.0, 1.0, (2.0 / 3.0) ** 4])
    assert np.isclose(w[0] / w[2], 81.0)
    # The same voxel seen from far (off by 1 mm) and then from close: the
    # weighted mean sits with the close pass, the count still says two.
    pc = PointCloud(voxel_mm=2.0)
    pc.add(np.array([[0.9, 0.0, 0.0]]), weights=w[2:3])
    pc.add(np.array([[0.1, 0.0, 0.0]]), weights=w[0:1])
    assert len(pc) == 1 and pc.counts().tolist() == [2]
    assert abs(pc.points()[0, 0] - 0.1) < 0.02                # 81:1, not 1:1
    assert np.isclose(pc.weights()[0], w[0] + w[2])
    # The confident cloud merges by the same weights, and reports the count.
    out = confident(np.array([[0.1, 0, 0], [0.9, 0, 0], [5.0, 0, 0], [5.4, 0, 0]]),
                    np.array([1, 1, 1, 1]), merge_mm=2.0, min_neighbours=0,
                    weights=np.array([81.0, 1.0, 1.0, 1.0]))
    assert abs(out.points[0, 0] - (0.1 * 81 + 0.9) / 82) < 1e-9 and out.obs[0] == 2
    # A still batch carries the weights along with the points.
    f = ScanFrame(points_board=np.array([[0, 0, 0.0], [1, 1, 1.0]]),
                  scanlines=np.array([1, 2]), weights=np.array([4.0, 0.25]))
    g = ScanFrame(points_board=np.array([[0, 0, 2.0], [1, 1, 3.0]]),
                  scanlines=np.array([1, 2]), weights=np.array([4.0, 0.25]))
    pts, rgb, wts = average_still([f, g])
    assert pts.tolist() == [[0, 0, 1.0], [1, 1, 2.0]] and wts.tolist() == [4.0, 0.25]
    assert rgb is None


def test_the_grid_counts_and_versions_its_voxels(tmp_path) -> None:
    pc = PointCloud(voxel_mm=1.0)
    v0 = pc.version
    pc.add(np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    assert pc.counts().tolist() == [2, 1] and pc.version == v0 + 1
    pc.clear()
    assert pc.version == v0 + 2 and len(pc.counts()) == 0
    path = tmp_path / "w.ply"
    assert write_ply(str(path), np.array([[1.0, 2.0, 3.0]]), np.array([[9, 8, 7]], np.uint8)) == 1
    got, rgb = read_ply(str(path))
    assert np.allclose(got, [[1, 2, 3]]) and rgb.tolist() == [[9, 8, 7]]
