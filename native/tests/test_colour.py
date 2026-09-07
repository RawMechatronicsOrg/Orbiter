"""Colour: read beside the stripe, averaged with the points, written out."""

from __future__ import annotations

import numpy as np

from orbiter_native.scan import CloudOverlay, PointCloud, ScanFrame, read_ply, sample_beside
from orbiter_native.scanworker import average_still

SURFACE_BGR = (200, 120, 40)          # a blue-ish surface: RGB (40, 120, 200)
SURFACE_RGB = [40, 120, 200]


def _frame_with_stripe(w=64, h=48, along_x=True, stripe_at=20, halo=3):
    """The surface with a red stripe across it, `halo` px to each side."""
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[:] = SURFACE_BGR
    if along_x:                         # scanlines are columns: the stripe runs along x
        bgr[stripe_at - halo: stripe_at + halo + 1, :] = (0, 0, 255)
    else:
        bgr[:, stripe_at - halo: stripe_at + halo + 1] = (0, 0, 255)
    return bgr


def test_colour_is_read_beside_the_stripe_not_under_it() -> None:
    bgr = _frame_with_stripe()
    pix = np.array([[10.0, 20.0], [30.5, 20.2]])
    rgb = sample_beside(bgr, pix, along_x=True, offset_px=6.0, wh=(64, 48))
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 3)
    assert rgb.tolist() == [SURFACE_RGB, SURFACE_RGB]
    # An offset inside the halo reads the laser, which is the failure the
    # default offset is chosen to avoid.
    assert sample_beside(bgr, pix, True, 1.0, (64, 48)).tolist() == [[255, 0, 0]] * 2


def test_colour_scales_to_a_half_size_image_and_the_other_orientation() -> None:
    # The image is half the frame (the GPU path's display copy); the stripe
    # runs along y, so "beside" is across x.
    bgr = _frame_with_stripe(w=32, h=24, along_x=False, stripe_at=10, halo=1)
    pix = np.array([[20.0, 30.0]])                        # full-frame coordinates
    assert sample_beside(bgr, pix, False, 6.0, (64, 48)).tolist() == [SURFACE_RGB]
    assert sample_beside(bgr, np.empty((0, 2)), True, 6.0, (64, 48)).shape == (0, 3)


def test_a_glint_on_one_side_does_not_tint_the_point() -> None:
    bgr = _frame_with_stripe()
    bgr[14, 9:12] = (255, 255, 255)                       # white glint, one side only
    rgb = sample_beside(bgr, np.array([[10.0, 20.0]]), True, 6.0, (64, 48))
    assert rgb.tolist() == [SURFACE_RGB]                  # the median holds


def test_point_cloud_keeps_a_mean_colour_per_voxel_and_writes_it(tmp_path) -> None:
    pc = PointCloud(voxel_mm=1.0)
    pc.add(np.array([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [5.0, 5.0, 5.0]]),
           np.array([[100, 0, 0], [200, 0, 0], [0, 0, 250]], np.uint8))
    assert len(pc) == 2
    assert pc.colors().tolist() == [[150, 0, 0], [0, 0, 250]]
    # A sweep without colour joins the positions and leaves the colour alone.
    pc.add(np.array([[0.1, 0.1, 0.1]]))
    assert pc.colors().tolist() == [[150, 0, 0], [0, 0, 250]]
    pts, rgb = pc.snapshot(10)
    assert pts.shape == (2, 3) and rgb.tolist() == [[150, 0, 0], [0, 0, 250]]

    path = tmp_path / "c.ply"
    assert pc.write_ply(str(path)) == 2
    got, got_rgb = read_ply(str(path))
    assert np.allclose(got[1], [5, 5, 5]) and got_rgb.tolist() == [[150, 0, 0], [0, 0, 250]]

    pc.clear()
    assert pc.colors() is None
    pc.add(np.zeros((1, 3)))
    assert pc.colors() is None and pc.snapshot(5)[1] is None


def test_point_cloud_grows_its_colour_arrays_with_the_rest() -> None:
    pc = PointCloud(voxel_mm=1.0)
    n = 5000                                              # past the first reserve
    pts = np.column_stack([np.arange(n, dtype=float) * 2.0, np.zeros(n), np.zeros(n)])
    pc.add(pts, np.full((n, 3), 7, np.uint8))
    assert len(pc) == n and (pc.colors() == 7).all()


def _sf(scan, pts, rgb=None) -> ScanFrame:
    return ScanFrame(points_board=np.asarray(pts, float), scanlines=np.asarray(scan, np.int64),
                     colours=None if rgb is None else np.asarray(rgb, np.uint8))


def test_still_average_carries_colour_with_the_points() -> None:
    frames = [_sf([1, 2], [[0, 0, 0], [10, 10, 10]], [[10, 20, 30], [200, 200, 200]]),
              _sf([1, 2], [[2, 0, 0], [10, 12, 10]], [[30, 20, 10], [200, 200, 200]])]
    pts, rgb, w = average_still(frames)
    assert pts.tolist() == [[1, 0, 0], [10, 11, 10]]
    assert w.tolist() == [1.0, 1.0]                         # frames without weights
    assert rgb.tolist() == [[20, 20, 20], [200, 200, 200]]
    # One frame without colour: the batch has no colour to give.
    frames[1].colours = None
    assert average_still(frames)[1] is None
    assert average_still(frames[:1])[1].tolist() == [[10, 20, 30], [200, 200, 200]]
    assert average_still([_sf([], np.empty((0, 3)), np.empty((0, 3)))] * 2)[1].shape == (0, 3)


def test_overlay_hands_out_colours_only_when_they_match_the_points() -> None:
    ov = CloudOverlay()
    assert ov.colors() is None
    ov.publish(np.zeros((3, 3)), np.ones((3, 3), np.uint8))
    assert ov.colors().shape == (3, 3)
    ov.publish(np.zeros((2, 3)))
    assert ov.colors() is None
