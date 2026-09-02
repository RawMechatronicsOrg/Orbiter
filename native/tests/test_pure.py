"""Tests for the parts that need neither a GUI nor the rig.

Deliberately narrow: orientation, the laser centroid, the MJPEG demultiplexer
and config parsing. Those are where a silent wrong answer is possible — a
mirrored preview, a centroid off by half a pixel, a frame boundary missed. The
rest of the app is threads and widgets, and is checked by running it.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbiter_native import config as cfgmod
from orbiter_native.laser import (
    find_laser_points,
    LaserParams,
    board_mask,
    find_laser_line,
    redness,
)
from orbiter_native.orient import Orientation, apply, map_point, map_points
from orbiter_native.source import MjpegReader

ALL_ORIENTATIONS = [
    Orientation(t, fh, fv)
    for t in range(4) for fh in (False, True) for fv in (False, True)
]


@pytest.mark.parametrize("o", ALL_ORIENTATIONS)
def test_orientation_is_a_pure_permutation(o: Orientation) -> None:
    """No pixel invented, none lost — flips and 90° steps only move them."""
    src = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    out = apply(src, o)
    assert sorted(out.ravel().tolist()) == sorted(src.ravel().tolist())
    expect = (8, 6) if o.swaps_axes else (6, 8)
    assert out.shape == expect


@pytest.mark.parametrize("o", ALL_ORIENTATIONS)
def test_map_point_follows_the_pixels(o: Orientation) -> None:
    """`map_point` must land on the pixel that actually moved there.

    This is the guard that keeps detections drawn in the right place once an
    eye is flipped or rotated — the failure it catches looks like a plausible
    overlay sitting a mirror-image away from the corners it describes.
    """
    src = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    out = apply(src, o)
    h, w = src.shape
    for y in range(h):
        for x in range(w):
            nx, ny = map_point(float(x), float(y), w, h, o)
            assert out[int(round(ny)), int(round(nx))] == src[y, x]


def test_orientation_matches_the_css_contract() -> None:
    """Flip-then-rotate, the order `StereoView.eyeTransform` renders.

    A frame flipped horizontally then rotated 90° CW is NOT the same as one
    rotated then flipped; pinning the composition here is what stops the
    native view and the web preview from drifting apart.
    """
    src = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)
    flip_then_rotate = apply(src, Orientation(quarter_turns_cw=1, flip_h=True))
    manual = np.rot90(src[:, ::-1], k=-1)      # mirror first, then 90° clockwise
    assert np.array_equal(flip_then_rotate, manual)

    rotate_then_flip = np.rot90(src, k=-1)[:, ::-1]
    assert not np.array_equal(flip_then_rotate, rotate_then_flip), (
        "the two orders must differ, or this test proves nothing"
    )


def _stripe_frame(y0: float = 60.0, slope: float = 0.05, w: int = 320, h: int = 200):
    """A red stripe on a bright neutral background, laid on a known line.

    Neutral-but-bright on purpose: it is the case a luminance detector gets
    wrong and this one must not — the background is as bright as the stripe.
    """
    import cv2
    bgr = np.full((h, w, 3), 200, np.uint8)          # bright, neutral
    for x in range(w):
        y = y0 + slope * x
        lo, hi = int(np.floor(y)), int(np.floor(y)) + 1
        frac = y - lo
        for row, weight in ((lo, 1.0 - frac), (hi, frac)):
            if 0 <= row < h:
                bgr[row, x, 2] = 255                                  # red up
                bgr[row, x, 0] = bgr[row, x, 1] = int(200 * (1 - weight))
    return bgr, cv2


def test_redness_ignores_bright_neutral_pixels() -> None:
    """White is not red. This is the property the whole detector rests on."""
    white = np.full((4, 4, 3), 255, np.uint8)
    assert int(redness(white).max()) == 0
    red = np.zeros((4, 4, 3), np.uint8)
    red[:, :, 2] = 255
    assert int(redness(red).min()) == 255


def test_laser_fits_a_known_line_subpixel() -> None:
    bgr, _ = _stripe_frame(y0=60.0, slope=0.05)
    line = find_laser_line(bgr, None, LaserParams(redness_min=40))
    assert line.ok, line.reason
    assert line.n_inliers > 250
    assert line.rms_px < 0.5
    # The synthetic slope is 0.05 -> atan(0.05) = 2.86 degrees.
    assert abs(line.angle_deg - np.degrees(np.arctan(0.05))) < 0.2
    # And the fitted line must pass through the stripe at both ends.
    p, d = line.point, line.direction
    for x in (10.0, 300.0):
        t = (x - p[0]) / d[0]
        assert abs((p + d * t)[1] - (60.0 + 0.05 * x)) < 0.5


def test_laser_needs_colour() -> None:
    with pytest.raises(ValueError):
        find_laser_line(np.zeros((16, 16), np.uint8), None)


def test_laser_reports_why_it_found_nothing() -> None:
    """Each refusal names its own cause — the fixes are different."""
    neutral = np.full((64, 64, 3), 180, np.uint8)
    assert find_laser_line(neutral, None).reason is not None

    bgr, _ = _stripe_frame()
    empty = np.zeros(bgr.shape[:2], bool)
    assert find_laser_line(bgr, empty).reason == "board not visible"


def test_laser_mask_confines_the_search() -> None:
    """Points outside the mask must not reach the fit.

    This is what keeps the stripe's continuation on the workbench — which is
    not on the board plane — out of the calibration sample.
    """
    bgr, _ = _stripe_frame(y0=60.0, slope=0.0)
    mask = np.zeros(bgr.shape[:2], bool)
    mask[:, 100:200] = True
    line = find_laser_line(bgr, mask, LaserParams(redness_min=40))
    assert line.ok, line.reason
    xs = line.points[:, 0]
    assert xs.min() >= 100 and xs.max() < 200


def test_ransac_rejects_an_outlier_cluster() -> None:
    """A bright blob off the line must not drag the fit.

    On real frames the stripe's centroid wanders where it crosses a dark
    square; measured there, robust fitting cut RMS from 1.88 px to 0.67 px.
    """
    bgr, _ = _stripe_frame(y0=60.0, slope=0.0)
    bgr[150:158, 40:70, 2] = 255            # a red blob far from the stripe
    bgr[150:158, 40:70, 0:2] = 0
    line = find_laser_line(bgr, None, LaserParams(redness_min=40))
    assert line.ok, line.reason
    assert line.rms_px < 0.5
    assert abs(line.point[1] + line.direction[1] * 0 - 60.0) < 1.5
    assert int((~line.inliers).sum()) >= 20   # the blob's columns were rejected


def test_laser_is_deterministic() -> None:
    """Same frame in, same fit out — RANSAC is seeded for this reason."""
    bgr, _ = _stripe_frame()
    a = find_laser_line(bgr, None)
    b = find_laser_line(bgr, None)
    assert a.n_inliers == b.n_inliers
    assert np.allclose(a.point, b.point) and np.allclose(a.direction, b.direction)


def test_board_mask_needs_enough_corners() -> None:
    assert board_mask(None, (32, 32)) is None
    assert board_mask(np.zeros((2, 1, 2), np.float32), (32, 32)) is None
    corners = np.array([[[4, 4]], [[28, 4]], [[28, 28]], [[4, 28]]], np.float32)
    m = board_mask(corners, (32, 32))
    assert m is not None and m[16, 16] and not m[0, 0]


def _part(payload: bytes, extra: bytes = b"") -> bytes:
    return (b"--camserverframe\r\nContent-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
            + extra + b"\r\n" + payload + b"\r\n")


def _jpeg(value: int = 40) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", np.full((16, 24), value, np.uint8))
    assert ok
    return buf.tobytes()


def test_demux_splits_frames_across_chunk_boundaries() -> None:
    """Frames must survive arriving in arbitrarily sized TCP chunks."""
    jpg = _jpeg()
    stream = _part(jpg, b"X-Capture-Monotonic: 123.5\r\nX-Frame-Seq: 7\r\n") + _part(jpg)
    chunks = [stream[i:i + 7] for i in range(0, len(stream), 7)]  # nastily small

    reader = MjpegReader("http://unused")
    frames = list(reader._demux(iter(chunks)))
    assert len(frames) == 2
    assert frames[0].gray.shape == (16, 24)
    assert frames[0].capture_mono == 123.5
    assert frames[0].seq == "7"
    assert frames[1].capture_mono is None      # absent header, not a crash


def test_demux_skips_a_torn_frame_and_keeps_going() -> None:
    """One undecodable payload must not end the stream."""
    stream = _part(b"not a jpeg at all") + _part(_jpeg())
    frames = list(MjpegReader("http://unused")._demux(iter([stream])))
    assert len(frames) == 1


def test_config_parse_reads_the_rig() -> None:
    cfg = cfgmod.parse({
        "stereo_rig": {
            "host": "http://cam:8088/",
            "token": "",
            "baseline_mm": 203.0,
            "left": {"camera_id": "cam2", "quarter_turns_cw": 1, "flip_h": True,
                     "flip_v": False},
            "right": {"camera_id": "cam4", "quarter_turns_cw": 0, "flip_h": False,
                      "flip_v": False},
        },
        "charuco_squares_x": 8, "charuco_squares_y": 8,
        "charuco_square_length_mm": 36.0, "charuco_marker_length_mm": 26.64,
        "aruco_dict_id": 5,
    })
    assert cfg.left.camera_id == "cam2"
    assert cfg.left.orientation == Orientation(1, True, False)
    assert cfg.board.squares_x == 8
    assert cfg.baseline_mm == 203.0
    # Trailing slash on the host must not produce a double slash in the URL.
    assert cfg.stream_url(cfg.left) == "http://cam:8088/stream/cam2?sync=1"


def test_config_parse_survives_an_empty_payload() -> None:
    """A server with no stereo_rig yet must yield unconfigured eyes, not raise."""
    cfg = cfgmod.parse({})
    assert cfg.left is not None and not cfg.left.configured
    assert cfg.board is None
    assert cfg.stream_url(cfg.left) is None


def test_intrinsics_absent_until_the_pair_is_calibrated() -> None:
    """The model's phone intrinsics must never stand in for an eye's own.

    Using them would yield a plausible but wrong pose — the failure this guards
    against is silent, which is why it is pinned by a test.
    """
    cfg = cfgmod.parse({
        "stereo_rig": {"host": "http://cam:8088",
                       "left": {"camera_id": "cam2"}, "right": {"camera_id": "cam4"}},
        "camera_fx": 1500.0, "camera_fy": 1500.0,
        "camera_cx": 960.0, "camera_cy": 540.0,
        "camera_distortion": [0.0] * 5,
    })
    assert not cfg.left.has_intrinsics
    assert not cfg.right.has_intrinsics
    assert cfg.left.intrinsics_for((1280, 720)) is None


def test_intrinsics_are_refused_at_a_different_resolution() -> None:
    """A camera matrix is only valid at the size it was solved at.

    Applying 1280x720 intrinsics to a 1080p frame moves the principal point
    and scales the focal length by two thirds — a pose that looks plausible
    and is wrong. Rescaling instead would be a guess about crop-vs-scale
    behaviour that nobody has verified, so a mismatch yields nothing.
    """
    k = {"fx": 900.0, "fy": 905.0, "cx": 646.0, "cy": 358.0,
         "dist": [-0.28, 0.09, 0.0, 0.0, 0.0], "width": 1280, "height": 720}
    cfg = cfgmod.parse({
        "stereo_rig": {"host": "http://cam:8088",
                       "left": {"camera_id": "cam2", "intrinsics": k},
                       "right": {"camera_id": "cam4"}},
    })
    assert cfg.left.has_intrinsics
    got = cfg.left.intrinsics_for((1280, 720))
    assert got is not None and got.fx == 900.0
    assert cfg.left.intrinsics_for((1920, 1080)) is None


@pytest.mark.parametrize("o", [
    Orientation(), Orientation(flip_h=True), Orientation(flip_v=True),
    Orientation(quarter_turns_cw=1), Orientation(quarter_turns_cw=3, flip_h=True),
    Orientation(quarter_turns_cw=2, flip_v=True),
])
def test_map_points_is_map_point_over_an_array(o: Orientation) -> None:
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 100, (40, 2))
    got = map_points(pts, 120, 80, o)
    assert got.shape == (40, 2)
    for (x, y), (mx, my) in zip(pts, got):
        assert (mx, my) == map_point(float(x), float(y), 120, 80, o)
    assert map_points(np.empty((0, 2)), 120, 80, o).shape == (0, 2)


def test_find_laser_points_recovers_a_curve_to_subpixel() -> None:
    """Scanning wants the stripe's actual shape, one centroid per column."""
    h, w = 240, 320
    bgr = np.zeros((h, w, 3), np.uint8)
    xs = np.arange(w)
    centre = 120 + 40 * np.sin(xs / 25.0)
    # A stripe two rows wide, weighted so the centroid is not on a pixel, and
    # never fainter than the threshold so both rows always count.
    lo = np.floor(centre).astype(int)
    frac = centre - lo
    r_lo = np.rint(160 * (1 - frac) + 60)
    r_hi = np.rint(160 * frac + 60)
    bgr[lo, xs, 2] = r_lo.astype(np.uint8)
    bgr[lo + 1, xs, 2] = r_hi.astype(np.uint8)

    got = find_laser_points(bgr, None, LaserParams(redness_min=40))
    assert got.ok and got.along_x and got.count == w
    assert np.array_equal(got.points[:, 0], xs)
    expected = (lo * r_lo + (lo + 1) * r_hi) / (r_lo + r_hi)
    assert np.abs(got.points[:, 1] - expected).max() < 1e-3


def test_find_laser_points_scans_rows_for_a_vertical_stripe() -> None:
    bgr = np.full((200, 100, 3), 20, np.uint8)
    ys = np.arange(200)
    bgr[ys, 50 + (ys // 40), 2] = 240
    got = find_laser_points(bgr, None, LaserParams(redness_min=40))
    assert got.ok and not got.along_x and got.count == 200
    assert np.array_equal(got.points[:, 1], ys)
    assert np.array_equal(got.points[:, 0], 50 + ys // 40)
    assert find_laser_points(np.full((8, 8, 3), 200, np.uint8), None).reason \
        == "no stripe above the redness threshold"
