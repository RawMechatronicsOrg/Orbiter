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
from orbiter_native.detect import LaserParams, find_laser
from orbiter_native.orient import Orientation, apply, map_point
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


def test_laser_finds_a_known_line_subpixel() -> None:
    """A synthetic stripe on a known row must come back at that row."""
    img = np.zeros((200, 320), np.uint8)
    img[100, :] = 255
    img[99, :] = 255            # symmetric two-pixel stripe → centroid at 99.5
    hit = find_laser(img, LaserParams(min_intensity=200))
    assert hit.count == 320
    assert np.allclose(hit.ys, 99.5, atol=1e-3)


def test_laser_ignores_a_scene_with_no_line() -> None:
    """A dim, textured frame must yield nothing at the shipped threshold.

    Regression guard for the default that was originally too low: on a lit
    workbench frame from the rig it reported a confident centroid in every one
    of 1280 columns, tracking the scene rather than a stripe.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 120, size=(200, 320), dtype=np.uint8)
    assert find_laser(img).count == 0


def test_laser_tolerates_a_blank_frame() -> None:
    assert find_laser(np.zeros((32, 32), np.uint8)).count == 0


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
    assert cfg.left.intrinsics is None
    assert cfg.right.intrinsics is None
