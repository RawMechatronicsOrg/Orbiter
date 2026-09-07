"""Story A1: keeping the camera's own JPEG, and measuring how sharp it is.

A photogrammetry pass needs the very frame the board pose was solved on, at
the resolution the intrinsics were solved at, without a re-encode — and that
is exactly what the reader throws away today. Retaining it is a copy per
frame on the reader thread and a Laplacian per frame on the detector thread,
against a 33 ms budget both are already spending, so both stay off until a
photo pass asks for them.

Half of what follows therefore tests the disarmed case: no bytes retained, no
focus measure taken, and the Laplacian not run at all.
"""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest

from orbiter_native import gpu
from orbiter_native import worker as workermod
from orbiter_native.detect import BoardDetector
from orbiter_native.scanworker import ScanWorker
from orbiter_native.source import MjpegReader
from orbiter_native.worker import EyeWorker

from test_pure import _part
from test_stereo_scan import _eye_result


def _texture(w: int = 96, h: int = 64) -> np.ndarray:
    """A frame with detail at every scale, so a blur has something to
    destroy: a two-pixel checkerboard laid over a smooth ramp."""
    y, x = np.mgrid[0:h, 0:w]
    check = ((x // 2 + y // 2) % 2 * 120).astype(np.int16)
    ramp = (x * 255 // max(w - 1, 1)).astype(np.int16)
    gray = np.clip((check + ramp) // 2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _encode(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def _read_one(payload: bytes, keep_jpeg: bool = False):
    """One frame out of a synthetic multipart stream, the way the reader
    itself gets it — `_demux` is where the retention decision is made."""
    reader = MjpegReader("http://unused")
    reader.keep_jpeg = keep_jpeg
    frames = list(reader._demux(iter([_part(payload)])))
    assert len(frames) == 1
    return frames[0]


def _detect(frame, armed: bool):
    """One detection pass, returning the result the sinks would have seen."""
    eye = EyeWorker("left")
    eye.set_photo_capture(armed)
    out = []
    eye.add_sink(out.append)
    eye._detect_one(BoardDetector(), frame)
    assert len(out) == 1
    return out[0]


# ── the reader ────────────────────────────────────────────────────────────


def test_the_reader_keeps_the_jpeg_only_when_armed() -> None:
    """Nothing is retained by default; armed, what is retained is byte for
    byte what came off the wire — no re-encode anywhere in between."""
    payload = _encode(_texture())
    assert _read_one(payload).jpeg is None
    assert _read_one(payload, keep_jpeg=True).jpeg == payload


def test_the_kept_jpeg_decodes_to_the_same_frame() -> None:
    """The bytes belong to this frame and not to a neighbour: decoding them
    again reproduces the pixels the detectors and the pose were given."""
    frame = _read_one(_encode(_texture()), keep_jpeg=True)
    again = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(again, frame.bgr)


# ── the focus measure ─────────────────────────────────────────────────────


def test_a_blurred_frame_scores_lower_than_a_sharp_one() -> None:
    """The whole point of the measure: it has to rank a smeared frame below
    a crisp one, which is what the photo policy rejects on."""
    sharp = _read_one(_encode(_texture()))
    blurred = _read_one(_encode(cv2.GaussianBlur(_texture(), (9, 9), 3.0)))
    assert workermod._sharpness(blurred) < workermod._sharpness(sharp)


def test_the_gpu_focus_measure_orders_blur_the_same_way() -> None:
    """The GPU path is the one a real scan runs, and it must agree with the
    CPU fallback about which of two frames is the sharper."""
    if not gpu.available():
        pytest.skip(gpu.describe())
    import torch

    def upload(bgr: np.ndarray):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1]).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(rgb)).cuda()

    sharp = _texture()
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3.0)
    assert gpu.sharpness(upload(blurred)) < gpu.sharpness(upload(sharp))


def test_the_gpu_path_still_carries_the_full_resolution_frame() -> None:
    """The case this whole story exists for. Scanning on the GPU, the full
    colour frame is never downloaded — `bgr` is None and `display` is a half
    size copy — so the retained JPEG is the only full-resolution view of the
    frame the pose was solved on, and it has to come through intact.
    """
    if not gpu.available():
        pytest.skip(gpu.describe())
    payload = _encode(_texture(640, 480))
    reader = MjpegReader("http://unused", gpu=True)
    reader.keep_jpeg = True
    reader.full_bgr = False                    # what a scan actually runs with
    frame = list(reader._demux(iter([_part(payload)])))[0]
    assert frame.rgb_gpu is not None and frame.bgr is None

    res = _detect(frame, armed=True)
    assert res.jpeg == payload
    assert res.sharpness > 0.0


# ── the detector pass ─────────────────────────────────────────────────────


def test_the_detector_pass_costs_nothing_while_disarmed(monkeypatch) -> None:
    """No bytes, no number — and, the part that matters for the 33 ms
    budget, no Laplacian run at all."""
    measured = []
    monkeypatch.setattr(workermod, "_sharpness",
                        lambda frame: measured.append(frame) or 0.0)
    res = _detect(_read_one(_encode(_texture())), armed=False)
    assert res.jpeg is None
    assert math.isnan(res.sharpness) and math.isnan(res.stats.sharpness)
    assert not measured


def test_the_detector_publishes_the_bytes_and_the_focus_once_armed() -> None:
    """Armed, the result carries this frame's own bytes and its score, and
    the overlay's copy of the score is the same number."""
    payload = _encode(_texture())
    res = _detect(_read_one(payload, keep_jpeg=True), armed=True)
    assert res.jpeg == payload
    assert res.sharpness > 0.0
    assert res.stats.sharpness == res.sharpness


def test_arming_an_eye_pushes_the_flag_to_its_live_reader() -> None:
    """The toggle is thrown mid-stream, so it has to reach the reader that is
    already running — the way `full_bgr` is already pushed to it."""
    eye = EyeWorker("left")
    eye.set_photo_capture(True)          # no reader yet: must not raise
    eye._reader = MjpegReader("http://unused")
    eye.set_photo_capture(True)
    assert eye._reader.keep_jpeg is True
    eye.set_photo_capture(False)
    assert eye._reader.keep_jpeg is False


# ── into the scan worker ──────────────────────────────────────────────────


def test_offer_carries_the_jpeg_and_sharpness_for_both_eyes() -> None:
    """Both eyes, not just the left. A right-hand photo is a photo in its own
    right and can only be made of the right eye's pixels; `bgr`, which the
    left eye keeps so points can be coloured, stays left-only as it was.
    """
    left, right = b"left jpeg bytes", b"right jpeg bytes"
    sw = ScanWorker()
    sw.set_active(True)
    sw.offer(replace(_eye_result("left", 0.000), jpeg=left, sharpness=140.0))
    sw.offer(replace(_eye_result("right", 0.0005), jpeg=right, sharpness=98.5))

    a, b, _, _, _ = sw._take_pair()
    # By reference: a photo pass may retain a hundred of these and must not
    # pay for a second copy of each.
    assert a.jpeg is left and b.jpeg is right
    assert (a.sharpness, b.sharpness) == (140.0, 98.5)
    assert a.bgr is not None and b.bgr is None
