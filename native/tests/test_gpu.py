"""The GPU path against its CPU reference. Skipped without torch and a CUDA
device: `gpu.available()` is the app's own switch, so a machine without a GPU
runs the OpenCV path and these tests do not apply to it."""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from orbiter_native import gpu
from orbiter_native.laser import LaserParams, find_stripe_pixels, stripe_score
from orbiter_native.source import MjpegReader

from test_pure import _stripe_frame

if not gpu.available():
    pytest.skip(gpu.describe(), allow_module_level=True)

import torch  # noqa: E402 — only after the skip


def _scene() -> np.ndarray:
    """A stripe across a bright neutral background AND a blue block, so both
    halves of the score — plain redness and the excess over a coloured
    background — carry pixels."""
    bgr, _ = _stripe_frame(y0=70.0, slope=0.08, w=400, h=240)
    bgr[:, 150:260] = (170, 60, 30)                      # blue, in BGR
    rows = (70.0 + 0.08 * np.arange(150, 260)).astype(int)
    bgr[rows, np.arange(150, 260), 2] = 255              # the stripe over it
    rng = np.random.default_rng(3)
    noisy = bgr.astype(np.int16) + rng.integers(-4, 5, bgr.shape, dtype=np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _upload(bgr: np.ndarray):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1]).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(rgb)).cuda()


def test_stripe_score_matches_the_cpu_reference() -> None:
    bgr = _scene()
    ref = stripe_score(bgr).astype(np.float32)
    got = gpu.stripe_score(_upload(bgr)).cpu().numpy()
    diff = np.abs(got - ref)
    # Rounding differs (float against uint8 stages), the construction does not.
    assert diff.max() <= 3.0, diff.max()
    assert diff.mean() < 0.5, diff.mean()


def test_stripe_pixels_agree_with_the_cpu_detector() -> None:
    bgr = _scene()
    p = LaserParams(redness_min=45)
    cpu = find_stripe_pixels(bgr, p)
    got = gpu.stripe_pixels(_upload(bgr), p)
    assert cpu.ok and got.ok
    assert got.along_x == cpu.along_x
    assert got.wh == cpu.wh
    a = set(zip(cpu.x.tolist(), cpu.y.tolist()))
    b = set(zip(got.x.tolist(), got.y.tolist()))
    assert len(a & b) >= 0.98 * max(len(a), len(b)), (len(a), len(b), len(a & b))
    # Row-major, like findNonZero: the scan step relies on nothing else, but
    # a different order would make the two paths' clouds differ in bit order.
    assert np.all(np.diff(got.y.astype(np.int64) * got.wh[0] + got.x) > 0)


def test_stripe_pixels_say_why_when_nothing_is_lit() -> None:
    white = np.full((3, 40, 60), 255, np.uint8)
    got = gpu.stripe_pixels(torch.from_numpy(white).cuda())
    assert not got.ok and got.reason == "no stripe above the redness threshold"


def test_gray_matches_opencv_to_a_level() -> None:
    """Same weights as `cv2.cvtColor`; its SIMD path rounds a level apart on
    a few pixels, which no corner detector can tell from sensor noise."""
    rng = np.random.default_rng(5)
    bgr = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    got_bgr, got_gray = gpu.to_cpu(_upload(bgr))
    assert np.array_equal(got_bgr, bgr)
    diff = np.abs(got_gray.astype(np.int16)
                  - cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16))
    assert diff.max() <= 1, diff.max()
    assert (diff > 0).mean() < 0.05, (diff > 0).mean()


def test_decode_is_close_to_opencv() -> None:
    """nvJPEG and libjpeg-turbo differ in chroma upsampling, not in content."""
    yy, xx = np.mgrid[0:96, 0:128]
    bgr = np.stack([(xx * 2) % 256, (yy * 2) % 256, ((xx + yy)) % 256], axis=2).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    ref = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = gpu.decode(bytearray(buf.tobytes()))
    assert rgb is not None and tuple(rgb.shape) == (3, 96, 128)
    got, _ = gpu.to_cpu(rgb)
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    assert diff.mean() < 2.0, diff.mean()
    assert diff.max() <= 24, diff.max()


def test_a_torn_frame_decodes_to_nothing() -> None:
    assert gpu.decode(bytearray(b"not a jpeg at all")) is None


def test_the_reader_hands_out_gpu_frames() -> None:
    ok, buf = cv2.imencode(".jpg", np.full((16, 24), 40, np.uint8))
    frame = MjpegReader("http://unused", gpu=True)._decode(bytearray(buf.tobytes()), {})
    assert frame is not None
    assert frame.rgb_gpu is not None and frame.rgb_gpu.is_cuda
    assert frame.bgr.shape == (16, 24, 3) and frame.gray.shape == (16, 24)
    assert MjpegReader("http://unused", gpu=False)._decode(
        bytearray(buf.tobytes()), {}).rgb_gpu is None
