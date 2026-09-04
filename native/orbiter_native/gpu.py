"""The frame on the GPU: nvJPEG decode and the stripe score, in torch.

Why here and not on the CPU. Measured at 1080p on this machine, single OpenCV
thread: JPEG decode 4.3 ms and the stripe score 13.5 ms per eye — 36 core-ms
per frame pair, most of what the detector threads still did once the board
was tracked. The same two stages on the RTX 5060 Ti, through torchvision's
nvJPEG binding and a torch port of `laser.stripe_score`: 0.9 ms and 1.7 ms of
GPU time, both eyes decoded and scored in 5 ms of wall time, and the CPU
mostly waits.

What stays on the CPU, and why. ChArUco has no GPU implementation, so the
luminance view is downloaded (2 MB); the view draws a half-size copy of the
frame (1.5 MB) as a texture, with the geometry as GL points. The
calibration-mode line fit (`laser.find_laser_line`) runs inside the board's
bounding box on the CPU — a few milliseconds, only while calibrating — and
is the one consumer of the full colour frame, downloaded only then.

Why MJPEG through nvJPEG rather than the cameras' H.264 through NVDEC. Each
JPEG stands alone: camserver's per-frame `X-Capture-Monotonic` header still
times it, a torn frame costs one frame, and the stripe — a thin red line — is
not smeared by inter-frame prediction.

Optional. `torch` and `torchvision` are the `gpu` extra in pyproject; without
them, or without a CUDA device, `available()` is False and `source.MjpegReader`
decodes with OpenCV as before. Everything downstream sees the same `Frame`.

Two things about nvJPEG worth knowing. Its chroma upsampling is not
libjpeg-turbo's, so a frame decoded here differs from the CPU decode by a level
or two along colour edges — fine for both detectors, not bit-identical. And a
JPEG without embedded Huffman tables (some UVC cameras) is refused; camserver
reports `needs_dht_fixup` per camera and patches those streams itself.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .laser import LaserParams, StripePixels

log = logging.getLogger("orbiter_native.gpu")

# Bound on first use, never at import: `--help` and the tests without a GPU
# must not pay for torch.
_torch = None
_io = None
_F = None
_device = None
_state: bool | None = None
_why = "not probed yet"


def available() -> bool:
    """True when torch sees a CUDA device and nvJPEG decodes on it. Probed
    once; the answer and the reason for a False are kept."""
    global _state
    if _state is None:
        _state = _probe()
    return _state


def describe() -> str:
    """What `available()` found, for the startup log."""
    available()
    return _why


def _probe() -> bool:
    global _torch, _io, _F, _device, _why
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import io
    except ImportError as exc:
        _why = f"CPU (OpenCV): {exc.name or exc} not installed — pip install -e ./native[gpu]"
        return False
    _sleep_while_waiting(torch)
    if not torch.cuda.is_available():
        _why = "CPU (OpenCV): torch has no CUDA device"
        return False
    _torch, _io, _F = torch, io, F
    _device = torch.device("cuda", 0)
    try:
        # Warm up here rather than on the first frame: the CUDA context and
        # nvJPEG take the better part of a second to come up.
        probe = torch.full((3, 16, 24), 90, dtype=torch.uint8)
        rgb = decode(bytearray(io.encode_jpeg(probe).numpy().tobytes()))
        if rgb is None or tuple(rgb.shape) != (3, 16, 24):
            raise RuntimeError("probe frame did not decode")
        stripe_pixels(rgb)
        torch.cuda.synchronize(_device)
    except Exception as exc:                                    # noqa: BLE001
        _why = f"CPU (OpenCV): GPU probe failed — {exc.__class__.__name__}: {exc}"
        _torch = _io = _F = _device = None
        return False
    _why = f"GPU: {torch.cuda.get_device_name(_device)} (nvJPEG decode + torch stripe score)"
    return True


def _sleep_while_waiting(torch) -> None:
    """Ask CUDA to sleep rather than spin while the CPU waits on the GPU.

    Every download here (`to_cpu`, the pixel list) waits for the GPU, and
    CUDA's default is to burn the core in that wait — which is the very thing
    this module exists to stop. `cudaDeviceScheduleBlockingSync` makes the
    wait a proper sleep. Measured per 1080p frame, decode + download + stripe:
    3.96 ms of CPU became 2.08 ms, for 0.7 ms more latency out of a 33 ms
    frame. It has to be set before the CUDA context exists, on the runtime
    torch actually links: the Windows wheels carry it in torch/lib. Where it
    is not found this does nothing and the wait spins as before.
    """
    import ctypes
    import glob
    import os

    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    for pattern in ("cudart64_*.dll", "libcudart.so*"):
        hits = sorted(glob.glob(os.path.join(lib_dir, pattern)))
        if not hits:
            continue
        try:
            rc = ctypes.CDLL(hits[-1]).cudaSetDeviceFlags(4)   # ScheduleBlockingSync
        except OSError as exc:
            log.debug("blocking sync not set: %s", exc)
            return
        if rc:
            log.debug("cudaSetDeviceFlags returned %d; the GPU wait will spin", rc)
        return
    log.debug("no CUDA runtime under %s; the GPU wait will spin", lib_dir)


# ── decode ────────────────────────────────────────────────────────────────


def decode(payload: bytearray):
    """One JPEG as a (3, H, W) RGB uint8 tensor on the GPU, or None for a
    torn frame. `payload` must be writable — `torch.frombuffer` wraps it
    without a copy and refuses read-only `bytes`."""
    data = _torch.frombuffer(payload, dtype=_torch.uint8)
    try:
        return _io.decode_jpeg(data, mode=_io.ImageReadMode.RGB, device=_device)
    except RuntimeError:
        return None


def to_cpu(rgb, full: bool = True) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """The CPU views the rest of the app works on: `(bgr, gray, display)`.

    `gray` (H, W) is full size always — ChArUco wants every pixel. With
    `full`, `bgr` (H, W, 3) is downloaded whole and doubles as `display`;
    without, `bgr` is None and `display` is the frame averaged 2×2 — the
    eyes are shown at about a third of their size, so a half-size copy is
    all the view can use, at a quarter of the 6 MB download and upload.
    The full copy is only for the calibration-mode line fit, which runs
    on the CPU. Gray uses OpenCV's own fixed-point BGR2GRAY weights; its
    SIMD path rounds a level apart on a few pixels, below anything ChArUco
    can tell from sensor noise."""
    torch = _torch
    r, g, b = (c.to(torch.int32) for c in rgb)
    gray = ((r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14).to(torch.uint8)
    if full:
        bgr = rgb.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
        return bgr, gray.cpu().numpy(), bgr
    small = _F.avg_pool2d(rgb.to(torch.float32)[None], 2)[0]
    display = (small.round_().clamp_(0, 255).to(torch.uint8)
               .flip(0).permute(1, 2, 0).contiguous().cpu().numpy())
    return None, gray.cpu().numpy(), display


# ── the stripe ────────────────────────────────────────────────────────────


def stripe_score(rgb, background_px: int = 15):
    """`laser.stripe_score` in torch: the larger of plain redness and redness
    in excess of the local background, as float32 (H, W) in 0..255.

    Same construction as the CPU version — background per channel from a
    morphological opening wider than the stripe, at half resolution — with
    the opening written as two max-pools (an erosion is a max-pool of the
    negative). Pooling pads with -inf, which is exactly OpenCV's default
    border for morphology: pixels outside the frame do not take part.
    """
    F = _F
    x = rgb.to(_torch.float32)
    r, g, b = x[0], x[1], x[2]
    m = _torch.maximum(g, b)
    red = (r - m).clamp_(min=0)
    k = int(background_px) | 1              # odd, so the pools keep the size
    pad = k // 2

    def excess(ch):
        small = F.avg_pool2d(ch[None, None], 2)              # INTER_AREA ×½
        eroded = -F.max_pool2d(-small, k, stride=1, padding=pad)
        opened = F.max_pool2d(eroded, k, stride=1, padding=pad)
        back = F.interpolate(opened, size=ch.shape, mode="bilinear",
                             align_corners=False)[0, 0]
        return (ch - back).clamp_(min=0)

    ex = (excess(r) - excess(m)).clamp_(min=0)
    return _torch.maximum(red, ex)


def stripe_pixels(rgb, p: LaserParams = LaserParams(),
                  rows: tuple[int, int] | None = None) -> StripePixels:
    """`laser.find_stripe_pixels` on a GPU frame: every pixel scoring at least
    `redness_min`, with its score, in the same row-major order OpenCV's
    `findNonZero` yields. Only the pixel list comes back to the CPU. `rows`
    limits the search to the band the sheet can appear in."""
    t0 = time.perf_counter()
    _, h, w = rgb.shape
    y0, y1 = (0, h) if rows is None else (max(0, int(rows[0])), min(h, int(rows[1])))
    if y1 <= y0:
        return StripePixels(wh=(w, h), reason="the sheet cannot appear in this frame",
                            ms=(time.perf_counter() - t0) * 1000.0)
    score = stripe_score(rgb[:, y0:y1], p.background_px)
    lit = score >= p.redness_min
    yx = lit.nonzero()
    if yx.shape[0] == 0:
        return StripePixels(wh=(w, h), reason="no stripe above the redness threshold",
                            ms=(time.perf_counter() - t0) * 1000.0)
    weight = score[lit].round_().clamp_(0, 255).to(_torch.uint8)
    yx_cpu = yx.cpu().numpy()
    x = yx_cpu[:, 1].astype(np.int32)
    y = (yx_cpu[:, 0] + y0).astype(np.int32)
    along_x = bool((x.max() - x.min()) >= (y.max() - y.min()))
    return StripePixels(x=x, y=y, w=weight.cpu().numpy(), wh=(w, h),
                        along_x=along_x, reason=None,
                        ms=(time.perf_counter() - t0) * 1000.0)
