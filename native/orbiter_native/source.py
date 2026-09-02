"""camserver MJPEG reader.

Reads the multipart stream itself instead of handing the URL to
`cv2.VideoCapture`. That is not stubbornness: the per-frame
`X-Capture-Monotonic` header is the only thing that makes a true end-to-end
latency number possible, and VideoCapture throws the headers away. camserver's
own web UI parses the stream for exactly this reason.

The reader decodes colour, and derives the grayscale view from it rather than
decoding twice. The laser is red and the board is black and white, so the
stripe is only separable in colour — see `laser.redness`. Measured at 720p on
this machine: colour decode 2.23 ms plus 0.20 ms for the BGR→GRAY conversion,
against 1.36 ms for a grayscale-only decode. About 1 ms per frame per camera
buys the only channel in which the stripe exists.

With `gpu=True` the JPEG is decoded by nvJPEG instead (`gpu.decode`), the
frame stays on the GPU for the stripe score, and `bgr`/`gray` are downloaded
copies of it. Same `Frame` either way.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from . import gpu

log = logging.getLogger("orbiter_native.source")

_BOUNDARY = b"--camserverframe"
_HEADER_END = b"\r\n\r\n"

#: Guard against a desynchronised stream growing the buffer without bound.
#: A 1080p MJPEG frame is a few hundred kB, so several MB means we have lost
#: the boundary and should start over rather than accumulate forever.
_MAX_BUFFER = 8 * 1024 * 1024

_CONNECT_TIMEOUT_S = 5.0
#: A synchronised 30 fps stream delivers every ~33 ms. Seconds of silence means
#: the connection is dead even though the socket is still open.
_READ_TIMEOUT_S = 10.0


@dataclass
class Frame:
    """One decoded frame plus what is needed to time it."""

    #: Colour, as decoded. The laser needs it; the board does not.
    bgr: np.ndarray
    #: Luminance view, derived from `bgr` — what ChArUco detection consumes.
    gray: np.ndarray
    #: camserver's capture instant on ITS monotonic clock. Not comparable to
    #: our clock directly — see `MjpegReader.age_ms` for why this app reports
    #: arrival intervals rather than pretending to know the clock offset.
    capture_mono: float | None
    seq: str | None
    #: Server-side age at send time, straight from `X-Age-Ms`.
    server_age_ms: float | None
    #: Our monotonic clock at the moment the frame finished arriving.
    recv_mono: float
    #: The decoded frame on the GPU — a (3, H, W) RGB uint8 torch tensor —
    #: when the reader decodes there; `bgr` and `gray` are then its copies.
    #: None on the OpenCV path.
    rgb_gpu: object | None = None


def _parse_headers(blob: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in blob.decode("latin-1", "replace").split("\r\n"):
        k, _, v = line.partition(":")
        if v:
            out[k.strip().lower()] = v.strip()
    return out


class MjpegReader:
    """Iterates decoded frames from one camserver stream URL.

    Not a thread itself — `worker.CameraWorker` drives it. Keeping the parsing
    free of Qt means it can be exercised from a plain script.
    """

    def __init__(self, url: str, gpu: bool = False) -> None:
        self.url = url
        self._gpu = gpu
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def frames(self):
        """Yield `Frame`s until stopped or the stream ends.

        Raises `httpx.HTTPError` if the stream cannot be opened or dies —
        reconnection policy belongs to the caller, not here.
        """
        timeout = httpx.Timeout(_READ_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", self.url) as resp:
                resp.raise_for_status()
                yield from self._demux(resp.iter_bytes())

    def _demux(self, chunks):
        buf = bytearray()
        for chunk in chunks:
            if self._stop:
                return
            buf += chunk
            while True:
                start = buf.find(_BOUNDARY)
                if start < 0:
                    break
                head_end = buf.find(_HEADER_END, start + len(_BOUNDARY))
                if head_end < 0:
                    break
                hdr = _parse_headers(bytes(buf[start + len(_BOUNDARY):head_end]))
                try:
                    length = int(hdr.get("content-length", "0"))
                except ValueError:
                    length = 0
                body = head_end + len(_HEADER_END)
                if not length or len(buf) < body + length:
                    # Incomplete part. Drop anything before the boundary so the
                    # buffer does not creep forward, then wait for more bytes.
                    if start:
                        del buf[:start]
                    break
                # A bytearray slice is already a copy, and writable — which
                # `torch.frombuffer` insists on for the GPU decode.
                payload = buf[body:body + length]
                del buf[:body + length]
                frame = self._decode(payload, hdr)
                if frame is not None:
                    yield frame
            if len(buf) > _MAX_BUFFER:
                log.warning("stream desynchronised (%d bytes buffered); resyncing",
                            len(buf))
                buf.clear()

    def _decode(self, payload: bytearray, hdr: dict[str, str]) -> Frame | None:
        rgb_gpu = None
        if self._gpu:
            rgb_gpu = gpu.decode(payload)
            if rgb_gpu is None:
                return None
            bgr, gray = gpu.to_cpu(rgb_gpu)
        else:
            bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                # A torn frame. Skip it; the next one is ~33 ms away and a
                # dropped frame is far better than propagating garbage into
                # the detectors.
                return None
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        def num(key: str) -> float | None:
            try:
                return float(hdr[key])
            except (KeyError, TypeError, ValueError):
                return None

        return Frame(
            bgr=bgr,
            gray=gray,
            capture_mono=num("x-capture-monotonic"),
            seq=hdr.get("x-frame-seq"),
            server_age_ms=num("x-age-ms"),
            recv_mono=time.monotonic(),
            rgb_gpu=rgb_gpu,
        )
