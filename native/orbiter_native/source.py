"""camserver MJPEG reader.

Reads the multipart stream itself instead of handing the URL to
`cv2.VideoCapture`. That is not stubbornness: the per-frame
`X-Capture-Monotonic` header is the only thing that makes a true end-to-end
latency number possible, and VideoCapture throws the headers away. camserver's
own web UI parses the stream for exactly this reason.

The reader also decodes straight to grayscale. Measured on this machine on a
live 1080p frame: 2.20 ms for `IMREAD_GRAYSCALE` against 4.02 ms for
`IMREAD_COLOR`. Both detectors downstream work on luminance, so the colour
frame would be decoded only to be discarded — the display gets grayscale
promoted back to BGR, which is cheaper than decoding colour and honest about
what the pipeline actually saw.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

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

    def __init__(self, url: str) -> None:
        self.url = url
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
                payload = bytes(buf[body:body + length])
                del buf[:body + length]
                frame = self._decode(payload, hdr)
                if frame is not None:
                    yield frame
            if len(buf) > _MAX_BUFFER:
                log.warning("stream desynchronised (%d bytes buffered); resyncing",
                            len(buf))
                buf.clear()

    @staticmethod
    def _decode(payload: bytes, hdr: dict[str, str]) -> Frame | None:
        gray = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            # A torn frame. Skip it; the next one is ~33 ms away and a dropped
            # frame is far better than propagating garbage into the detectors.
            return None

        def num(key: str) -> float | None:
            try:
                return float(hdr[key])
            except (KeyError, TypeError, ValueError):
                return None

        return Frame(
            gray=gray,
            capture_mono=num("x-capture-monotonic"),
            seq=hdr.get("x-frame-seq"),
            server_age_ms=num("x-age-ms"),
            recv_mono=time.monotonic(),
        )
