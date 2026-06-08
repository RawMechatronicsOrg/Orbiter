"""MJPEG live-frame reader for the IP-Webcam phone camera.

Companion to `camera_io.py` (the still-image client). `camera_io` GETs
`/photoaf.jpg` per shot; this module opens a long-lived `/video`
connection and keeps the LATEST decoded JPEG frame in memory, dropping
older frames whenever a new one arrives. The still-image path is
untouched — both can run concurrently against the same phone since the
Android IP-Webcam app multiplexes them.

Consumer: `routes/stream.py::camera_mjpeg` — proxies the latest frame as
a re-multiplexed MJPEG stream to the UI hover preview.

Backpressure: writers (the HTTP read loop) never block; readers always
see "the freshest frame available." If a consumer is slow, intermediate
frames are silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx

from orbiter_model import model

log = logging.getLogger("orbiter.camera_stream")

# Stream URL derivation: take the scheme+netloc of `model.camera_url` (the
# live, UI-editable camera address — the SAME source `phone_sensor` uses),
# append `/video`. The IP-Webcam app serves MJPEG on /video as
# multipart/x-mixed-replace.
_STREAM_PATH = "/video"

# Re-connect backoff (s). Capped so we don't hammer the phone if it's
# rebooting / out of range, but small enough to recover within a second
# once the phone is back.
_RETRY_INITIAL_S = 0.5
_RETRY_MAX_S = 10.0

# Chunk size for the MJPEG read. Big enough that an entire small JPEG
# usually arrives in one read on LAN; not so big that we sit on RAM.
_READ_CHUNK = 65536


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """(width, height) read from a JPEG's SOF marker WITHOUT decoding the image.
    Lets us know how frames actually arrive (portrait vs landscape) cheaply.
    Returns None if no SOF segment is found."""
    n = len(data)
    i = 2  # skip the SOI (0xFFD8)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0..SOF15 carry the frame size; skip the non-SOF C-markers
        # (DHT 0xC4, JPG 0xC8, DAC 0xCC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return (w, h) if (w and h) else None
        if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
            i += 2  # standalone marker, no length field
            continue
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def stream_url() -> str | None:
    """Derive the MJPEG stream URL from `model.camera_url`.

    `model.camera_url` is the live, UI-editable camera address (e.g.
    `http://<phone-ip>:<port>/`); we strip the path and append `/video`.
    Returns None when the camera URL isn't configured. This MUST read the
    model (not the static `settings.camera_url` env default) or the video
    preview and the phone-sensor IMU disagree about whether the camera is
    online — the bug this fixes: env unset, but model URL set via the UI.
    """
    base = model.camera_url
    if not base:
        return None
    try:
        p = urlparse(base)
    except Exception:  # noqa: BLE001
        return None
    if not p.scheme or not p.netloc:
        return None
    return f"{p.scheme}://{p.netloc}{_STREAM_PATH}"


class CameraStreamReader:
    """Singleton — one phone, one running stream task.

    `latest()` is sync and never blocks.
    `wait_for_new(last_seq, timeout)` is async and resolves as soon as
    a frame newer than `last_seq` is available (or `timeout` elapses).
    """

    def __init__(self) -> None:
        self._latest: bytes | None = None
        self._seq: int = 0
        self._cond = asyncio.Condition()
        self._task: asyncio.Task | None = None
        # Set by the read loop on a successful connect, cleared on
        # disconnect — handy for the UI to render a connection light.
        self._connected: bool = False
        self._url_last: str | None = None
        #: (width, height) of the most recent frame — lets the UI size the live
        #: camera frustum to how frames actually arrive (portrait vs landscape).
        self._frame_wh: tuple[int, int] | None = None

    # ── public API ──────────────────────────────────────────────────────

    def latest(self) -> tuple[bytes | None, int]:
        """Return `(jpeg_bytes, seq)`. `jpeg_bytes` is None until the
        first frame arrives; `seq` increments by one per new frame."""
        return self._latest, self._seq

    async def wait_for_new(
        self, last_seq: int, timeout: float = 5.0,
    ) -> tuple[bytes | None, int]:
        """Resolve as soon as `self._seq > last_seq`, or after `timeout`."""
        async with self._cond:
            try:
                await asyncio.wait_for(
                    self._cond.wait_for(lambda: self._seq > last_seq),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                pass
            return self._latest, self._seq

    def status(self) -> dict:
        return {
            "connected": self._connected,
            "seq": self._seq,
            "have_frame": self._latest is not None,
            "url": self._url_last,
            "frame_wh": list(self._frame_wh) if self._frame_wh else None,
        }

    def frame_aspect(self) -> float | None:
        """width / height of the latest frame (>1 landscape, <1 portrait), or
        None until a frame has arrived. Used to size the live camera frustum."""
        if not self._frame_wh:
            return None
        w, h = self._frame_wh
        return (w / h) if h else None

    async def start(self) -> None:
        """Start the background read loop (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._loop(), name="orbiter.camera_stream",
            )

    async def stop(self) -> None:
        """Cancel the read loop and wait briefly for it to exit. We hard-cap
        the wait because httpx can hold the cancellation inside a TLS/connect
        teardown longer than uvicorn's --timeout-graceful-shutdown; the task
        is daemon-ish, so leaking it on shutdown is acceptable."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    # ── internals ───────────────────────────────────────────────────────

    async def _loop(self) -> None:
        backoff = _RETRY_INITIAL_S
        while True:
            url = stream_url()
            self._url_last = url
            if not url:
                # No camera configured — sleep and re-check periodically;
                # operator may set ORBITER_CAMERA_URL at runtime via .env.
                self._connected = False
                await asyncio.sleep(_RETRY_MAX_S)
                continue
            try:
                # `timeout=None` on the body read — MJPEG is by design an
                # endless stream; httpx's default 5 s read timeout would
                # kill us between frames if the phone hiccups.
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(5.0, read=None),
                ) as client:
                    async with client.stream("GET", url) as response:
                        if response.status_code != 200:
                            log.warning(
                                "camera_stream: %s returned HTTP %d",
                                url, response.status_code,
                            )
                            self._connected = False
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2.0, _RETRY_MAX_S)
                            continue
                        self._connected = True
                        backoff = _RETRY_INITIAL_S
                        log.info("camera_stream: connected to %s", url)
                        await self._consume(response)
            except asyncio.CancelledError:
                self._connected = False
                raise
            except Exception as exc:  # noqa: BLE001
                self._connected = False
                log.warning("camera_stream: %s — %s", url, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, _RETRY_MAX_S)

    async def _consume(self, response: httpx.Response) -> None:
        """Pull bytes, slice out complete JPEGs by SOI/EOI markers,
        publish each one as `latest`. Never blocks the writer — slow
        consumers just see the freshest available frame."""
        # We don't bother parsing multipart boundaries — we scan for the
        # JPEG SOI (0xFFD8) / EOI (0xFFD9) markers directly. Robust to
        # any wrapping the phone uses and never desynchronises after a
        # partial-frame drop.
        buf = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=_READ_CHUNK):
            # camera_url changed under us (UI edit) → drop this connection so
            # `_loop` re-derives stream_url() and reconnects to the new phone.
            if stream_url() != self._url_last:
                return
            if not chunk:
                continue
            buf += chunk
            while True:
                soi = buf.find(b"\xff\xd8")
                if soi < 0:
                    # No JPEG marker in buffer — discard junk (multipart
                    # headers between frames) and keep accumulating.
                    buf.clear()
                    break
                eoi = buf.find(b"\xff\xd9", soi + 2)
                if eoi < 0:
                    # Drop pre-SOI junk; keep the partial JPEG for the
                    # next iteration.
                    if soi > 0:
                        del buf[:soi]
                    break
                # Complete JPEG.
                jpeg = bytes(buf[soi : eoi + 2])
                del buf[: eoi + 2]
                await self._publish(jpeg)

    async def _publish(self, jpeg: bytes) -> None:
        wh = _jpeg_dimensions(jpeg)
        async with self._cond:
            self._latest = jpeg
            if wh is not None:
                self._frame_wh = wh
            self._seq += 1
            self._cond.notify_all()


# Module-level singleton — one phone per process.
stream = CameraStreamReader()
