"""HTTP endpoints for the live camera stream.

  * `GET /camera/stream.mjpeg` — re-multiplexes the latest JPEG from
    `camera_stream` as a long-lived multipart/x-mixed-replace stream.
    Drop-in for an `<img src=...>` on the frontend; no JS decoder needed.
  * `GET /camera/stream/status` — connectivity snapshot.

Everything is read-only against the model.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from camera_stream import stream

log = logging.getLogger("orbiter.stream")

router = APIRouter(tags=["stream"])

# Boundary marker for the MJPEG multipart response. Anything unique works
# — clients only need consistency within one response.
_BOUNDARY = "orbiterframe"


# ── HTTP MJPEG proxy ────────────────────────────────────────────────────


async def _mjpeg_iter():
    """Yield boundary-framed JPEGs as the upstream stream produces them.

    The phone's IP-Webcam app already streams MJPEG; we don't simply
    forward it byte-for-byte because:
      * `camera_stream` already parses out clean JPEGs (it can recover
        from partial-frame drops); reusing them avoids a second parser.
      * Multiple browser tabs can attach without each opening its own
        upstream HTTP connection to the phone.
    """
    boundary = f"--{_BOUNDARY}\r\n".encode("ascii")
    header_tmpl = b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
    last_seq = -1
    try:
        while True:
            jpeg, seq = await stream.wait_for_new(last_seq, timeout=5.0)
            if jpeg is None:
                # No frame yet — quick re-poll. Don't yield empty data
                # because some browsers close the stream on it.
                await asyncio.sleep(0.1)
                continue
            last_seq = seq
            yield boundary + header_tmpl % len(jpeg) + jpeg + b"\r\n"
    except asyncio.CancelledError:
        return


@router.get("/camera/stream.mjpeg")
def camera_mjpeg() -> StreamingResponse:
    """Live MJPEG, drop-in for `<img src=>`. One connection per viewer,
    all sharing the same upstream phone stream."""
    return StreamingResponse(
        _mjpeg_iter(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/camera/stream/status")
def camera_status() -> dict:
    """Upstream connection + frame counter. Lets the UI show a green/red
    light beside the live preview."""
    return stream.status()
