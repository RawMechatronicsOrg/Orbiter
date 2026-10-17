"""HTTP endpoints for the phone camera.

  * `GET /camera/stream.mjpeg` — re-multiplexes the latest JPEG from
    `camera_stream` as a long-lived multipart/x-mixed-replace stream.
    Drop-in for an `<img src=...>` on the frontend; no JS decoder needed.
  * `GET /camera/stream/status` — connectivity snapshot (URL, last error).
  * `POST /camera/node/{lock|profile|unlock}` — a profile operation on the
    CameraStreamer capture node (`camera_io.node_control`); the body is passed
    through. Its state reaches the UI as `model.camera_node`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse

import camera_io
from camera_stream import stream
from orbiter_model import model

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


@router.post("/camera/node/{op}")
async def camera_node_op(op: str, body: dict[str, Any] = Body(default_factory=dict)) -> dict:
    """Lock (meter once and freeze), re-apply / patch the profile, or unlock
    the capture node. Returns the phone's apply result; 409 while a scan
    runs; 502 with the phone's reason when it refuses or does not answer."""
    if op not in camera_io.NODE_OPS:
        raise HTTPException(404, f"unknown operation {op!r}")
    if model.scan_running:
        # A re-metered or unlocked camera would shoot the rest of the scan
        # with other settings (or not at all).
        raise HTTPException(409, "a scan is running — the camera profile stays "
                                 "as it is until the scan ends")
    try:
        return await camera_io.node_control(op, body)
    except camera_io.CameraError as exc:
        raise HTTPException(502, str(exc)) from exc
