"""HTTP surface for the binocular camserver pair.

Thin CORS-dodging passthrough to `stereo_proxy`. The browser cannot call
camserver's JSON API directly — camserver runs on a separate box and sends no
CORS headers for this origin — so the small JSON round-trips come through
here. The MJPEG streams do NOT: the Stereo tab points its `<img>` tags
straight at `http://<camserver>/stream/{id}`, which keeps two 1080p streams
off this server entirely.

Every upstream failure collapses to a 502 carrying the reason, so a camserver
that is powered off or has moved shows a readable message in the UI instead
of a hung fetch.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

import stereo_proxy
from stereo_proxy import UpstreamError

log = logging.getLogger("orbiter.routes.stereo")


def _no_store(response: Response) -> None:
    """camserver's numbers move every frame and its knobs are live device
    state; nothing here is ever cacheable. Applied router-wide so a stale
    snapshot can never freeze the overlays."""
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(
    prefix="/stereo/upstream", tags=["stereo"], dependencies=[Depends(_no_store)]
)


def _bad_gateway(exc: UpstreamError) -> HTTPException:
    log.warning("camserver upstream failed: %s", exc)
    return HTTPException(status_code=502, detail=str(exc))


@router.get("/state")
async def upstream_state() -> Any:
    """camserver `/api/state` — camera list, per-camera stats, pair sync.

    The Stereo tab polls this ~1 Hz to drive the fps / Mbit/s / drop overlays
    and the pair-sync chip, and to populate the left/right camera selects.
    """
    try:
        return await stereo_proxy.state()
    except UpstreamError as exc:
        raise _bad_gateway(exc) from exc


@router.get("/controls/{cam_id}")
async def upstream_controls(cam_id: str, probe: bool = False) -> Any:
    """V4L2 knobs for one camera. `?probe=1` makes camserver verify each knob
    is actually honoured by the firmware (slower — it writes and reads back)."""
    try:
        return await stereo_proxy.controls(cam_id, probe=probe)
    except UpstreamError as exc:
        raise _bad_gateway(exc) from exc


@router.post("/controls/{cam_id}")
async def upstream_set_controls(cam_id: str, body: dict[str, Any]) -> Any:
    """Write V4L2 knobs. Body `{"controls": {...}}`, or a bare mapping of
    knob → value, which is wrapped for the caller's convenience."""
    values = body.get("controls") if isinstance(body.get("controls"), dict) else body
    if not isinstance(values, dict) or not values:
        raise HTTPException(status_code=400, detail="expected a non-empty controls mapping")
    try:
        return await stereo_proxy.set_controls(cam_id, values)
    except UpstreamError as exc:
        raise _bad_gateway(exc) from exc


@router.get("/formats/{cam_id}")
async def upstream_formats(cam_id: str) -> Any:
    """Available fourcc + sizes, plus the format currently in use."""
    try:
        return await stereo_proxy.formats(cam_id)
    except UpstreamError as exc:
        raise _bad_gateway(exc) from exc


@router.post("/format/{cam_id}")
async def upstream_set_format(cam_id: str, body: dict[str, Any]) -> Any:
    """Reconfigure one camera: `{"fourcc": "MJPG", "size": "1920x1080", "fps": 30}`.

    Keep the pair on MJPG for metrology work. H.264 is offered by these
    cameras and is far lighter on the USB bus, but its interframe compression
    smears exactly the high-frequency edges that ChArUco corners and a laser
    line peak live on.
    """
    fourcc = str(body.get("fourcc") or "").strip()
    size = str(body.get("size") or "").strip()
    if not fourcc or not size:
        raise HTTPException(status_code=400, detail="'fourcc' and 'size' are required")
    try:
        fps = int(body.get("fps", 30))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="'fps' must be an integer")
    try:
        return await stereo_proxy.set_format(cam_id, fourcc, size, fps)
    except UpstreamError as exc:
        raise _bad_gateway(exc) from exc
