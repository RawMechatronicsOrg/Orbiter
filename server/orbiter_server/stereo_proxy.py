"""Upstream client for the binocular camserver.

`camserver/1.0` runs on a separate box and owns the two UVC cameras of the
stereo pair. This module is the ONLY place that talks to it. Companion to
`routes/stereo.py`, which exposes these calls over HTTP to the browser.

Why proxy the JSON at all? The browser pulls the MJPEG streams **directly**
from camserver (an `<img src=...>` is not subject to CORS as long as nobody
reads its pixels), which keeps two ~30-90 Mbit/s streams off this server
entirely. The small JSON API *is* subject to CORS, and camserver is not our
process — we cannot add headers to it. So the pixels go direct and only the
~2 KB of JSON comes through here.

camserver's API surface, in full:

  * `GET  /api/state`                 all cameras + pair sync + server clock
  * `GET  /api/stats`                 the same, stats only
  * `GET  /snapshot/{id}`             one JPEG frame
  * `GET  /stream/{id}?sync=1`        MJPEG multipart (boundary `camserverframe`)
  * `GET  /stream/{id}.mp4?sync=1`    fragmented MP4, H.264 rewrapped, no transcode
  * `GET  /api/controls/{id}?probe=1` V4L2 knobs; `probe` verifies the firmware
                                      actually honours each one
  * `POST /api/controls/{id}`         `{"controls": {"brightness": 60}}`
  * `GET  /api/formats/{id}`          fourcc + size list
  * `POST /api/format/{id}`           `{"size": "1280x720", "fourcc": "MJPG", "fps": 30}`

An optional `?token=` is appended to every request when the rig config
carries one.

Note there is no `hflip`/`vflip` among the V4L2 controls of these cameras —
image orientation is ours to apply, not something we can push upstream. See
`commands._cmd_set_stereo_rig` for where that policy lives.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from orbiter_model import model

log = logging.getLogger("orbiter.stereo_proxy")

#: Hard timeouts (s). Every call here is a small JSON round-trip on the LAN;
#: a format change makes camserver reopen the V4L2 device, which is slower.
_READ_TIMEOUT_S = 4.0
_FORMAT_TIMEOUT_S = 10.0


class UpstreamError(RuntimeError):
    """camserver was unreachable, timed out, or answered with an error."""


def _rig() -> dict[str, Any]:
    rig = model.stereo_rig or {}
    return rig if isinstance(rig, dict) else {}


def base_url() -> str:
    """`scheme://host:port` of the configured camserver, no trailing slash."""
    return str(_rig().get("host") or "").strip().rstrip("/")


def _params(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(extra or {})
    token = str(_rig().get("token") or "").strip()
    if token:
        params["token"] = token
    return params


async def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float = _READ_TIMEOUT_S,
) -> Any:
    """One JSON round-trip to camserver. Raises `UpstreamError` on any failure.

    Failures are collapsed into a single exception type on purpose: the route
    layer turns it into one 502 with the reason attached, so a camserver that
    is off or renumbered produces a readable message in the UI instead of a
    hung request.
    """
    base = base_url()
    if not base:
        raise UpstreamError("no camserver host configured (Stereo tab → host)")
    url = base + path
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, params=_params(params), json=json)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise UpstreamError(
            f"{method} {url} → HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamError(f"{method} {url} → {exc.__class__.__name__}: {exc}") from exc
    except ValueError as exc:  # non-JSON body
        raise UpstreamError(f"{method} {url} → malformed JSON: {exc}") from exc


async def state() -> Any:
    """Full `/api/state`: camera list, per-camera stats, pair sync, clock."""
    return await _request("GET", "/api/state")


async def camera_ids() -> list[str]:
    """Ids camserver currently exposes, in its own enumeration order.

    The Stereo tab seeds its left/right selects from this. Order follows
    /dev/videoN and carries no meaning about physical placement.
    """
    data = await state()
    cams = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(cams, list):
        return []
    return [str(c["id"]) for c in cams if isinstance(c, dict) and c.get("id")]


async def controls(cam_id: str, probe: bool = False) -> Any:
    """V4L2 knobs for one camera. `probe=True` asks camserver to verify that
    the firmware actually honours each knob (slower — it writes and reads back)."""
    return await _request(
        "GET", f"/api/controls/{cam_id}", params={"probe": 1} if probe else None
    )


async def set_controls(cam_id: str, values: dict[str, Any]) -> Any:
    """Write V4L2 knobs, e.g. `{"power_line_frequency": 1}`."""
    return await _request("POST", f"/api/controls/{cam_id}", json={"controls": values})


async def formats(cam_id: str) -> Any:
    """Available fourcc + sizes, plus the format currently in use."""
    return await _request("GET", f"/api/formats/{cam_id}")


async def set_format(
    cam_id: str, fourcc: str, size: str, fps: int = 30
) -> Any:
    """Reconfigure one camera. camserver reopens the V4L2 device, so this is
    the one call here that can take seconds rather than milliseconds."""
    return await _request(
        "POST",
        f"/api/format/{cam_id}",
        json={"fourcc": fourcc, "size": size, "fps": fps},
        timeout=_FORMAT_TIMEOUT_S,
    )
