"""Camera HTTP I/O.

Talks to the phone on `model.camera_url`, which runs one of two apps. Only the
`scheme://host:port` of that URL matters — any path is stripped.

**CameraStreamer** (`labs/orbiter-lab/camerastreamer`, recognised by
`GET /status` → `"app": "camerastreamer"`) — the capture node. The camera sits
on a locked manual profile; `trigger_shot` returns as soon as the frame is
exposed (the rig may move), and `fetch_shot` later pulls the JPEG, checks its
SHA-256, reads the per-frame verification and frees the phone's copy. See its
docs/CAPTURE_API.md.

**IP Webcam** (Pavel Khlebovich) — the legacy source, documented endpoints:

  * `/photoaf.jpg`  — trigger autofocus + capture (slow, sharp)
  * `/photo.jpg`    — capture with whatever focus is currently set (fast)
  * `/focus`        — fire single-shot AF (no photo)
  * `/nofocus`      — freeze the current focus distance (locks AF result)
  * `/enabletorch` / `/disabletorch` — camera LED
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

from orbiter_model import model

log = logging.getLogger("orbiter.camera_io")

#: Hard timeouts (s). `/photoaf.jpg` blocks the phone until AF settles —
#: full-resolution shots on the camera side can take 2-3 s plus transfer.
_PHOTOAF_TIMEOUT_S = 15.0
_CONTROL_TIMEOUT_S = 3.0

# Throttle the "no camera configured" warning — it would otherwise spam
# the log once per fetch (~65× per sweep). One line per process is enough
# to make the placeholder mode obvious without flooding.
_warned_no_camera = False


def _placeholder_jpeg() -> bytes:
    """A small valid JPEG, used when no camera URL is configured."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (96, 72), (40, 44, 52)).save(buf, "JPEG")
    return buf.getvalue()


PLACEHOLDER = _placeholder_jpeg()


def camera_base() -> str | None:
    """`scheme://netloc` of `model.camera_url`, or None if not configured.
    Any path component of `camera_url` is ignored — we know the endpoints.

    Reads the LIVE model (UI-editable), NOT the static `settings.camera_url`
    env default, to match `phone_sensor` / `camera_stream` — otherwise scan
    captures silently become placeholders whenever the URL was set via the UI
    rather than `ORBITER_CAMERA_URL`."""
    url = model.camera_url
    if not url:
        return None
    try:
        p = urlparse(url)
    except Exception:  # noqa: BLE001
        return None
    if not p.scheme or not p.netloc:
        return None
    return f"{p.scheme}://{p.netloc}"


def reset_focus_lock() -> None:
    """No-op — kept for source compatibility with existing callers."""
    return None


#: Focus-distance sweep ranges (Pavel Khlebovich `/settings/focus_distance`).
#: The unit is opaque (looks like "diopters-ish": small = close, large = far).
#: Empirically the board sits at ~2-4 on our bench; we coarse-scan a wider
#: band first, then refine ±0.4 around the peak.
_FOCUS_COARSE_STEPS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0)
_FOCUS_FINE_OFFSETS = (-0.4, -0.2, 0.0, 0.2, 0.4)

#: Settle delay after setting focus_distance — lens motor needs ~300-500 ms
#: to physically move + the image pipeline needs another frame to stabilise.
_FOCUS_SETTLE_S = 0.45


async def _sweep_sharpness(
    client: "httpx.AsyncClient",
    base: str,
    values: "Iterable[float]",
) -> list[tuple[float, float]]:
    """Set each focus distance, capture, compute Laplacian-variance sharpness.

    Returns ``[(distance, lap_var), ...]``. Failures contribute a 0.0
    sharpness so the caller can still pick the best of the rest.
    """
    import cv2  # local import — camera_io stays lean for callers that skip the focus sweep
    import numpy as np

    out: list[tuple[float, float]] = []
    for d in values:
        try:
            await client.post(
                f"{base}/settings/focus_distance?set={d:.3f}",
                timeout=_CONTROL_TIMEOUT_S,
            )
            await asyncio.sleep(_FOCUS_SETTLE_S)
            resp = await client.get(base + "/photo.jpg", timeout=_PHOTOAF_TIMEOUT_S)
            arr = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                out.append((d, 0.0))
                continue
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            out.append((d, var))
        except Exception:  # noqa: BLE001
            out.append((d, 0.0))
    return out


async def lock_focus() -> tuple[float | None, float | None]:
    """Manual-focus sweep: find peak-sharpness distance + freeze.

    Returns ``(best_distance, best_sharpness)`` — caller can stash in
    job logs for visibility. Returns ``(None, None)`` on any failure /
    no camera configured.

    Pavel Khlebovich's stock AF can't be trusted to hold a stable focal
    length across a long capture run. This routine bypasses that by
    going full manual:

      1. ``GET /focus`` — engages the manual-focus lock mode (Pavel
         Khlebovich now ignores AF entirely until ``/nofocus``).
      2. ``POST /settings/focus_distance?set=X`` over a coarse range +
         capture + measure Laplacian-variance sharpness for each. Pick
         the X with the highest sharpness.
      3. Refine around the coarse peak with +/-0.4 in 0.2 steps for
         sub-grid accuracy.
      4. Set the final distance and leave the camera there.

    Total time: ~30 s (9 + 5 captures with 0.45 s settle each). One-time
    cost in exchange for a single stable focal length across every frame.
    """
    base = camera_base()
    if base is None:
        return None, None
    try:
        async with httpx.AsyncClient(timeout=_PHOTOAF_TIMEOUT_S) as client:
            # Step 1: engage manual focus mode.
            await client.get(base + "/focus", timeout=_CONTROL_TIMEOUT_S)
            await asyncio.sleep(0.2)

            # Step 2: coarse sweep — find rough peak.
            coarse = await _sweep_sharpness(client, base, _FOCUS_COARSE_STEPS)
            if not any(v > 0 for _, v in coarse):
                log.warning("focus sweep returned no sharpness data — "
                            "focal length may drift across captures")
                return None, None
            best_d, best_v = max(coarse, key=lambda x: x[1])
            log.info(
                "focus coarse peak: d=%.2f sharp=%.0f  (curve: %s)",
                best_d, best_v,
                ", ".join(f"{d:.1f}→{v:.0f}" for d, v in coarse),
            )

            # Step 3: fine sweep around the coarse peak.
            fine_values = [
                round(best_d + off, 3)
                for off in _FOCUS_FINE_OFFSETS
                if best_d + off > 0
            ]
            fine = await _sweep_sharpness(client, base, fine_values)
            best_d, best_v = max(fine, key=lambda x: x[1])
            log.info(
                "focus fine peak: d=%.2f sharp=%.0f  (curve: %s)",
                best_d, best_v,
                ", ".join(f"{d:.2f}→{v:.0f}" for d, v in fine),
            )

            # Step 4: leave the lens at the best distance.
            await client.post(
                f"{base}/settings/focus_distance?set={best_d:.3f}",
                timeout=_CONTROL_TIMEOUT_S,
            )
            await asyncio.sleep(_FOCUS_SETTLE_S)
        log.info("camera focus locked at distance=%.2f (sharp=%.0f)",
                 best_d, best_v)
        return best_d, best_v
    except Exception as exc:  # noqa: BLE001
        log.warning("lock_focus failed (%s) — focal length may drift", exc)
        return None, None


async def unlock_focus() -> None:
    """Release the focus lock — call after the LAST shot of a sweep.

    Restores normal continuous-AF behaviour so the camera can re-focus
    for subsequent UI previews / single captures / new sessions.
    Idempotent and best-effort.
    """
    base = camera_base()
    if base is None:
        return
    try:
        async with httpx.AsyncClient(timeout=_CONTROL_TIMEOUT_S) as client:
            await client.get(base + "/nofocus")
        log.info("camera focus unlocked")
    except Exception:  # noqa: BLE001
        pass


async def fetch_photo(el_deg: float | None = None) -> bytes:
    """GET a still image at the camera's current (possibly locked) focus.

    Args:
      el_deg: ignored. Kept in the signature for source compatibility with
              callers that still pass it.

    On a capture node: one shot on its locked profile, waited for and pulled
    (raises on failure — a blank stand-in would poison calibration or a scan).

    On IP Webcam: hits ``/photo.jpg`` (no AF trigger) — every shot uses
    whatever focal distance the lens currently holds. Callers that need a
    stable focal length across views must call ``lock_focus()`` first. Falls
    back to PLACEHOLDER on any camera/network error so a sweep job still
    writes valid (if blank) pairs instead of crashing."""
    global _warned_no_camera
    if await is_capture_node():
        shot_id = new_shot_id("photo")
        await trigger_shot(shot_id, {"source": "orbiter-server fetch_photo"})
        jpeg, _record = await fetch_shot(shot_id)
        await release_shot(shot_id)
        return jpeg
    base = camera_base()
    if base is None:
        if not _warned_no_camera:
            log.warning(
                "camera_url not configured — every capture will be a 96×72 "
                "placeholder JPEG (RGB 40,44,52). Set the camera URL in the UI "
                "(Machine config) or via ORBITER_CAMERA_URL to enable real captures."
            )
            _warned_no_camera = True
        return PLACEHOLDER

    try:
        async with httpx.AsyncClient(timeout=_PHOTOAF_TIMEOUT_S) as client:
            resp = await client.get(base + "/photo.jpg")
            resp.raise_for_status()
            return resp.content
    except Exception as exc:  # noqa: BLE001
        log.warning("camera fetch failed (%s) — using placeholder", exc)
        return PLACEHOLDER


async def set_torch(on: bool) -> None:
    """Toggle the phone LED. No-op + warning on failure (never fatal).

    On a capture node the torch is part of the locked profile: the change is
    applied and settled before this returns."""
    base = camera_base()
    if base is None:
        return
    if await is_capture_node():
        try:
            resp = await _client().post(f"{base}/profile", json={"torch": on},
                                        timeout=_PROFILE_TIMEOUT_S)
            if resp.status_code != 200 or not _body(resp).get("settled"):
                log.warning("torch %s: HTTP %d %s", on, resp.status_code, _error(resp))
        except Exception as exc:  # noqa: BLE001
            log.warning("torch %s failed: %s", on, exc)
        return
    endpoint = "/enabletorch" if on else "/disabletorch"
    try:
        async with httpx.AsyncClient(timeout=_CONTROL_TIMEOUT_S) as client:
            resp = await client.get(base + endpoint)
            resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("torch %s failed: %s", endpoint, exc)


# ── CameraStreamer capture node ─────────────────────────────────────────────

#: `POST /shots` returns at "exposed": the HAL queue plus the exposure, well
#: under a second; the phone waits up to 3 s (longer for long exposures) for
#: the camera to settle first.
_SHOT_TIMEOUT_S = 20.0
#: `GET /shots/{id}.jpg` waits on the phone until the file is written.
_FETCH_WAIT_MS = 30_000
_FETCH_TIMEOUT_S = 60.0
_PROFILE_TIMEOUT_S = 10.0
_NODE_APP = "camerastreamer"
#: How long an answer to "is this a capture node?" holds for one URL.
_NODE_CACHE_S = 30.0

_node_cache: dict[str, tuple[bool, float]] = {}
_shot_counter = iter(range(1, 1 << 62))


class ShotError(RuntimeError):
    """A capture-node shot did not happen or could not be fetched.

    ``definitive`` — the phone says the shot itself failed: retrying the SAME
    id returns that failure again, a retry needs a new id. Otherwise (busy,
    not settled, network) the same id is safe to retry: the phone never
    shoots a known id twice and records nothing for a 503."""

    def __init__(self, message: str, definitive: bool = False):
        super().__init__(message)
        self.definitive = definitive


_shared: tuple[asyncio.AbstractEventLoop, httpx.AsyncClient] | None = None


def _client() -> httpx.AsyncClient:
    """One keep-alive client per event loop for every capture-node call: a new
    connection costs the phone a thread and a WiFi round trip (20–70 ms on the
    bench), which would sit on the scan's critical path at every pose."""
    global _shared
    loop = asyncio.get_running_loop()
    if _shared is None or _shared[0] is not loop:
        _shared = (loop, httpx.AsyncClient(timeout=_SHOT_TIMEOUT_S))
    return _shared[1]


async def aclose() -> None:
    """Close the shared client (app shutdown)."""
    global _shared
    if _shared is not None:
        client, _shared = _shared[1], None
        await client.aclose()


def _body(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _error(resp: httpx.Response) -> str:
    return str(_body(resp).get("error") or resp.text[:200])


def _require_base() -> str:
    base = camera_base()
    if base is None:
        raise ShotError("camera_url is not configured")
    return base


def new_shot_id(prefix: str) -> str:
    """A process-unique shot id (the phone accepts ``[A-Za-z0-9_-]``)."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{next(_shot_counter)}"


async def is_capture_node() -> bool:
    """True when ``camera_url`` points at a CameraStreamer capture node.

    Answers are cached per URL for `_NODE_CACHE_S`; an unreachable phone is
    not cached, so the next call asks again."""
    base = camera_base()
    if base is None:
        return False
    hit = _node_cache.get(base)
    now = time.monotonic()
    if hit is not None and now - hit[1] < _NODE_CACHE_S:
        return hit[0]
    try:
        resp = await _client().get(f"{base}/status", timeout=_CONTROL_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — unreachable: no answer to cache
        return False
    is_node = resp.status_code == 200 and _body(resp).get("app") == _NODE_APP
    _node_cache[base] = (is_node, now)
    return is_node


async def require_locked_node() -> dict[str, Any]:
    """The node's ``GET /status``; raises unless it is ready and locked on a
    profile — a scan must never start on auto exposure."""
    base = _require_base()
    resp = await _client().get(f"{base}/status", timeout=_CONTROL_TIMEOUT_S)
    status = _body(resp)
    if resp.status_code != 200:
        raise ShotError(f"phone status: HTTP {resp.status_code} {_error(resp)}")
    if status.get("state") != "ready":
        raise ShotError(f"phone camera is {status.get('state')}: {status.get('last_error')}")
    if status.get("mode") != "locked":
        raise ShotError("phone camera is not locked on a profile — lock it first "
                        "(camerastreamer: POST /lock, or `camctl lock`)")
    return status


async def trigger_shot(shot_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Shoot ``shot_id`` and return once the frame is exposed — the rig may
    move as soon as this returns. The JPEG stays on the phone until
    `fetch_shot`. Raises `ShotError`."""
    base = _require_base()
    try:
        resp = await _client().post(f"{base}/shots",
                                    json={"id": shot_id, "meta": meta, "wait": "exposed"},
                                    timeout=_SHOT_TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise ShotError(f"shot {shot_id}: {exc!r}") from exc
    record = _body(resp)
    if resp.status_code not in (200, 201) or record.get("state") == "failed":
        raise ShotError(f"shot {shot_id}: HTTP {resp.status_code} "
                        f"{record.get('error') or _error(resp)}",
                        definitive=record.get("state") == "failed")
    if record.get("state") not in ("exposed", "stored"):
        raise ShotError(f"shot {shot_id}: not exposed in time (state {record.get('state')})")
    return record


async def fetch_shot(shot_id: str) -> tuple[bytes, dict[str, Any]]:
    """Wait for ``shot_id``'s JPEG on the phone and take it (SHA-256 checked)
    with its final record (the per-frame verification). The phone keeps its
    copy until `release_shot` — call that once the photo is safely stored.
    Safe to retry. Raises `ShotError`."""
    base = _require_base()
    try:
        client = _client()
        resp = await client.get(f"{base}/shots/{shot_id}.jpg",
                                params={"wait_ms": _FETCH_WAIT_MS}, timeout=_FETCH_TIMEOUT_S)
        if resp.status_code != 200:
            raise ShotError(f"fetch {shot_id}: HTTP {resp.status_code} {_error(resp)}",
                            definitive=resp.status_code in (404, 409))
        jpeg = resp.content
        want = resp.headers.get("x-shot-sha256")
        if want and hashlib.sha256(jpeg).hexdigest() != want:
            raise ShotError(f"fetch {shot_id}: sha256 mismatch")
        record = _body(await client.get(f"{base}/shots/{shot_id}", timeout=_CONTROL_TIMEOUT_S))
    except httpx.HTTPError as exc:
        raise ShotError(f"fetch {shot_id}: {exc!r}") from exc
    if record.get("state") == "failed":
        log.warning("shot %s recovered after a phone-side failure (%s): unverified",
                    shot_id, record.get("error"))
    elif record.get("deviations"):
        log.warning("shot %s was not shot as locked: %s", shot_id, "; ".join(record["deviations"]))
    return jpeg, record


async def release_shot(shot_id: str) -> None:
    """Free the phone's copy of a stored shot. Best effort: a shot left behind
    only costs phone storage (it shows up in ``GET /shots?state=stored``)."""
    try:
        resp = await _client().delete(f"{_require_base()}/shots/{shot_id}",
                                      timeout=_CONTROL_TIMEOUT_S)
        if resp.status_code not in (200, 404):
            log.warning("shot %s stays on the phone: delete HTTP %d %s",
                        shot_id, resp.status_code, _error(resp))
    except Exception as exc:  # noqa: BLE001
        log.warning("shot %s stays on the phone: %s", shot_id, exc)


def shot_summary(record: dict[str, Any]) -> dict[str, Any]:
    """What a capture keeps of the phone's shot record: identity, the profile
    it was shot with, and whether the camera reported exactly that."""
    frame = record.get("frame") or {}
    return {
        "id": record.get("id"),
        "state": record.get("state"),
        "error": record.get("error"),
        "profile_fingerprint": record.get("profile_fingerprint"),
        "verified": record.get("verified"),
        "deviations": record.get("deviations") or [],
        "sensor_timestamp_ns": frame.get("sensor_timestamp_ns"),
        "exposure_ns": frame.get("exposure_ns"),
        "iso": frame.get("iso"),
        "focus_diopters": frame.get("focus_diopters"),
        "physical_camera_id": frame.get("physical_camera_id"),
        "latency_ms": record.get("latency_ms"),
    }
