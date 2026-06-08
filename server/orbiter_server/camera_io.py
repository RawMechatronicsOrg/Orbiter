"""Camera HTTP I/O.

Targets the IP Webcam (Pavel Khlebovich) Android app on `settings.camera_url`.
Only the `scheme://host:port` of that URL matters — any path is stripped and
we hit the documented endpoints directly:

  * `/photoaf.jpg`  — trigger autofocus + capture (slow, sharp)
  * `/photo.jpg`    — capture with whatever focus is currently set (fast)
  * `/focus`        — fire single-shot AF (no photo)
  * `/nofocus`      — freeze the current focus distance (locks AF result)
  * `/enabletorch` / `/disabletorch` — camera LED
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Iterable
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

    Hits ``/photo.jpg`` (no AF trigger) — every shot uses whatever focal
    distance the lens currently holds. Callers that need a stable focal
    length across views must call ``lock_focus()`` first.

    Falls back to PLACEHOLDER on any camera/network error so a sweep job
    still writes valid (if blank) pairs instead of crashing."""
    global _warned_no_camera
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
    """Toggle the phone LED. No-op + warning on failure (never fatal)."""
    base = camera_base()
    if base is None:
        return
    endpoint = "/enabletorch" if on else "/disabletorch"
    try:
        async with httpx.AsyncClient(timeout=_CONTROL_TIMEOUT_S) as client:
            resp = await client.get(base + endpoint)
            resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("torch %s failed: %s", endpoint, exc)
