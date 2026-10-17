"""Firmware diagnostics and updates — the UI's way to the ESP32 itself.

The browser never talks to the firmware (Viser pattern); these routes proxy
the firmware's own read-only endpoints and stream an update image to it:

  * `GET  /firmware/health`  — firmware `GET /health` (chip, IDF, uptime, heap)
  * `GET  /firmware/encoder` — firmware `GET /test/encoder` (per-axis AS5600
    reading, magnet status, idle levels of the I2C lines)
  * `GET  /firmware/motor`   — firmware `GET /motor/calibration` (the motor
    self-calibration run in flight and each axis' stored result)
  * `POST /firmware/ota`     — body = the raw app image (`build/orbiter.bin`),
    streamed to firmware `POST /ota`; its status code and message come back
    unchanged.

An update mutates the device, not the server model, and its payload is a
binary file, so it is a route rather than a WS command.
"""

from __future__ import annotations

import logging
import struct
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from esp_proxy import EspBusy, EspError, esp
from orbiter_model import model

log = logging.getLogger("orbiter.routes.firmware")

router = APIRouter(prefix="/firmware", tags=["firmware"])

#: First byte of every ESP32 app image (esp_image_header_t.magic).
_ESP_IMAGE_MAGIC = 0xE9
#: Bigger than either OTA slot (1.875 MiB on the 4 MB layout); refuse before
#: the upload ties up the device's HTTP server.
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
#: esp_app_desc_t follows the image header (24 bytes) and the first segment
#: header (8 bytes) of every ESP-IDF app image.
_APP_DESC_OFFSET = 32
_APP_DESC_MAGIC = 0xABCD5432


def app_description(image: bytes) -> dict[str, str] | None:
    """The app descriptor baked into an ESP-IDF app image (version, project,
    build time, IDF, ELF hash), or None when there is none — a bootloader or
    partition-table image also starts with 0xE9 but carries no descriptor."""
    off = _APP_DESC_OFFSET
    if len(image) < off + 176 or struct.unpack_from("<I", image, off)[0] != _APP_DESC_MAGIC:
        return None

    def text(start: int, size: int) -> str:
        raw = image[off + start: off + start + size]
        return raw.split(b"\0", 1)[0].decode("ascii", "replace")

    return {
        "version": text(16, 32),
        "project": text(48, 32),
        "built": f"{text(96, 16)} {text(80, 16)}",
        "idf": text(112, 32),
        "elf_sha256": image[off + 144: off + 176].hex()[:16],
    }


async def _proxy_get(path: str) -> dict[str, Any]:
    try:
        return await esp.get_json(path)
    except EspBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EspError as exc:
        raise HTTPException(status_code=502, detail=f"firmware not reachable: {exc}") from exc


@router.get("/health")
async def firmware_health() -> dict[str, Any]:
    """The firmware's liveness/identity record, plus the address it answered on."""
    data = await _proxy_get("/health")
    return {**data, "esp_ip": esp.base_url.removeprefix("http://")}


@router.get("/encoder")
async def firmware_encoder() -> dict[str, Any]:
    """Per-axis encoder diagnostics: `{azimuth, elevation}`, each `{ok, raw,
    angle_deg, magnet_missing, magnet_weak, magnet_strong, agc, magnitude}`
    (the reading only when `ok`) and `lines: {sda, scl, external_pullups}` —
    1 high, 0 LOW, -1 unknown."""
    return await _proxy_get("/test/encoder")


@router.get("/motor")
async def firmware_motor() -> dict[str, Any]:
    """The motor self-calibration: `run` {active, axis, phase, progress,
    speed_dps, message} and per axis {valid, steps_per_deg, max_speed_dps,
    dir_sign, compensation, profile_*...}. Mirrored into `model.motor_cal`
    so the axes' steps/° follow it without waiting for a /state read."""
    data = await _proxy_get("/motor/calibration")
    esp.apply_motor_calibration(data)
    return data


@router.post("/ota")
async def firmware_ota(request: Request) -> JSONResponse:
    """Stream a firmware image to the device. On 200 it reboots into it; the
    new image confirms itself once its HTTP server is up, else the bootloader
    rolls back after 120 s (poll `/firmware/health` to see which). The
    response adds `image`: the app descriptor of what was sent."""
    busy = model.busy_reason()
    if busy:
        raise HTTPException(status_code=409, detail=busy)
    image = await request.body()
    if not image:
        raise HTTPException(status_code=400, detail="empty body — send build/orbiter.bin")
    if len(image) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{len(image)} bytes is larger than any OTA slot")
    desc = app_description(image) if image[0] == _ESP_IMAGE_MAGIC else None
    if desc is None:
        raise HTTPException(
            status_code=400,
            detail="not an ESP32 app image — send the .bin from firmware/build/, "
                   "not the bootloader, partition table or an .elf",
        )
    log.warning("firmware update: %d bytes → %s/ota", len(image), esp.base_url)
    try:
        status, data = await esp.ota(image)
    except EspError as exc:
        raise HTTPException(status_code=502, detail=f"firmware not reachable: {exc}") from exc
    log.warning("firmware update (%s %s): HTTP %d %s",
                desc["project"], desc["version"], status, data)
    return JSONResponse(status_code=status, content={**data, "image": desc})
