"""Orbiter server — FastAPI app.

Run locally:
    uvicorn orbiter_server.app:app --reload --port 8000

The app brings up:
  * the ESP32 proxy (REST + WebSocket to the firmware),
  * the WebSocket hub (broadcasts scene + model patches to browsers),
  * the live camera MJPEG stream,
  * the scan-task autosave loop.

v0.1 deliberately excludes the live triangulator and the photogrammetry
job orchestration found in the parent storage-api. The ChArUco hand-eye
geometry calibration *is* included here (see calibration.py and the
`calibrate_geometry` command).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import camera_adapter
import discovery
import scan_task
from camera_stream import stream as camera_stream
from config import settings
from esp_proxy import esp
from orbiter_model import PERSISTED_FIELDS, model
from phone_sensor import phone_sensor
from routes import captures, photos, scans, stream, ws
from ws_hub import hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# The ESP proxy polls /state ~4 Hz — silence httpx's per-request INFO line.
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: bring up the WS hub, the ESP32 proxy and the camera stream;
    launch the scan autosave loop. Shutdown unwinds in reverse so each
    component's teardown can still talk to its dependencies."""
    hub.start(model)
    scan_task.publish_scan_list()   # populate model.scans so the Library lists
                                    # already-saved scans on first connect
    esp.on_log = hub.emit_log   # firmware log frames -> /ws/scene `log` messages
    await esp.start()
    await discovery.start()     # mDNS browser; respects model.esp_autodiscover
    await phone_sensor.start()  # polls camera_url/sensors.json for IMU tilt
    scan_task.start_autosave()
    await camera_stream.start()
    try:
        yield
    finally:
        await scan_task.stop_autosave()
        await camera_stream.stop()
        await phone_sensor.stop()
        await discovery.stop()
        await esp.stop()
        await hub.stop()
        camera_adapter.shutdown_thumb_pool()


app = FastAPI(
    title="Orbiter Server",
    description=(
        "FastAPI service for the 2-axis Orbiter turntable: ESP32 proxy, "
        "scan-session manifests, photo storage, live camera adapter, "
        "WebSocket scene graph."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scans.router)
app.include_router(photos.router)
app.include_router(captures.router)
app.include_router(stream.router)
app.include_router(ws.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "service": "orbiter-server",
        "storage_dir": str(settings.storage_dir.resolve()),
        "captures_dir": str(settings.captures_dir.resolve()),
    }


@app.get("/debug/model", tags=["meta"])
def debug_model() -> dict:
    """Read-only snapshot of the authoritative model state. Mutations go
    through the WS command channel."""
    return model.to_dict()


@app.get("/config", tags=["meta"])
def get_config() -> dict:
    """The machine configuration — rig build params, camera preset, render
    preferences (the persisted, config-like model fields)."""
    return {k: getattr(model, k) for k in sorted(PERSISTED_FIELDS)}


def main() -> None:
    """Console-script entry point: run uvicorn with this app."""
    import uvicorn

    uvicorn.run(
        "orbiter_server.app:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
