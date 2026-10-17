# orbiter-server

FastAPI service for the two-axis Orbiter turntable.

Compared to the parent research rig it intentionally omits the laser
triangulator and the photogrammetry job orchestration — what remains is
the minimum required to drive the rig, capture (manual or motion-planned)
photo sessions, and run ChArUco hand-eye geometry calibration.

## What it does

- Proxies the ESP32 firmware (`/move`, `/calibrate`, `/motors`, `/spin`,
  `/reboot`, `/ota`, `/health`, `/test/encoder`) over REST + a `/ws/log`
  WebSocket for live pose / task / log frames.
- Owns the authoritative model state (`ModelState`) and broadcasts
  scene-graph + model diffs over `/ws/scene`.
- Stores scan sessions on disk as a single JSON manifest per scan plus a
  global capture pool (`captures/<id>/{original,thumb*}.jpg`).
- Re-multiplexes the phone's MJPEG stream as
  `GET /camera/stream.mjpeg` for the browser preview.
- Exports per-photo COLMAP priors via `POST /scans/{sid}/sfm_priors`.

## Running locally

```bash
cd OrbiterV0.1/server
pip install -e .
uvicorn orbiter_server.app:app --reload --port 8000
```

`python -c "import orbiter_server.app"` (or the console entry
`orbiter-server`) is enough to smoke-test the import graph.

## Running in Docker

```bash
docker build -t orbiter-server .
docker run --rm -p 8000:8000 \
    -e ORBITER_ESP_IP=192.168.1.50 \
    -e ORBITER_CAMERA_URL=http://192.168.1.51:8080 \
    -v $(pwd)/data:/data \
    orbiter-server
```

## Routes

| Method | Path | Notes |
|--------|------|-------|
| `GET`  | `/health` | Service heartbeat. |
| `GET`  | `/debug/model` | Read-only snapshot of the full model. |
| `GET`  | `/config` | The persisted, config-like model fields. |
| `GET`  | `/scans` | List stored scan summaries. |
| `POST` | `/scans` | Create a scan manifest. |
| `GET`  | `/scans/{sid}` | Read a manifest. |
| `GET`  | `/scans/{sid}/download` | Zip-archive of the scan + photos. |
| `POST` | `/scans/{sid}/archive` | Mark a scan archived (kept on disk). |
| `DELETE` | `/scans/{sid}` | Remove a scan manifest. |
| `POST` | `/scans/{sid}/sfm_priors` | Write `sfm_priors.json` (see `docs/COLMAP.md`). |
| `POST` | `/scans/{sid}/photos` | Upload a photo for a scan. |
| `GET`  | `/scans/{sid}/photos/{idx}/...` | Thumbnails / full / meta. |
| `GET`  | `/captures/{cid}/...` | Capture pool (immutable). |
| `GET`  | `/camera/stream.mjpeg` | Live MJPEG preview. |
| `GET`  | `/camera/stream/status` | Camera connection status: URL, frame size, last error. |
| `POST` | `/camera/node/{lock\|profile\|unlock}` | Capture node only: lock / re-lock / unlock its manual profile. |
| `GET`  | `/firmware/health` | The firmware's `GET /health`, proxied. |
| `GET`  | `/firmware/encoder` | The firmware's `GET /test/encoder`, proxied. |
| `GET`  | `/firmware/motor` | The firmware's `GET /motor/calibration` (motor self-calibration run + per-axis results), proxied and mirrored into `model.motor_cal`. |
| `POST` | `/firmware/ota` | Body = `firmware/build/orbiter.bin`, streamed to the firmware's `POST /ota`; its status is relayed. |
| `GET`  | `/command` | The command names the server accepts. |
| `POST` | `/command/{name}` | Run a command over HTTP (same handlers as the WS channel). |
| `WS`   | `/ws/scene` | Scene + model diffs, command channel. |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORBITER_STORAGE_DIR` | `./data` | Root for `scans/` + `captures/`. |
| `ORBITER_ESP_IP` | `orbiter.local` | ESP32 firmware host (mDNS auto-discovery or the UI override it). |
| `ORBITER_CAMERA_URL` | empty | The phone, `http://phone-ip:8080`: a CameraStreamer capture node (recognised automatically) or IP Webcam. |
| `ORBITER_PORT` | `8000` | Bound port. |
| `ORBITER_CORS_ORIGINS` | localhost:5173/5174 | Comma-separated allowed origins. |
| `ORBITER_DEFAULT_CAMERA_PRESET` | `native` | Pixel-rotation preset for new captures. |

## Data layout

```
<ORBITER_STORAGE_DIR>/
  orbiter_state.json          # persisted ModelState subset
  scans/
    <scan_id>/
      manifest.json           # the scan document
      sfm_priors.json         # written by POST /sfm_priors
  captures/
    <capture_id>/
      original.jpg
      thumb.jpg               # medium tier
      thumb_small.jpg         # sidebar tier
      thumb_tiny.jpg          # in-scene texture tier
      meta.json
```

## Out of scope

The parent research rig carries a live laser-stripe triangulator and a
photogrammetry job orchestrator. Neither is shipped here. The kit does
ship a ChArUco hand-eye solver (`calibration.py`) so the rig geometry —
and, since v0.2, the camera intrinsics, the turntable axis, the EL-axis
tilt and per-view diagnostics — can be derived from calibration photos
rather than measured with calipers and entered by hand. The SfM-priors
exporter writes COLMAP-ready poses from that derived geometry; refining
them further is a downstream COLMAP-side step.
