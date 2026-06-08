# orbiter-server

FastAPI service for the two-axis Orbiter turntable.

This is the **v0.1** server. Compared to the parent repo's `storage-api`
it intentionally omits the laser triangulator and the photogrammetry job
orchestration — what remains is the minimum required to drive the rig,
capture (manual or motion-planned) photo sessions, and run ChArUco
hand-eye geometry calibration.

## What it does

- Proxies the ESP32 firmware (`/move`, `/calibrate`, `/motors`,
  `/reboot`) over REST + a `/ws/log` WebSocket for live pose / task /
  log frames.
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
| `GET`  | `/camera/stream/status` | Camera connection status. |
| `WS`   | `/ws/scene` | Scene + model diffs, command channel. |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORBITER_STORAGE_DIR` | `./data` | Root for `scans/` + `captures/`. |
| `ORBITER_ESP_IP` | `192.168.1.50` | ESP32 firmware host. |
| `ORBITER_CAMERA_URL` | empty | IP-Webcam HTTP base URL (e.g. `http://phone-ip:8080`). |
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

## Out of scope for v0.1

The parent repo's `storage-api` carries a live laser-stripe triangulator
and a photogrammetry job orchestrator. Neither is shipped here. The v0.1
slice does ship a ChArUco hand-eye solver (`calibration.py`) so the rig
geometry can be derived from a calibration board rather than measured
with calipers and entered by hand. The SfM-priors exporter writes
COLMAP-ready poses from that derived geometry; refining them further is
a downstream COLMAP-side step.
