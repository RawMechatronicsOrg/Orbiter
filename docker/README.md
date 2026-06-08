# Orbiter v0.1 — Docker stack

This directory holds the Compose configuration that ties the **server**,
**UI**, and the optional **COLMAP** container together.

```
docker/
├── docker-compose.yml   ← service definitions
├── .env.example         ← env template (copy to .env)
├── .gitignore           ← keeps .env and data/ out of git
└── README.md            ← this file
```

The service images are built from sibling directories:

| Service | Build context | Image tag           | Host port |
|---------|---------------|---------------------|-----------|
| server  | `../server`   | `orbiter/server:dev`| `8000`    |
| ui      | `../ui`       | `orbiter/ui:dev`    | `5173`    |
| colmap  | `../colmap`   | `orbiter/colmap:dev`| —         |

## First-time setup

```bash
cd OrbiterV0.1/docker
cp .env.example .env
# edit .env to set ESP32_IP and CAMERA_URL
```

Then start the stack:

```bash
# Full app — server + UI
docker compose up

# Background
docker compose up -d
```

Open the UI at <http://localhost:5173>. The UI proxies API calls to the
server at <http://localhost:8000>.

### Server-only or UI-only

```bash
docker compose up server
docker compose up ui server      # ui depends_on server, so this works too
```

### COLMAP (opt-in profile)

The `colmap` service has the `colmap` profile, so it stays out of the
default `up`. Bring it in explicitly:

```bash
# One-off reconstruction for session <sid>
docker compose --profile colmap run --rm colmap run_colmap_session.sh <sid>

# Or keep a long-running shell to poke around
docker compose --profile colmap run --rm colmap

# Build the image without running it
docker compose --profile colmap build colmap
```

See [`../colmap/README.md`](../colmap/README.md) for the pipeline details.

## Where data lives

Everything the server reads or writes lives under `./data` on the host
(mounted as `/data` in both the `server` and `colmap` containers). The
server uses `ORBITER_STORAGE_DIR=/data`. Scan sessions land under
`./data/scans/<session_id>/`.

The `data/` directory is **gitignored** — wipe it freely to start clean.

## Common gotchas

### Windows / Docker Desktop

- The compose file uses **relative bind mounts** (`./data:/data`). On
  Docker Desktop for Windows this works in both PowerShell and WSL2
  shells, but only if the `data/` folder is on a drive Docker has been
  granted access to (Settings → Resources → File Sharing).
- Backslash paths (`C:\Users\...`) in `.env` need to be quoted or escaped
  if you put any. Prefer forward slashes.
- The server's MJPEG/HTTP camera fetch goes to `CAMERA_URL` *as seen
  from inside the container*. `localhost` from inside a container is the
  container, not your phone — use the LAN IP of the phone instead.

### Resource limits

Docker Desktop defaults to a small CPU/RAM budget. For a serious COLMAP
run, bump it: Settings → Resources → Advanced (CPUs ≥ 4, Memory ≥ 8 GB
recommended for ~50-image sessions; more for larger ones).

### GPU passthrough for COLMAP

The `deploy.resources.reservations.devices` stanza requests an NVIDIA
GPU. It only does anything if:

- **Linux native** — the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  is installed.
- **Windows + WSL2** — Docker Desktop's WSL2 backend with NVIDIA drivers
  exposed; see Docker Desktop's GPU support docs.
- **macOS** — no NVIDIA GPU passthrough; COLMAP runs CPU-only.

If the GPU isn't available, the `run_colmap_session.sh` script falls
back to CPU SIFT extraction automatically (see `--gpu` flag).

### Rebuilding after code changes

The `server` and `ui` images cache aggressively. After a code change in
those subprojects:

```bash
docker compose build server
docker compose up server
```

### Logs

```bash
docker compose logs -f server
docker compose logs -f ui
```

## Stopping and cleaning up

```bash
docker compose down              # stop containers, keep volumes
docker compose down -v           # also drop named volumes (no effect — we use bind mounts)
rm -rf ./data                    # nuke storage (irreversible)
```
