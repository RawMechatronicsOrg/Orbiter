# Orbiter v0.1 UI

Slim Vite + React + three.js front-end for the Orbiter photogrammetry rig.
This is the v0.1 slice — only the two tabs the operator needs to drive a
camera-only photogrammetry pass:

- **Scaner** — live 3D view of the rig, motion controls (jog, absolute
  move, navball), motion-planner (discrete/continuous AZ-EL sweep), scan
  session controls (new / save / take shot / start scan / stop).
- **Library** — list of saved scans with their captured photos, plus
  per-scan "Export SfM priors" (POST `/scans/{sid}/sfm_priors`) and a
  disabled "Run COLMAP" button reserved for v0.2.

What was intentionally removed from the upstream `ui/` codebase: the live
laser-scan panel, the laser-scans Library section, and the laser-points
overlay. The ChArUco-based geometry calibration was kept (Machine config
panel → Calibrate from board).

## How it talks to the server

One persistent WebSocket to the storage-api (`ws://server:8000/ws/scene`)
drives everything in the Scaner tab: server pushes `scene_snapshot` +
`scene_update` diffs (rendered by `nodeRegistry.ts`) and model state;
browser pushes `command` messages (one channel for mutating state). The
Library tab is REST: `GET /scans`, `GET /config`,
`POST /scans/{sid}/sfm_priors`, `DELETE /scans/{sid}`.

In dev (`npm run dev`) Vite proxies `/ws`, `/scans`, `/captures`, `/config`
to `http://127.0.0.1:8000` so the browser sees a same-origin API. In
production the bundle hits `http://<host>:8000` directly — same-origin
deployments are best done with an nginx reverse proxy in front of both.

## Development

```bash
npm install
npm run dev
# open http://localhost:5173
```

The storage-api must be running on `localhost:8000` for the proxy to
resolve. See `../server/README.md` (owned by the server agent).

Type-check the tree without building:

```bash
npx tsc --noEmit
```

## Production build (Docker)

```bash
docker build -t orbiter-v0.1-ui .
docker run --rm -p 8080:80 orbiter-v0.1-ui
# open http://localhost:8080
```

Two-stage `Dockerfile`: `node:20-alpine` runs `npm ci` + `npm run build`,
then `nginx:alpine` serves the static `dist/` on port 80. The default
nginx config is enough — no custom routing needed for the SPA.

## COLMAP integration

Not in v0.1. Each scan row in the Library tab has a placeholder "Run
COLMAP" button that's disabled with a tooltip pointing to v0.2. The
adjacent "Export SfM priors" button does work — it POSTs to
`/scans/{sid}/sfm_priors` on the storage-api, which writes a JSON sidecar
of per-capture poses suitable for feeding into COLMAP's exhaustive
matcher / point triangulator as known camera extrinsics.

The v0.2 work plan is: invoke `colmap` from a sibling Docker service
against a loaded scan + its priors, stream progress over the same
WebSocket, and surface the resulting sparse / dense reconstruction back
into the scene as a `point_cloud` node.
