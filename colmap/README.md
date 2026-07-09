# Orbiter — COLMAP container

A thin layer on top of the official **`colmap/colmap:latest`** image that
makes it trivial to run a full SfM + MVS reconstruction on a single
Orbiter scan session, using the per-image poses from the scan as priors.

```
colmap/
├── Dockerfile                  ← base image + Python + scripts
├── run_colmap_session.sh       ← end-to-end wrapper (CPU/GPU)
├── sfm_priors_to_colmap.py     ← sfm_priors.json -> COLMAP text model
├── masks/                      ← object-mask tool (own README)
├── .dockerignore
└── README.md                   ← this file
```

## What's in the container

| Tool                          | Where                                            |
|-------------------------------|--------------------------------------------------|
| `colmap` (CUDA build)         | inherited from `colmap/colmap:latest`            |
| `python3` + OpenCV + NumPy    | `apt` / `pip` packages                           |
| `run_colmap_session.sh`       | `/usr/local/bin/run_colmap_session.sh` (on PATH) |
| `sfm_priors_to_colmap.py`     | `/usr/local/bin/sfm_priors_to_colmap.py` (on PATH)|
| `generate_colmap_masks`       | `/usr/local/bin/…` (on PATH) → `/opt/orbiter/masks` |

### Why the official COLMAP image as the base

- Upstream maintains it; new COLMAP releases drop in by changing the tag.
- The CUDA runtime is already wired up, so GPU SIFT and PatchMatch work
  out of the box when the host has the NVIDIA Container Toolkit.
- Saves us from reproducing a ~30-line `apt` install of OpenGL, Ceres,
  FreeImage, glog/gflags, etc.
- Fallback for CPU-only hosts is built into the same binary — no
  separate image.

If `colmap/colmap:latest` is ever pulled or breaks, alternatives include
`colmap/colmap:3.9.1` (or whichever tag is current at the time you read
this) — pin the digest in production.

## Running it

### Via Compose (recommended)

```bash
cd docker

# CPU SIFT (default)
docker compose --profile colmap run --rm colmap \
    run_colmap_session.sh <session-id>

# GPU SIFT (needs NVIDIA passthrough — see ../docker/README.md)
docker compose --profile colmap run --rm colmap \
    run_colmap_session.sh <session-id> --gpu

# Dry-run: print the command plan without touching anything
docker compose --profile colmap run --rm colmap \
    run_colmap_session.sh <session-id> --dry-run
```

### Standalone (no Compose)

```bash
# Build once
docker build -t colmap-orbiter colmap/

# Run
docker run --rm \
    -v "$(pwd)/data:/data" \
    colmap-orbiter \
    run_colmap_session.sh <session-id>

# With GPU (Linux + NVIDIA Container Toolkit)
docker run --rm --gpus all \
    -v "$(pwd)/data:/data" \
    colmap-orbiter \
    run_colmap_session.sh <session-id> --gpu
```

### Interactive shell

`docker compose --profile colmap run --rm colmap` (no command) drops
you into bash with `/data` mounted. Useful for one-off `colmap gui` or
running individual steps by hand.

## Pipeline

The wrapper executes seven steps, halting on the first error:

| # | Step                              | Notes                                                  |
|---|-----------------------------------|--------------------------------------------------------|
| 1 | `colmap feature_extractor`        | Camera model/params pinned from the priors; adds `--ImageReader.mask_path` automatically when the session has `masks/` (see `masks/README.md`); `--gpu` enables `FeatureExtraction.use_gpu`. |
| 2 | `colmap exhaustive_matcher`       | Pairwise match across all images.                      |
| 3 | `sfm_priors_to_colmap.py --database` | Priors JSON -> COLMAP text model, image/frame ids synced to the database; also fills `pose_priors` and emits `rigs.txt`/`frames.txt` (COLMAP >= 3.13). |
| 4 | `colmap point_triangulator`       | Uses the prior sparse as input, triangulates points.   |
| 5 | `colmap image_undistorter`        | Sparse -> dense workspace, undistorted images.         |
| 6 | `colmap patch_match_stereo`       | Per-image depth maps (slow; benefits hugely from GPU). |
| 7 | `colmap stereo_fusion`            | Depth maps -> single fused `.ply` point cloud.         |

Each step prints `[PROFILE]` start/end markers with durations. Frames are
read from the session's `photos/` dir — `POST /scans/<sid>/sfm_priors`
materializes it alongside the priors.

### CPU vs GPU notes

- **Feature extraction** — CPU is fine for 10–100 images; GPU is ~5–10×
  faster on a discrete NVIDIA card. The script defaults to CPU SIFT to
  stay portable across hosts.
- **PatchMatch stereo** — this is the slow stage. On CPU it can be tens
  of minutes per image; on GPU it's seconds per image. If you have an
  NVIDIA GPU available, pass `--gpu` (and check the deploy stanza in
  `../docker/docker-compose.yml`).
- **Matching** — `exhaustive_matcher` accepts the same `use_gpu` flag.
  We pass the same value the user set on `--gpu`.

## Inputs and outputs

```
/data/scans/<sid>/
├── sfm_priors.json                          (input, written by the server)
├── photos/001_az000_el+15.jpg, ...          (input, materialized by the export)
├── masks/                                   (optional — masks/README.md)
└── colmap/                                  (created by this script)
    ├── sparse_priors/
    │   ├── cameras.txt                      (one shared camera, model from priors)
    │   ├── images.txt                       (priors quat/translation per image)
    │   ├── rigs.txt, frames.txt             (COLMAP >= 3.13 bindings)
    │   └── points3D.txt                     (empty)
    ├── database.db                          (SIFT features + matches)
    ├── sparse/0/                            (triangulated sparse model)
    └── dense/
        ├── images/                          (undistorted source images)
        ├── sparse/                          (undistorted sparse)
        ├── stereo/                          (depth + normal maps)
        └── fused.ply                        (FINAL — open in MeshLab / CloudCompare)
```

## Sample failure modes

| Symptom                                                   | Likely cause                                                                   | Fix                                                                                                  |
|-----------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `sfm_priors.json missing` immediately                     | Priors weren't exported.                                                       | UI → Library → session → Export → SfM priors. Or call the server's export endpoint.                  |
| `feature_extractor` aborts with `cudaError`               | `--gpu` was set but the host has no usable NVIDIA GPU.                         | Re-run without `--gpu`, or fix host GPU passthrough (see `../docker/README.md`).                     |
| `point_triangulator` registers 0 images                   | Image filenames in priors don't match files on disk.                           | Check the `file` fields in `sfm_priors.json` vs the actual layout under the session dir.             |
| `patch_match_stereo` runs forever on CPU                  | Expected behaviour — it's the slow stage.                                      | Use `--gpu`, or reduce image count, or accept the wait.                                              |
| `stereo_fusion` produces a near-empty PLY                 | Too few overlapping views, or noisy priors.                                    | Add more shots, especially with small angular gaps. Verify priors with `--dry-run` and a manual run. |
| Out-of-memory mid-PatchMatch                              | Docker Desktop's RAM cap is too low.                                           | Bump RAM in Docker Desktop → Settings → Resources, or downsample input images.                       |
| `permission denied` writing to `/data/scans/<sid>/colmap` | Host-side `data/` was created by a different UID and the container runs as root| Either `chown` the host dir, or wipe and let the container recreate it.                              |

## See also

- [`OrbiterV0.1/docs/COLMAP.md`](../docs/COLMAP.md) — the canonical
  description of `sfm_priors.json` and the manual COLMAP recipe.
- [`OrbiterV0.1/docker/README.md`](../docker/README.md) — Compose
  stack overview, GPU passthrough notes.
