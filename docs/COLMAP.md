# Using a session with COLMAP

A scan session is just a folder of photos with poses. That makes it a
near-perfect input for [COLMAP](https://colmap.github.io/) — you can hand
COLMAP your photos *and* the camera positions, and it will skip pose
estimation and go straight to dense reconstruction.

This guide assumes you have a session saved by Orbiter (see
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the data model).

## Option A — Containerised COLMAP

The easiest path. Use the [`colmap/`](../colmap/) container we ship.

```bash
cd docker
docker compose up colmap
```

From the UI, in the **Library** tab → pick a session → **Export → COLMAP
priors**. Then in the same panel: **Run COLMAP**.

The container mounts the storage directory read-only, reads the session
manifest, builds the COLMAP database + priors, and runs the reconstruction.
Output ends up in `<storage>/scans/<session_id>/colmap/`.

Progress is streamed back to the UI. Hit cancel and the container is
killed.

## Option B — Hand-off to your own COLMAP

If you'd rather drive COLMAP yourself, ask Orbiter to write the priors
file and walk in by hand.

From the UI: **Library → session → Export → SfM priors** writes
`<storage>/scans/<session_id>/sfm_priors.json`:

```jsonc
{
  "schema": "orbiter.sfm_priors.v1",
  "camera_intrinsics": {
    "model": "PINHOLE",
    "width":  1920,
    "height": 1080,
    "fx": 1500,
    "fy": 1500,
    "cx":  960,
    "cy":  540
  },
  "images": [
    {
      "file": "c_001/photo.jpg",
      "qw":  0.707, "qx": 0, "qy": 0.707, "qz": 0,
      "tx":   220, "ty": 0,  "tz":  45
    }
    /* ... */
  ]
}
```

Quaternions are **Hamilton** convention (w, x, y, z). Translations are in
**millimetres** in the world frame defined in
[`ARCHITECTURE.md`](ARCHITECTURE.md). The transform takes world points
into camera space (COLMAP's convention).

Camera intrinsics are guessed from the IP Webcam stream by default — if
you've calibrated your phone separately, override them via the **Camera
config** panel before exporting.

To use the priors in your COLMAP run:

```bash
colmap feature_extractor \
    --database_path session.db \
    --image_path photos/

colmap exhaustive_matcher --database_path session.db

# load priors as a fixed sparse model, then triangulate / refine
colmap point_triangulator \
    --database_path session.db \
    --image_path photos/ \
    --input_path priors_sparse/ \
    --output_path sparse/0

colmap image_undistorter \
    --image_path photos/ \
    --input_path sparse/0 \
    --output_path dense/

colmap patch_match_stereo  --workspace_path dense/
colmap stereo_fusion        --workspace_path dense/ \
                            --output_path dense/fused.ply
```

The conversion from `sfm_priors.json` to COLMAP's `cameras.txt` /
`images.txt` is a one-liner in Python — see
[`server/orbiter_server/sfm_export.py`](../server/orbiter_server/sfm_export.py)
for our implementation.

## How accurate are the priors?

It depends on how carefully you measured `arm_radius`, `base_height`,
`camera_offset` and how accurate your encoders are. A reasonable build
with a calliper-measured arm and AS5600 + AS5048A encoders should give:

| Quantity | Typical |
|----------|---------|
| Per-shot rotation error | 0.5° – 1.5° |
| Per-shot position error | 2 – 10 mm |

That's not enough for "feature-free" reconstruction, but it's plenty as a
warm start. COLMAP's bundle adjustment will polish them.

If you want **better** priors, the parent repo (`Orbiter/`) has a ChArUco
hand-eye calibration that gets the per-shot error down to ~10 mm / ~0.5°
after solving. That's not part of v0.1.

## What this is *not* good for

- **Single-image NeRF** — the priors are good enough for COLMAP to seed,
  not good enough as ground truth for NeRF/Gaussian-splat training.
- **Metric reconstruction without a calibrated intrinsic** — bring your
  own intrinsics if you care about absolute scale.
