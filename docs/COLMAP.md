# Using a session with COLMAP

A scan session is just a folder of photos with poses. That makes it a
near-perfect input for [COLMAP](https://colmap.github.io/) — you can hand
COLMAP your photos *and* the camera positions, and it will skip pose
estimation and go straight to dense reconstruction.

This guide assumes you have a session saved by Orbiter (see
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the data model).

## Option A — Containerised COLMAP

The easiest path. Use the [`colmap/`](../colmap/) container we ship.

From the UI, in the **Library** tab → pick a session → **Export → SfM
priors**. That writes `sfm_priors.json` *and* materializes the capture
originals into `scans/<session_id>/photos/` — the layout the runner reads.
Then:

```bash
cd docker
docker compose --profile colmap run --rm colmap \
    run_colmap_session.sh <session_id>          # add --gpu for CUDA
```

The runner pins the COLMAP camera model to the priors intrinsics, picks up
object masks automatically when `scans/<session_id>/masks/` exists (see
[`colmap/masks/README.md`](../colmap/masks/README.md)), syncs the priors
into the COLMAP database (`pose_priors`, `rigs.txt`/`frames.txt`), and runs
triangulation + dense reconstruction. Output ends up in
`<storage>/scans/<session_id>/colmap/`.

## Option B — Hand-off to your own COLMAP

If you'd rather drive COLMAP yourself, ask Orbiter to write the priors
file and walk in by hand.

From the UI: **Library → session → Export → SfM priors** writes
`<storage>/scans/<session_id>/sfm_priors.json`:

```jsonc
{
  "schema": "orbiter.sfm_priors.v1",
  "camera_intrinsics": {
    "model": "PINHOLE",              // "OPENCV" once calibrated with distortion
    "width":  4080,                  // stored-photo pixels
    "height": 3060,
    "fx": 3122.9,
    "fy": 3122.1,
    "cx": 2039.5,
    "cy": 1529.5
  },
  "images": [
    {
      "file": "photos/001_az000_el+15.jpg",
      "qw":  0.707, "qx": 0, "qy": 0.707, "qz": 0,
      "tx":   220, "ty": 0,  "tz":  45
    }
    /* ... */
  ],
  "turntable": {                     // present once the rig is calibrated
    "axis_xy_mm": [1.8, 0.0],
    "board": { "rvec": [/*…*/], "t": [/*…*/],
               "width_mm": 288, "height_mm": 288 }
  }
}
```

Quaternions are **Hamilton** convention (w, x, y, z). Translations are in
**millimetres** in the world frame defined in
[`ARCHITECTURE.md`](ARCHITECTURE.md). The transform takes world points
into camera space (COLMAP's convention).

Once the rig has been ChArUco-calibrated, the exporter embeds the solved
intrinsics (rotated to match the stored-pixel orientation) instead of the
IP-Webcam guess — if you've calibrated your phone separately, override via
the **Camera config** panel before exporting.

To use the priors in your COLMAP run
([`colmap/sfm_priors_to_colmap.py`](../colmap/sfm_priors_to_colmap.py) does
the JSON → text-model conversion):

```bash
# Pin the database camera to the priors intrinsics — without this COLMAP
# defaults to SIMPLE_RADIAL and point_triangulator aborts on the mismatch.
CAM_FLAGS=$(python3 sfm_priors_to_colmap.py --emit-extractor-flags sfm_priors.json)

colmap feature_extractor \
    --database_path session.db \
    --image_path photos/ \
    --ImageReader.single_camera=1 \
    $CAM_FLAGS
    # add --ImageReader.mask_path masks/ if you generated object masks

colmap exhaustive_matcher --database_path session.db

# priors -> text model, image/frame ids synced to the database (also fills
# COLMAP's pose_priors table and emits rigs.txt/frames.txt for COLMAP >= 3.13)
python3 sfm_priors_to_colmap.py sfm_priors.json priors_sparse/ --database session.db

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

If you want **better** priors, run the ChArUco hand-eye calibration that
ships with the kit (**Machine config → Calibrate from board**) — it refines
the machine geometry, solves the camera intrinsics from the same photos and
gets the per-shot error down to ~10 mm / ~0.5°.

## What this is *not* good for

- **Single-image NeRF** — the priors are good enough for COLMAP to seed,
  not good enough as ground truth for NeRF/Gaussian-splat training.
- **Metric reconstruction without a calibrated intrinsic** — bring your
  own intrinsics if you care about absolute scale.
