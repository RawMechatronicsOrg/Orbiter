# Object masks for COLMAP

Generate binary masks for a turntable photoset so COLMAP's feature extractor
matches on what **rotates with the table** — the object, the ChArUco board and
the turntable itself — and ignores the **static room**. Static surroundings
are a feature-matching trap: they line up perfectly between views and pull the
reconstruction toward the *room* instead of the rotating assembly. The board
and table are kept deliberately: they rotate rigidly with the object and give
texture for registration.

Per frame, the calibrated rig geometry provides the prompt: the turntable
working volume (a cylinder around the rotation axis) is projected through the
frame's world→camera pose from `sfm_priors.json` into a box + positive /
negative points, and **SAM 2.1** turns that into a pixel-accurate foreground.
The projected silhouette doubles as a no-ML fallback (`--method geometric`)
and as a clamp so SAM2 can never leak into the room.

```
sfm_priors.json ─┐
                 ├─ geom_prompt.py: project volume → box + points + silhouette
photos/*.jpg ────┤
                 └─ sam2_masker.py: SAM 2.1 predict
                        └─ mask_pipeline.py: union components → fill holes
                           → dilate → fail-open → masks/*.png (white = keep)
```

## Layout

The tool reads and writes next to the scan session — the layout
`POST /scans/<sid>/sfm_priors` stages and `run_colmap_session.sh` consumes:

```
scans/<sid>/
  photos/         001_az000_el+15.jpg …  # materialized frames
  sfm_priors.json                        # per-image poses + intrinsics
->
  masks/          001_az000_el+15.jpg.png    # white = keep, black = suppressed
  masks_preview/  001_az000_el+15_overlay.jpg # mask painted over the frame (debug)
```

`masks/<name>.png` is exactly the name COLMAP expects for `photos/<name>`;
subfolders are mirrored — right where `--ImageReader.mask_path` looks.
`run_colmap_session.sh` adds the flag automatically when `masks/` is present,
and the server serves each frame's mask/overlay at
`/scans/<sid>/photos/<idx>/mask` / `…/mask_preview` for eyeballing.

## Install

The `geometric` and `motion` methods need only OpenCV + NumPy — the
`orbiter/colmap` Docker image ships both, so `--method motion`
(edge-refine falls open when no checkpoint is around) and
`--method geometric` run right in the container. SAM2 — `--method sam2`,
or motion's edge snap — needs torch + the `sam2` package and the weights;
for that host venv:

```bash
# CUDA torch FIRST (RTX 50xx needs cu128+ / torch >= 2.7):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --no-build-isolation
```

### Weights

Download the SAM 2.1 checkpoint once into `models/` (gitignored):

```bash
curl -L -o models/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

The matching hydra config (`configs/sam2.1/sam2.1_hiera_b+.yaml`) ships inside
the installed `sam2` package — no separate download. Alternates: `tiny`/
`small` for CPU, `large` for max quality (pass `--checkpoint` /
`--model-cfg`).

## Run

```bash
python generate_colmap_masks.py \
    --images   ./scans/<sid>/photos \
    --masks    ./scans/<sid>/masks \
    --poses    ./scans/<sid>/sfm_priors.json \
    --preview  ./scans/<sid>/masks_preview \
    --checkpoint ./models/sam2.1_hiera_base_plus.pt
```

| Flag                | Default    | Meaning                                              |
|---------------------|------------|------------------------------------------------------|
| `--images`          | required   | Folder of frames (recursed).                         |
| `--masks`           | required   | Output folder for COLMAP masks.                      |
| `--poses`           | required   | `sfm_priors.json` (poses + intrinsics).              |
| `--preview`         | *(off)*    | Output folder for debug overlays.                    |
| `--method`          | `sam2`     | `sam2`, `geometric` (projected silhouette only), or `motion` (background subtraction per EL ring). |
| `--checkpoint`      | *(none)*   | SAM2.1 weights (required for `--method sam2`).       |
| `--model-cfg`       | `configs/sam2.1/sam2.1_hiera_b+.yaml` | Hydra config in the `sam2` package. |
| `--device`          | `cuda`     | `cuda` or `cpu` (falls back to CPU with a warning).  |
| `--volume-radius-mm`| `180`      | Working-volume cylinder radius.                      |
| `--volume-height-mm`| `350`      | Cylinder height above its base.                      |
| `--volume-base-mm`  | `0`        | Cylinder base Z (0 = table level).                   |
| `--volume-center-xy-mm` | `0,0`  | Cylinder axis XY (world mm).                         |
| `--dilate`          | `10`       | Grow the mask N px so edges aren't clipped.          |
| `--min-area`        | `0.005`    | Drop components smaller than this frame fraction.    |
| `--limit`           | `0`        | Process only the first N frames (debugging).         |
| `--sam-long-side`   | `1024`     | Inference resolution, long side (model-native 1024). |
| `--containment-min` | `0.8`      | Keep a probe mask when this fraction lies inside the grown silhouette. |
| `--probe-max-fraction` | `0.35`  | Reject probe masks larger than this frame fraction.  |
| `--top-edge-band-frac` | `0.01`  | Reject probe masks touching the top band of the frame. |
| `--clamp-grow-frac` | `0.05`     | Silhouette dilation for the hard clamp on SAM2 output. |
| `--grid-steps`      | `8`        | Probe grid density (N×N over the projected bbox).    |
| `--neg-radius-mult` | `2.0`      | Negative-ring radius (× volume radius).              |
| `--connect-overlap-min` | `0.10` | Growth join: fraction of a segment that must overlap the dilated union. |
| `--geometry-composite` | auto    | Stamp the projected disc + nominal board into every mask. Default ON for sam2/geometric, **OFF for motion** (the projected disc drifts off the real one); override either way. |
| `--disc-radius-mm`  | `180`      | Physical disc radius for the stamp (0 disables it).  |
| `--plate-long-side` | `1024`     | (motion) Plate / motion working resolution, long side. |
| `--plate-iters`     | `2`        | (motion) 1 = plain median plate, 2 = + one robust rebuild (kills the on-axis ghost). |
| `--motion-k`        | `2.0`      | (motion) Threshold = `max(floor, k·noise σ)`. Low by design — discard a pixel only when confidently identical to the plate. |
| `--motion-floor`    | `6.0`      | (motion) Absolute threshold floor (colour-distance units, 0–441). |
| `--motion-open-px`  | `2`        | (motion) Despeckle open radius (working res).        |
| `--motion-close-px` | `4`        | (motion) Gap-bridge close radius (working res).      |
| `--edge-refine`     | on         | (motion) Snap edges with SAM2 (`--no-edge-refine` = pure motion). |
| `--edge-band-frac`  | `0.02`     | (motion) How far SAM2 may move the motion boundary.  |

`--volume-center-xy-mm` defaults to `auto` — the calibrated turntable axis
from the priors.

### Geometry composite (deterministic stamps)

Regions KNOWN to rotate with the table are unioned into the final mask, so
the segmenter gets no vote on them:

* the **disc** — a circle of `--disc-radius-mm` around the calibrated
  turntable axis, stamped at the top surface and one slab-thickness below
  (covers the rotating side band);
* the **board** — the ChArUco board's NOMINAL calibrated pose
  (`calib_board_world`, embedded into `sfm_priors.json` as the
  `turntable.board` block). Per-frame detection is deliberately not used:
  during an object scan the board is mostly occluded by the subject.

The final mask is also cut back to the (grown) geometric silhouette after
post-processing: white pixels readmit static-room features to COLMAP, so
the tuning bias is *under-mask the room*, never over-grow into it.

A `masks/_masks_manifest.json` records the settings and a per-image row
(coverage, SAM score, prompt kind, degenerate flag) for review.

### Motion method (background subtraction per EL ring)

The rig captures each elevation ring (the az-sweep at a fixed EL) with a
**physically static camera** — AZ rotates the platform, not the camera. So the
ring's frames share a pixel-identical background and only the assembly moves.
`--method motion` exploits this directly:

1. **Group** frames into rings by the `_elNN` tag in the filename (falls back to
   a single ring when untagged).
2. **Plate** — a per-pixel temporal median over the ring reconstructs the *empty
   scene* (no empty-table shot needed). `--plate-iters 2` then rebuilds it
   ignoring first-pass motion pixels, removing the on-axis "ghost" a plain
   median bakes in for a tall, near-axis object.
3. **Motion** — `‖frame − plate‖` (colour distance) thresholded at
   `max(--motion-floor, --motion-k · noise σ)` is the moving assembly. σ is a
   robust per-ring noise estimate, so the threshold tracks each ring's exposure.
4. **Clamp** — the result is cut to the projected working volume (`motion ∩
   clamp`) to drop any far-room motion. The deterministic disc/board **stamp is
   OFF by default** here (`--geometry-composite` to force it on): the projected
   disc follows the calibration, which drifts off the real disc and drags
   background in, and the high-contrast speckle disc is captured by motion
   directly anyway.
5. **Edge refine** (`--edge-refine`, default on) — SAM2 snaps the motion mask's
   boundary, *bounded* to a band so it can only refine edges (positives from the
   motion interior, negatives from the static plate). `--no-edge-refine` runs
   pure motion with no GPU.

This is the most robust method on a calibrated rig: the foreground signal is
physical (not an ML guess) and independent of pose accuracy, and the static room
gives a zero diff so it cannot leak into the mask — the degenerate "reconstructed
the room" solution masks exist to prevent. Per-ring σ / threshold and per-frame
motion coverage land in `_masks_manifest.json` (`output.rings`,
`per_image[].motion_coverage`). A rotationally near-symmetric object centred on
the axis barely changes between frames — fall back to `sam2` for those.

### How the SAM2 pass works (probe grid + disc connectivity)

One big box-plus-points query makes SAM2 hallucinate texture-noise masks on
busy scenes, so the tool runs many cheap single-point queries against one
image embedding instead:

1. **Seed** — probes on the projected turntable disc; their segments anchor
   the assembly (they are physically the rotating table).
2. **Growth** — probes on the volume axis and on a grid inside the projected
   silhouette join the union only if their segment touches the assembly so
   far — a lamp or shelf that merely *projects* inside the volume stays out.
3. Every probe segment must stay mostly inside the silhouette, must not be
   frame-sized and must not touch the top frame edge (the camera always looks
   down — anything reaching the frame top is the room).
4. The union is clamped to the dilated silhouette, then post-processed.

### Prompts and fallbacks

- **geometry** — the frame has a pose in the priors and the projected volume
  lands in the frame: the probe-grid pass above. This is the normal path on a
  calibrated rig.
- **heuristic** — no usable geometry (frame missing from the priors, or the
  volume projects off-frame): a centre box covering ~60% of the frame, one
  positive point in the middle, corner negatives. Logged in the manifest as
  `prompt: "heuristic"`.
- `--method geometric` skips SAM2 entirely and writes the projected silhouette
  (convex hull of the volume) — the emergency mode when torch/weights are
  unavailable.

### Degenerate frames (fail-open)

If nothing survives cleaning, the tool writes an **all-white** mask — COLMAP
then uses the whole frame, which is far safer than an all-black mask that
would silently drop the image from the reconstruction. Every such frame is
logged to stderr and flagged `degenerate: true` in the manifest. If *every*
frame is degenerate the tool exits `3`.

## Using the masks with COLMAP

```bash
colmap feature_extractor \
    --database_path ./colmap/database.db \
    --image_path    ./photos \
    --ImageReader.mask_path ./masks
```

`run_colmap_session.sh` wires this automatically when `masks/` exists in
the session dir.

## Tuning

| Symptom                                  | Fix                                                       |
|------------------------------------------|-----------------------------------------------------------|
| Top of a tall object clipped             | Raise `--volume-height-mm`.                               |
| Disc edge cut out of the mask            | Raise `--volume-radius-mm` — the cylinder must enclose the physical disc with margin. |
| Room furniture leaks into the mask       | Tighten the volume to the real assembly; check the calibration / priors — the silhouette clamp depends on correct poses. |
| Object edges clipped                     | Raise `--dilate`.                                         |
| Many `heuristic` prompts in the manifest | The scan has no `camera_quat` poses — recapture with a calibrated rig, or accept centre-box prompts. |
| CPU run too slow                         | Use the `tiny`/`small` checkpoint (`--checkpoint`), or `--method motion --no-edge-refine` (no ML at all). |

## Return codes

`0` ok · `1` bad input / SAM2 unavailable · `3` nothing usable produced ·
`4` unhandled exception.
