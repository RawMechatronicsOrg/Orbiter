#!/usr/bin/env bash
# run_colmap_session.sh
# =====================
# End-to-end COLMAP run for one Orbiter scan session.
#
# Reads:    /data/scans/<sid>/sfm_priors.json       (from the server)
#           /data/scans/<sid>/photos/               (materialized frames)
#           /data/scans/<sid>/masks/                (optional — see colmap/masks/)
# Writes:   /data/scans/<sid>/colmap/
#               sparse_priors/                       (cameras/images/rigs/frames + points3D)
#               database.db                          (COLMAP working DB)
#               sparse/0/                            (triangulated sparse)
#               dense/                               (undistorted + stereo workspace)
#               dense/fused.ply                      (final point cloud)
#
# Usage:    run_colmap_session.sh <sid> [--dry-run] [--gpu]
#
#   --dry-run   Print the planned command sequence and exit 0.
#   --gpu       Enable GPU feature extraction/matching and PatchMatch stereo
#               (default: CPU features, CPU/GPU PatchMatch as COLMAP decides).

set -euo pipefail

SID="${1:-}"
shift || true

USE_GPU=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --gpu)     USE_GPU=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *)
            echo "run_colmap_session.sh: unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$SID" ]]; then
    cat >&2 <<'EOF'
usage: run_colmap_session.sh <session-id> [--dry-run] [--gpu]
  session-id    Subdirectory under /data/scans/ to operate on.
  --dry-run     Print planned commands and exit.
  --gpu         Use the GPU for feature extraction/matching (requires NVIDIA passthrough).
EOF
    exit 2
fi

SESSION_DIR="/data/scans/${SID}"
PRIORS_JSON="${SESSION_DIR}/sfm_priors.json"
COLMAP_DIR="${SESSION_DIR}/colmap"
PRIORS_SPARSE="${COLMAP_DIR}/sparse_priors"
DATABASE="${COLMAP_DIR}/database.db"
SPARSE_OUT="${COLMAP_DIR}/sparse/0"
DENSE_DIR="${COLMAP_DIR}/dense"
FUSED_PLY="${DENSE_DIR}/fused.ply"

# COLMAP reads frames from photos/ (names in priors are basenames under that
# dir). `POST /scans/<sid>/sfm_priors` materializes it next to the manifest.
IMAGE_PATH="${SESSION_DIR}/photos"

# colmap/colmap:latest (3.13+) — FeatureExtraction/FeatureMatching, not Sift*.
COLMAP_GPU_INDEX="${ORBITER_COLMAP_GPU_INDEX:-0}"
COLMAP_MAX_IMAGE_SIZE="${ORBITER_COLMAP_MAX_IMAGE_SIZE:-3200}"
if (( USE_GPU == 1 )); then
    EXTRACT_GPU_FLAG="--FeatureExtraction.use_gpu=1 --FeatureExtraction.gpu_index=${COLMAP_GPU_INDEX} --FeatureExtraction.max_image_size=${COLMAP_MAX_IMAGE_SIZE}"
    MATCH_GPU_FLAG="--FeatureMatching.use_gpu=1 --FeatureMatching.gpu_index=${COLMAP_GPU_INDEX}"
else
    EXTRACT_GPU_FLAG="--FeatureExtraction.use_gpu=0"
    MATCH_GPU_FLAG="--FeatureMatching.use_gpu=0"
fi

# Object masks (colmap/masks/generate_colmap_masks.py) — picked up
# automatically when the session has a non-empty masks/ dir. White = keep
# features, black = suppress (the static room).
MASKS_DIR="${SESSION_DIR}/masks"
MASK_FLAG=""
if [[ -d "$MASKS_DIR" ]] && [[ -n "$(ls -A "$MASKS_DIR" 2>/dev/null)" ]]; then
    MASK_FLAG="--ImageReader.mask_path=${MASKS_DIR}"
fi

# Camera-model flags derived from the priors intrinsics. Without them COLMAP
# defaults the database camera to SIMPLE_RADIAL while sparse_priors says
# PINHOLE/OPENCV — point_triangulator then aborts on the model mismatch.
CAMERA_MODEL_FLAGS=""
if [[ -f "$PRIORS_JSON" ]]; then
    CAMERA_MODEL_FLAGS="$(python3 /usr/local/bin/sfm_priors_to_colmap.py --emit-extractor-flags "$PRIORS_JSON")"
fi

# ---- Validation -----------------------------------------------------------

if (( DRY_RUN == 0 )); then
    if [[ ! -d "$SESSION_DIR" ]]; then
        echo "run_colmap_session.sh: session dir not found: $SESSION_DIR" >&2
        exit 1
    fi
    if [[ ! -f "$PRIORS_JSON" ]]; then
        echo "run_colmap_session.sh: sfm_priors.json missing — export it from the UI first." >&2
        echo "  expected at: $PRIORS_JSON" >&2
        exit 1
    fi
    if [[ ! -d "$IMAGE_PATH" ]]; then
        echo "run_colmap_session.sh: photos/ missing — POST /scans/${SID}/sfm_priors" >&2
        echo "  (the export also materializes the frames) or copy them in manually." >&2
        exit 1
    fi
fi

# ---- Helper: run-or-echo --------------------------------------------------
#
# Echoes a step header, then either runs the command or, in --dry-run
# mode, prints what would have run.

step() {
    local title="$1"; shift
    echo
    echo "=== ${title} ==="
    if (( DRY_RUN )); then
        printf '  +'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    # Profiling: emit start/end/duration markers around each step. The span
    # name is the colmap subcommand (or "priors" for the converter) so the
    # lines are trivially greppable across runs. The command runs as "$@";
    # capturing its rc via &&/|| keeps `set -e` from aborting before we
    # print the end marker, and `return $r` re-arms the abort at the call site.
    local name="$1"
    case "$1" in
        colmap)         name="$2" ;;
        python3|python) name="priors" ;;
    esac
    local s e r
    s=$(date +%s.%N)
    echo "[PROFILE] start name=${name} at=$(date -u +%FT%TZ)"
    "$@" && r=0 || r=$?
    e=$(date +%s.%N)
    echo "[PROFILE] end name=${name} rc=${r} dur=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.3f", b-a}') at=$(date -u +%FT%TZ)"
    printf '  (%s took %ss)\n' "${name}" "$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.1f", b-a}')"
    return "$r"
}

# ---- Pipeline plan --------------------------------------------------------

echo "Orbiter COLMAP runner"
echo "  session id : ${SID}"
echo "  session dir: ${SESSION_DIR}"
echo "  use GPU    : $([[ $USE_GPU == 1 ]] && echo yes || echo no)"
echo "  masks      : $([[ -n $MASK_FLAG ]] && echo "$MASKS_DIR" || echo none)"
echo "  dry run    : $([[ $DRY_RUN == 1 ]] && echo yes || echo no)"

if (( DRY_RUN == 0 )); then
    mkdir -p "$PRIORS_SPARSE" "$SPARSE_OUT" "$DENSE_DIR"
fi

# Wall-clock start for the end-of-run total (per-step times come from step()).
SCRIPT_START=$(date +%s.%N)

step "1/7 feature_extractor (gpu=$USE_GPU)" \
    colmap feature_extractor \
        --database_path "$DATABASE" \
        --image_path    "$IMAGE_PATH" \
        --ImageReader.single_camera=1 \
        $CAMERA_MODEL_FLAGS \
        $MASK_FLAG \
        $EXTRACT_GPU_FLAG

step "2/7 exhaustive_matcher (gpu=$USE_GPU)" \
    colmap exhaustive_matcher \
        --database_path "$DATABASE" \
        $MATCH_GPU_FLAG

step "3/7 convert sfm_priors.json -> COLMAP text model (sync to database)" \
    python3 /usr/local/bin/sfm_priors_to_colmap.py "$PRIORS_JSON" "$PRIORS_SPARSE" --database "$DATABASE"

step "4/7 point_triangulator (priors -> sparse/0)" \
    colmap point_triangulator \
        --database_path "$DATABASE" \
        --image_path    "$IMAGE_PATH" \
        --input_path    "$PRIORS_SPARSE" \
        --output_path   "$SPARSE_OUT"

step "5/7 image_undistorter (sparse/0 -> dense/)" \
    colmap image_undistorter \
        --image_path    "$IMAGE_PATH" \
        --input_path    "$SPARSE_OUT" \
        --output_path   "$DENSE_DIR" \
        --output_type   COLMAP

step "6/7 patch_match_stereo" \
    colmap patch_match_stereo \
        --workspace_path        "$DENSE_DIR" \
        --workspace_format      COLMAP \
        --PatchMatchStereo.geom_consistency true

step "7/7 stereo_fusion -> fused.ply" \
    colmap stereo_fusion \
        --workspace_path        "$DENSE_DIR" \
        --workspace_format      COLMAP \
        --input_type            geometric \
        --output_path           "$FUSED_PLY"

if (( DRY_RUN )); then
    echo
    echo "(dry-run) would write: $FUSED_PLY"
else
    echo
    echo "done. fused point cloud: $FUSED_PLY"
    echo "total: $(awk -v a="$SCRIPT_START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')s"
fi
