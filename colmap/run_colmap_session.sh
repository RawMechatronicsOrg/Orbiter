#!/usr/bin/env bash
# run_colmap_session.sh
# =====================
# End-to-end COLMAP run for one Orbiter scan session.
#
# Reads:    /data/scans/<sid>/sfm_priors.json       (from the server)
#           /data/scans/<sid>/<image files...>      (referenced by priors)
# Writes:   /data/scans/<sid>/colmap/
#               sparse_priors/                       (cameras.txt, images.txt, points3D.txt)
#               database.db                          (COLMAP working DB)
#               sparse/0/                            (triangulated sparse)
#               dense/                               (undistorted + stereo workspace)
#               dense/fused.ply                      (final point cloud)
#
# Usage:    run_colmap_session.sh <sid> [--dry-run] [--gpu]
#
#   --dry-run   Print the planned command sequence and exit 0.
#   --gpu       Enable GPU SIFT extraction and PatchMatch stereo
#               (default: CPU SIFT, CPU/GPU PatchMatch as COLMAP decides).

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
  --gpu         Use GPU for SIFT extraction (requires NVIDIA passthrough).
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

# Image paths in sfm_priors.json are relative to the session dir.
IMAGE_PATH="${SESSION_DIR}"

# Toggle the SIFT-on-GPU flag based on --gpu.
SIFT_GPU_FLAG="--SiftExtraction.use_gpu=$USE_GPU"

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
    else
        "$@"
    fi
}

# ---- Pipeline plan --------------------------------------------------------

echo "Orbiter COLMAP runner"
echo "  session id : ${SID}"
echo "  session dir: ${SESSION_DIR}"
echo "  use GPU    : $([[ $USE_GPU == 1 ]] && echo yes || echo no)"
echo "  dry run    : $([[ $DRY_RUN == 1 ]] && echo yes || echo no)"

if (( DRY_RUN == 0 )); then
    mkdir -p "$PRIORS_SPARSE" "$SPARSE_OUT" "$DENSE_DIR"
fi

step "1/7 convert sfm_priors.json -> COLMAP text model" \
    python3 /usr/local/bin/sfm_priors_to_colmap.py "$PRIORS_JSON" "$PRIORS_SPARSE"

step "2/7 feature_extractor (sift_gpu=$USE_GPU)" \
    colmap feature_extractor \
        --database_path "$DATABASE" \
        --image_path    "$IMAGE_PATH" \
        "$SIFT_GPU_FLAG"

step "3/7 exhaustive_matcher" \
    colmap exhaustive_matcher \
        --database_path "$DATABASE" \
        "$SIFT_GPU_FLAG"

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
fi
