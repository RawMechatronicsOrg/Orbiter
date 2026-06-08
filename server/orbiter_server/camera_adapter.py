"""Camera adapter — single source of truth for every photo transformation.

This module owns:
- Orientation policy per phone model (preset)
- EXIF / ICC preservation across rotation
- Multi-tier thumbnail generation (tiny / small / medium), parallelised
- Photo basename construction shared across storage-api / tools / UI

Why a preset system? Phones lie. Samsung Galaxy S22 (SM-S921B) saves
landscape pixels with EXIF Orientation=1 ("no rotation needed") even when
the operator framed the shot in portrait. EXIF is unreliable, so each
preset hard-codes the right transformation for a specific device.

See `AAAphotoaf.jpg` at the project root for the canonical Sm22 sample.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

# ─────────────────────────────────────────────────────────────────────────
# Preset registry
# ─────────────────────────────────────────────────────────────────────────

PresetName = Literal["native", "sm22"]


@dataclass(frozen=True)
class PresetSpec:
    """Per-device rules for turning raw JPEG bytes into oriented pixels."""

    name: PresetName
    # ↻ 90° clockwise rotations to apply *to the decoded RGB image*, in
    # addition to whatever exif_transpose did. Multiples of 90° so we can
    # use Image.transpose (exact pixel-wise, no resampling).
    extra_cw_quarter_turns: int
    # If True, apply ImageOps.exif_transpose first (trust the EXIF tag).
    # If False, ignore EXIF Orientation entirely — the phone lied.
    honor_exif_orientation: bool
    # If True, strip EXIF/XMP from original.jpg. Keep False unless you
    # know none of the downstream SfM tools (Meshroom, RealityCapture)
    # need focal-length priors from the JPEG.
    strip_metadata_from_original: bool


PRESETS: dict[PresetName, PresetSpec] = {
    "native": PresetSpec(
        name="native",
        extra_cw_quarter_turns=0,
        honor_exif_orientation=True,
        strip_metadata_from_original=False,
    ),
    # Galaxy S22 (SM-S921B): writes 1920×1080 landscape pixels with
    # EXIF Orientation=1, but the subject is actually portrait with top
    # on the LEFT. Ignore EXIF, force one 90° CW rotation.
    "sm22": PresetSpec(
        name="sm22",
        extra_cw_quarter_turns=1,
        honor_exif_orientation=False,
        strip_metadata_from_original=False,
    ),
}


def get_preset(name: str | None) -> PresetSpec:
    """Lookup with safe fallback. Unknown / None → native."""
    if not name:
        return PRESETS["native"]
    return PRESETS.get(name, PRESETS["native"])  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────
# Thumbnail tier specification
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TierSpec:
    """One thumbnail tier — filename on disk + resize parameters."""

    name: str
    filename: str
    max_width: int
    quality: int


# Order matters only for log readability — they all run in parallel.
THUMB_TIERS: tuple[TierSpec, ...] = (
    TierSpec("medium", "thumb.jpg",       240, 80),
    TierSpec("small",  "thumb_small.jpg", 512, 78),
    TierSpec("tiny",   "thumb_tiny.jpg",  128, 55),
)


# ─────────────────────────────────────────────────────────────────────────
# Basename — shared across storage-api / tools / build_priors_sfm
# ─────────────────────────────────────────────────────────────────────────


def photo_basename(index: int, az_deg: float, el_deg: float) -> str:
    """`000_az020_el+00.jpg` — the canonical Orbiter photo filename.

    Used by:
    - storage._photo_filename (ZIP entries / on-disk photos/ dir)
    - SfmCameraOverlay.thumbByBasename (UI side, lowercase match)
    - build_priors_sfm._photo_stem_from_capture (SfM-from-machine writer)

    All callers MUST go through this function so the format never drifts.
    """
    az_lbl = f"{round(az_deg):03d}"
    el_lbl = f"{round(el_deg):+03d}".replace("+-", "-")
    return f"{int(index):03d}_az{az_lbl}_el{el_lbl}.jpg"


# ─────────────────────────────────────────────────────────────────────────
# Pixel transformations
# ─────────────────────────────────────────────────────────────────────────


def _apply_quarter_turns_cw(im: Image.Image, turns: int) -> Image.Image:
    """Apply N×90° clockwise rotations using exact pixel-wise transpose
    (no resampling, no quality loss). turns is taken mod 4."""
    turns = turns % 4
    if turns == 0:
        return im
    if turns == 1:
        return im.transpose(Image.Transpose.ROTATE_270)  # 90° CW
    if turns == 2:
        return im.transpose(Image.Transpose.ROTATE_180)
    return im.transpose(Image.Transpose.ROTATE_90)  # 270° CW = 90° CCW


def orient(im: Image.Image, preset: PresetSpec) -> Image.Image:
    """Apply the preset's full orientation policy to a decoded RGB image.

    Order:
      1. exif_transpose (if honor_exif_orientation) — physically rotate
         pixels per the EXIF Orientation tag, clearing the tag.
      2. extra_cw_quarter_turns — preset's forced rotation (for phones
         that report wrong EXIF).
    """
    if preset.honor_exif_orientation:
        im = ImageOps.exif_transpose(im)
    if preset.extra_cw_quarter_turns:
        im = _apply_quarter_turns_cw(im, preset.extra_cw_quarter_turns)
    return im


def _write_thumb_tier(im: Image.Image, path: Path, tier: TierSpec) -> None:
    """Resize and save a single thumbnail tier. Safe to call concurrently
    from a ThreadPoolExecutor — PIL releases the GIL during LANCZOS."""
    w, h = im.size
    if w > tier.max_width:
        scale = tier.max_width / w
        out = im.resize((tier.max_width, int(h * scale)), Image.LANCZOS)
    else:
        out = im
    out.save(path, "JPEG", quality=tier.quality, optimize=True)


# Lazily-created executor — reused across requests. 3 workers covers the
# three tiers; if more presets are added, set max_workers=len(THUMB_TIERS).
_THUMB_POOL: ThreadPoolExecutor | None = None


def _thumb_pool() -> ThreadPoolExecutor:
    global _THUMB_POOL
    if _THUMB_POOL is None:
        _THUMB_POOL = ThreadPoolExecutor(
            max_workers=max(3, len(THUMB_TIERS)),
            thread_name_prefix="thumb",
        )
    return _THUMB_POOL


def shutdown_thumb_pool() -> None:
    """Tear down the lazily-created thumbnail thread pool. Called from
    app.py's lifespan finally so a still-running resize doesn't keep the
    process alive after uvicorn shuts down."""
    global _THUMB_POOL
    if _THUMB_POOL is not None:
        _THUMB_POOL.shutdown(wait=False, cancel_futures=True)
        _THUMB_POOL = None


# ─────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProcessedCapture:
    """Everything a caller needs to know after process_capture finishes."""

    original_path: Path
    thumb_path: Path
    thumb_small_path: Path
    thumb_tiny_path: Path
    # Post-rotation dimensions of the stored original.jpg. Persist on the
    # capture record so the 3D viewer can size frustums correctly without
    # re-decoding the JPEG.
    width: int
    height: int


def process_capture(
    raw_bytes: bytes,
    *,
    preset: PresetSpec,
    out_dir: Path,
    original_name: str = "original.jpg",
) -> ProcessedCapture:
    """Decode JPEG → orient → write original.jpg + 3 thumbnail tiers.

    The three thumbnails are resized in parallel from independent copies
    of the full-resolution PIL image, so quality doesn't degrade through
    a chain of LANCZOS passes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    original_path = out_dir / original_name
    paths = {tier.name: out_dir / tier.filename for tier in THUMB_TIERS}

    with Image.open(io.BytesIO(raw_bytes)) as im_in:
        exif_bytes = im_in.info.get("exif") if not preset.strip_metadata_from_original else None
        icc_profile = im_in.info.get("icc_profile") if not preset.strip_metadata_from_original else None

        im = im_in.convert("RGB")
        im = orient(im, preset)
        w, h = im.size

        # ── Write original.jpg ──
        if preset.extra_cw_quarter_turns == 0 and preset.honor_exif_orientation is False:
            # Special case: no transformation at all. Could byte-copy
            # input → original, but we already decoded; re-encode is fine
            # and ensures EXIF Orientation isn't relied upon downstream.
            pass
        save_kwargs: dict[str, object] = {"quality": 92, "optimize": True}
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        im.save(original_path, "JPEG", **save_kwargs)

        # ── Three thumbnail tiers in parallel ──
        # We pass `im` (a single PIL.Image) to all three workers. PIL's
        # resize() is read-only on the source image, so this is safe.
        pool = _thumb_pool()
        futures = [
            pool.submit(_write_thumb_tier, im, paths[tier.name], tier)
            for tier in THUMB_TIERS
        ]
        for f in futures:
            f.result()  # surfaces exceptions

    return ProcessedCapture(
        original_path=original_path,
        thumb_path=paths["medium"],
        thumb_small_path=paths["small"],
        thumb_tiny_path=paths["tiny"],
        width=w,
        height=h,
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI self-check (handy for verifying a new preset on a sample image)
# ─────────────────────────────────────────────────────────────────────────


def _self_check(path: Path, preset_name: PresetName) -> None:
    """Decode a sample image, run the preset, print resulting dimensions
    and the path to a temp original.jpg so you can inspect it visually."""
    import tempfile
    import shutil

    preset = get_preset(preset_name)
    raw = path.read_bytes()
    tmp = Path(tempfile.mkdtemp(prefix="camera-adapter-check-"))
    try:
        result = process_capture(raw, preset=preset, out_dir=tmp)
        print(f"preset:     {preset.name}")
        print(f"input:      {path} ({len(raw)} bytes)")
        print(f"output W×H: {result.width}×{result.height}")
        print(f"original:   {result.original_path}")
        print(f"thumb:      {result.thumb_path}")
        print(f"thumb_small:{result.thumb_small_path}")
        print(f"thumb_tiny: {result.thumb_tiny_path}")
        print(f"(leaving temp dir in place: {tmp})")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="JPEG to test")
    p.add_argument("--preset", default="sm22", choices=sorted(PRESETS.keys()))
    args = p.parse_args()
    _self_check(args.image, args.preset)
