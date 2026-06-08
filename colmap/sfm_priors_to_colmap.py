#!/usr/bin/env python3
"""
sfm_priors_to_colmap.py
=======================

Convert an Orbiter ``sfm_priors.json`` (schema ``orbiter.sfm_priors.v1``)
into COLMAP's text-format ``cameras.txt`` + ``images.txt``.

The source schema is documented in ``OrbiterV0.1/docs/COLMAP.md``::

    {
      "schema": "orbiter.sfm_priors.v1",
      "camera_intrinsics": {
        "model": "PINHOLE",
        "width":  1920, "height": 1080,
        "fx": 1500, "fy": 1500, "cx": 960, "cy": 540
      },
      "images": [
        {
          "file": "c_001/photo.jpg",
          "qw": 0.707, "qx": 0, "qy": 0.707, "qz": 0,
          "tx": 220,   "ty": 0, "tz":  45
        }
      ]
    }

Quaternions are Hamilton convention (w, x, y, z). Translations in
millimetres. The transform takes world points into camera space (COLMAP's
convention) — so we can write them straight into ``images.txt`` without
inversion.

Usage::

    sfm_priors_to_colmap.py <input_json> <output_dir>

Writes ``<output_dir>/cameras.txt``, ``<output_dir>/images.txt``, and an
empty ``<output_dir>/points3D.txt`` (COLMAP expects the file to exist).

Stdlib-only (json, sys, pathlib, math).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


SUPPORTED_SCHEMA = "orbiter.sfm_priors.v1"

# Map our schema's camera model name to (colmap_name, expected param keys
# in order). COLMAP's text format wants the parameters as a space-
# separated list whose meaning depends on MODEL. References:
#   https://colmap.github.io/cameras.html
_CAMERA_MODELS = {
    "PINHOLE": ("PINHOLE", ("fx", "fy", "cx", "cy")),
    "SIMPLE_PINHOLE": ("SIMPLE_PINHOLE", ("f", "cx", "cy")),
    "SIMPLE_RADIAL": ("SIMPLE_RADIAL", ("f", "cx", "cy", "k")),
    "RADIAL": ("RADIAL", ("f", "cx", "cy", "k1", "k2")),
    "OPENCV": ("OPENCV", ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")),
}


def _die(msg: str, code: int = 1) -> None:
    print(f"sfm_priors_to_colmap: error: {msg}", file=sys.stderr)
    sys.exit(code)


def _normalize_quat(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float, float]:
    """Renormalize the quaternion. Priors come from finite-precision
    encoders, so small drift is expected."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        _die("zero-norm quaternion encountered")
    return qw / n, qx / n, qy / n, qz / n


def _camera_line(cam_id: int, intrinsics: dict) -> str:
    model = intrinsics.get("model", "PINHOLE")
    if model not in _CAMERA_MODELS:
        _die(f"unsupported camera model {model!r} (known: {sorted(_CAMERA_MODELS)})")

    colmap_name, param_keys = _CAMERA_MODELS[model]
    try:
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        params = [float(intrinsics[k]) for k in param_keys]
    except KeyError as e:
        _die(f"camera_intrinsics missing required key: {e}")
    except (TypeError, ValueError) as e:
        _die(f"camera_intrinsics has non-numeric value: {e}")

    params_str = " ".join(f"{p:.10g}" for p in params)
    return f"{cam_id} {colmap_name} {width} {height} {params_str}"


def _image_lines(image_id: int, img: dict, camera_id: int) -> list[str]:
    try:
        qw = float(img["qw"])
        qx = float(img["qx"])
        qy = float(img["qy"])
        qz = float(img["qz"])
        tx = float(img["tx"])
        ty = float(img["ty"])
        tz = float(img["tz"])
        name = str(img["file"])
    except KeyError as e:
        _die(f"image entry missing required key: {e}")
    except (TypeError, ValueError) as e:
        _die(f"image entry has non-numeric value: {e}")

    qw, qx, qy, qz = _normalize_quat(qw, qx, qy, qz)

    # Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    line1 = (
        f"{image_id} "
        f"{qw:.10g} {qx:.10g} {qy:.10g} {qz:.10g} "
        f"{tx:.10g} {ty:.10g} {tz:.10g} "
        f"{camera_id} {name}"
    )
    # Line 2: POINTS2D — empty (we'll triangulate from features later).
    line2 = ""
    return [line1, line2]


def convert(priors_path: Path, out_dir: Path) -> None:
    if not priors_path.is_file():
        _die(f"priors file not found: {priors_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with priors_path.open("r", encoding="utf-8") as fh:
        priors = json.load(fh)

    schema = priors.get("schema")
    if schema != SUPPORTED_SCHEMA:
        _die(
            f"unsupported schema {schema!r}; expected {SUPPORTED_SCHEMA!r}"
        )

    intrinsics = priors.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        _die("'camera_intrinsics' missing or not an object")

    images = priors.get("images")
    if not isinstance(images, list) or not images:
        _die("'images' missing or empty")

    # We emit a single shared camera. All images reference camera_id=1.
    camera_id = 1
    cameras_txt = out_dir / "cameras.txt"
    images_txt = out_dir / "images.txt"
    points3d_txt = out_dir / "points3D.txt"

    header_cameras = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: 1",
    ]
    with cameras_txt.open("w", encoding="utf-8") as fh:
        for line in header_cameras:
            fh.write(line + "\n")
        fh.write(_camera_line(camera_id, intrinsics) + "\n")

    header_images = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images)}, mean observations per image: 0",
    ]
    with images_txt.open("w", encoding="utf-8") as fh:
        for line in header_images:
            fh.write(line + "\n")
        for idx, img in enumerate(images, start=1):
            for line in _image_lines(idx, img, camera_id):
                fh.write(line + "\n")

    # COLMAP expects points3D.txt to exist even when empty (priors model
    # has no triangulated points yet).
    with points3d_txt.open("w", encoding="utf-8") as fh:
        fh.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n"
        )

    print(f"wrote {cameras_txt}")
    print(f"wrote {images_txt}")
    print(f"wrote {points3d_txt}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: sfm_priors_to_colmap.py <input_json> <output_dir>",
            file=sys.stderr,
        )
        return 2
    convert(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
