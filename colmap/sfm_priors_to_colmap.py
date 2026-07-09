#!/usr/bin/env python3
"""
sfm_priors_to_colmap.py
=======================

Convert an Orbiter ``sfm_priors.json`` (schema ``orbiter.sfm_priors.v1``)
into COLMAP's text-format sparse model.

The source schema is documented in ``docs/COLMAP.md``::

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

    sfm_priors_to_colmap.py <input_json> <output_dir> [--database database.db]

Without ``--database``, image IDs are assigned sequentially (1..N) in JSON
order — fine for quick inspection only.

With ``--database`` (after ``feature_extractor``), image/frame IDs and names
are taken from the COLMAP SQLite database. Images missing from the database
are dropped. ``rigs.txt`` and ``frames.txt`` are emitted for COLMAP 4.

Stdlib-only (json, sys, pathlib, math, sqlite3, argparse).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _FrameBinding:
    frame_id: int
    rig_id: int
    sensor_id: int
    data_id: int


def _die(msg: str, code: int = 1) -> None:
    print(f"sfm_priors_to_colmap: error: {msg}", file=sys.stderr)
    sys.exit(code)


def _colmap_image_name(file_field: str) -> str:
    """COLMAP image names are relative to ``--image_path`` (our ``photos/`` dir)."""
    path = file_field.replace("\\", "/").lstrip("./")
    if path.startswith("photos/"):
        return path[len("photos/") :]
    return path


def _normalize_db_name(name: str) -> str:
    return _colmap_image_name(name)


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


def _read_pose(img: dict) -> tuple[float, float, float, float, float, float, float]:
    try:
        qw = float(img["qw"])
        qx = float(img["qx"])
        qy = float(img["qy"])
        qz = float(img["qz"])
        tx = float(img["tx"])
        ty = float(img["ty"])
        tz = float(img["tz"])
    except KeyError as e:
        _die(f"image entry missing required key: {e}")
    except (TypeError, ValueError) as e:
        _die(f"image entry has non-numeric value: {e}")
    return _normalize_quat(qw, qx, qy, qz) + (tx, ty, tz)


def _image_lines(image_id: int, name: str, pose: tuple[float, ...], camera_id: int) -> list[str]:
    qw, qx, qy, qz, tx, ty, tz = pose
    line1 = (
        f"{image_id} "
        f"{qw:.10g} {qx:.10g} {qy:.10g} {qz:.10g} "
        f"{tx:.10g} {ty:.10g} {tz:.10g} "
        f"{camera_id} {name}"
    )
    return [line1, ""]


def _load_database_bindings(db_path: Path) -> tuple[dict[str, int], dict[int, _FrameBinding], int, int]:
    if not db_path.is_file():
        _die(f"database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        name_to_image_id = {
            _normalize_db_name(name): int(image_id)
            for image_id, name in conn.execute("SELECT image_id, name FROM images")
        }
        camera_row = conn.execute("SELECT camera_id FROM cameras LIMIT 1").fetchone()
        if camera_row is None:
            _die("database has no cameras row — run feature_extractor first")
        camera_id = int(camera_row[0])

        rig_row = conn.execute("SELECT rig_id FROM rigs LIMIT 1").fetchone()
        if rig_row is None:
            _die("database has no rigs row — run feature_extractor first")
        default_rig_id = int(rig_row[0])

        image_to_frame: dict[int, _FrameBinding] = {}
        for frame_id, rig_id in conn.execute("SELECT frame_id, rig_id FROM frames"):
            row = conn.execute(
                "SELECT data_id, sensor_id FROM frame_data WHERE frame_id = ?",
                (int(frame_id),),
            ).fetchone()
            if row is None:
                _die(f"frame {frame_id} has no frame_data row in database")
            data_id, sensor_id = int(row[0]), int(row[1])
            image_to_frame[data_id] = _FrameBinding(
                frame_id=int(frame_id),
                rig_id=int(rig_id),
                sensor_id=sensor_id,
                data_id=data_id,
            )
    finally:
        conn.close()

    return name_to_image_id, image_to_frame, default_rig_id, camera_id


def _write_rigs_txt(path: Path, rig_id: int, sensor_id: int) -> None:
    header = [
        "# Rig list with one line of data per rig:",
        "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, "
        "SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])",
        "# Number of rigs: 1",
    ]
    with path.open("w", encoding="utf-8") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(f"{rig_id} 1 CAMERA {sensor_id}\n")


def _write_frames_txt(
    path: Path,
    frames: list[tuple[_FrameBinding, tuple[float, ...]]],
) -> None:
    header = [
        "# Frame list with one line of data per frame:",
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], "
        "NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)",
        f"# Number of frames: {len(frames)}",
    ]
    with path.open("w", encoding="utf-8") as fh:
        for line in header:
            fh.write(line + "\n")
        for binding, pose in frames:
            qw, qx, qy, qz, tx, ty, tz = pose
            fh.write(
                f"{binding.frame_id} {binding.rig_id} "
                f"{qw:.10g} {qx:.10g} {qy:.10g} {qz:.10g} "
                f"{tx:.10g} {ty:.10g} {tz:.10g} "
                f"1 CAMERA {binding.sensor_id} {binding.data_id}\n"
            )


def _camera_center(pose: tuple[float, ...]) -> tuple[float, float, float]:
    """Camera centre ``C = -Rᵀ t`` for a world→camera Hamilton quaternion +
    translation (the pose tuple ``(qw,qx,qy,qz,tx,ty,tz)``)."""
    qw, qx, qy, qz, tx, ty, tz = pose
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    r = (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )
    t = (tx, ty, tz)
    return tuple(-sum(r[i][k] * t[i] for i in range(3)) for k in range(3))


def _write_db_pose_priors(db_path: Path, kept: list) -> int:
    """Fill the database ``pose_priors`` table with prior camera centres
    (CARTESIAN) so ``colmap pose_prior_mapper`` can softly anchor the poses.

    COLMAP's feature_extractor seeds empty placeholder rows (position 0,0,0,
    WGS84); we overwrite them by ``corr_data_id`` (= image_id for our
    single-sensor rig). The actual prior covariance is supplied at run time via
    ``--overwrite_priors_covariance`` + ``--prior_position_std_*``. Returns the
    number of rows updated; 0 if the table is absent (older COLMAP)."""
    conn = sqlite3.connect(db_path)
    try:
        have = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='pose_priors'"
        ).fetchone()
        if have is None:
            return 0
        n = 0
        for image_id, _name, pose, _binding in kept:
            cx, cy, cz = _camera_center(pose)
            cur = conn.execute(
                "UPDATE pose_priors SET position=?, coordinate_system=1 "
                "WHERE corr_data_id=?",
                (struct.pack("<3d", cx, cy, cz), image_id),
            )
            n += cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def convert(priors_path: Path, out_dir: Path, database_path: Path | None = None) -> None:
    if not priors_path.is_file():
        _die(f"priors file not found: {priors_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with priors_path.open("r", encoding="utf-8") as fh:
        priors = json.load(fh)

    schema = priors.get("schema")
    if schema != SUPPORTED_SCHEMA:
        _die(f"unsupported schema {schema!r}; expected {SUPPORTED_SCHEMA!r}")

    intrinsics = priors.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        _die("'camera_intrinsics' missing or not an object")

    images = priors.get("images")
    if not isinstance(images, list) or not images:
        _die("'images' missing or empty")

    camera_id = 1
    name_to_image_id: dict[str, int] | None = None
    image_to_frame: dict[int, _FrameBinding] | None = None
    rig_id = 1
    ref_sensor_id = 1
    dropped = 0

    if database_path is not None:
        name_to_image_id, image_to_frame, rig_id, camera_id = _load_database_bindings(database_path)

    cameras_txt = out_dir / "cameras.txt"
    images_txt = out_dir / "images.txt"
    points3d_txt = out_dir / "points3D.txt"
    rigs_txt = out_dir / "rigs.txt"
    frames_txt = out_dir / "frames.txt"

    header_cameras = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: 1",
    ]
    with cameras_txt.open("w", encoding="utf-8") as fh:
        for line in header_cameras:
            fh.write(line + "\n")
        fh.write(_camera_line(camera_id, intrinsics) + "\n")

    kept: list[tuple[int, str, tuple[float, ...], _FrameBinding | None]] = []
    if database_path is not None:
        assert name_to_image_id is not None and image_to_frame is not None
        for img in images:
            name = _colmap_image_name(str(img["file"]))
            image_id = name_to_image_id.get(name)
            if image_id is None:
                dropped += 1
                continue
            binding = image_to_frame.get(image_id)
            if binding is None:
                _die(f"database has image {name!r} (id={image_id}) but no frame_data binding")
            kept.append((image_id, name, _read_pose(img), binding))
        kept.sort(key=lambda row: row[0])
        if not kept:
            _die("no priors images matched the database — check photos/ materialization")
        ref_sensor_id = kept[0][3].sensor_id  # type: ignore[index]
    else:
        for idx, img in enumerate(images, start=1):
            name = _colmap_image_name(str(img["file"]))
            kept.append((idx, name, _read_pose(img), None))

    header_images = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(kept)}, mean observations per image: 0",
    ]
    with images_txt.open("w", encoding="utf-8") as fh:
        for line in header_images:
            fh.write(line + "\n")
        for image_id, name, pose, _binding in kept:
            for line in _image_lines(image_id, name, pose, camera_id):
                fh.write(line + "\n")

    if database_path is not None:
        frame_rows = [
            (binding, pose)
            for _image_id, _name, pose, binding in kept
            if binding is not None
        ]
        _write_rigs_txt(rigs_txt, rig_id, ref_sensor_id)
        _write_frames_txt(frames_txt, frame_rows)
        n_pp = _write_db_pose_priors(database_path, kept)

    with points3d_txt.open("w", encoding="utf-8") as fh:
        fh.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n"
        )

    print(f"wrote {cameras_txt}")
    print(f"wrote {images_txt}")
    print(f"wrote {points3d_txt}")
    if database_path is not None:
        print(f"wrote {rigs_txt}")
        print(f"wrote {frames_txt}")
        print(f"synced {len(kept)} priors to database, dropped {dropped} (not in database)")
        print(f"populated {n_pp} pose_priors rows (CARTESIAN centres)")


def extractor_flags(priors_path: Path) -> str:
    """``feature_extractor`` flags matching the priors' intrinsics.

    Without these COLMAP defaults the database camera to SIMPLE_RADIAL while
    the priors text model says PINHOLE/OPENCV — ``point_triangulator`` then
    aborts on the model mismatch. Deriving the extractor flags, the database
    and the text model from the same ``camera_intrinsics`` block keeps all
    three in agreement (``run_colmap_session.sh`` consumes this via
    ``--emit-extractor-flags``)."""
    if not priors_path.is_file():
        _die(f"priors file not found: {priors_path}")
    try:
        priors = json.loads(priors_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _die(f"invalid JSON in {priors_path}: {e}")
    intrinsics = priors.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        _die("'camera_intrinsics' missing or not an object")
    model = intrinsics.get("model", "PINHOLE")
    if model not in _CAMERA_MODELS:
        _die(f"unsupported camera model {model!r}")
    colmap_name, param_keys = _CAMERA_MODELS[model]
    try:
        params = ",".join(f"{float(intrinsics[k]):.10g}" for k in param_keys)
    except (KeyError, TypeError, ValueError) as e:
        _die(f"camera_intrinsics missing/bad key: {e}")
    return (f"--ImageReader.camera_model {colmap_name} "
            f"--ImageReader.camera_params {params}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Orbiter sfm_priors.json to COLMAP text model.")
    parser.add_argument("input_json", type=Path, help="path to sfm_priors.json")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="output directory for COLMAP text files")
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="COLMAP database.db — align image/frame IDs after feature_extractor",
    )
    parser.add_argument(
        "--emit-extractor-flags",
        action="store_true",
        help="print feature_extractor camera-model flags for the priors and "
             "exit (no files written)",
    )
    args = parser.parse_args(argv)
    if args.emit_extractor_flags:
        print(extractor_flags(args.input_json))
        return 0
    if args.output_dir is None:
        parser.error("output_dir is required unless --emit-extractor-flags")
    convert(args.input_json, args.output_dir, args.database)
    return 0


if __name__ == "__main__":
    sys.exit(main())
