"""Disk operations: global capture pool, scan manifests, atomic JSON writes, thumbnails.

Pixel-level photo transforms (rotation, EXIF, multi-tier resize) live in
camera_adapter.py — this module just orchestrates disk layout, manifest
state, and scan-archive (zip) packing.
"""

from __future__ import annotations

import io
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import camera_adapter
from camera_adapter import (
    THUMB_TIERS,
    PresetName,
    get_preset,
    photo_basename,
    process_capture,
)
from config import settings
from models import Capture, CaptureMeta, CreateScanReq, Manifest

# Filenames are owned by camera_adapter.THUMB_TIERS; aliases kept for the
# few legacy callers that still want a bare constant.
ORIGINAL_NAME = "original.jpg"
THUMB_NAME = THUMB_TIERS[0].filename         # medium tier
THUMB_SMALL_NAME = THUMB_TIERS[1].filename   # sidebar preview tier
THUMB_TINY_NAME = THUMB_TIERS[2].filename    # in-scene texture tier
META_NAME = "meta.json"


def _new_scan_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{ts}_{secrets.token_hex(3)}"


def _new_capture_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{ts}_{secrets.token_hex(4)}"


def _scan_dir(scan_id: str) -> Path:
    return settings.scans_dir / scan_id


def _capture_dir(capture_id: str) -> Path:
    return settings.captures_dir / capture_id


def _manifest_path(scan_id: str) -> Path:
    return _scan_dir(scan_id) / "manifest.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def create_scan(req: CreateScanReq) -> Manifest:
    scan_id = _new_scan_id()
    sdir = _scan_dir(scan_id)
    sdir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = Manifest(
        scan_id=scan_id,
        created=now,
        updated=now,
        machine_captured=req.machine_captured,
        params=req.params,
        geometry=req.geometry,
        encoder_zero=req.encoder_zero,
        captures=[],
        path=req.path,
        motion_plan=req.motion_plan,
    )
    _atomic_write_json(_manifest_path(scan_id), manifest.model_dump(mode="json"))
    return manifest


def write_manifest(manifest: Manifest) -> None:
    """Atomically overwrite a scan's manifest.json, stamping ``updated``.

    The single 'save the scan document' primitive — used by the explicit
    Save command and the debounced autosave (see scan_task.py)."""
    stamped = manifest.model_copy(update={
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    _scan_dir(stamped.scan_id).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_manifest_path(stamped.scan_id),
                       stamped.model_dump(mode="json"))


def list_scans() -> list[Manifest]:
    if not settings.scans_dir.exists():
        return []
    out: list[Manifest] = []
    for d in sorted(settings.scans_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        try:
            out.append(read_manifest(d.name))
        except FileNotFoundError:
            continue
    return out


def read_manifest(scan_id: str) -> Manifest:
    p = _manifest_path(scan_id)
    if not p.exists():
        raise FileNotFoundError(scan_id)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return Manifest.model_validate(raw)


def archive_scan(scan_id: str) -> Manifest:
    """Mark scan as archived (read-only history). Captures on disk are kept.

    Idempotent: archiving an already-archived scan returns it unchanged.
    """
    manifest = read_manifest(scan_id)
    if manifest.archived:
        return manifest
    manifest = manifest.model_copy(
        update={
            "archived": True,
            "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    _atomic_write_json(_manifest_path(scan_id), manifest.model_dump(mode="json"))
    return manifest


def delete_scan(scan_id: str) -> None:
    """Remove scan manifest folder only; capture files under captures/ are kept."""
    sdir = _scan_dir(scan_id)
    if not sdir.exists():
        raise FileNotFoundError(scan_id)
    for root, dirs, files in os.walk(sdir, topdown=False):
        for fn in files:
            try:
                os.unlink(Path(root) / fn)
            except OSError:
                pass
        for dn in dirs:
            try:
                (Path(root) / dn).rmdir()
            except OSError:
                pass
    sdir.rmdir()


def save_capture_with_thumb(
    scan_id: str, meta: CaptureMeta, raw_bytes: bytes
) -> tuple[str, Path, Path, Path, int, int]:
    """Write captures/<id>/{original,thumb,thumb_small,thumb_tiny}.jpg + meta.json.

    All pixel-level work is delegated to ``camera_adapter.process_capture``
    using the preset named on the capture (``meta.camera_preset``) or the
    server default (``settings.default_camera_preset``) when the caller
    didn't specify one.

    Returns ``(capture_id, original_path, thumb_path, meta_path,
    stored_width, stored_height)`` — width/height come AFTER preset
    rotation so callers can persist them on the capture record.
    """
    if not _scan_dir(scan_id).exists():
        raise FileNotFoundError(scan_id)

    capture_id = _new_capture_id()
    cdir = _capture_dir(capture_id)

    preset_name: PresetName = meta.camera_preset or settings.default_camera_preset
    preset = get_preset(preset_name)

    result = process_capture(raw_bytes, preset=preset, out_dir=cdir,
                             original_name=ORIGINAL_NAME)

    meta_path = cdir / META_NAME
    disk_meta = {
        **meta.model_dump(mode="json", exclude_none=True),
        "capture_id": capture_id,
        "scan_id": scan_id,
        "camera_preset": preset.name,
        "stored_width": result.width,
        "stored_height": result.height,
    }
    meta_path.write_text(json.dumps(disk_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return capture_id, result.original_path, result.thumb_path, meta_path, result.width, result.height


def delete_capture_media(capture_id: str) -> bool:
    """Remove one capture's whole pool directory (original.jpg + every thumb
    tier + meta.json) by capture_id.

    The mirror of ``save_capture_with_thumb`` on the way out. Like
    ``delete_scan`` it is OSError-tolerant — already-missing files are fine,
    and a stray file left in the dir doesn't stop the rmdir — so a partially
    written capture (e.g. a crash mid-``process_capture``) still cleans up.

    Returns True if the pool directory existed and was removed, False if there
    was nothing on disk for this capture_id (idempotent — safe to call twice).
    """
    cdir = _capture_dir(capture_id)
    if not cdir.exists():
        return False
    # Walk bottom-up so child files are unlinked before their dir is removed.
    for root, dirs, files in os.walk(cdir, topdown=False):
        for fn in files:
            try:
                os.unlink(Path(root) / fn)
            except OSError:
                pass
        for dn in dirs:
            try:
                (Path(root) / dn).rmdir()
            except OSError:
                pass
    try:
        cdir.rmdir()
    except OSError:
        # A leftover (locked/undeletable) file keeps the dir around — the
        # capture is still functionally gone from the pool; don't crash.
        return False
    return True


def capture_path(capture_id: str, kind: str) -> Path:
    """kind ∈ {'full', 'thumb', 'thumb_small', 'thumb_tiny', 'meta'}."""
    if kind == "full":
        p = _capture_dir(capture_id) / ORIGINAL_NAME
    elif kind == "thumb":
        p = _capture_dir(capture_id) / THUMB_NAME
    elif kind == "thumb_small":
        p = _capture_dir(capture_id) / THUMB_SMALL_NAME
    elif kind == "thumb_tiny":
        p = _capture_dir(capture_id) / THUMB_TINY_NAME
    elif kind == "meta":
        p = _capture_dir(capture_id) / META_NAME
    else:
        raise ValueError(f"invalid kind: {kind}")
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _photo_filename(capture: Capture) -> str:
    """Pose-encoded basename. Single source of truth lives in
    ``camera_adapter.photo_basename`` — keep this thin wrapper so the
    rest of storage.py reads naturally."""
    return photo_basename(capture.index, capture.az_deg, capture.el_deg)


def build_scan_archive(scan_id: str) -> bytes:
    """Pack a scan into an in-memory ZIP and return its bytes.

    Layout:
        manifest.json
        sfm_priors.json                        — COLMAP-ready per-capture poses
        photos/{NNN}_az{AAA}_el{EEE}.jpg     — full-res originals
        photos/{NNN}_az{AAA}_el{EEE}.meta.json — per-capture metadata
    """
    import zipfile

    manifest = read_manifest(scan_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
        )
        # SfM priors so the archive is COLMAP-ready out of the box. Lazy import
        # avoids a storage <-> sfm_export cycle; a failure is non-fatal — the
        # archive is still useful without them.
        try:
            from sfm_export import build_sfm_priors
            zf.writestr(
                "sfm_priors.json",
                json.dumps(build_sfm_priors(scan_id), indent=2, ensure_ascii=False),
            )
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger("orbiter.storage").warning(
                "scan archive %s: sfm_priors export failed", scan_id, exc_info=True,
            )
        for c in sorted(manifest.captures, key=lambda c: c.index):
            fname = _photo_filename(c)
            stem = fname[:-4]  # strip ".jpg"
            try:
                full = capture_path(c.capture_id, "full")
                zf.write(full, arcname=f"photos/{fname}")
            except FileNotFoundError:
                continue
            try:
                meta = capture_path(c.capture_id, "meta")
                zf.write(meta, arcname=f"photos/{stem}.meta.json")
            except FileNotFoundError:
                pass
    buf.seek(0)
    return buf.getvalue()


def photo_path(scan_id: str, idx: int, kind: str) -> Path:
    """Resolve scan index → capture_id (manifest); then file on disk."""
    if kind not in ("full", "thumb", "thumb_small", "thumb_tiny"):
        raise ValueError(f"invalid kind: {kind}")
    manifest = read_manifest(scan_id)
    cap = next((c for c in manifest.captures if c.index == idx), None)
    if cap is None:
        raise FileNotFoundError(f"capture {idx} in scan {scan_id}")
    return capture_path(cap.capture_id, kind)
