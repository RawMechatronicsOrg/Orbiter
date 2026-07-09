from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import storage
from models import Capture, CaptureMeta, Manifest
from orbiter_model import model

router = APIRouter(prefix="/scans/{scan_id}/photos", tags=["photos"])

# Photo media is immutable per capture — cache forever, skip the 304 roundtrip.
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _reject_if_active(scan_id: str) -> None:
    """409 while the scan is the device's active session — the scan loop
    rebuilds the manifest from memory (write-through), so a REST mutation
    would be resurrected with its files already gone. Use the
    ``delete_capture`` WS command for the live session instead."""
    if model.current_scan_id == scan_id:
        raise HTTPException(
            status_code=409,
            detail=f"scan {scan_id} is the active session — "
                   "mutate it through the device commands instead",
        )


@router.post("", response_model=Capture)
async def upload_photo(
    scan_id: str,
    file: UploadFile = File(...),
    meta: str = Form(...),
) -> Capture:
    try:
        meta_obj = CaptureMeta.model_validate_json(meta)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid meta JSON: {exc}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        capture_id, _orig, _thumb, _meta, stored_w, stored_h = storage.save_capture_with_thumb(
            scan_id, meta_obj, raw
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"scan {scan_id} not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"save failed: {exc}")

    cid = capture_id
    capture = Capture(
        **meta_obj.model_dump(),
        capture_id=cid,
        thumb_url=f"/captures/{cid}/thumb",
        thumb_small_url=f"/captures/{cid}/thumb/small",
        thumb_tiny_url=f"/captures/{cid}/thumb/tiny",
        full_url=f"/captures/{cid}/full",
        meta_url=f"/captures/{cid}/meta",
        stored_width=stored_w,
        stored_height=stored_h,
    )
    storage.append_capture(scan_id, capture)
    return capture


@router.delete("/{capture_id}", response_model=Manifest)
def delete_photo(scan_id: str, capture_id: str) -> Manifest:
    """Delete one capture from a stored scan.

    Removes the manifest entry, the capture's pool files, the materialized
    ``photos/`` link, any mask/preview built from it, and the stale priors
    file. Returns the updated manifest so the UI can refresh without a
    second GET."""
    _reject_if_active(scan_id)
    try:
        return storage.delete_scan_capture(scan_id, capture_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"capture {capture_id} not found in scan {scan_id}",
        )


@router.get("/{idx}/thumb")
def get_thumb(scan_id: str, idx: int):
    try:
        p = storage.photo_path(scan_id, idx, "thumb")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="thumb not found")
    return FileResponse(p, media_type="image/jpeg", headers=_IMMUTABLE)


@router.get("/{idx}/thumb/tiny")
def get_thumb_tiny(scan_id: str, idx: int):
    # See captures.py::get_thumb_tiny — falls back to the medium tier when
    # the tier files haven't been backfilled yet for legacy scans.
    for kind in ("thumb_tiny", "thumb"):
        try:
            return FileResponse(storage.photo_path(scan_id, idx, kind),
                                media_type="image/jpeg", headers=_IMMUTABLE)
        except FileNotFoundError:
            continue
    raise HTTPException(status_code=404, detail="thumb not found")


@router.get("/{idx}/thumb/small")
def get_thumb_small(scan_id: str, idx: int):
    for kind in ("thumb_small", "thumb"):
        try:
            return FileResponse(storage.photo_path(scan_id, idx, kind),
                                media_type="image/jpeg", headers=_IMMUTABLE)
        except FileNotFoundError:
            continue
    raise HTTPException(status_code=404, detail="thumb not found")


@router.get("/{idx}/full")
def get_full(scan_id: str, idx: int):
    try:
        p = storage.photo_path(scan_id, idx, "full")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="full not found")
    return FileResponse(p, media_type="image/jpeg", headers=_IMMUTABLE)


# Mask media: regenerated in place by the masks tool between runs — always
# revalidate (cheap 304s keep repeat visits snappy without staleness).
_MASK_REVALIDATE = {"Cache-Control": "no-cache"}


@router.get("/{idx}/mask")
def get_mask(scan_id: str, idx: int):
    """COLMAP mask for one capture (white = features kept). 404 until the
    masks tool has produced it (see colmap/masks/README.md)."""
    try:
        p = storage.photo_mask_path(scan_id, idx)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="mask not found")
    return FileResponse(p, media_type="image/png", headers=_MASK_REVALIDATE)


@router.get("/{idx}/mask_preview")
def get_mask_preview(scan_id: str, idx: int):
    """Debug overlay (mask painted over the frame, green outline) written by
    the masks tool next to the masks. 404 when the run skipped previews or
    the frame has none."""
    try:
        p = storage.photo_mask_preview_path(scan_id, idx)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="mask preview not found")
    return FileResponse(p, media_type="image/jpeg", headers=_MASK_REVALIDATE)
