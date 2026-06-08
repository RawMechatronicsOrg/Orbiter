from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import storage
from models import Capture, CaptureMeta

router = APIRouter(prefix="/scans/{scan_id}/photos", tags=["photos"])

# Photo media is immutable per capture — cache forever, skip the 304 roundtrip.
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


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
