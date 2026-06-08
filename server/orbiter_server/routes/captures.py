from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import storage

router = APIRouter(prefix="/captures", tags=["captures"])

# Capture media is immutable: a capture_id never changes its pixels. Tell the
# browser it can cache forever and skip the revalidation roundtrip (304).
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


@router.get("/{capture_id}/thumb")
def get_thumb(capture_id: str):
    try:
        p = storage.capture_path(capture_id, "thumb")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"thumb not found for {capture_id}")
    return FileResponse(p, media_type="image/jpeg", headers=_IMMUTABLE)


@router.get("/{capture_id}/thumb/tiny")
def get_thumb_tiny(capture_id: str):
    # Graceful fallback to the medium tier if the tiny file hasn't been
    # generated yet (legacy captures, before tools/regenerate_thumbs.py was
    # run). The URL contract is "always returns an image", so the 3D viewer
    # never hits a hard 404 → useLoader error → Canvas crash.
    for kind in ("thumb_tiny", "thumb"):
        try:
            return FileResponse(storage.capture_path(capture_id, kind),
                                media_type="image/jpeg", headers=_IMMUTABLE)
        except FileNotFoundError:
            continue
    raise HTTPException(status_code=404, detail=f"thumb not found for {capture_id}")


@router.get("/{capture_id}/thumb/small")
def get_thumb_small(capture_id: str):
    # Same fallback as /thumb/tiny — see comment there.
    for kind in ("thumb_small", "thumb"):
        try:
            return FileResponse(storage.capture_path(capture_id, kind),
                                media_type="image/jpeg", headers=_IMMUTABLE)
        except FileNotFoundError:
            continue
    raise HTTPException(status_code=404, detail=f"thumb not found for {capture_id}")


@router.get("/{capture_id}/full")
def get_full(capture_id: str):
    try:
        p = storage.capture_path(capture_id, "full")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"full image not found for {capture_id}")
    return FileResponse(p, media_type="image/jpeg", headers=_IMMUTABLE)


@router.get("/{capture_id}/meta")
def get_meta(capture_id: str):
    try:
        p = storage.capture_path(capture_id, "meta")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"meta not found for {capture_id}")
    return FileResponse(p, media_type="application/json", headers=_IMMUTABLE)
