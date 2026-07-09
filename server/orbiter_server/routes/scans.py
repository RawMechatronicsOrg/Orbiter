import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import scan_task
import storage
from models import CreateScanReq, Manifest, ScanSummary
from orbiter_model import model
from sfm_export import write_sfm_priors

router = APIRouter(prefix="/scans", tags=["scans"])


class ScanPatch(BaseModel):
    """PATCH /scans/{id} body — only the user-editable catalog fields."""

    notes: str | None = None
    tags: list[str] | None = None


@router.post("", response_model=Manifest)
def create_scan(req: CreateScanReq) -> Manifest:
    return storage.create_scan(req)


@router.get("", response_model=list[ScanSummary])
def list_scans() -> list[ScanSummary]:
    return [
        ScanSummary(
            scan_id=m.scan_id,
            created=m.created,
            captures_count=len(m.captures),
            archived=m.archived,
            archived_at=m.archived_at,
        )
        for m in storage.list_scans()
    ]


@router.get("/{scan_id}", response_model=Manifest)
def get_scan(scan_id: str) -> Manifest:
    try:
        return storage.read_manifest(scan_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"scan {scan_id} not found")


@router.patch("/{scan_id}", response_model=Manifest)
def patch_scan(scan_id: str, patch: ScanPatch) -> Manifest:
    """Update the user-editable catalog fields (notes / tags) of a scan.

    409 while the scan is the device's active session — this is a whole-
    document read-modify-write that could drop a capture appended in
    between; the Scaner tab edits the active session's notes instead."""
    if model.current_scan_id == scan_id:
        raise HTTPException(
            status_code=409,
            detail=f"scan {scan_id} is the active session — "
                   "edit its notes in the Scaner tab instead",
        )
    try:
        manifest = storage.read_manifest(scan_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"scan {scan_id} not found")
    update = patch.model_dump(exclude_none=True)
    if update:
        manifest = manifest.model_copy(update=update)
        storage.write_manifest(manifest)
        manifest = storage.read_manifest(scan_id)  # pick up the stamped `updated`
    return manifest


@router.get("/{scan_id}/download")
def download_scan(scan_id: str) -> Response:
    """Download a ZIP with the scan's manifest + full-res photos + per-photo meta."""
    try:
        data = storage.build_scan_archive(scan_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"scan {scan_id} not found")
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{scan_id}.zip"',
            "Content-Length": str(len(data)),
        },
    )


# NOTE: these mutate model.scans via publish_scan_list(), so they MUST be
# `async def`. A sync route runs in FastAPI's threadpool — no running event
# loop — and the model's change broadcast bails out there (it needs the loop),
# so the deletion/archival would never reach the UI (the Library kept showing
# the "deleted" scan). Running on the loop makes the broadcast actually fire.
@router.post("/{scan_id}/archive", response_model=Manifest)
async def archive_scan(scan_id: str) -> Manifest:
    """Mark scan as archived. Files on disk are preserved; UI hides it from active workflows."""
    try:
        manifest = await asyncio.to_thread(storage.archive_scan, scan_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"scan {scan_id} not found")
    scan_task.publish_scan_list()   # reflect the archived flag in the Library
    return manifest


@router.delete("/{scan_id}")
async def delete_scan(scan_id: str) -> dict:
    """Hard-delete the manifest folder (captures/ on disk are NOT removed)."""
    try:
        await asyncio.to_thread(storage.delete_scan, scan_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"scan {scan_id} not found")
    scan_task.publish_scan_list()   # drop it from the Library list (broadcasts now)
    return {"status": "ok"}


@router.post("/{scan_id}/sfm_priors")
def export_sfm_priors(scan_id: str) -> dict:
    """Export SfM priors for the given scan as `sfm_priors.json` and stage
    the session for COLMAP.

    Reads the scan manifest, computes per-photo camera poses (Hamilton
    quaternion + translation in mm, world->camera direction, COLMAP
    convention) and writes the result next to the manifest. Also
    materializes the capture originals into ``scans/<id>/photos/`` — the
    layout `colmap/run_colmap_session.sh` (and the masks tool) read. See
    `docs/COLMAP.md` for the schema.
    """
    try:
        photos = storage.materialize_scan_photos(scan_id)
        path = write_sfm_priors(scan_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"scan {scan_id} not found")
    return {
        "path": str(path),
        "photos_ready": photos.ready,
        "photos_missing": photos.missing,
    }
