from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class Geometry(BaseModel):
    arm_radius_mm: float
    camera_offset_mm: float
    base_height_mm: float
    look_at_xyz_mm: Vec3 = Vec3(x=0, y=0, z=0)
    frame: str = (
        "right-handed; +Z up (az axis), +X = arm direction at az=0,el=0, "
        "origin at axis intersection"
    )


class EncoderZero(BaseModel):
    """Firmware encoder zero values.

    Distinct from any board / hand-eye procedure: these are the raw-encoder
    offsets persisted on the ESP32 (`/calibrate` endpoint) so user-space
    angles read zero at the physical home pose.
    """

    az_zero_raw_deg: float
    el_zero_raw_deg: float


class ScanParams(BaseModel):
    el_start_deg: float
    el_max_deg: float
    el_steps: int
    az_step_deg: float


class ScanPathPoint(BaseModel):
    """One planned point on the scan trajectory, in execution order."""

    index: int
    az_deg: float
    el_deg: float


class CreateScanReq(BaseModel):
    geometry: Geometry
    encoder_zero: EncoderZero

    # Automated-scan parameters — omitted for a manual shot-by-shot session.
    params: ScanParams | None = None
    # True = automated scan-loop run; False = manual session (the default).
    machine_captured: bool = False
    path: list[ScanPathPoint] = Field(default_factory=list)
    # Snapshot of the MotionPlanner that drove a machine scan (None for manual
    # sessions). Stored as a dict so the schema can evolve without migrations.
    motion_plan: dict[str, Any] | None = None


class CaptureMeta(BaseModel):
    """Pose metadata from client (e.g. encoder-backed angles from GET /state)."""

    model_config = ConfigDict(extra="ignore")

    index: int
    az_deg: float
    el_deg: float
    camera_xyz_mm: Vec3
    look_at_xyz_mm: Vec3
    optical_axis_unit: Vec3
    # Full camera orientation as a three.js object-frame quaternion
    # [x, y, z, w] (-Z along the optical axis). Optional.
    camera_quat: list[float] | None = None
    timestamp: str
    planned_az_deg: float | None = None
    planned_el_deg: float | None = None
    # Legacy hint — kept for back-compat with older clients that still send
    # portrait/landscape. New clients use camera_preset instead.
    orientation: Literal["landscape", "portrait"] | None = None
    camera_preset: str | None = None
    # MotionPlanner action that produced this capture (`photo`, `photo_flash`).
    # None for manual `Take shot` and legacy captures.
    action: str | None = None


class Capture(CaptureMeta):
    capture_id: str
    thumb_url: str
    # New tiers — see storage.save_capture_with_thumb. Optional in the schema
    # so that old manifests on disk (written before the multi-tier change)
    # still validate; the validator below derives them from capture_id.
    thumb_small_url: str | None = None
    thumb_tiny_url: str | None = None
    full_url: str
    meta_url: str
    # Post-rotation pixel dimensions of the stored `original.jpg`. The 3D
    # viewer reads these to size camera frustums correctly when the
    # `camera_preset` rotates the file.
    stored_width: int | None = None
    stored_height: int | None = None

    @model_validator(mode="after")
    def _fill_thumb_urls(self) -> "Capture":
        if self.thumb_small_url is None:
            self.thumb_small_url = f"/captures/{self.capture_id}/thumb/small"
        if self.thumb_tiny_url is None:
            self.thumb_tiny_url = f"/captures/{self.capture_id}/thumb/tiny"
        return self


class Manifest(BaseModel):
    """Scan manifest on disk — the scan *document*.

    A scan is a single JSON document tying together its photos (each with
    pose metadata) and a snapshot of the rig geometry + firmware encoder
    zero values in effect when it was captured.

    `machine_captured` distinguishes an automated scan-loop run (True) from a
    manual shot-by-shot session (False — the current priority workflow).

    `archived=True` marks the scan as read-only history: files and capture
    references are kept, the scan is just hidden from active workflows.
    Archiving never touches `captures/` on disk.
    """

    model_config = ConfigDict(extra="ignore")

    scan_id: str
    created: str
    updated: str = ""
    # False = manual shot-by-shot session; True = automated scan-loop run.
    machine_captured: bool = False
    # Snapshot of the rig config + firmware encoder zeros in effect at capture time.
    geometry: Geometry
    encoder_zero: EncoderZero
    # Automated-scan parameters — None for a manual session.
    params: ScanParams | None = None
    captures: list[Capture] = Field(default_factory=list)
    path: list[ScanPathPoint] = Field(default_factory=list)
    # Snapshot of the MotionPlanner used (machine scans only). See MotionPlan.
    motion_plan: dict[str, Any] | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    archived: bool = False
    archived_at: str | None = None


class ScanSummary(BaseModel):
    scan_id: str
    created: str
    captures_count: int
    archived: bool = False
    archived_at: str | None = None


# ── MotionPlanner ──────────────────────────────────────────────────────────
#
# A scan's "plan" — the discrete sweep:
#
#   discrete   — elevation rings x azimuth step; at each ring point the loop
#                runs every checked `action` (one stored capture per action).
#
# `MotionPlan` wraps the discrete sub-plan. `mode` is retained (always
# "discrete") so the WS plan payload keeps a stable shape for the UI.

CaptureAction = Literal[
    "photo",                # plain photo (no flash)
    "photo_flash",          # phone-camera torch on for the shot
]


class DiscretePlan(BaseModel):
    """Elevation rings x azimuth step, with a per-point checked action list."""

    model_config = ConfigDict(extra="ignore")

    el_start_deg: float = 0.0
    el_max_deg: float = 60.0
    el_steps: int = 4
    az_step_deg: float = 20.0
    actions: list[CaptureAction] = Field(default_factory=lambda: ["photo"])


class MotionPlan(BaseModel):
    """The scan-loop plan owned by the server (`orbiter_model.motion_plan`)."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["discrete"] = "discrete"
    discrete: DiscretePlan = Field(default_factory=DiscretePlan)
