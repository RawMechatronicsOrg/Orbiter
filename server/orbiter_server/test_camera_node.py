"""The CameraStreamer capture-node path: camera_io against a stand-in phone,
and the sweep moving the rig on while photos are still being pulled."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx
import pytest

import camera_io
import scan_task
from models import DiscretePlan
from orbiter_model import model

BASE = "http://phone.test:8080"


class FakePhone:
    """Just enough of docs/CAPTURE_API.md: status, shoot, file, record, delete."""

    def __init__(self, app: str = "camerastreamer", mode: str = "locked") -> None:
        self.app = app
        self.mode = mode
        self.shots: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.fail_with: dict[str, int] = {}  # shot id -> HTTP status for POST /shots
        self.bad_sha = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        if path == "/status":
            return httpx.Response(200, json={"app": self.app, "state": "ready", "mode": self.mode})
        if request.method == "POST" and path == "/shots":
            body = json.loads(request.content)
            sid = body["id"]
            if sid in self.fail_with:
                code = self.fail_with[sid]
                state = "failed" if code == 500 else None
                return httpx.Response(code, json={"error": "nope", "id": sid, "state": state})
            rec = self.shots.setdefault(sid, {
                "id": sid, "state": "exposed", "meta": body.get("meta"),
                "profile_fingerprint": "abc123", "verified": True, "deviations": [],
                "frame": {"exposure_ns": 10_000_000, "iso": 100, "focus_diopters": 3.0},
                "jpeg": b"\xff\xd8jpeg-" + sid.encode() + b"\xff\xd9",
            })
            return httpx.Response(201, json={k: v for k, v in rec.items() if k != "jpeg"})
        if path.startswith("/shots/"):
            name = path[len("/shots/"):]
            sid = name.removesuffix(".jpg")
            rec = self.shots.get(sid)
            if rec is None:
                return httpx.Response(404, json={"error": "not found"})
            if request.method == "DELETE":
                del self.shots[sid]
                return httpx.Response(200, json={"deleted": sid})
            if name.endswith(".jpg"):
                sha = "0" * 64 if self.bad_sha else hashlib.sha256(rec["jpeg"]).hexdigest()
                return httpx.Response(200, content=rec["jpeg"], headers={"X-Shot-Sha256": sha})
            return httpx.Response(200, json={**{k: v for k, v in rec.items() if k != "jpeg"},
                                             "state": "stored"})
        return httpx.Response(404, json={"error": "no route"})


@pytest.fixture
def phone(monkeypatch: pytest.MonkeyPatch) -> FakePhone:
    fake = FakePhone()
    transport = httpx.MockTransport(fake.handler)
    monkeypatch.setattr(camera_io, "_client",
                        lambda: httpx.AsyncClient(transport=transport))
    camera_io._node_cache.clear()
    previous = model.camera_url
    model.update(camera_url=BASE)
    yield fake
    model.update(camera_url=previous)
    camera_io._node_cache.clear()


def test_a_camerastreamer_phone_is_recognised(phone: FakePhone) -> None:
    assert asyncio.run(camera_io.is_capture_node()) is True


def test_ip_webcam_is_not_a_capture_node(phone: FakePhone) -> None:
    phone.app = "ipwebcam"
    assert asyncio.run(camera_io.is_capture_node()) is False


def test_a_scan_refuses_an_unlocked_phone(phone: FakePhone) -> None:
    phone.mode = "metering"
    with pytest.raises(camera_io.ShotError, match="not locked"):
        asyncio.run(camera_io.require_locked_node())


def test_busy_is_retryable_but_a_failed_shot_is_definitive(phone: FakePhone) -> None:
    phone.fail_with = {"busy": 503, "broken": 500}
    with pytest.raises(camera_io.ShotError) as busy:
        asyncio.run(camera_io.trigger_shot("busy", {}))
    assert busy.value.definitive is False
    with pytest.raises(camera_io.ShotError) as broken:
        asyncio.run(camera_io.trigger_shot("broken", {}))
    assert broken.value.definitive is True


def test_a_corrupted_transfer_is_caught(phone: FakePhone) -> None:
    asyncio.run(camera_io.trigger_shot("s1", {}))
    phone.bad_sha = True
    with pytest.raises(camera_io.ShotError, match="sha256"):
        asyncio.run(camera_io.fetch_shot("s1"))


def test_fetch_photo_shoots_pulls_and_then_frees_the_phone(phone: FakePhone) -> None:
    jpeg = asyncio.run(camera_io.fetch_photo())
    assert jpeg.startswith(b"\xff\xd8")
    assert phone.shots == {}
    methods = [c.split()[0] for c in phone.calls if c != "GET /status"]
    assert methods == ["POST", "GET", "GET", "DELETE"]


def test_retries_reuse_the_shot_id_until_the_phone_says_it_failed() -> None:
    seen: list[str] = []

    async def take(shot_id: str) -> str:
        seen.append(shot_id)
        if len(seen) == 1:
            raise camera_io.ShotError("timeout")
        if len(seen) == 2:
            raise camera_io.ShotError("failed on the phone", definitive=True)
        return shot_id

    attempt = scan_task._shot_attempts("scan-0001", take)

    async def run() -> str:
        for _ in range(3):
            try:
                return await attempt()
            except camera_io.ShotError:
                continue
        raise AssertionError("no attempt succeeded")

    assert asyncio.run(run()) == "scan-0001-r1"
    assert seen == ["scan-0001", "scan-0001", "scan-0001-r1"]


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A sweep with every piece of hardware and storage replaced by a timeline."""
    timeline: list[str] = []

    async def move(az: float, el: float) -> None:
        timeline.append(f"move {az:.0f}")
        await asyncio.sleep(0.01)

    async def trigger(shot_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        timeline.append(f"shot {meta['index']}")
        return {"id": shot_id, "state": "exposed"}

    async def fetch(shot_id: str) -> tuple[bytes, dict[str, Any]]:
        await asyncio.sleep(0.05)  # the transfer outlasts the next move
        index = int(shot_id.rsplit("-", 1)[1])
        if index == 2:
            raise camera_io.ShotError("wifi dropped")
        timeline.append(f"stored {index}")
        return b"jpeg", {"id": shot_id, "verified": True, "deviations": [],
                         "profile_fingerprint": "abc123", "frame": {"iso": 100}}

    async def release(shot_id: str) -> None:
        timeline.append(f"released {int(shot_id.rsplit('-', 1)[1])}")

    async def is_node() -> bool:
        return True

    async def nothing(*_args: Any) -> None:
        return None

    def save(scan_id: str, meta: Any, raw: bytes) -> tuple[str, None, None, None, int, int]:
        return f"cap-{meta.index}", None, None, None, 4032, 3024

    monkeypatch.setattr(scan_task, "esp_move", move)
    monkeypatch.setattr(camera_io, "is_capture_node", is_node)
    monkeypatch.setattr(camera_io, "trigger_shot", trigger)
    monkeypatch.setattr(camera_io, "fetch_shot", fetch)
    monkeypatch.setattr(camera_io, "release_shot", release)
    monkeypatch.setattr(scan_task.storage, "save_capture_with_thumb", save)
    monkeypatch.setattr(scan_task, "_persist_capture_now", nothing)
    monkeypatch.setattr(scan_task, "_OP_BACKOFF_S", (0.0, 0.0))
    model.update(captures=[], scan_failed_points=[], scan_progress=0)
    scan_task._abort.clear()
    return timeline


def test_the_rig_moves_on_while_photos_are_still_being_pulled(rig: list[str]) -> None:
    plan = DiscretePlan(el_start_deg=0, el_max_deg=0, el_steps=0, az_step_deg=120, actions=["photo"])
    asyncio.run(scan_task._run_discrete("scan", plan))

    # The second move starts before the first photo is stored.
    assert rig.index("move 60") < rig.index("stored 0")
    captures = model.captures
    assert [c["index"] for c in captures] == [0, 1, 3, 4, 5]
    assert all(c["camera_shot"]["verified"] for c in captures)
    # A photo is released on the phone only after it is stored.
    assert rig.index("stored 0") < rig.index("released 0")
    assert "released 2" not in rig
    failed = model.scan_failed_points
    assert len(failed) == 1
    assert failed[0]["index"] == 2 and "left on the phone" in failed[0]["stage"]
