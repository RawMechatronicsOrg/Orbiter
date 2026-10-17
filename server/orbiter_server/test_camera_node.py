"""The CameraStreamer capture-node path: camera_io against a stand-in phone,
and the sweep moving the rig on while photos are still being pulled. Also the
IP Webcam path where it shares code: a failed still raises, never a blank."""

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
WEBCAM_JPEG = b"\xff\xd8webcam\xff\xd9"


class FakePhone:
    """Just enough of docs/CAPTURE_API.md: status, capabilities, profile
    operations, shoot, file, record, delete. With any other ``app`` it plays
    IP Webcam: no ``/status``, stills on ``/photo.jpg``."""

    def __init__(self, app: str = "camerastreamer", mode: str = "locked") -> None:
        self.app = app
        self.mode = mode
        self.sensor_orientation = 90
        self.shots: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.fail_with: dict[str, int] = {}  # shot id -> HTTP status for POST /shots
        self.busy_times = 0  # POST /shots answers 503 this many times first
        self.bad_sha = False
        self.photo = httpx.Response(200, content=WEBCAM_JPEG,
                                    headers={"content-type": "image/jpeg"})

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        if self.app != "camerastreamer":
            return self.photo if path == "/photo.jpg" else httpx.Response(404, text="Not found")
        if path == "/status":
            return httpx.Response(200, json={
                "app": self.app, "version": "0.1.0", "state": "ready", "mode": self.mode,
                "settled": self.mode == "locked", "camera_id": "0", "still_size": [4080, 3060],
                "profile": {"exposure_ns": 10_000_000, "iso": 100, "focus_diopters": 3.0,
                            "wb_mode": "manual", "torch": False},
                "profile_fingerprint": "abc123", "shots": {"stored": len(self.shots)},
                "frame": {"sensor_timestamp_ns": 123},
                "device": {"battery_pct": 80, "charging": True, "thermal": "none"},
            })
        if path == "/capabilities":
            return httpx.Response(200, json={"camera_id": "0",
                                             "sensor_orientation": self.sensor_orientation})
        if request.method == "POST" and path in ("/lock", "/profile", "/unlock"):
            self.mode = "metering" if path == "/unlock" else "locked"
            return httpx.Response(200, json={"settled": True, "body": json.loads(request.content)})
        if request.method == "POST" and path == "/shots":
            body = json.loads(request.content)
            sid = body["id"]
            if self.mode != "locked":
                return httpx.Response(409, json={"error": "no locked profile"})
            if self.busy_times > 0:
                self.busy_times -= 1
                return httpx.Response(503, json={"error": "not settled: focus"})
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
    camera_io._caps_cache.clear()
    monkeypatch.setattr(camera_io, "_PHOTO_BACKOFF_S", 0.0)
    previous = model.camera_url, model.camera_preset
    model.update(camera_url=BASE, camera_preset="native")
    yield fake
    model.update(camera_url=previous[0], camera_preset=previous[1],
                 camera_kind="none", camera_node=None, camera_quarter_turns=0)
    camera_io._node_cache.clear()
    camera_io._caps_cache.clear()


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

    async def preset() -> str:
        return "sensor90"

    async def nothing(*_args: Any) -> None:
        return None

    def save(scan_id: str, meta: Any, raw: bytes) -> tuple[str, None, None, None, int, int]:
        return f"cap-{meta.index}", None, None, None, 4032, 3024

    monkeypatch.setattr(scan_task, "esp_move", move)
    monkeypatch.setattr(camera_io, "is_capture_node", is_node)
    monkeypatch.setattr(camera_io, "capture_preset", preset)
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
    assert all(c["camera_preset"] == "sensor90" for c in captures)
    # A photo is released on the phone only after it is stored.
    assert rig.index("stored 0") < rig.index("released 0")
    assert "released 2" not in rig
    failed = model.scan_failed_points
    assert len(failed) == 1
    assert failed[0]["index"] == 2 and "left on the phone" in failed[0]["stage"]


# ── a failed still is an error, never a stand-in ───────────────────────────


def test_no_camera_url_is_an_error_not_a_placeholder(phone: FakePhone) -> None:
    model.update(camera_url="")
    with pytest.raises(camera_io.CameraError, match="not configured"):
        asyncio.run(camera_io.fetch_photo())


def test_ip_webcam_photo_is_returned(phone: FakePhone) -> None:
    phone.app = "ipwebcam"
    assert asyncio.run(camera_io.fetch_photo()) == WEBCAM_JPEG


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="camera busy"),
    httpx.Response(200, content=b"<html>login</html>", headers={"content-type": "text/html"}),
])
def test_an_ip_webcam_failure_raises(phone: FakePhone, response: httpx.Response) -> None:
    phone.app = "ipwebcam"
    phone.photo = response
    with pytest.raises(camera_io.CameraError):
        asyncio.run(camera_io.fetch_photo())


def test_an_unlocked_phone_refuses_the_shot_with_a_hint(phone: FakePhone) -> None:
    phone.mode = "metering"
    with pytest.raises(camera_io.ShotError, match="not locked") as exc:
        asyncio.run(camera_io.fetch_photo())
    assert exc.value.definitive is False


def test_fetch_photo_retries_a_busy_phone_with_the_same_id(phone: FakePhone) -> None:
    phone.busy_times = 2
    assert asyncio.run(camera_io.fetch_photo()).startswith(b"\xff\xd8")
    posted = [c for c in phone.calls if c == "POST /shots"]
    assert len(posted) == 3
    assert phone.shots == {}


def test_a_scan_refuses_to_start_without_a_camera(phone: FakePhone) -> None:
    previous = model.arm_radius_mm
    model.update(camera_url="", arm_radius_mm=max(previous, 100.0))
    try:
        with pytest.raises(RuntimeError, match="camera_url is not configured"):
            asyncio.run(scan_task.start_scan())
    finally:
        model.update(arm_radius_mm=previous)


# ── orientation, camera kind, node state ───────────────────────────────────


def test_a_node_capture_is_oriented_by_the_sensor(phone: FakePhone) -> None:
    assert asyncio.run(camera_io.capture_preset()) == "sensor90"
    phone.sensor_orientation = 270
    camera_io._caps_cache.clear()
    assert asyncio.run(camera_io.capture_preset()) == "sensor270"


def test_ip_webcam_keeps_the_operator_preset(phone: FakePhone) -> None:
    phone.app = "ipwebcam"
    model.update(camera_preset="sm22")
    assert asyncio.run(camera_io.capture_preset()) == "sm22"


def test_the_monitor_publishes_the_node(phone: FakePhone) -> None:
    asyncio.run(camera_io.monitor.refresh())
    assert model.camera_kind == "capture_node"
    assert model.camera_quarter_turns == 1
    node = model.camera_node
    assert node is not None
    assert node["mode"] == "locked" and node["sensor_orientation"] == 90
    assert node["profile"]["iso"] == 100 and node["battery_pct"] == 80
    assert "frame" not in node  # per-frame values would broadcast every tick


@pytest.mark.parametrize("app, url, kind", [
    ("ipwebcam", BASE, "ipwebcam"),
    ("camerastreamer", "", "none"),
])
def test_the_monitor_tells_the_kinds_apart(phone: FakePhone, app: str, url: str, kind: str) -> None:
    phone.app = app
    model.update(camera_url=url, camera_preset="sm22")
    asyncio.run(camera_io.monitor.refresh())
    # Without a node the preview turns like the operator's preset.
    assert (model.camera_kind, model.camera_node, model.camera_quarter_turns) == (kind, None, 1)


def test_an_unreachable_phone_is_offline(phone: FakePhone, monkeypatch: pytest.MonkeyPatch) -> None:
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    transport = httpx.MockTransport(down)
    monkeypatch.setattr(camera_io, "_client", lambda: httpx.AsyncClient(transport=transport))
    asyncio.run(camera_io.monitor.refresh())
    assert model.camera_kind == "offline" and model.camera_node is None


def test_lock_and_unlock_go_through_to_the_node(phone: FakePhone) -> None:
    phone.mode = "metering"
    answer = asyncio.run(camera_io.node_control("lock", {"iso": 100}))
    assert answer["body"] == {"iso": 100}
    assert model.camera_node["mode"] == "locked"
    asyncio.run(camera_io.node_control("unlock", {}))
    assert model.camera_node["mode"] == "metering"
    with pytest.raises(camera_io.CameraError, match="unknown"):
        asyncio.run(camera_io.node_control("reboot", {}))


def test_node_operations_refuse_ip_webcam(phone: FakePhone) -> None:
    phone.app = "ipwebcam"
    with pytest.raises(camera_io.CameraError, match="not a capture node"):
        asyncio.run(camera_io.node_control("lock", {}))
