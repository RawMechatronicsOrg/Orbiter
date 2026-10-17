"""Firmware proxy: /firmware/{health,encoder,ota}, the spin commands and the
EL spin guard, against a mocked firmware HTTP server."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import commands
from config import settings
from drive import axes_info
from esp_proxy import EspError, esp
from geom.angles import EL_MIN
from orbiter_model import model
from routes import firmware

HEALTH = {"status": "ok", "device": "Orbiter", "chip": "ESP32-D0WD-V3", "revision": 3,
          "cores": 2, "mac": "d4:e9:f4:88:f3:e8", "free_heap_bytes": 214320,
          "uptime_ms": 83400, "idf_version": "v6.0.1"}
ENCODER = {
    "azimuth": {"ok": True, "raw": 1024, "angle_deg": 90.0,
                "lines": {"sda": 1, "scl": 1, "external_pullups": 1}},
    "elevation": {"ok": False, "frerr": 1, "invcmd": 0, "parity": 0, "magnet_missing": 1,
                  "magnet_weak": 0, "magnet_strong": 0,
                  "lines": {"sda": 0, "scl": 1, "external_pullups": 0}},
}
IMAGE = bytes([0xE9]) + b"\x00" * 1023


def axis_result(valid: bool, spd: float = 0.0) -> dict[str, Any]:
    return {"valid": valid, "steps_per_deg": spd, "max_speed_dps": 40.0 if valid else 0.0,
            "dir_sign": -1, "compensation": True, "has_profile": valid,
            "profile_peak_deg": 0.4, "profile_rms_deg": 0.2, "profile_deg": [0.0] * 72}


MOTOR = {"run": {"active": False, "axis": "az", "phase": "done", "progress": 1.0,
                 "speed_dps": 0.0, "message": "done"},
         "az": axis_result(True, 61.08), "el": axis_result(False)}
STATE = {"state": "idle", "motors_enabled": True, "spinning_az": False, "spinning_el": False,
         "azimuth": {"angle_deg": 1.0}, "elevation": {"angle_deg": 2.0},
         "calibration": {"az_zero_raw_deg": 0.0, "el_zero_raw_deg": 138.296},
         "runner": {"id": 7, "kind": "motor_cal", "status": "done", "result": {}},
         "motor_cal": {"active": False,
                       "az": {"valid": True, "steps_per_deg": 61.2, "max_speed_dps": 40.0,
                              "dir_sign": -1},
                       "el": {"valid": True, "steps_per_deg": 90.0, "max_speed_dps": 6.0,
                              "dir_sign": 1}}}


class FakeFirmware:
    """Answers like the ESP32 HTTP server; records what reached it."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes, str]] = []
        self.ota_reply: tuple[int, dict[str, Any]] = (
            200, {"status": "ok", "message": "updated, rebooting"})
        self.down = False

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("unreachable", request=req)
        body = req.read()
        self.requests.append((req.method, req.url.path, body, req.headers.get("content-type", "")))
        route = (req.method, req.url.path)
        if route == ("GET", "/health"):
            return httpx.Response(200, json=HEALTH)
        if route == ("GET", "/test/encoder"):
            return httpx.Response(200, json=ENCODER)
        if route == ("GET", "/motor/calibration"):
            return httpx.Response(200, json=MOTOR)
        if route == ("GET", "/state"):
            return httpx.Response(200, json=STATE)
        if route in (("POST", "/move"), ("POST", "/motor/calibrate")):
            return httpx.Response(202, json={"status": "accepted", "task_id": 8})
        if route in (("POST", "/motor/calibrate/cancel"), ("POST", "/motor/compensation")):
            return httpx.Response(200, json={"status": "ok"})
        if route == ("POST", "/ota"):
            status, data = self.ota_reply
            return httpx.Response(status, json=data)
        if route in (("POST", "/spin"), ("POST", "/spin/stop")):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    def posted(self, path: str) -> list[dict[str, Any]]:
        return [json.loads(b) for m, p, b, _ in self.requests if m == "POST" and p == path]


@pytest.fixture
def fw(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    fake = FakeFirmware()
    monkeypatch.setattr(esp, "_client", httpx.AsyncClient(
        transport=httpx.MockTransport(fake), base_url="http://esp.test"))
    monkeypatch.setattr(esp, "_http_lock", asyncio.Lock(), raising=False)
    monkeypatch.setattr(esp, "base_url", "http://esp.test")
    monkeypatch.setattr(esp, "_last_el", None)
    yield fake
    model.update(scan_running=False, spinning_az=False, spinning_el=False,
                 encoder_zero_initialized=False, el=0.0, motor_cal=None, axes=axes_info())


@pytest.fixture
def client(fw) -> TestClient:
    app = FastAPI()
    app.include_router(firmware.router)
    return TestClient(app)


def test_health_and_encoder_pass_through(client, fw) -> None:
    r = client.get("/firmware/health")
    assert r.status_code == 200
    assert r.json() == {**HEALTH, "esp_ip": "esp.test"}
    assert client.get("/firmware/encoder").json() == ENCODER
    fw.down = True
    r = client.get("/firmware/health")
    assert r.status_code == 502
    assert "not reachable" in r.json()["detail"]


def test_ota_streams_the_image_and_relays_the_answer(client, fw) -> None:
    r = client.post("/firmware/ota", content=IMAGE,
                    headers={"Content-Type": "application/octet-stream"})
    assert (r.status_code, r.json()) == (200, {"status": "ok", "message": "updated, rebooting"})
    assert fw.requests[-1] == ("POST", "/ota", IMAGE, "application/octet-stream")

    fw.ota_reply = (409, {"status": "error", "message": "axes are moving, stop them first"})
    r = client.post("/firmware/ota", content=IMAGE)
    assert r.status_code == 409
    assert r.json()["message"].startswith("axes are moving")


@pytest.mark.parametrize(("body", "status"), [(b"", 400), (b"\x7fELF" + b"\x00" * 64, 400)])
def test_ota_refuses_what_is_not_an_app_image(client, fw, body, status) -> None:
    assert client.post("/firmware/ota", content=body).status_code == status
    assert fw.requests == []


def test_ota_refused_during_a_scan(client, fw) -> None:
    model.update(scan_running=True)
    assert client.post("/firmware/ota", content=IMAGE).status_code == 409
    assert fw.requests == []


def test_spin_commands(fw) -> None:
    asyncio.run(commands.dispatch("spin", {"axis": "az", "dir": "ccw", "step_hz": 800}))
    asyncio.run(commands.dispatch("spin_stop", {}))
    assert fw.posted("/spin") == [{"axis": "az", "dir": "ccw", "step_hz": 800}]
    assert fw.posted("/spin/stop") == [{"axis": "both"}]
    for bad in ({"axis": "x"}, {"axis": "az", "dir": "up"}, {"axis": "az", "step_hz": 9000},
                {"axis": "el", "step_hz": 300}):  # EL without its zero
        with pytest.raises(commands.CommandError):
            asyncio.run(commands.dispatch("spin", bad))
    assert len(fw.posted("/spin")) == 1


def test_pose_frames_keep_the_spin_flags_and_guard_el(fw) -> None:
    def pose(el: float, sp_el: bool) -> str:
        return json.dumps({"kind": "pose", "az": 1.0, "el": el, "st": "spinning",
                           "motors": True, "sp_az": False, "sp_el": sp_el})

    async def spin_down() -> None:
        for el in (0.0, -10.0, -20.0, EL_MIN + 2.0):
            esp._handle_frame(pose(el, True))
            await asyncio.sleep(0)
        if esp._el_stop_task is not None:
            await esp._el_stop_task

    asyncio.run(spin_down())
    assert model.spinning_el is True
    assert fw.posted("/spin/stop") == [{"axis": "el"}]

    # Spinning up, away from the lower limit: no stop.
    async def spin_up() -> None:
        esp._last_el = None
        for el in (EL_MIN + 1.0, EL_MIN + 2.0, EL_MIN + 4.0):
            esp._handle_frame(pose(el, True))
            await asyncio.sleep(0)

    asyncio.run(spin_up())
    assert fw.posted("/spin/stop") == [{"axis": "el"}]


def test_motor_report_drives_the_axes(client) -> None:
    r = client.get("/firmware/motor")
    assert r.status_code == 200
    assert r.json()["run"]["phase"] == "done"
    assert model.motor_cal["az"]["valid"] is True
    assert model.axes["az"]["steps_per_deg"] == 61.08
    assert model.axes["az"]["motor_calibrated"] is True
    assert model.axes["el"]["motor_calibrated"] is False


def test_a_motor_calibration_task_frame_refreshes_from_state(fw) -> None:
    async def finish() -> None:
        esp._handle_frame(json.dumps({"kind": "task", "task_id": 7, "status": "done",
                                      "result": {"axis": "el", "phase": "done",
                                                 "steps_per_deg": 90.0}}))
        await esp._hydrate_task

    asyncio.run(finish())
    assert model.axes["el"]["steps_per_deg"] == 90.0
    assert model.axes["el"]["max_speed_dps"] == 6.0


def test_moves_wait_for_the_motor_calibration(fw) -> None:
    async def move(**kw: float) -> None:
        await esp.move(**kw)

    asyncio.run(move(azimuth_deg=10.0))  # firmware reports no motor_cal: no check
    esp.apply_motor_calibration(MOTOR)   # AZ calibrated, EL not
    asyncio.run(move(azimuth_deg=20.0))
    with pytest.raises(EspError, match="EL not motor-calibrated"):
        asyncio.run(move(azimuth_deg=20.0, elevation_deg=5.0))
    esp.apply_motor_calibration({**MOTOR, "run": {**MOTOR["run"], "active": True}})
    with pytest.raises(EspError, match="running"):
        asyncio.run(move(azimuth_deg=30.0))
    assert [b["azimuth_deg"] for b in fw.posted("/move")] == [10.0, 20.0]


def test_motor_calibration_commands(fw) -> None:
    with pytest.raises(commands.CommandError, match="remove the arm"):
        asyncio.run(commands.dispatch("motor_calibrate", {"axis": "el"}))
    asyncio.run(commands.dispatch("motor_calibrate", {"axis": "el", "arm_removed": True}))
    asyncio.run(commands.dispatch("motor_calibrate", {"axis": "az"}))
    asyncio.run(commands.dispatch("motor_calibrate_cancel", {}))
    with pytest.raises(commands.CommandError):
        asyncio.run(commands.dispatch("motor_compensation", {"axis": "az", "enabled": "yes"}))
    asyncio.run(commands.dispatch("motor_compensation", {"axis": "az", "enabled": False}))
    assert fw.posted("/motor/calibrate") == [{"axis": "el", "arm_removed": True},
                                            {"axis": "az", "arm_removed": False}]
    assert fw.posted("/motor/compensation") == [{"axis": "az", "enabled": False}]
    assert len(fw.posted("/motor/calibrate/cancel")) == 1
