"""Encoder zero (Machine config → Step 2): the EL zero — read from the offsets
the firmware reports — gates the calibration sweep, and the command validates
what it forwards to the firmware."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import commands
from config import settings
from encoder_init import FIRMWARE_DEFAULT_EL_ZERO_RAW, el_zero_set
from esp_proxy import esp
from orbiter_model import model


@pytest.fixture(autouse=True)
def firmware(tmp_path, monkeypatch):
    """`/calibrate` as the firmware answers it: the zero offsets after the op."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    zeros = {"az_zero_raw_deg": 0.0, "el_zero_raw_deg": FIRMWARE_DEFAULT_EL_ZERO_RAW}
    sent: list[dict[str, Any]] = []

    async def fake_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/calibrate"
        sent.append(body)
        for axis, current in (("az", 12.5), ("el", 331.0)):
            if body["axis"] in (axis, "both"):
                zeros[f"{axis}_zero_raw_deg"] = (
                    current if body["mode"] == "current"
                    else (0.0 if axis == "az" else FIRMWARE_DEFAULT_EL_ZERO_RAW))
        return {"status": "ok", **body, **zeros}

    monkeypatch.setattr(esp, "_post", fake_post)
    model.update(encoder_zero={}, encoder_zero_initialized=False)
    yield sent
    model.update(encoder_zero={}, encoder_zero_initialized=False)


def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(commands.dispatch(name, args))


def test_el_zero_is_read_from_the_reported_offsets() -> None:
    assert el_zero_set(None) is None
    assert el_zero_set({"el_zero_raw_deg": FIRMWARE_DEFAULT_EL_ZERO_RAW}) is False
    assert el_zero_set({"el_zero_raw_deg": 126.211}) is True


def test_zeroing_el_initializes_and_reset_clears(firmware) -> None:
    run("calibrate_encoder", {"axis": "az", "mode": "current"})
    assert model.encoder_zero_initialized is False  # AZ alone gates nothing
    run("calibrate_encoder", {"axis": "el", "mode": "current"})
    assert model.encoder_zero_initialized is True
    run("calibrate_encoder", {"axis": "both", "mode": "reset"})
    assert model.encoder_zero_initialized is False
    assert [(b["axis"], b["mode"]) for b in firmware] == [
        ("az", "current"), ("el", "current"), ("both", "reset")]


def test_an_erased_firmware_zero_clears_the_gate() -> None:
    model.update(encoder_zero={"az_zero_raw_deg": 0.0, "el_zero_raw_deg": 126.2})
    assert model.encoder_zero_initialized is True
    # e.g. an ESP swap or an erased NVS: /state reports the default again
    esp._apply_state({"calibration": {"az_zero_raw_deg": 0.0,
                                      "el_zero_raw_deg": FIRMWARE_DEFAULT_EL_ZERO_RAW}})
    assert model.encoder_zero_initialized is False


@pytest.mark.parametrize("args", [{"axis": "x"}, {"axis": "el", "mode": "home"}])
def test_bad_axis_or_mode_never_reaches_the_firmware(firmware, args) -> None:
    with pytest.raises(commands.CommandError):
        run("calibrate_encoder", args)
    assert firmware == []


def test_calibration_sweep_needs_the_el_zero() -> None:
    with pytest.raises(commands.CommandError, match="Zero EL"):
        run("calibrate_geometry", {"preset": "fast"})
