"""Encoder zero (Machine config → Step 0): the EL zero gates the calibration
sweep, and the command validates what it forwards to the firmware."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import commands
from config import settings
from esp_proxy import esp
from orbiter_model import model


@pytest.fixture(autouse=True)
def _rig(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    sent: list[dict[str, Any]] = []

    async def fake_calibrate(axis: str, mode: str = "current", **_: Any) -> dict[str, Any]:
        sent.append({"axis": axis, "mode": mode})
        return {"status": "ok", "az_zero_raw_deg": 1.0, "el_zero_raw_deg": 2.0}

    monkeypatch.setattr(esp, "calibrate", fake_calibrate)
    model.update(encoder_zero_initialized=False)
    yield sent
    model.update(encoder_zero_initialized=False)


def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(commands.dispatch(name, args))


def test_zeroing_el_initializes_and_reset_clears(_rig) -> None:
    run("calibrate_encoder", {"axis": "az", "mode": "current"})
    assert model.encoder_zero_initialized is False  # AZ alone gates nothing
    run("calibrate_encoder", {"axis": "el", "mode": "current"})
    assert model.encoder_zero_initialized is True
    run("calibrate_encoder", {"axis": "both", "mode": "reset"})
    assert model.encoder_zero_initialized is False
    assert _rig == [
        {"axis": "az", "mode": "current"},
        {"axis": "el", "mode": "current"},
        {"axis": "both", "mode": "reset"},
    ]


@pytest.mark.parametrize("args", [{"axis": "x"}, {"axis": "el", "mode": "home"}])
def test_bad_axis_or_mode_never_reaches_the_firmware(_rig, args) -> None:
    with pytest.raises(commands.CommandError):
        run("calibrate_encoder", args)
    assert _rig == []


def test_calibration_sweep_needs_the_el_zero() -> None:
    with pytest.raises(commands.CommandError, match="Zero EL"):
        run("calibrate_geometry", {"preset": "fast"})
