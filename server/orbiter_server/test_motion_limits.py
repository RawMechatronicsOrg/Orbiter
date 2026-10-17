"""Speed knobs and EL travel: one set of numbers (drive.py, geom.angles) that
the server clamps and validates with and publishes to the UI."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

import commands
from config import settings
from drive import AXES, axes_info
from esp_proxy import esp
from geom.angles import EL_MAX, EL_MIN
from models import DiscretePlan
from orbiter_model import model


@pytest.fixture(autouse=True)
def moves(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    sent: list[dict[str, Any]] = []

    async def fake_move(azimuth_deg=None, elevation_deg=None, timeout_ms=30_000):
        sent.append({"az": azimuth_deg, "el": elevation_deg})
        return {"status": "done"}

    monkeypatch.setattr(esp, "move_and_await", fake_move)
    yield sent


def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(commands.dispatch(name, args))


def test_axes_info_carries_the_travel_and_the_drive() -> None:
    info = axes_info()
    assert (info["el"]["min_deg"], info["el"]["max_deg"]) == (EL_MIN, EL_MAX)
    assert "min_deg" not in info["az"]
    assert info["az"]["steps_per_deg_nominal"] == pytest.approx(62.2, abs=0.05)
    assert info["el"]["steps_per_deg_nominal"] == pytest.approx(88.9, abs=0.05)
    assert AXES["az"].deg_per_s(2100) == pytest.approx(2100 / 61.1)
    assert model.axes == info


def test_speed_knobs_clamp_to_the_published_band() -> None:
    out = run("set_move_speed", {"az_hz_max": 99999, "el_hz_max": 1})
    assert out == {"move_hz_max_az": AXES["az"].hz_max, "move_hz_max_el": AXES["el"].hz_min}
    with pytest.raises(commands.CommandError):
        run("set_move_speed", {"az_hz_max": "fast"})


def test_move_refuses_el_outside_the_travel(moves) -> None:
    for el in (EL_MAX + 1, EL_MIN - 1, "nan"):
        with pytest.raises(commands.CommandError):
            run("move", {"az": 10, "el": el})
    with pytest.raises(commands.CommandError):
        run("move", {})
    run("move", {"az": 10, "el": None})
    run("move", {"el": EL_MIN})
    assert moves == [{"az": 10.0, "el": None}, {"az": None, "el": EL_MIN}]


@pytest.mark.parametrize(
    ("el", "delta", "target"),
    [(88.0, 5.0, EL_MAX), (-20.0, -10.0, EL_MIN), (-26.0, 5.0, -21.0), (10.0, -5.0, 5.0)],
)
def test_jog_stops_at_the_limit(moves, monkeypatch, el, delta, target) -> None:
    monkeypatch.setattr(model, "el", el)
    run("jog", {"axis": "el", "delta_deg": delta})
    assert moves == [{"az": None, "el": target}]


@pytest.mark.parametrize(("el", "delta"), [(EL_MAX, 5.0), (EL_MIN, -1.0), (-300.0, 5.0)])
def test_jog_refuses_at_the_limit_or_from_far_outside(moves, monkeypatch, el, delta) -> None:
    monkeypatch.setattr(model, "el", el)
    with pytest.raises(commands.CommandError):
        run("jog", {"axis": "el", "delta_deg": delta})
    assert moves == []


def test_plan_rings_must_lie_in_the_travel() -> None:
    DiscretePlan(el_start_deg=EL_MIN, el_max_deg=EL_MAX)
    for bad in ({"el_max_deg": EL_MAX + 0.5}, {"el_start_deg": -30.0},
                {"az_step_deg": 0.0}, {"el_steps": -1}):
        with pytest.raises(ValidationError):
            DiscretePlan(**bad)
    with pytest.raises(commands.CommandError, match="outside the EL travel"):
        run("set_motion_plan", {"discrete": {"el_start_deg": -36.0}})
