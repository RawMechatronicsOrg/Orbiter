"""`set_stereo_rig` stores a per-eye rolling-shutter readout time."""

from __future__ import annotations

import pytest

import orbiter_server  # noqa: F401 — puts the package's bare-name modules on sys.path

import commands


def test_readout_is_coerced_with_its_stats() -> None:
    out = commands._coerce_readout({
        "seconds": "0.0213", "width": 1920, "height": 1080,
        "sigma_s": 0.0004, "skew_px": 12.5, "rms_px": 0.21, "views": 150, "junk": 1,
    })
    assert out == {"seconds": 0.0213, "width": 1920, "height": 1080,
                   "sigma_s": 0.0004, "skew_px": 12.5, "rms_px": 0.21, "views": 150}


def test_readout_accepts_a_negative_sign_and_null() -> None:
    assert commands._coerce_readout({"seconds": -0.0213, "width": 1280, "height": 720})[
        "seconds"] == -0.0213
    assert commands._coerce_readout(None) is None


@pytest.mark.parametrize("bad", [
    {"seconds": 0.0, "width": 1920, "height": 1080},
    {"seconds": 0.5, "width": 1920, "height": 1080},
    {"seconds": 0.02, "width": 0, "height": 1080},
    {"seconds": 0.02},
    "0.02",
])
def test_readout_rejects_what_is_not_one(bad) -> None:
    with pytest.raises(commands.CommandError):
        commands._coerce_readout(bad)


def test_readout_is_an_eye_field() -> None:
    merged = commands._merge_eye({"camera_id": "cam2"},
                                 {"readout": {"seconds": 0.02, "width": 1920, "height": 1080}},
                                 "left")
    assert merged["camera_id"] == "cam2"
    assert merged["readout"] == {"seconds": 0.02, "width": 1920, "height": 1080}
