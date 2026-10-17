"""A geometry calibration is stamped with the mechanics it was solved on.

The persisted state outlives a change of mechanics (v1 to v2): the stamp is
what lets the server and the UI tell a current calibration from one that
describes another rig, and `reset_calibration` is the way back to the CAD.
"""

from __future__ import annotations

import asyncio

import pytest

import commands
import scan_task
from config import settings
from geom.rig_model import rig_identity
from orbiter_model import CALIBRATION_FIELDS, ModelState, model, motor_signature


@pytest.fixture(autouse=True)
def _tmp_storage(tmp_path, monkeypatch):
    """Model updates persist; keep them out of the checkout."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)


def _calibrated(state: ModelState, rig: str | None) -> None:
    state.update(
        calibrated=True, calib_rig_id=rig, arm_radius_mm=292.2,
        camera_offset_mm=41.0, turntable_axis=(3.0, 0.0),
        calib_extrinsic={"rvec": [0.1, 0.0, 0.0], "t": [1.0, 2.0, 3.0]},
        calib_board_world={"rvec": [0.0, 0.0, 0.0], "t": [0.0, 0.0, 0.0]},
        rocker_correction={"rvec": [0.01, 0.0, 0.0], "t": [0.0, 0.0, 0.0]},
        el_encoder_scale=1.02, az_encoder_correction=[0.1, 0.2],
        calib_test_msg="dR 0.1 deg", camera_fx=1234.0,
    )


def test_rig_identity_is_a_content_hash(tmp_path) -> None:
    a = tmp_path / "a.glb"
    b = tmp_path / "b.glb"
    a.write_bytes(b"glTF one")
    b.write_bytes(b"glTF two")
    assert len(rig_identity(a)) == 12
    assert rig_identity(a) == rig_identity(str(a))
    assert rig_identity(a) != rig_identity(b)


def test_status_follows_the_stamp() -> None:
    state = ModelState(rig_id="aaaa")
    assert state.calibration_status == "none"
    _calibrated(state, "aaaa")
    assert state.calibration_status == "current"
    state.update(rig_id="bbbb")
    assert state.calibration_status == "stale"
    # A calibration from before the stamp existed is stale too.
    state.update(calib_rig_id=None, rig_id="aaaa")
    assert state.calibration_status == "stale"


def test_reset_restores_the_defaults_but_keeps_intrinsics() -> None:
    state = ModelState(rig_id="aaaa")
    _calibrated(state, "zzzz")
    state.reset_calibration()
    fresh = ModelState()
    for name in CALIBRATION_FIELDS:
        assert getattr(state, name) == getattr(fresh, name), name
    assert state.calibration_status == "none"
    assert state.camera_fx == 1234.0


def test_reset_command_and_scan_guard(monkeypatch) -> None:
    monkeypatch.setattr(model, "rig_id", "aaaa")
    _calibrated(model, "old-mechanics")
    try:
        with pytest.raises(RuntimeError, match="no longer matches"):
            asyncio.run(scan_task.start_scan())
        asyncio.run(commands.dispatch("reset_calibration", {}))
        assert model.calibration_status == "none"
        assert model.arm_radius_mm == 0.0
    finally:
        model.reset_calibration()


def test_a_re_zeroed_encoder_makes_the_calibration_stale() -> None:
    state = ModelState(rig_id="aaaa")
    state.update(encoder_zero={"az_zero_raw_deg": 0.0, "el_zero_raw_deg": 126.2})
    _calibrated(state, "aaaa")
    state.update(calib_encoder_zero=dict(state.encoder_zero))
    assert state.calibration_status == "current"
    state.update(encoder_zero={"az_zero_raw_deg": 0.0, "el_zero_raw_deg": 131.0})
    assert state.calibration_status == "stale"
    assert "EL encoder zero changed" in state.calibration_stale_reason
    state.update(encoder_zero={"az_zero_raw_deg": 0.0, "el_zero_raw_deg": 126.2})
    assert state.calibration_status == "current"


def test_a_changed_motor_calibration_makes_the_calibration_stale() -> None:
    def mc(spd: float, comp: bool | None) -> dict:
        axis = {"valid": True, "steps_per_deg": spd, "max_speed_dps": 40.0,
                "dir_sign": 1, "compensation": comp}
        return {"active": False, "az": dict(axis), "el": dict(axis)}

    state = ModelState(rig_id="aaaa")
    state.update(motor_cal=mc(61.08, True))
    _calibrated(state, "aaaa")
    state.update(calib_motor_cal=motor_signature(state.motor_cal))
    assert state.calibration_status == "current"
    state.update(motor_cal=mc(61.08, None))     # /state carries no compensation
    assert state.calibration_status == "current"
    state.update(motor_cal=mc(61.08, False))    # profile switched off
    assert state.calibration_status == "stale"
    assert "motor calibration changed" in state.calibration_stale_reason
    state.update(motor_cal=mc(61.2, True))      # a new motor calibration
    assert state.calibration_status == "stale"


def test_busy_reason_covers_scan_calibration_and_motor_calibration() -> None:
    state = ModelState()
    assert state.busy_reason() is None
    state.update(calibration_running=True)
    assert "geometry calibration" in state.busy_reason()
    state.update(calibration_running=False, motor_cal={"active": True})
    assert "motor calibration" in state.busy_reason()
