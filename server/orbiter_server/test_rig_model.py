"""The v2 rig GLB (`cad/v2/assembly/Orbiter.glb`) as the server reads it.

Pins the axes the scene graph and the camera math rely on: the azimuth axis is
the world Z axis, the elevation axis runs along world X through it, and the
orbit arm points along +Y at rest. Numbers are from the Fusion model, v225.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from geom.rig_model import load_rig, to_three_name

GLB = Path(__file__).resolve().parents[2] / "cad" / "v2" / "assembly" / "Orbiter.glb"


@pytest.fixture(scope="module")
def rig():
    return load_rig(GLB)


def test_azimuth_axis_is_world_z(rig) -> None:
    assert rig.node_origin("ORB_ROTOR_base") == pytest.approx((0.0, 0.0, 0.0), abs=1e-3)


def test_elevation_axis_crosses_the_azimuth_axis(rig) -> None:
    x, y, z = rig.node_origin("ORB_ROTOR_elevation")
    assert y == pytest.approx(0.0, abs=0.01)
    assert z == pytest.approx(127.178, abs=0.01)
    assert x > 150.0  # the elevation drive stands beside the platform, on +X


def test_rotors_own_the_moving_parts(rig) -> None:
    def under(rotor: str) -> set[str]:
        return set(rig._children[rotor])

    assert {"BottomBody:1 / BaseSpindle:1", "BottomBody:1 / BasePulley GT2 112T:1",
            "BottomBody:1 / BasePlatformInsert Taper:1"} <= under("ORB_ROTOR_base")
    assert {"ElevationDrive:1 / ElevationShaft:1", "ElevationDrive:1 / ReplacablePulley:1",
            "ElevationDrive:1 / BarMountInsert:1", "OrbitArm:1"} <= under("ORB_ROTOR_elevation")


def test_platform_is_the_120_mm_disc_on_the_axis(rig) -> None:
    box = rig.part_aabb("BottomBody:1 / BasePlatformInsert Taper:1")
    assert box.extents[0] == pytest.approx(120.0, abs=0.1)
    assert box.extents[1] == pytest.approx(120.0, abs=0.1)
    assert box.centre[:2] == pytest.approx((0.0, 0.0), abs=0.01)
    assert box.max[2] == pytest.approx(118.64, abs=0.01)


def test_arm_tip_is_on_the_tube_centreline(rig) -> None:
    arm = rig.part_aabb("OrbitArm:1")
    tip = rig.part_end_centre("OrbitArm:1", axis=1, sign=+1)
    assert tip[0] == pytest.approx((arm.min[0] + arm.max[0]) / 2, abs=0.01)
    assert tip[1] == pytest.approx(arm.max[1], abs=1.0)
    pivot = rig.node_origin("ORB_ROTOR_elevation")
    reach = math.hypot(tip[1] - pivot[1], tip[2] - pivot[2])
    assert reach == pytest.approx(225.0, abs=0.1)
    # The tube's centreline passes through the axis; at rest the arm droops
    # 0.84° below the horizontal.
    droop = math.degrees(math.atan2(tip[2] - pivot[2], tip[1] - pivot[1]))
    assert droop == pytest.approx(-0.84, abs=0.05)


def test_scene_spins_the_rotor_subtrees(monkeypatch) -> None:
    import camera_stream
    import scene_graph
    from orbiter_model import ModelState

    monkeypatch.setattr(scene_graph, "load_rig", lambda: load_rig(GLB))
    monkeypatch.setattr(camera_stream.stream, "frame_aspect", lambda: None)
    nodes = {n["id"]: n for n in scene_graph.build_scene(
        ModelState(el=30.0, camera_offset_mm=0.0))}

    rig_glb = load_rig(GLB)
    hidden = nodes["rig_cad"]["props"]["hide_parts"]
    assert hidden == ["ORB_ROTOR_base", "ORB_ROTOR_elevation"]
    assert all(to_three_name(n) == n for n in hidden)
    assert nodes["part_orb_rotor_base"]["parent"] == "platform_spin"
    assert nodes["part_orb_rotor_elevation"]["parent"] == "orbit_spin"
    assert nodes["orbit_spin"]["transform"]["position"] == pytest.approx(
        list(rig_glb.node_origin("ORB_ROTOR_elevation")))
    # The math anchor is the AZ-EL axes intersection.
    assert nodes["rig_math"]["transform"]["position"] == pytest.approx([0.0, 0.0, 127.178], abs=0.01)
    # With no arm radius set the camera sits at the CAD arm's reach, and the
    # arm is turned to point at it: el = 30° lifts it by 30° + the 0.84° droop.
    cam = nodes["rig_camera"]["transform"]["position"]
    assert math.hypot(*cam) == pytest.approx(225.0, abs=0.1)
    qx, _, _, qw = nodes["orbit_spin"]["transform"]["quaternion"]
    assert math.degrees(2 * math.atan2(qx, qw)) == pytest.approx(30.84, abs=0.05)
