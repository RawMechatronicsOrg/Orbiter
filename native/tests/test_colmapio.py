"""The COLMAP sparse model we write, and the binary one we read back.

Every file here is a convention someone else defined, so the tests are golden
lines and round trips rather than properties: the only way to be sure a column
order is right is to write the column order down. The binary fixtures are packed
by hand with `struct` for the same reason — a reader tested against its own
writer proves nothing about the bytes `image_undistorter` actually produces.
"""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orbiter_native.colmapio import (
    EyeCamera,
    ImageRecord,
    Point3D,
    quat_wxyz,
    read_images_bin,
    read_pinhole_bin,
    write_cameras,
    write_frames,
    write_images,
    write_points3d,
    write_rigs,
)

LEFT = EyeCamera(camera_id=1, side="left", width=1920, height=1080,
                 fx=1234.5, fy=1235.75, cx=959.5, cy=539.5,
                 dist=(-0.0125, 0.00375, 0.0001, -0.0002, 0.0))
RIGHT = EyeCamera(camera_id=2, side="right", width=1920, height=1080,
                  fx=1240.0, fy=1241.5, cx=955.0, cy=545.0,
                  dist=(-0.0131, 0.00402, -0.0003, 0.0001, 0.0))

#: Four photos, two pairs, alternating eyes. The first is deliberately the
#: identity rotation so its written line has exact goldens to compare against.
POSES = [
    ((0.0, 0.0, 0.0), (12.5, -4.25, 305.0)),
    ((0.05, 0.62, 0.29), (-131.0, -3.9, 288.0)),
    ((0.02, -0.55, 0.30), (9.0, -4.0, 312.0)),
    ((0.06, 0.10, 0.28), (-134.0, -3.4, 295.0)),
]
NAMES = ["left_0001.jpg", "right_0001.jpg", "left_0002.jpg", "right_0002.jpg"]


def _model() -> tuple[list[ImageRecord], list[Point3D]]:
    """A two-camera, four-image model whose POINTS2D and tracks are built from
    ONE list of observations — the property the written files must preserve."""
    images = [
        ImageRecord(image_id=i + 1, name=name, camera_id=1 + i % 2,
                    R=Rotation.from_rotvec(rvec).as_matrix(), t=np.array(t, float))
        for i, (name, (rvec, t)) in enumerate(zip(NAMES, POSES))
    ]
    seeds = [((0.0, 0.0, 10.0), (200, 100, 50)),
             ((25.0, -12.0, 40.0), (128, 128, 128)),
             ((-31.5, 8.25, 22.0), (10, 240, 33))]
    points = [Point3D(point_id=i + 1, xyz=np.array(xyz, float), rgb=rgb)
              for i, (xyz, rgb) in enumerate(seeds)]
    # (point index, image index, x, y) — one list, both sides derived from it.
    observations = [
        (0, 0, 900.5, 512.25), (0, 1, 812.0, 498.75), (0, 2, 941.5, 505.0),
        (1, 0, 1020.0, 430.5), (1, 3, 705.25, 455.0),
        (2, 1, 640.75, 611.0), (2, 2, 1103.5, 590.25), (2, 3, 688.0, 602.5),
    ]
    for pi, ii, x, y in observations:
        image, point = images[ii], points[pi]
        point.track.append((image.image_id, len(image.points2d)))
        image.points2d.append((x, y, point.point_id))
    return images, points


def _body(path) -> list[str]:
    """A written model's data lines — comments dropped, blanks kept, because in
    `images.txt` a blank line is an image with no observations."""
    return [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if not ln.startswith("#")]


def _read_images_txt(path) -> dict[int, dict]:
    """`images.txt` back as `{image_id: {...}}`, the way COLMAP reads it: a pose
    line and an observation line per image."""
    rows = _body(path)
    out = {}
    for head, obs in zip(rows[0::2], rows[1::2]):
        f = head.split()
        vals = obs.split()
        out[int(f[0])] = {
            "pose": f[1:8], "camera_id": int(f[8]), "name": f[9],
            "points2d": [(float(vals[i]), float(vals[i + 1]), int(vals[i + 2]))
                         for i in range(0, len(vals), 3)],
        }
    return out


def _read_points3d_txt(path) -> dict[int, dict]:
    out = {}
    for line in _body(path):
        f = line.split()
        rest = [int(v) for v in f[8:]]
        out[int(f[0])] = {
            "xyz": [float(v) for v in f[1:4]], "rgb": tuple(int(v) for v in f[4:7]),
            "error": float(f[7]),
            "track": list(zip(rest[0::2], rest[1::2])),
        }
    return out


def test_quaternion_is_scalar_first_and_round_trips() -> None:
    R = Rotation.from_rotvec([0.3, -0.7, 1.1]).as_matrix()
    w, x, y, z = quat_wxyz(R)
    assert w >= 0.0                                   # canonical sign, so a line is stable
    # scipy is scalar-LAST, so the reorder is what has to reproduce R.
    assert np.allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), R, atol=1e-12)
    # Handing our four numbers to a scalar-last reader unchanged is the bug this
    # function exists to prevent, and it does not give R back.
    assert not np.allclose(Rotation.from_quat([w, x, y, z]).as_matrix(), R, atol=1e-6)


def test_quaternion_canonicalises_a_rotation_past_half_a_turn() -> None:
    # A rotation of 3.5 rad has cos(1.75) < 0, so the raw quaternion's scalar is
    # negative; the same rotation must still be written one way.
    R = Rotation.from_rotvec([0.0, 0.0, 3.5]).as_matrix()
    w, x, y, z = quat_wxyz(R)
    assert w > 0.0
    assert np.allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), R, atol=1e-12)


def test_cameras_txt_matches_a_golden_line(tmp_path) -> None:
    path = tmp_path / "cameras.txt"
    write_cameras(path, [LEFT, RIGHT])
    assert path.read_text(encoding="utf-8") == (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 2\n"
        "1 OPENCV 1920 1080 1234.5 1235.75 959.5 539.5 -0.0125 0.00375 0.0001 -0.0002\n"
        "2 OPENCV 1920 1080 1240 1241.5 955 545 -0.0131 0.00402 -0.0003 0.0001\n"
    )


def test_refuses_mixed_resolutions_within_one_side(tmp_path) -> None:
    mixed = replace(LEFT, photo_wh=((1920, 1080), (1280, 720)))
    with pytest.raises(ValueError) as excinfo:
        write_cameras(tmp_path / "cameras.txt", [mixed, RIGHT])
    message = str(excinfo.value)
    assert "left" in message and "1920x1080" in message and "1280x720" in message
    assert not (tmp_path / "cameras.txt").exists()     # refused before anything was written
    # The same side at the size its intrinsics were solved at is fine.
    write_cameras(tmp_path / "cameras.txt", [replace(LEFT, photo_wh=((1920, 1080),)), RIGHT])
    assert len(_body(tmp_path / "cameras.txt")) == 2


def test_images_txt_writes_two_lines_per_image_with_its_own_points2d(tmp_path) -> None:
    images, _points = _model()
    path = tmp_path / "images.txt"
    write_images(path, images)
    rows = _body(path)
    assert len(rows) == 2 * len(images)
    # The identity-rotation photo, verbatim: pose, camera and name in one line.
    assert rows[0] == "1 1 0 0 0 12.5 -4.25 305 1 left_0001.jpg"
    assert rows[1] == "900.5 512.25 1 1020 430.5 2"
    read = _read_images_txt(path)
    assert [im["name"] for im in read.values()] == NAMES
    assert read[1]["points2d"] == [(900.5, 512.25, 1), (1020.0, 430.5, 2)]


def test_images_txt_uses_camera_2_for_right_photos(tmp_path) -> None:
    images, _points = _model()
    path = tmp_path / "images.txt"
    write_images(path, images)
    read = _read_images_txt(path)
    assert {im["name"]: im["camera_id"] for im in read.values()} == {
        "left_0001.jpg": 1, "right_0001.jpg": 2,
        "left_0002.jpg": 1, "right_0002.jpg": 2,
    }
    # And the two eyes of one pair carry different poses — a right photo given
    # the left's pose is displaced by the whole stereo baseline.
    assert read[1]["pose"] != read[2]["pose"]


def test_points2d_and_tracks_are_generated_from_one_list(tmp_path) -> None:
    images, points = _model()
    write_images(tmp_path / "images.txt", images)
    write_points3d(tmp_path / "points3D.txt", points)
    read_images = _read_images_txt(tmp_path / "images.txt")
    read_points = _read_points3d_txt(tmp_path / "points3D.txt")

    # Every track entry indexes the matching observation...
    for point_id, point in read_points.items():
        assert len(point["track"]) >= 2
        for image_id, idx in point["track"]:
            assert read_images[image_id]["points2d"][idx][2] == point_id
    # ...and every observation is named back by that point's track.
    for image_id, image in read_images.items():
        for idx, (_x, _y, point_id) in enumerate(image["points2d"]):
            assert point_id != -1
            assert (image_id, idx) in read_points[point_id]["track"]

    assert read_points[1]["rgb"] == (200, 100, 50)
    assert read_points[1]["error"] == 1.0
    assert read_points[3]["xyz"] == [-31.5, 8.25, 22.0]


def test_rigs_and_frames_match_a_golden_pair(tmp_path) -> None:
    images, _points = _model()
    write_rigs(tmp_path / "rigs.txt", [LEFT, RIGHT])
    write_frames(tmp_path / "frames.txt", images)

    assert (tmp_path / "rigs.txt").read_text(encoding="utf-8") == (
        "# Rig list with one line of data per rig:\n"
        "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, "
        "SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n"
        "# Number of rigs: 2\n"
        "1 1 CAMERA 1\n"
        "2 1 CAMERA 2\n"
    )

    text = (tmp_path / "frames.txt").read_text(encoding="utf-8")
    assert text.startswith(
        "# Frame list with one line of data per frame:\n"
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], "
        "NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n"
        "# Number of frames: 4\n"
    )
    rows = _body(tmp_path / "frames.txt")
    assert len(rows) == 4
    assert rows[0] == "1 1 1 0 0 0 12.5 -4.25 305 1 CAMERA 1 1"
    for row, image in zip(rows, images):
        f = row.split()
        assert int(f[0]) == image.image_id              # FRAME_ID is the image's
        assert int(f[1]) == image.camera_id             # one rig per camera
        assert f[9:] == ["1", "CAMERA", str(image.camera_id), str(image.image_id)]


def test_frame_pose_equals_the_images_txt_pose(tmp_path) -> None:
    images, _points = _model()
    write_images(tmp_path / "images.txt", images)
    write_frames(tmp_path / "frames.txt", images)
    poses = {im: row["pose"] for im, row in _read_images_txt(tmp_path / "images.txt").items()}
    for row in _body(tmp_path / "frames.txt"):
        f = row.split()
        # Byte for byte, not merely close: a single-sensor rig has no offset, so
        # the two files describe the same pose and must print it identically.
        assert f[2:9] == poses[int(f[0])]


def _camera_bin(camera_id: int, model_id: int, w: int, h: int, params) -> bytes:
    return (struct.pack("<Ii", camera_id, model_id) + struct.pack("<QQ", w, h)
            + struct.pack(f"<{len(params)}d", *params))


def test_read_pinhole_bin_matches_a_golden_struct(tmp_path) -> None:
    path = tmp_path / "cameras.bin"
    path.write_bytes(
        struct.pack("<Q", 2)
        + _camera_bin(1, 1, 1600, 900, (1300.0, 1301.5, 800.0, 450.0))
        + _camera_bin(2, 1, 1584, 880, (1290.0, 1292.25, 792.0, 440.0)))

    cams = read_pinhole_bin(path)
    assert sorted(cams) == [1, 2]
    # Each eye's own newK and its own undistorted size — using the first
    # camera's for both would warp every right-eye mask.
    K1, w1, h1 = cams[1]
    K2, w2, h2 = cams[2]
    assert (w1, h1) == (1600, 900) and (w2, h2) == (1584, 880)
    assert np.allclose(K1, [[1300.0, 0.0, 800.0], [0.0, 1301.5, 450.0], [0.0, 0.0, 1.0]])
    assert np.allclose(K2, [[1290.0, 0.0, 792.0], [0.0, 1292.25, 440.0], [0.0, 0.0, 1.0]])


def test_read_pinhole_bin_refuses_a_model_it_cannot_measure(tmp_path) -> None:
    # OPENCV (model 4) has eight parameters; guessing four would mis-parse every
    # camera after it rather than fail, so the reader says what it found.
    path = tmp_path / "cameras.bin"
    path.write_bytes(struct.pack("<Q", 1) + _camera_bin(1, 4, 1600, 900, (0.0,) * 8))
    with pytest.raises(ValueError, match="model id 4"):
        read_pinhole_bin(path)


def _image_bin(image_id: int, camera_id: int, name: str, n_points2d: int) -> bytes:
    return (struct.pack("<I", image_id)
            + struct.pack("<7d", 1.0, 0.0, 0.0, 0.0, 5.0, -6.0, 300.0)
            + struct.pack("<I", camera_id)
            + name.encode("utf-8") + b"\x00"
            + struct.pack("<Q", n_points2d)
            + b"".join(struct.pack("<ddQ", 1.5, 2.5, 7) for _ in range(n_points2d)))


def test_read_images_bin_maps_names_to_camera_ids(tmp_path) -> None:
    path = tmp_path / "images.bin"
    # Non-zero observation counts on purpose: the reader has to skip exactly
    # 24 bytes each to find the next image at all.
    path.write_bytes(
        struct.pack("<Q", 3)
        + _image_bin(1, 1, "left_0001.jpg", 3)
        + _image_bin(2, 2, "right_0001.jpg", 0)
        + _image_bin(3, 1, "left_0002.jpg", 5))
    assert read_images_bin(path) == {
        "left_0001.jpg": 1, "right_0001.jpg": 2, "left_0002.jpg": 1,
    }
