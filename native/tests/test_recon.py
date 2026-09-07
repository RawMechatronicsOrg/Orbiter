"""The offline chain: what it runs, in what order, and what it refuses.

Every test here drives `recon.run` through `FakeBackend`, which records the
argv it is handed and can act on it — creating the files a real COLMAP would
leave behind. That is what makes the two checks worth having testable: a
workspace holding the wrong images, and a `model_converter` that says no.

The session under `tmp_path` is a real one, written the way the app writes one:
twelve places around the subject, photographed twice — once with the laser on
and once with it off — and the box fixture of `test_views` exported as
`laser.ply`, which is what the app does when Reconstruct is pressed. Twenty-four
photographs survive selection, which clears the twenty a reconstruction needs
with enough margin that a test watching a refusal is watching the rule.

Nothing here starts a container, and nothing here needs a GPU — which is the
point of Milestone 1 and is asserted rather than assumed.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import struct
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from orbiter_native import recon
from orbiter_native.photos import PhotoCandidate, EyePhoto, PhotoSession, PhotoWriter
from orbiter_native.recon import (
    CANONICAL_STEPS,
    DIRS_CREATED,
    MODE_IMAGE_PATH,
    MODE_STEPS,
    DockerBackend,
    FakeBackend,
    GpuSpec,
    ReconCancelled,
    ReconRefused,
    State,
    StepFailed,
    container_path,
    disk_estimate_bytes,
    undistorted_wh_for,
)
from orbiter_native.scan import write_ply
from orbiter_native.stereo import compose_right_pose

from test_colmapio import _image_bin
from test_stereo_scan import WH
from test_stereo_scan import _rig as _stereo_rig
from test_views import BOX_XYZ, _pair_pose, _rig

#: Twelve places round the subject, none of them on a view-bucket boundary
#: (the boundaries are multiples of 15 degrees).
PLACES = tuple(float(az) for az in np.linspace(11.0, 341.0, 12))


# ── a session on disk ────────────────────────────────────────────────────


def _record(writer: PhotoWriter, az: float, k: int, *, pass_id: int,
            laser_on: bool) -> None:
    """One pair — both eyes, the right one's pose composed across the pair —
    written straight through the writer's own `_write`.

    Not through the queue: it is bounded at 32 and drops the OLDEST entry to
    keep the scan thread free, which is exactly right live and exactly wrong
    for a fixture that has to contain what it wrote. Forty-eight photographs
    would lose some of themselves on the way in.
    """
    geom = _stereo_rig().geom
    R, t = _pair_pose(az)
    R_r, t_r = compose_right_pose(R, t, geom)
    cand = PhotoCandidate(
        left=EyePhoto(camera_id="cam2", jpeg=b"\xff\xd8left\xff\xd9", wh=WH,
                      capture_mono=1000.0 + k, R=R, t_mm=t,
                      sharpness=100.0 + k),
        right=EyePhoto(camera_id="cam4", jpeg=b"\xff\xd8right\xff\xd9", wh=WH,
                       capture_mono=1000.0 + k - 0.012, R=R_r, t_mm=t_r,
                       sharpness=90.0 + k, stripe_shifted=True),
        pair_capture_mono=1000.0 + k, pose_source="left+right",
        pose_rms_px=0.31, pose_gap_deg=0.2, pose_gap_mm=1.0, pose_corners=24,
        pose_smooth_mm=0.21, pass_id=pass_id, laser_on=laser_on)
    writer._write(cand.record("left"))
    writer._write(cand.record("right"))


def _session(tmp_path, *, places=PLACES, clean: bool = True,
             cloud: np.ndarray | None = None) -> Path:
    """A session directory the chain can be run against.

    The clean pass stands in the same places as the laser pass, so the
    selection — which prefers a clean photograph wherever both stand together —
    comes out as twenty-four clean ones, covering every bucket it covers. That
    is `texture_source: "clean"`, the undegraded path, which is what most of
    these tests want to be looking at.
    """
    session = PhotoSession(tmp_path, rig=_rig())
    writer = PhotoWriter(session)
    k = 0
    for pass_id, laser_on in ((0, True),) + (((1, False),) if clean else ()):
        for az in places:
            _record(writer, az, k, pass_id=pass_id, laser_on=laser_on)
            k += 1
    write_ply(str(session.path / recon.LASER_PLY),
              BOX_XYZ if cloud is None else cloud)
    return session.path


# ── a backend that leaves the right things behind ────────────────────────


def _texture_workspace(session_dir: Path, names: list[str]) -> None:
    """What `image_undistorter --image_list_path` leaves behind: one file per
    listed image and a binary model naming exactly those."""
    images = session_dir / "colmap" / "texture" / "images"
    sparse = session_dir / "colmap" / "texture" / "sparse"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    for name in names:
        (images / name).write_bytes(b"\xff\xd8undistorted\xff\xd9")
    (sparse / "images.bin").write_bytes(
        struct.pack("<Q", len(names))
        + b"".join(_image_bin(i + 1, 1 + (i % 2), name, 0)
                   for i, name in enumerate(names)))


def _listed(session_dir: Path) -> list[str]:
    return [line for line in (session_dir / "colmap" / "texture_images.txt")
            .read_text(encoding="utf-8").splitlines() if line.strip()]


def _textured_ply(session_dir: Path) -> None:
    """What `mesh_texturer` leaves behind: a binary PLY with per-face
    `texcoord` naming its atlas in a header comment, and the atlas."""
    from PIL import Image

    mesh = session_dir / "mesh"
    mesh.mkdir(parents=True, exist_ok=True)
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    faces = [(0, 1, 2), (0, 2, 3)]
    uvs = [(0.0, 0.0, 1.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0)]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment TextureFile texture.png\n"
              f"element vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\n"
              "property list uchar float texcoord\n"
              "end_header\n")
    body = b"".join(struct.pack("<3f", *v) for v in verts)
    for face, uv in zip(faces, uvs):
        body += struct.pack("<B3i", 3, *face) + struct.pack("<B6f", 6, *uv)
    (mesh / "mesh.ply").write_bytes(header.encode("ascii") + body)
    Image.new("RGB", (8, 8), (200, 120, 60)).save(mesh / "texture.png")


def _effects(session_dir: Path, *, texture_names: list[str] | None = None
             ) -> dict[str, Callable[[list[str]], None]]:
    """The files each COLMAP tool would leave behind, keyed by tool.

    `texture_names` overrides what the undistorter puts in the workspace, which
    is how the post-condition is made to fire without a real COLMAP that
    ignores `--image_list_path`.
    """
    return {
        "model_converter": lambda argv: (
            session_dir / "colmap" / "_validate.ply").write_bytes(b"ply\n"),
        "image_undistorter": lambda argv: _texture_workspace(
            session_dir,
            _listed(session_dir) if texture_names is None else texture_names),
        "poisson_mesher": lambda argv: (
            session_dir / "mesh" / "meshed-poisson.ply").write_bytes(b"ply\n"),
        "mesh_texturer": lambda argv: _textured_ply(session_dir),
    }


def _fake(session_dir: Path, **kw) -> FakeBackend:
    exits = kw.pop("exits", None)
    output = kw.pop("output", None)
    return FakeBackend(exits=exits, output=output,
                       effects=_effects(session_dir, **kw))


def _run(session_dir: Path, backend: FakeBackend | None = None, **kw):
    backend = backend or _fake(session_dir)
    result = recon.run(session_dir, kw.pop("mode", "texture-only"),
                       backend=backend, **kw)
    return result, backend


# ── the argv, exactly ────────────────────────────────────────────────────


def _expected_argv(mode: str = "texture-only") -> list[list[str]]:
    """Every command the chain sends to COLMAP, in order.

    `write_sparse` and `glb` are host steps and send none — which is itself
    part of the shape being asserted.
    """
    poisson_input = ("/data/laser.ply" if mode == "texture-only"
                     else "/data/colmap/dense/merged.ply")
    return [
        ["colmap", "-h"],
        ["colmap", "model_converter",
         "--input_path", "/data/colmap/sparse",
         "--output_path", "/data/colmap/_validate.ply",
         "--output_type", "PLY"],
        ["colmap", "image_undistorter",
         "--image_path", f"/data/{MODE_IMAGE_PATH[mode]}",
         "--input_path", "/data/colmap/sparse",
         "--output_path", "/data/colmap/texture",
         "--output_type", "COLMAP",
         "--max_image_size", "-1",
         "--image_list_path", "/data/colmap/texture_images.txt"],
        ["colmap", "poisson_mesher",
         "--input_path", poisson_input,
         "--output_path", "/data/mesh/meshed-poisson.ply",
         "--PoissonMeshing.trim", "7"],
        ["colmap", "mesh_texturer",
         "--workspace_path", "/data/colmap/texture",
         "--input_path", "/data/mesh/meshed-poisson.ply",
         "--output_path", "/data/mesh",
         "--MeshTextureMapping.texture_scale_factor", "1",
         "--MeshTextureMapping.atlas_patch_padding", "4",
         "--MeshTextureMapping.inpaint_radius", "5",
         "--MeshTextureMapping.min_visible_vertices", "3",
         "--MeshTextureMapping.apply_color_correction", "1"],
    ]


def test_step_order_and_exact_argv(tmp_path) -> None:
    """The texture-only chain, argument for argument — and the dense chain's
    step names in canonical order.

    The argv is golden because every one of these flags was read off COLMAP
    4.2.0 and a silent change to any of them is a run that finishes and is
    wrong. The dense chain is asserted by NAME only: its argv belongs to D1b's
    own fixture, so the two stories cannot fight over one golden list.
    """
    session_dir = _session(tmp_path)
    result, backend = _run(session_dir)

    assert result.ran == MODE_STEPS["texture-only"]
    assert backend.calls == _expected_argv()
    assert result.texture_source == "clean"
    assert result.degraded == []

    assert MODE_STEPS["dense"] == CANONICAL_STEPS
    assert MODE_STEPS["texture-only"] == (
        "write_sparse", "validate_sparse", "texture_workspace",
        "poisson_mesher", "mesh_texturer", "glb")
    # `clean_images` is third, before both dense consumers of its output.
    assert CANONICAL_STEPS.index("clean_images") < \
        CANONICAL_STEPS.index("texture_workspace") < \
        CANONICAL_STEPS.index("image_undistorter")


def test_write_sparse_writes_the_model_and_the_verbatim_images(tmp_path) -> None:
    """Step one's whole output: five text files, one copy per selected
    photograph under the name `images.txt` knows it by, the texture list, and
    the `reconstruct` block."""
    session_dir = _session(tmp_path)
    result, _ = _run(session_dir)

    sparse = session_dir / "colmap" / "sparse"
    for name in ("cameras.txt", "images.txt", "points3D.txt", "rigs.txt",
                 "frames.txt"):
        assert (sparse / name).stat().st_size > 0

    names = [line.split()[-1] for line
             in (sparse / "images.txt").read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#") and line.endswith(".jpg")]
    assert len(names) == result.n_selected == 24
    copied = sorted(p.name for p in (session_dir / "colmap" / "images").iterdir())
    assert copied == sorted(names)
    # Verbatim: the intrinsics were solved against these exact pixels.
    for name in names:
        side = name.split("_")[0]
        assert (session_dir / "colmap" / "images" / name).read_bytes() == \
            f"\xff\xd8{side}\xff\xd9".encode("latin-1")

    assert _listed(session_dir) == names

    block = json.loads((session_dir / "session.json")
                       .read_text(encoding="utf-8"))["reconstruct"]
    assert block["mode"] == "texture-only"
    assert block["texture_source"] == "clean"
    assert block["selected"]["clean"] == 24
    assert block["buckets"]["clean_coverage"] == 1.0
    assert block["disk"]["estimate_bytes"] == recon.TEXTURE_ONLY_DISK_BYTES
    assert block["disk"]["estimate_inputs"]["n_selected"] == 24
    assert block["started_utc"] and block["finished_utc"]


def test_write_sparse_creates_the_output_directories(tmp_path) -> None:
    """COLMAP writes into the path it is given and creates no parents, so step
    one makes its own. Asserted on the host side: the fake backend does not
    stat a parent, so a missing one would show up only in a real run."""
    session_dir = _session(tmp_path)
    _run(session_dir, only="write_sparse")

    for relative in DIRS_CREATED["write_sparse"]:
        assert session_dir.joinpath(*relative.split("/")).is_dir(), relative
    assert set(DIRS_CREATED["write_sparse"]) == {
        "colmap", "colmap/images", "colmap/sparse", "mesh"}


def test_write_sparse_adds_normals_when_the_cloud_has_none(tmp_path) -> None:
    """Poisson is undefined without normals, and a cloud exported straight off
    the scan carries none — so step one writes them, oriented toward the
    nearest camera that was actually selected."""
    session_dir = _session(tmp_path)
    _run(session_dir, only="write_sparse")

    xyz, _, normals = recon.read_ply_full(str(session_dir / recon.LASER_PLY))
    assert normals is not None and len(normals) == len(xyz)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    # The box's four walls face outward, so a wall point's normal points away
    # from the axis rather than into it.
    wall = np.abs(xyz[:, 0]) > 109.0
    assert float(np.mean(np.sign(xyz[wall, 0]) * normals[wall, 0])) > 0.8


# ── refusals ─────────────────────────────────────────────────────────────


def test_write_sparse_refuses_a_thin_cloud_and_too_few_photos_by_name(
        tmp_path) -> None:
    """Two floors, both named, both hit before a container starts: a cloud too
    thin to be a surface, and a selection too small to be an atlas."""
    thin = _session(tmp_path / "thin", cloud=BOX_XYZ[:400])
    with pytest.raises(ReconRefused, match=r"400 confident points, below the 1000"):
        _run(thin)

    few = _session(tmp_path / "few", places=PLACES[:4])
    with pytest.raises(ReconRefused,
                       match=r"8 of 16 photographs were selected, below the 20"):
        _run(few)

    missing = _session(tmp_path / "missing")
    (missing / recon.LASER_PLY).unlink()
    with pytest.raises(ReconRefused, match=r"laser\.ply does not exist"):
        _run(missing)


def test_unknown_step_for_the_mode_is_refused_by_name(tmp_path) -> None:
    """`--mode texture-only --to patch_match_stereo` names a step of the other
    chain. It fails immediately and prints the chain it does have, rather than
    running the short chain to its end and looking like it obeyed."""
    session_dir = _session(tmp_path)
    with pytest.raises(ReconRefused) as caught:
        _run(session_dir, to_step="patch_match_stereo")
    message = str(caught.value)
    assert "patch_match_stereo is a step of the dense chain" in message
    assert " -> ".join(MODE_STEPS["texture-only"]) in message

    with pytest.raises(ReconRefused, match="no step called wobble"):
        _run(session_dir, only="wobble")


def test_dense_mode_is_refused_by_name_until_its_steps_exist(tmp_path) -> None:
    """Dense's steps are named in `CANONICAL_STEPS` and are not implemented
    here. That is a sentence naming them, not a stub that would fail somewhere
    less legible — and it names the mode that does work."""
    session_dir = _session(tmp_path)
    with pytest.raises(ReconRefused) as caught:
        _run(session_dir, mode="dense")
    message = str(caught.value)
    for name in ("clean_images", "image_undistorter", "undistort_masks",
                 "write_patch_match_cfg", "patch_match_stereo",
                 "stereo_fusion", "merge"):
        assert name in message
    assert "--mode texture-only" in message
    assert set(MODE_STEPS["texture-only"]) <= set(recon.STEPS)


# ── disk ─────────────────────────────────────────────────────────────────


def test_disk_estimate_scales_with_selection_and_image_size() -> None:
    """The §2.8 formula, evaluated: 132 images at 1600x900 is 8 981 069 824
    bytes. A flat figure is simultaneously far too much for a 40-photograph
    capture and nowhere near enough for a 150-photograph one, which is how a
    run fills a disk while its own check says it is fine."""
    assert disk_estimate_bytes("dense", 132, (1600, 900)) == 8_981_069_824

    headroom = recon.DISK_HEADROOM_BYTES
    one = disk_estimate_bytes("dense", 1, (1600, 900)) - headroom
    assert disk_estimate_bytes("dense", 132, (1600, 900)) - headroom == 132 * one
    # Half the pixels, half the maps.
    assert disk_estimate_bytes("dense", 132, (800, 900)) - headroom == \
        (disk_estimate_bytes("dense", 132, (1600, 900)) - headroom) // 2

    # Milestone 1 writes no depth maps, so there is nothing to scale.
    assert disk_estimate_bytes("texture-only", 132, (1600, 900)) == 1 << 30
    assert disk_estimate_bytes("texture-only", 12, (640, 480)) == 1 << 30

    # 1920x1080 undistorted at --max_image_size 1600 is the pair above.
    assert undistorted_wh_for((1920, 1080), 1600) == (1600, 900)
    assert undistorted_wh_for((1920, 1080), -1) == (1920, 1080)
    assert undistorted_wh_for((1280, 720), 1600) == (1280, 720)


def test_run_refuses_below_the_disk_estimate_unless_forced(
        tmp_path, monkeypatch) -> None:
    """A run that cannot fit says so by name, names the remedy, and stops
    before a container starts. `--force` turns the refusal into a warning,
    which is what an operator who knows their disk better than we do needs."""
    session_dir = _session(tmp_path)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda path: usage(1 << 40, 1 << 40, 800 * (1 << 20)))

    backend = _fake(session_dir)
    with pytest.raises(ReconRefused) as caught:
        _run(session_dir, backend)
    message = str(caught.value)
    assert "refusing to start: 0.8 GB free" in message
    assert "ORBITER_SESSIONS_DIR" in message and "--force" in message
    assert backend.calls == [["colmap", "-h"]]      # nothing else ran

    result, backend = _run(session_dir, force=True)
    assert result.ran == MODE_STEPS["texture-only"]
    assert any("Forced." in w for w in result.warnings)


# ── the checks that turn assumptions into facts ──────────────────────────


def test_validate_sparse_runs_model_converter_and_aborts_on_failure(
        tmp_path) -> None:
    """Five seconds against an hour: COLMAP is asked whether it accepts the
    model before anything expensive reads it. On success the smoke-test PLY
    does not survive the step; on failure the chain stops there."""
    session_dir = _session(tmp_path)
    _, backend = _run(session_dir)
    assert backend.argv_for("model_converter")[2:] == [
        "--input_path", "/data/colmap/sparse",
        "--output_path", "/data/colmap/_validate.ply",
        "--output_type", "PLY"]
    assert not (session_dir / "colmap" / "_validate.ply").exists()

    refused = _session(tmp_path / "refused")
    backend = _fake(refused, exits={"model_converter": 1})
    with pytest.raises(StepFailed, match="COLMAP refused the model"):
        _run(refused, backend)
    assert backend.tools == ["-h", "model_converter"]
    state = State.read(refused)
    assert state.finished("write_sparse")
    assert not state.finished("validate_sparse")


def test_texture_workspace_postcondition_catches_an_unfiltered_workspace(
        tmp_path) -> None:
    """`--image_list_path` is the one flag the whole texture story rests on. A
    workspace that quietly held every image would texture the atlas from laser
    stripes while `session.json` reported the clean set — so the step asserts
    what it just did, and the chain stops before the mesh is ever painted."""
    session_dir = _session(tmp_path)
    every = [f"{side}_{n:04d}.jpg" for n in range(1, 25) for side in ("left", "right")]
    backend = _fake(session_dir, texture_names=every)

    with pytest.raises(StepFailed) as caught:
        _run(session_dir, backend)
    message = str(caught.value)
    assert "the workspace holds 48 images but the list named 24" in message
    assert "--image_list_path did not restrict the model" in message
    assert "texture_sparse" in message
    assert "poisson_mesher" not in backend.tools


def test_texture_workspace_postcondition_catches_renamed_images(tmp_path) -> None:
    """The same count, different names. `--image_list_path` and the mask paths
    both match by string, so a workspace that normalised an extension would
    silently texture from images nobody chose."""
    session_dir = _session(tmp_path)
    _run(session_dir, only="write_sparse")
    renamed = [name.replace(".jpg", ".png") for name in _listed(session_dir)]
    backend = _fake(session_dir, texture_names=renamed)

    with pytest.raises(StepFailed, match="not the 24 that were listed"):
        _run(session_dir, backend)


# ── no GPU anywhere in Milestone 1 ───────────────────────────────────────


def test_texture_only_argv_carries_no_gpus_flag(tmp_path) -> None:
    """Not one M1 invocation asks for a GPU, and the runner never offers one.

    A broken nvidia container runtime would otherwise fail `write_sparse`, in a
    chain that needs no GPU at any step — which is a real way for a CPU-only
    run to die at step one on a machine that never had a card.
    """
    session_dir = _session(tmp_path)
    _, backend = _run(session_dir)

    assert backend.gpus == [None] * len(backend.calls)
    flat = " ".join(part for argv in backend.calls for part in argv)
    assert "--gpus" not in flat and "jitcache" not in flat and "nvidia" not in flat

    docker = DockerBackend(session_dir)
    for argv in backend.calls:
        assert "--gpus" not in docker.command(argv)


def test_texture_only_chain_never_touches_colmap_dense(tmp_path) -> None:
    """`colmap/dense/` is Milestone 2's alone, and the GPU fallback deletes it
    wholesale — so a mesh written there would be either impossible in M1 or
    destructible in M2. `poisson_mesher` writes into `mesh/`, and
    `mesh_texturer` reads it from there."""
    session_dir = _session(tmp_path)
    _, backend = _run(session_dir)

    for argv in backend.calls:
        assert not any("colmap/dense" in part for part in argv), argv
    assert not (session_dir / "colmap" / "dense").exists()
    assert backend.argv_for("poisson_mesher")[2:] == [
        "--input_path", "/data/laser.ply",
        "--output_path", "/data/mesh/meshed-poisson.ply",
        "--PoissonMeshing.trim", "7"]
    assert "/data/mesh/meshed-poisson.ply" in backend.argv_for("mesh_texturer")


# ── resume, ranges and the mode ──────────────────────────────────────────


def test_resume_skips_completed_steps(tmp_path) -> None:
    """A finished step is skipped, which is what makes a five-hour dense run
    survivable — and a resumed run sends no command at all beyond asking which
    COLMAP it is talking to."""
    session_dir = _session(tmp_path)
    _run(session_dir)

    result, backend = _run(session_dir)
    assert result.ran == ()
    assert result.skipped == MODE_STEPS["texture-only"]
    assert backend.calls == [["colmap", "-h"]]

    state = State.read(session_dir)
    assert set(state.steps) == set(MODE_STEPS["texture-only"])
    assert state.mode == "texture-only"
    assert state.max_image_size == recon.ReconParams().max_image_size
    assert all("done_utc" in v and "seconds" in v for v in state.steps.values())


def test_restart_clears_the_state(tmp_path) -> None:
    """`--restart` ignores what finished and runs the lot again."""
    session_dir = _session(tmp_path)
    _run(session_dir)

    result, backend = _run(session_dir, restart=True)
    assert result.ran == MODE_STEPS["texture-only"]
    assert result.skipped == ()
    assert backend.calls == _expected_argv()


def test_to_stops_after_the_named_step(tmp_path) -> None:
    session_dir = _session(tmp_path)
    result, backend = _run(session_dir, to_step="texture_workspace")

    assert result.ran == ("write_sparse", "validate_sparse", "texture_workspace")
    assert backend.tools == ["-h", "model_converter", "image_undistorter"]
    assert not (session_dir / "mesh" / "meshed-poisson.ply").exists()


def test_from_resumes_at_the_named_step(tmp_path) -> None:
    """`--from` starts there, whatever the state file says about it. An
    operator who types a step name means "run this one", not "skip it if the
    file claims it is done"."""
    session_dir = _session(tmp_path)
    _run(session_dir)

    result, backend = _run(session_dir, from_step="texture_workspace")
    assert result.ran == ("texture_workspace",)
    assert result.skipped == ("poisson_mesher", "mesh_texturer", "glb")
    assert backend.tools == ["-h", "image_undistorter"]


def test_only_runs_one_step(tmp_path) -> None:
    session_dir = _session(tmp_path)
    _run(session_dir, only="write_sparse")

    result, backend = _run(session_dir, only="validate_sparse")
    assert result.ran == ("validate_sparse",)
    assert backend.tools == ["-h", "model_converter"]
    assert not (session_dir / "colmap" / "texture").exists()


def _stub_dense(monkeypatch) -> list[str]:
    """Register the dense steps D1b owns as recording no-ops.

    The mode-invalidation rule is `run`'s, not theirs, and it cannot be
    exercised at all while `--mode dense` refuses for want of an
    implementation. The stubs live here rather than in `recon.py`, where a
    placeholder would be a lie about what ships.
    """
    seen: list[str] = []
    for name in CANONICAL_STEPS:
        if name in recon.STEPS:
            continue
        monkeypatch.setitem(recon.STEPS, name,
                            (lambda n: lambda run: seen.append(n))(name))
    return seen


def test_switching_mode_invalidates_the_shared_tail(tmp_path, monkeypatch) -> None:
    """`texture_workspace` is the first step whose input directory depends on
    the mode — verbatim `colmap/images` against inpainted `colmap/images_clean`
    — so a `--mode` change invalidates it and everything after it, in BOTH
    directions, and says so by name.

    Asserting only the mesh tail would pass while the workspace stayed built
    from the wrong pixels, which is precisely the bug this rule exists for.
    """
    stubbed = _stub_dense(monkeypatch)
    session_dir = _session(tmp_path)
    _run(session_dir)                                    # texture-only, complete

    result, backend = _run(session_dir, mode="dense")
    assert "write_sparse" in result.skipped and "validate_sparse" in result.skipped
    assert "texture_workspace" in result.ran
    assert backend.argv_for("image_undistorter")[3] == "/data/colmap/images_clean"
    assert backend.argv_for("poisson_mesher")[3] == "/data/colmap/dense/merged.ply"
    assert stubbed == ["clean_images", "image_undistorter", "undistort_masks",
                       "write_patch_match_cfg", "patch_match_stereo",
                       "stereo_fusion", "merge"]

    log = (session_dir / recon.LOG_NAME).read_text(encoding="utf-8")
    line = next(ln for ln in log.splitlines() if "mode changed" in ln)
    assert "texture-only -> dense" in line
    # The count is computed from what was actually dropped, and the same names
    # are listed beside it.
    assert "invalidating texture_workspace and the 3 steps after it" in line
    assert "(poisson_mesher, mesh_texturer, glb)" in line

    # ...and back the other way, which is the direction revision 4 missed.
    result, backend = _run(session_dir, mode="texture-only")
    assert "texture_workspace" in result.ran
    assert backend.argv_for("image_undistorter")[3] == "/data/colmap/images"
    assert backend.argv_for("poisson_mesher")[3] == "/data/laser.ply"
    back = next(ln for ln in (session_dir / recon.LOG_NAME)
                .read_text(encoding="utf-8").splitlines()
                if "dense -> texture-only" in ln)
    assert "invalidating texture_workspace and the" in back


def test_from_image_undistorter_is_refused_when_clean_images_never_ran(
        tmp_path, monkeypatch) -> None:
    """`colmap/images_clean/` is `clean_images`' output and the dense
    undistorter reads it, so starting after that step without one on record
    would undistort a directory that is missing or stale — and a stale
    workspace is one nobody can tell is stale by looking at it."""
    _stub_dense(monkeypatch)
    session_dir = _session(tmp_path)

    with pytest.raises(ReconRefused) as caught:
        _run(session_dir, mode="dense", from_step="image_undistorter")
    message = str(caught.value)
    assert "clean_images' output" in message
    assert "--from clean_images" in message and "--mode texture-only" in message

    with pytest.raises(ReconRefused, match="no finished clean_images"):
        _run(session_dir, mode="dense", only="patch_match_stereo")

    # Once it is on record the same range is allowed.
    _run(session_dir, mode="dense", to_step="clean_images")
    result, _ = _run(session_dir, mode="dense", from_step="image_undistorter")
    assert result.ran[0] == "image_undistorter"


# ── the container's view of the host ─────────────────────────────────────


def test_docker_maps_windows_paths_to_slash_data_posix(tmp_path) -> None:
    """One mapper, one shape: everything under the session becomes
    `/data/<posix relative>`, and the bind mount is written with forward
    slashes because Docker Desktop takes `D:/sessions/...` and does not take
    the backslash form."""
    session_dir = tmp_path / "20260906-213500"
    (session_dir / "colmap" / "sparse").mkdir(parents=True)

    assert container_path(session_dir, session_dir) == "/data"
    assert container_path(session_dir, session_dir / "colmap" / "sparse") == \
        "/data/colmap/sparse"
    # A session-relative string, the way a step names its own output.
    assert container_path(session_dir, "mesh/meshed-poisson.ply") == \
        "/data/mesh/meshed-poisson.ply"
    if os.name == "nt":
        # A backslash-joined child, the way a Windows caller writes one. The
        # container never sees a backslash, which is the whole point.
        assert container_path(session_dir,
                              str(session_dir) + "\\mesh\\model.glb") == \
            "/data/mesh/model.glb"

    # Only the session is mounted, so nothing outside it can be named at all.
    with pytest.raises(ValueError, match="outside the session directory"):
        container_path(session_dir, tmp_path / "elsewhere.ply")

    docker = DockerBackend(session_dir, image="colmap/colmap:latest")
    cmd = docker.command(["colmap", "poisson_mesher"])
    assert cmd[:4] == ["docker", "run", "--rm", "-v"]
    assert cmd[4] == f"{session_dir.resolve().as_posix()}:/data"
    assert "\\" not in cmd[4]
    assert cmd[5:] == ["colmap/colmap:latest", "colmap", "poisson_mesher"]

    # The GPU and its JIT cache ride on the one invocation that asks for them.
    with_gpu = docker.command(["colmap", "patch_match_stereo"],
                              GpuSpec(uuid="GPU-d5d6", name="RTX 5060 Ti",
                                      jit_volume="orbiter-colmap-jit"))
    assert "--gpus" in with_gpu
    assert with_gpu[with_gpu.index("--gpus") + 1] == "device=GPU-d5d6"
    assert "orbiter-colmap-jit:/jitcache" in with_gpu
    assert "CUDA_CACHE_PATH=/jitcache" in with_gpu


def test_local_backend_only_runs_what_orbiter_colmap_names(tmp_path) -> None:
    """No PATH search: a `colmap` picked up by accident is of unknown version
    and unknown CUDA architectures, and every flag in this chain was read off
    4.2.0. Unset, it says so; set, it unwinds `/data/...` back to the session,
    because a local binary reads the host filesystem and there is no mount."""
    session_dir = tmp_path / "20260906-213500"
    session_dir.mkdir()
    exe = tmp_path / "colmap.exe"

    assert "ORBITER_COLMAP is not set" in \
        recon.LocalBackend(session_dir, executable="").available()
    assert "which is not a file" in \
        recon.LocalBackend(session_dir, executable=str(exe)).available()

    exe.write_bytes(b"MZ")
    backend = recon.LocalBackend(session_dir, executable=str(exe))
    assert backend.available() is None
    cmd = backend.command(["colmap", "poisson_mesher",
                           "--input_path", "/data/laser.ply",
                           "--PoissonMeshing.trim", "7"])
    assert cmd == [str(exe), "poisson_mesher",
                   "--input_path", str(session_dir / "laser.ply"),
                   "--PoissonMeshing.trim", "7"]


# ── the log, and stopping ────────────────────────────────────────────────


def test_recon_log_carries_every_command_and_line(tmp_path) -> None:
    """`recon.log` is authoritative: every command and every line of its
    output, appended so a resumed run sits in the same file as the run it
    resumed. The panel's sink sees exactly the same lines."""
    session_dir = _session(tmp_path)
    seen: list[str] = []
    backend = _fake(session_dir, output={
        "model_converter": ["Reading reconstruction", "Writing PLY"],
        "image_undistorter": ["Undistorting image [1/24]"]})
    recon.run(session_dir, "texture-only", backend=backend,
              on_line=seen.append)

    text = (session_dir / recon.LOG_NAME).read_text(encoding="utf-8")
    assert "$ colmap-fake model_converter --input_path /data/colmap/sparse" in text
    assert "$ colmap-fake image_undistorter" in text
    assert "Reading reconstruction" in text and "Writing PLY" in text
    assert "Undistorting image [1/24]" in text
    for step in MODE_STEPS["texture-only"]:
        assert f"{step}: running" in text and f"{step}: done in" in text
    assert "Reading reconstruction" in seen
    assert any(line.startswith("$ ") for line in seen)

    # Appended, not truncated: the second run's lines join the first's.
    before = len(text.splitlines())
    _run(session_dir)
    assert len((session_dir / recon.LOG_NAME)
               .read_text(encoding="utf-8").splitlines()) > before


def test_cancel_stops_between_steps(tmp_path) -> None:
    """Abort stops the chain at the next step boundary, and what finished stays
    recorded — so the next run resumes there rather than starting over."""
    session_dir = _session(tmp_path)
    stop = threading.Event()
    effects = _effects(session_dir)
    converted = effects["model_converter"]
    effects["model_converter"] = lambda argv: (converted(argv), stop.set())[0]
    backend = FakeBackend(effects=effects)

    with pytest.raises(ReconCancelled, match="cancelled before texture_workspace"):
        recon.run(session_dir, "texture-only", backend=backend, cancel=stop)

    assert backend.tools == ["-h", "model_converter"]
    state = State.read(session_dir)
    assert state.finished("write_sparse") and state.finished("validate_sparse")
    assert not state.finished("texture_workspace")


# ── the GLB, which is allowed to fail ────────────────────────────────────


def test_glb_packing_is_best_effort(tmp_path) -> None:
    """A textured PLY plus its atlas becomes one `model.glb` — and when it
    cannot, the run still finishes.

    Blender's PLY importer drops per-face UVs, so the mesh COLMAP writes loads
    untextured almost everywhere; a GLB is the one container every viewer opens
    with the texture already on. But losing an hour of reconstruction to a
    packaging library is not a trade worth making, so a failure leaves
    `mesh.ply` and `texture.png` on disk and logs a warning.
    """
    session_dir = _session(tmp_path)
    result, _ = _run(session_dir)
    glb = session_dir / "mesh" / "model.glb"
    assert glb.stat().st_size > 0
    assert glb.read_bytes()[:4] == b"glTF"
    assert result.warnings == []

    # The same chain with nothing for the packer to read.
    broken = _session(tmp_path / "broken")
    effects = _effects(broken)
    effects["mesh_texturer"] = lambda argv: (
        broken / "mesh" / "mesh.ply").write_bytes(b"not a ply")
    result, _ = _run(broken, FakeBackend(effects=effects))

    assert result.ran == MODE_STEPS["texture-only"]         # it finished
    assert not (broken / "mesh" / "model.glb").exists()
    assert any("was not packed into model.glb" in w for w in result.warnings)
    assert (broken / "mesh" / "mesh.ply").exists()


# ── the pass-agreement warning ───────────────────────────────────────────


def test_silhouette_score_reads_the_edge_under_the_projected_boundary(
        tmp_path) -> None:
    """The check behind the pass-agreement warning, on its own: the cloud's
    projected silhouette scores high where a real edge lies under it and low
    where the frame is flat.

    Flipping the physical laser switch can nudge the object relative to the
    board, and view selection *prefers* the clean photographs — so that
    corruption would land precisely where the atlas is most confident.
    """
    import cv2

    from orbiter_native.views import load_session
    session_dir = _session(tmp_path)
    info, photos = load_session(session_dir)
    photo = photos[0]
    xyz, _, _ = recon.read_ply_full(str(session_dir / recon.LASER_PLY))

    eye = info.eye(photo.side)
    cam = xyz @ np.asarray(photo.R).T + np.asarray(photo.t_mm)
    u = eye.fx * cam[:, 0] / cam[:, 2] + eye.cx

    flat = np.full((WH[1], WH[0]), 90, np.uint8)
    edged = flat.copy()
    lo, hi = int(u.min()), int(u.max())
    edged[:, max(0, lo - 30):lo + 1] = 240          # hard edges where the
    edged[:, hi:min(WH[0], hi + 30)] = 240          # silhouette's sides land

    blank = tmp_path / "flat.png"
    marked = tmp_path / "edged.png"
    cv2.imwrite(str(blank), flat)
    cv2.imwrite(str(marked), edged)

    quiet = recon._silhouette_score(info, photo, xyz, blank)
    loud = recon._silhouette_score(info, photo, xyz, marked)
    assert quiet == 0.0
    assert loud > 1.0
    # An unreadable file is not a score of zero — it is no score at all, and a
    # pass with no readable samples must not be reported as a bad one.
    assert recon._silhouette_score(info, photo, xyz, tmp_path / "absent.png") is None
