"""`orbiter-recon`: what `--check` asks, and what the flags do with the runner.

No Docker daemon is started anywhere here. Every probe goes through the runner
seam `check()` takes, so the tests answer for Docker themselves — which is the
only way to assert what the check does with an answer it was given rather than
with whatever this machine happens to be able to say today.

The session fixture is `test_recon`'s: a real session on disk, twelve places
round the subject photographed twice, with the box cloud exported as
`laser.ply`. It is shared rather than rebuilt so the CLI is tested against the
same thing the runner is tested against, which is what makes "the dry run
prints the same argv as the runner" a comparison of two real answers.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from orbiter_native import recon, reconcli
from orbiter_native.recon import (
    COLMAP_IMAGE_ENV,
    COLMAP_LOCAL_ENV,
    DEFAULT_COLMAP_IMAGE,
    MODE_STEPS,
    FakeBackend,
    disk_estimate_bytes,
    undistorted_wh_for,
)
from orbiter_native.reconcli import COLMAP_GPU_ENV
from orbiter_native.views import load_session

from test_recon import _fake, _run, _session

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: Both cards of this machine, as `nvidia-smi -L` prints them inside the
#: container: the sm_120 one the runner prefers and the sm_75 one it falls back
#: to.
SMI = ("GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-11111111-2222-3333-4444-555555555555)\n"
       "GPU 1: NVIDIA GeForce GTX 1650 SUPER (UUID: GPU-66666666-7777-8888-9999-aaaaaaaaaaaa)\n")


#: What the image's entrypoint prints before it runs the command it was given.
#: Every canned answer carries it, because a probe that reads the banner
#: instead of the answer is the bug this shape of fixture exists to catch: `ls`
#: reading these words as filenames would pass a mount that maps to nothing.
BANNER = ("==========\n== CUDA ==\n==========\n\nCUDA Version 12.9.1\n\n"
          "WARNING: The NVIDIA Driver was not detected.\n\n")


def _runner(**overrides):
    """A stand-in for Docker: one canned answer per probe, keyed by a fragment
    of the argv that identifies it.

    Every call is recorded, so a test can assert that a probe was skipped by
    the absence of its call rather than only by the absence of its line.
    """
    answers: dict[str, tuple[int, str]] = {
        "version": (0, "27.4.0\n"),
        "inspect": (0, "sha256:0123456789abcdef0123456789abcdef\n"),
        "nvidia-smi": (0, BANNER + SMI),
        "id -u": (0, BANNER + "orbiter-probe\n0\n"),
        "ls /data": (0, BANNER + "orbiter-probe\nsession.json\nphotos.jsonl\n"
                        "laser.ply\nphotos\n"),
        "-h": (0, BANNER + "COLMAP 4.2.0\n"),
    }
    answers.update(overrides)
    calls: list[list[str]] = []

    def run(argv):
        calls.append(list(argv))
        for key, answer in answers.items():
            if any(key in part for part in argv):
                return answer
        raise AssertionError(f"no canned answer for {argv}")

    run.calls = calls
    return run


@pytest.fixture(autouse=True)
def _neutral_env(monkeypatch):
    """The check reads three environment variables, and this machine's values
    for them would make these tests pass or fail for reasons of their own."""
    for name in (COLMAP_IMAGE_ENV, COLMAP_LOCAL_ENV, COLMAP_GPU_ENV):
        monkeypatch.delenv(name, raising=False)


def _plenty(monkeypatch) -> None:
    """Enough free space that the disk probe is never the thing under test."""
    monkeypatch.setattr(recon, "free_space_bytes", lambda path: 500 << 30)


def _lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


# ── --check ──────────────────────────────────────────────────────────────


def test_check_reports_each_probe(tmp_path, monkeypatch, capsys) -> None:
    """One line per question, and a verdict on the lot.

    Dense with a session is the shape that asks everything: texture-only skips
    the GPU and no session skips the mount, and both of those are their own
    tests below.
    """
    _plenty(monkeypatch)
    session = _session(tmp_path)
    runner = _runner()

    code = reconcli.check("dense", session, runner=runner)
    out = "\n".join(_lines(capsys))

    assert code == 0
    for probe in ("docker:", "image:", "colmap:", "gpu:", "user:", "mount:",
                  "local:", "disk:"):
        assert probe in out, f"{probe} was never reported\n{out}"
    assert DEFAULT_COLMAP_IMAGE in out
    assert "COLMAP 4.2.0" in out or "colmap: 4.2.0" in out
    assert "user: root" in out                    # the JIT volume is writable
    assert "nothing in the way of a dense run" in out

    # The listing probe is the one the plan says has to work, and it is the one
    # that ran — not a per-UUID `--gpus device=...` confirmation of it.
    assert ["docker", "run", "--rm", "--gpus", "all", DEFAULT_COLMAP_IMAGE,
            "nvidia-smi", "-L"] in runner.calls
    assert not any("device=" in part for call in runner.calls for part in call)

    # sm_120 is stated as unknowable rather than guessed at, in both directions.
    assert "cannot be probed" in out and "compute_90" in out


def test_check_names_the_primary_and_the_fallback_gpu(tmp_path, monkeypatch,
                                                      capsys) -> None:
    """Which card `patch_match_stereo` will ask for, and which one it retries
    on — by name and by UUID, because slot order decides nothing here."""
    _plenty(monkeypatch)
    session = _session(tmp_path)

    reconcli.check("dense", session, runner=_runner())
    out = "\n".join(_lines(capsys))

    assert "primary is NVIDIA GeForce RTX 5060 Ti " \
           "(GPU-11111111-2222-3333-4444-555555555555)" in out
    assert "fallback is NVIDIA GeForce GTX 1650 SUPER " \
           "(GPU-66666666-7777-8888-9999-aaaaaaaaaaaa)" in out
    assert f"{COLMAP_GPU_ENV}=5060" in out

    # One card is a different answer, not a shorter one: there is nothing to
    # retry on, and the check has to say so rather than stay quiet.
    one = SMI.splitlines()[1] + "\n"
    reconcli.check("dense", session, runner=_runner(**{"nvidia-smi": (0, one)}))
    alone = "\n".join(_lines(capsys))
    assert "primary is NVIDIA GeForce GTX 1650 SUPER" in alone
    assert "no fallback card" in alone


def test_check_reports_free_space_against_the_mode_estimate(
        tmp_path, monkeypatch, capsys) -> None:
    """The number printed is the estimate the runner will refuse on, computed
    from this session's own photograph count and undistorted size."""
    session = _session(tmp_path)
    info, photos = load_session(session)
    wh = undistorted_wh_for(info.eye("left").wh,
                            recon.ReconParams().max_image_size)
    estimate = disk_estimate_bytes("dense", len(photos), wh)

    monkeypatch.setattr(recon, "free_space_bytes", lambda path: estimate // 4)
    code = reconcli.check("dense", session, runner=_runner())
    out = "\n".join(_lines(capsys))

    assert code == 1
    assert f" X disk: {reconcli._gb(estimate // 4)} free" in out
    assert f"wants about {reconcli._gb(estimate)}" in out
    assert f"for {len(photos)} images at {wh[0]}x{wh[1]}" in out
    assert "in the way of a run:" in out

    # Texture-only asks for the flat gigabyte instead, and the same free space
    # is then plenty — the estimate is per mode, not per machine.
    monkeypatch.setattr(recon, "free_space_bytes",
                        lambda path: recon.TEXTURE_ONLY_DISK_BYTES * 2)
    assert reconcli.check("texture-only", session, runner=_runner()) == 0
    flat = "\n".join(_lines(capsys))
    assert f"wants about {reconcli._gb(recon.TEXTURE_ONLY_DISK_BYTES)}" in flat


def test_check_skips_the_bind_mount_probe_without_a_session_and_says_so(
        tmp_path, monkeypatch, capsys) -> None:
    """A skipped probe is a reported probe. The mount is the one check that
    catches a Windows path mapping to nothing, so silence about it would read
    as a pass."""
    _plenty(monkeypatch)
    runner = _runner()

    code = reconcli.check("dense", None, runner=runner)
    out = "\n".join(_lines(capsys))

    assert code == 0
    assert "mount: skipped, no --session was given" in out
    assert not any("-v" in call for call in runner.calls)

    # And with a session it is not skipped, on the session's own path.
    session = _session(tmp_path)
    reconcli.check("dense", session, runner=_runner())
    mounted = "\n".join(_lines(capsys))
    assert f"mount: {session.resolve().as_posix()} -> /data" in mounted

    # A mount that resolves to nothing is the failure this probe is for, and
    # the image's CUDA banner must not be read as its contents: `ls` printing
    # the banner and nothing else is an empty mount, not sixteen entries.
    empty = reconcli.check("dense", session,
                           runner=_runner(**{"ls /data":
                                             (0, BANNER + "orbiter-probe\n")}))
    assert empty == 1
    assert "mounted empty" in "\n".join(_lines(capsys))


def test_check_skips_the_gpu_probe_for_texture_only(tmp_path, monkeypatch,
                                                    capsys) -> None:
    """No texture-only step is given `--gpus`, so a broken nvidia runtime
    cannot break the run and is not asked about."""
    _plenty(monkeypatch)
    runner = _runner()

    assert reconcli.check("texture-only", _session(tmp_path), runner=runner) == 0
    out = "\n".join(_lines(capsys))

    assert "gpu: skipped" in out
    assert not any("nvidia-smi" in call for call in runner.calls)


# ── --dry-run ────────────────────────────────────────────────────────────


def test_dry_run_prints_the_same_argv_as_the_runner(tmp_path, capsys) -> None:
    """The commands the dry run prints are the commands the runner sends — and
    the session it was pointed at is untouched afterwards.

    The dry run ends where a step's post-condition needs output only a real
    COLMAP produces (`texture_workspace` asserts that `--image_list_path`
    restricted the workspace it built), so what is compared is every command up
    to that point, command for command.
    """
    real_dir = _session(tmp_path / "real")
    _, backend = _run(real_dir, _fake(real_dir))    # the runner's own answer
    expected = [reconcli._shown(recon.DockerBackend(real_dir).command(argv, gpu))
                for argv, gpu in zip(backend.calls, backend.gpus)]

    dry_dir = _session(tmp_path / "dry")
    before = sorted(p.relative_to(dry_dir).as_posix() for p in dry_dir.rglob("*"))

    assert reconcli.dry_run(dry_dir, "texture-only") == 0
    printed = [line[2:] for line in _lines(capsys) if line.startswith("  docker")]

    assert printed, "the dry run printed no commands"
    assert printed == [text.replace(real_dir.as_posix(), dry_dir.as_posix())
                       for text in expected[:len(printed)]]
    assert "colmap image_undistorter" in " ".join(printed)

    # Writes nothing: not the model, not the log, and above all not a state
    # file claiming validate_sparse finished when nothing validated anything.
    after = sorted(p.relative_to(dry_dir).as_posix() for p in dry_dir.rglob("*"))
    assert after == before
    assert not (dry_dir / recon.STATE_NAME).exists()


# ── the flags, and the exit codes ────────────────────────────────────────


def test_session_dir_must_exist(tmp_path, capsys) -> None:
    """Named before anything else happens, because every other failure this
    would cause is reported from somewhere further from the cause."""
    missing = tmp_path / "20260907-000000"

    assert reconcli.main(["--session", str(missing)]) == 2
    assert "is not a directory" in "\n".join(_lines(capsys))

    # Including for --check, which would otherwise bind-mount a path Docker
    # would create as an empty directory.
    assert reconcli.main(["--check", "--session", str(missing)]) == 2


def test_named_refusal_exits_two(tmp_path, monkeypatch, capsys) -> None:
    """A step of the other mode's chain is a refusal, not a run that quietly
    honours nothing — and a refusal is exit 2, not exit 1."""
    monkeypatch.setattr(recon, "DockerBackend",
                        lambda session_dir, *a, **kw: FakeBackend())
    session = _session(tmp_path)

    code = reconcli.main(["--session", str(session), "--mode", "texture-only",
                          "--to", "patch_match_stereo"])
    out = "\n".join(_lines(capsys))

    assert code == 2
    assert "refused:" in out
    assert "patch_match_stereo is a step of the dense chain" in out
    assert " -> ".join(MODE_STEPS["texture-only"]) in out


def test_list_steps_prints_mode_steps(capsys) -> None:
    """One line per mode, in the only order the steps may run in."""
    assert reconcli.main(["--list-steps"]) == 0
    out = "\n".join(_lines(capsys))

    for mode, steps in MODE_STEPS.items():
        assert f"{mode}: {' -> '.join(steps)}" in out


# ── what ships ───────────────────────────────────────────────────────────


def test_console_script_is_declared() -> None:
    """`orbiter-recon` is the name the plan's live-bench checklist types, so it
    ships with the module it names."""
    raw = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = raw["project"]["scripts"]

    assert scripts["orbiter-recon"] == "orbiter_native.reconcli:main"


def test_scipy_is_declared_as_a_dependency() -> None:
    """colmapio, photos, rolling, scan, scanworker and views all import it at
    module scope, so installing this package alone — without orbiter-server,
    which is what was dragging it in — fails on the first import of any of
    them."""
    raw = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    names = [dep.split(">")[0].split("=")[0].split("[")[0].strip()
             for dep in raw["project"]["dependencies"]]

    assert "scipy" in names
