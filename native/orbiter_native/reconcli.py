"""What the offline chain needs, asked one question at a time — and the chain.

    orbiter-recon --check --mode dense
    orbiter-recon --session ~/.orbiter-native/sessions/20260907-051417 --mode texture-only

Every way a reconstruction dies before it starts looks the same from the app: a
step exits non-zero an unknown number of minutes in. Docker missing, the image
not pulled, a COLMAP whose flags moved, a bind mount that maps a Windows path to
nothing, a container that cannot write the JIT cache, a disk with no room for
the depth maps — one line each, with a verdict, so the answer is read rather
than inferred. `--check` starts nothing longer than a second and is the first
line of the live-bench checklist.

The run half is a thin wrapper: `recon.run` owns the chain, the resume, the
refusals and the log, and this module owns the argument names, the streaming of
lines to stdout and the exit code. Three codes, and they mean different things —
0 the chain finished, 1 a step ran and failed, 2 a refusal an operator can act
on before anything starts (too little disk, a step that is not in this mode's
chain, a session that is not there).

One thing `--check` deliberately does not claim: whether an RTX 5060 Ti will run
PatchMatch. The public image carries SASS to sm_90 and PTX for compute_90, and
sm_120 is neither, so the answer depends on the driver JIT-compiling that PTX
forward — which nothing but a real dense run can find out. The check says so and
names the fallback rather than guessing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from . import recon
from .photos import sessions_root
from .recon import (
    COLMAP_GPU_ENV,
    COLMAP_IMAGE_ENV,
    COLMAP_LOCAL_ENV,
    DEFAULT_COLMAP_IMAGE,
    DEFAULT_GPU_MATCH,
    MODE_STEPS,
    ReconCancelled,
    ReconRefused,
    StepFailed,
    _VERSION_RE,
    _shown,
    disk_estimate_bytes,
    gpu_choice,
    gpu_entries,
    undistorted_wh_for,
)
from .views import SelectParams, load_session

#: Every flag in this chain was read off COLMAP 4.2.0. A different version is
#: not a refusal — it may well work — but it is the first thing to suspect when
#: a step rejects an option, so it is said out loud.
EXPECTED_COLMAP = "4.2.0"

# Which card `patch_match_stereo` asks for, how `nvidia-smi -L` is parsed and
# which of its entries wins are all `recon`'s, imported above rather than
# restated here. This module's whole claim is that it reports the decision the
# runner will make, and the only way to keep that claim true is to make it with
# the runner's own code.

#: What the dense estimate is evaluated at when there is no session to measure:
#: a full capture at the selection's cap, at the size this rig's sensors shoot.
DEFAULT_DENSE_PHOTOS = 150
DEFAULT_SENSOR_WH = (1920, 1080)

#: The chain's own output trees. A dry run plans a *fresh* run, so its mirror of
#: the session leaves them behind — and `colmap/dense/` is tens of gigabytes,
#: which is not something a dry run may copy.
_CHAIN_OUTPUTS = ("colmap", "mesh", "clean", recon.LOG_NAME, recon.STATE_NAME)
#: Directories the chain only ever reads. They are hard-linked into the mirror,
#: so a session of photographs costs no bytes and no time. Everything else is
#: copied, because a hard link IS the file: `write_sparse` rewrites `laser.ply`
#: when the cloud carries no normals, and the report rewrites `session.json`.
_LINKED = ("photos", "stripe")

#: A line's mark, as `orbiter-rigcheck` prints it: nothing to see, worth
#: knowing, in the way of a run.
OK, WARN, BAD = "ok", "warn", "bad"
MARK = {OK: "  ", WARN: " !", BAD: " X"}

#: A probe: argv in, (exit code, merged output) out. Injected so the tests can
#: answer for Docker on a machine that has none.
Runner = Callable[[Sequence[str]], "tuple[int, str]"]


def say(state: str, line: str) -> None:
    print(f"{MARK[state]} {line}")


#: The image's entrypoint is NVIDIA's, and it prints a CUDA banner before it
#: runs anything it is given. A probe whose answer is plain text — a uid, a
#: directory listing — would read that banner instead, and `ls /data` reading
#: the banner's words as filenames would pass a mount that maps to nothing,
#: which is the one failure that probe exists to catch. So those probes echo a
#: marker of their own first and the answer is whatever follows the last one.
_MARK = "orbiter-probe"


def _first(text: str, otherwise: str = "") -> str:
    """The first non-empty line of a probe's output, which is where both Docker
    and COLMAP put the sentence worth reading."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return otherwise


def _after_mark(text: str) -> str:
    """Everything the container printed after the marker it was told to echo."""
    _, found, tail = text.rpartition(_MARK)
    return tail if found else text


def _capture(argv: Sequence[str], timeout: float = 120.0) -> tuple[int, str]:
    """Run one probe and return its code and its merged output.

    stdout and stderr are merged for the reason the runner merges them: Docker
    writes its refusals to stderr and the container writes its answer to stdout,
    and a probe wants whichever arrived. A probe that hangs is a probe that
    failed, so the timeout is part of the answer rather than an exception every
    caller would have to know about.
    """
    kwargs: dict[str, object] = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        done = subprocess.run(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            **kwargs)
    except FileNotFoundError:
        return 127, f"{argv[0]} is not on PATH"
    except OSError as exc:
        return 126, str(exc)
    except subprocess.TimeoutExpired:
        return 124, f"no answer in {timeout:.0f} s"
    return done.returncode, done.stdout or ""


# ── the probes ───────────────────────────────────────────────────────────


def _probe_docker(runner: Runner, docker: str, trouble: list[str]) -> bool:
    code, out = runner([docker, "version", "--format", "{{.Server.Version}}"])
    if code != 0:
        say(BAD, f"docker: no server — {_first(out, f'exit {code}')}")
        trouble.append("docker is not answering")
        return False
    say(OK, f"docker: server {_first(out, '?')}")
    return True


def _probe_image(runner: Runner, docker: str, image: str,
                 trouble: list[str]) -> bool:
    code, out = runner([docker, "image", "inspect", image, "--format", "{{.Id}}"])
    if code != 0:
        say(BAD, f"image: {image} is not on this machine — docker pull {image}")
        trouble.append(f"{image} is not pulled")
        return False
    say(OK, f"image: {image} {_first(out)[:19]}")
    return True


def _probe_version(runner: Runner, docker: str, image: str,
                   trouble: list[str]) -> None:
    code, out = runner([docker, "run", "--rm", image, "colmap", "-h"])
    found = _VERSION_RE.search(out)
    if code != 0 or not found:
        say(BAD, f"colmap: the image ran `colmap -h` and said nothing about a "
                 f"version (exit {code})")
        trouble.append("the image's colmap did not answer")
        return
    version = found.group(1)
    if version != EXPECTED_COLMAP:
        say(WARN, f"colmap: {version}, not the {EXPECTED_COLMAP} every flag in "
                  "this chain was read off — suspect this first if a step "
                  "rejects an option")
        return
    say(OK, f"colmap: {version}")


def _probe_gpu(runner: Runner, docker: str, image: str, mode: str,
               trouble: list[str]) -> None:
    """The nvidia runtime, the cards, and what cannot be known from here.

    The listing call is the probe on purpose: it is exactly what the runner does
    to resolve a UUID before `patch_match_stereo`, so a listing that works is
    the call that has to work. Addressing one card by UUID afterwards only
    confirms what the listing already said, and is skipped.

    The answer is parsed by `recon.gpu_entries` and decided by
    `recon.gpu_choice` — the same two the runner calls — so the card named
    here is the card the run will pin, rather than the card a second copy of
    the rule would have picked.
    """
    if mode != "dense":
        say(OK, "gpu: skipped — no texture-only step is given --gpus, so the "
                "nvidia runtime cannot break this run")
        return
    code, out = runner([docker, "run", "--rm", "--gpus", "all", image,
                        "nvidia-smi", "-L"])
    if code != 0:
        say(BAD, f"gpu: `--gpus all` failed — {_first(out, f'exit {code}')}")
        trouble.append("the nvidia container runtime is not working")
        return
    entries = gpu_entries(out)
    if not entries:
        say(BAD, "gpu: nvidia-smi -L listed no card")
        trouble.append("no GPU is visible to the container")
        return
    want = os.environ.get(COLMAP_GPU_ENV, DEFAULT_GPU_MATCH)
    primary, fallback = gpu_choice(entries, want)
    say(OK, f"gpu: {len(entries)} card(s) — "
            + "; ".join(f"{card.name} ({card.uuid})" for card in entries))
    say(OK, f"gpu: primary is {primary.name} ({primary.uuid}), chosen on "
            f"{COLMAP_GPU_ENV}={want}")
    if fallback is None:
        say(WARN, "gpu: no fallback card — a missing kernel aborts instead of "
                  "retrying, and the remedy is native/docker/colmap-cuda128")
    else:
        say(OK, f"gpu: fallback is {fallback.name} ({fallback.uuid}) — "
                "patch_match_stereo retries there once")
    say(WARN, "gpu: whether the primary runs PatchMatch cannot be probed from "
              "here — the image carries SASS to sm_90 and PTX for compute_90 "
              "only, so an sm_120 card runs it solely by a driver JIT of that "
              "PTX, and only a real dense run finds out")


def _probe_container_user(runner: Runner, docker: str, image: str) -> None:
    """Whether the container is root, because the JIT cache is a volume.

    A driver JIT of the compute_90 PTX costs minutes and is thrown away by
    `--rm` unless it lands in a mounted volume — and a volume is only writable
    if the container's user can write it.
    """
    code, out = runner([docker, "run", "--rm", image,
                        "sh", "-c", f"echo {_MARK}; id -u"])
    uid = _first(_after_mark(out))
    if code != 0 or not uid:
        say(WARN, f"user: the image would not say who it runs as (exit {code}) "
                  "— chown the JIT volume if patch_match_stereo cannot cache")
        return
    if uid == "0":
        say(OK, "user: root — the JIT cache volume is writable as it is")
        return
    say(WARN, f"user: uid {uid}, not root — chown the JIT volume once, or the "
              "PTX compile is paid for again on every run")


def _probe_bind_mount(runner: Runner, docker: str, image: str,
                      session: Path | None, trouble: list[str]) -> None:
    """The one probe that catches a Windows path that maps to nothing.

    Docker Desktop takes `D:/sessions/...` and does not take the backslash form,
    and a mount that resolves to an empty directory fails an hour in, silently,
    having written into a container nobody looks at.
    """
    if session is None:
        say(OK, "mount: skipped, no --session was given — pass one to "
                "bind-mount a real session and prove the path maps")
        return
    mount = session.resolve().as_posix()
    code, out = runner([docker, "run", "--rm", "-v", f"{mount}:/data", image,
                        "sh", "-c", f"echo {_MARK}; ls /data"])
    if code != 0:
        say(BAD, f"mount: {mount} did not mount — {_first(out, f'exit {code}')}")
        trouble.append("the session does not bind-mount")
        return
    seen = [name for name in _after_mark(out).split() if name.strip()]
    if not seen:
        say(BAD, f"mount: {mount} mounted empty — the container sees none of "
                 "the session's files")
        trouble.append("the session mounts empty")
        return
    say(OK, f"mount: {mount} -> /data, {len(seen)} entries")


def _probe_local(runner: Runner, trouble: list[str]) -> None:
    """A local COLMAP, and only when `ORBITER_COLMAP` names one.

    There is no PATH search anywhere in this chain: a `colmap` picked up by
    accident is of unknown version and unknown CUDA architectures, and a clear
    "set ORBITER_COLMAP" beats a run that dies on an option that moved.
    """
    exe = os.environ.get(COLMAP_LOCAL_ENV, "")
    if not exe:
        say(OK, f"local: {COLMAP_LOCAL_ENV} is unset — the Docker backend runs "
                "every step")
        return
    code, out = runner([exe, "-h"])
    if code != 0:
        say(BAD, f"local: {exe} would not run (exit {code})")
        trouble.append(f"{COLMAP_LOCAL_ENV} names something that will not run")
        return
    found = _VERSION_RE.search(out)
    version = found.group(1) if found else "of an unstated version"
    say(OK, f"local: {exe} is COLMAP {version}")


def _disk_inputs(mode: str, session: Path | None,
                 max_image_size: int) -> tuple[int, tuple[int, int], str]:
    """How many images the estimate is evaluated at, at what size, and why.

    The exact selection is `write_sparse`'s to make — it needs the cloud, the
    poses and a KD tree, which is not a probe — so this counts what the manifest
    holds and caps it the way the selection caps it. That is an upper bound on
    the selection and therefore on the disk, which is the direction a check
    wants to be wrong in.
    """
    if mode != "dense":
        return 0, (0, 0), "a flat gigabyte: texture-only writes no depth maps"
    if session is None:
        return (DEFAULT_DENSE_PHOTOS,
                undistorted_wh_for(DEFAULT_SENSOR_WH, max_image_size),
                f"no --session: {DEFAULT_DENSE_PHOTOS} photographs at "
                f"{DEFAULT_SENSOR_WH[0]}x{DEFAULT_SENSOR_WH[1]}")
    try:
        info, photos = load_session(session)
    except (OSError, ValueError, KeyError):
        return (DEFAULT_DENSE_PHOTOS,
                undistorted_wh_for(DEFAULT_SENSOR_WH, max_image_size),
                "the session's manifest could not be read, so this assumes a "
                "full capture")
    cap = SelectParams().cap
    sizes = [eye.wh for eye in (info.eye("left"), info.eye("right")) if eye]
    source = max(sizes, key=lambda wh: wh[0] * wh[1]) if sizes else DEFAULT_SENSOR_WH
    return (min(len(photos), cap),
            undistorted_wh_for((int(source[0]), int(source[1])), max_image_size),
            f"{len(photos)} photographs in the manifest, capped at the "
            f"selection's {cap}")


def _probe_disk(mode: str, session: Path | None, max_image_size: int,
                trouble: list[str]) -> None:
    """Free space where the run will write, against what this mode wants.

    The session root is the place to ask about even when no session is named:
    that is the filesystem every session lands on, and a root that does not
    exist yet is asked about through the nearest parent that does.
    """
    probe = Path(session if session is not None else sessions_root())
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = recon.free_space_bytes(probe)
    n_images, wh, why = _disk_inputs(mode, session, max_image_size)
    estimate = disk_estimate_bytes(mode, n_images, wh)
    shape = f" for {n_images} images at {wh[0]}x{wh[1]}" if mode == "dense" else ""
    line = (f"disk: {_gb(free)} free on {probe}, {mode} wants about "
            f"{_gb(estimate)}{shape} ({why})")
    if free < estimate:
        say(BAD, line)
        trouble.append(f"{_gb(free)} free is below the {_gb(estimate)} "
                       f"{mode} wants")
        return
    say(OK, line)


def _gb(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.1f} GB"


def check(mode: str = "texture-only", session: Path | None = None, *,
          runner: Runner = _capture, docker: str = "docker") -> int:
    """Ask every question a run depends on, print one line each, and return 0
    when nothing is in the way.

    The order is the order the answers gate each other: with no Docker the
    questions that run inside the image cannot be asked at all, and asking them
    anyway would print four more failures about the first one.
    """
    image = os.environ.get(COLMAP_IMAGE_ENV, DEFAULT_COLMAP_IMAGE)
    trouble: list[str] = []
    print(f"mode {mode} · image {image} · "
          f"session {session if session is not None else '(none)'}\n")

    if (_probe_docker(runner, docker, trouble)
            and _probe_image(runner, docker, image, trouble)):
        _probe_version(runner, docker, image, trouble)
        _probe_gpu(runner, docker, image, mode, trouble)
        _probe_container_user(runner, docker, image)
        _probe_bind_mount(runner, docker, image, session, trouble)
    else:
        say(WARN, "colmap, gpu, user and mount: not asked — every one of them "
                  "runs inside the image")
    _probe_local(runner, trouble)
    _probe_disk(mode, session, recon.ReconParams().max_image_size, trouble)

    print()
    if trouble:
        print("in the way of a run: " + "; ".join(trouble))
        return 1
    print(f"nothing in the way of a {mode} run")
    return 0


# ── the dry run ──────────────────────────────────────────────────────────


def _mirror(session_dir: Path, into: Path) -> Path:
    """A copy of the session's inputs that a dry run may write into freely.

    The point is that `--dry-run` writes nothing in the session itself: a chain
    driven through a fake backend still runs its host steps — that is where the
    argv comes from — and those steps write a model, a state file and a log. A
    state file claiming `validate_sparse` finished, written by a run that
    validated nothing, is the worst outcome available here, so the whole thing
    happens somewhere else and is thrown away.
    """
    root = into / session_dir.name
    root.mkdir(parents=True)
    for src in sorted(session_dir.rglob("*")):
        rel = src.relative_to(session_dir)
        if rel.parts[0] in _CHAIN_OUTPUTS:
            continue
        dst = root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts[0] in _LINKED:
            try:
                os.link(src, dst)
                continue
            except OSError:                 # another volume, or no hard links
                pass
        shutil.copy2(src, dst)
    return root


def dry_run(session: Path, mode: str, *, from_step: str | None = None,
            to_step: str | None = None, only: str | None = None,
            params: recon.ReconParams | None = None) -> int:
    """Print the command line of every COLMAP invocation this chain would make.

    The commands come from the chain itself rather than from a second copy of
    the argv kept here, which is the only arrangement in which the two cannot
    drift apart. Steps whose post-condition reads output that only a real COLMAP
    produces end the dry run — `texture_workspace` asserts that
    `--image_list_path` restricted the workspace it built, and a workspace
    nobody built holds nothing — so the last line names where it stopped instead
    of leaving the chain looking shorter than it is.

    What it shows is the whole chain, not a resume: the mirror carries the
    session's inputs and none of its outputs, so there is no finished step for
    it to skip and it would be dishonest to imply otherwise.
    """
    fake = recon.FakeBackend()
    speller = recon.DockerBackend(session)
    stopped = ""
    print(f"dry run: {mode} — no container is started, nothing under {session} "
          f"is written, and every step is shown whether or not a resume would "
          f"skip it\n")
    with tempfile.TemporaryDirectory(prefix="orbiter-dry-") as tmp:
        mirror = _mirror(Path(session), Path(tmp))
        try:
            recon.run(mirror, mode, from_step=from_step, to_step=to_step,
                      only=only, restart=True, force=True, backend=fake,
                      on_line=lambda line: None, params=params)
        except ReconRefused as exc:
            print(f"refused: {exc}")
            return 2
        except (StepFailed, ReconCancelled) as exc:
            stopped = str(exc)

    for argv, gpu in zip(fake.calls, fake.gpus):
        print("  " + _shown(speller.command(argv, gpu)))
    if stopped:
        print(f"\nstopped there: {stopped}\n"
              "A dry run produces no COLMAP output for the steps after it to "
              "read; the commands above are the ones the runner sends.")
    return 0


# ── the run ──────────────────────────────────────────────────────────────


def _run_chain(session: Path, mode: str, args: argparse.Namespace,
               params: recon.ReconParams | None) -> int:
    """Hand the whole thing to `recon.run` and turn what comes back into an
    exit code. Every line the chain logs is printed as it arrives, because an
    hour-long step with nothing on stdout is indistinguishable from a hang."""
    try:
        result = recon.run(session, mode, from_step=args.from_step,
                           to_step=args.to_step, only=args.only,
                           restart=args.restart, force=args.force,
                           on_line=print, params=params)
    except ReconRefused as exc:
        print(f"refused: {exc}")
        return 2
    except StepFailed as exc:
        print(f"failed: {exc}")
        return 1
    except ReconCancelled as exc:
        print(f"cancelled: {exc}")
        return 1

    print(f"\nran {len(result.ran)}, skipped {len(result.skipped)}"
          + (f", texture_source={result.texture_source}"
             if result.texture_source else ""))
    for text in result.warnings:
        say(WARN, text)
    if result.degraded:
        say(WARN, "degraded: " + ", ".join(result.degraded)
                  + " — the run finished as something less than it set out to "
                    "be, which is not a pass")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Session ids, atlas names and this module's own separators are not ASCII,
    # and a Windows console is cp866 or cp1251 until it is told otherwise — at
    # which point printing a line raises instead of printing it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        prog="orbiter-recon", description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="ask what a run depends on and print one line per "
                         "question; starts no reconstruction")
    ap.add_argument("--session", metavar="DIR",
                    help="the session's own directory, the one holding "
                         "session.json")
    ap.add_argument("--mode", choices=sorted(MODE_STEPS), default="texture-only",
                    help="texture-only meshes the laser cloud and paints it "
                         "with the photographs; dense adds PatchMatch "
                         "(default: texture-only)")
    ap.add_argument("--from", dest="from_step", metavar="STEP",
                    help="start at this step, running it even if it finished")
    ap.add_argument("--to", dest="to_step", metavar="STEP",
                    help="stop after this step")
    ap.add_argument("--only", metavar="STEP",
                    help="run this step and nothing else")
    ap.add_argument("--restart", action="store_true",
                    help="clear the state file, so every step runs again")
    ap.add_argument("--force", action="store_true",
                    help="run even when free disk is below the estimate")
    ap.add_argument("--max-image-size", type=int, metavar="N",
                    help="the dense image_undistorter's limit (default "
                         f"{recon.ReconParams().max_image_size})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command line of every COLMAP invocation "
                         "the chain would make, and write nothing")
    ap.add_argument("--list-steps", action="store_true",
                    help="print each mode's chain and exit")
    a = ap.parse_args(argv)

    if a.list_steps:
        for mode in sorted(MODE_STEPS):
            print(f"{mode}: {' -> '.join(MODE_STEPS[mode])}")
        return 0

    session: Path | None = None
    if a.session:
        session = Path(a.session).expanduser()
        if not session.is_dir():
            print(f"refused: {session} is not a directory — pass the session's "
                  "own directory, the one holding session.json.")
            return 2

    if a.check:
        return check(a.mode, session)

    if session is None:
        ap.error("--session is required unless --check or --list-steps is given")

    params = (recon.ReconParams(max_image_size=a.max_image_size)
              if a.max_image_size is not None else None)
    if a.dry_run:
        return dry_run(session, a.mode, from_step=a.from_step,
                       to_step=a.to_step, only=a.only, params=params)
    return _run_chain(session, a.mode, a, params)


if __name__ == "__main__":
    sys.exit(main())
