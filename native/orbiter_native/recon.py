"""The offline chain: a session directory in, a textured mesh out.

Everything the scan knows — the confident cloud, a pose for every photograph,
the intrinsics both eyes were solved at — is already on disk when this module
starts. What is left is to hand COLMAP a model it accepts, run its steps in an
order that cannot be got wrong, and check the two or three things that would
otherwise fail silently an hour later.

**One `docker run --rm` per step, and nothing kept warm between them.** A
long-lived container would save a second or two of start-up per step and cost a
lifetime to reason about: which step left which file open, what a crash halfway
leaves behind, whether the mount is still there after a Docker restart. The
chain has thirteen steps and runs for an hour; the start-ups are noise. One
process per step also means one exit code per step, which is what `recon-state`
records and what a resume trusts.

**The GPU is asked for by exactly one step.** `--gpus device=GPU-<uuid>` and the
JIT-cache volume go on `patch_match_stereo`'s invocation and on no other. That
is not tidiness: a broken nvidia container runtime would otherwise fail
`write_sparse`, which needs no GPU at all, in a chain that in texture-only mode
never needs one anywhere. Milestone 1 therefore runs on a machine with no card.

**Two steps are checked rather than trusted.** `validate_sparse` asks COLMAP
whether it accepts our text model at all — five seconds against an hour of
compute — and `texture_workspace` asserts afterwards that `--image_list_path`
really restricted the workspace it built. A workspace that quietly held every
image would texture the atlas from laser stripes while the manifest claimed it
came from the clean photographs, and nothing downstream could tell.

**The state file records the mode.** `colmap/images/` (verbatim) and
`colmap/images_clean/` (inpainted) are different pixels under the same names, so
a resumed run that changed `--mode` must not reuse a workspace undistorted from
the other one. `texture_workspace` is the first step whose input directory
depends on the mode, so a mode change invalidates it and everything after it, in
both directions, and says so by name.

The steps of Milestone 2 — `clean_images`, the dense block and the merge — are
named in `CANONICAL_STEPS` but are not registered here. `run(mode="dense")`
refuses by name until they are, which is a plain sentence rather than a stub
that would fail somewhere less legible.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np

from . import colmapio
from .colmapio import ImageRecord, read_images_bin
from .scan import normals_pca, orient_normals, read_ply_full, write_ply
from .views import (
    MODES,
    PhotoMeta,
    Selection,
    SelectParams,
    SessionInfo,
    buckets,
    load_session,
    seed_points,
    select,
    texture_set,
    tracks,
    write_image_list,
)

log = logging.getLogger("orbiter_native.recon")


# ── the chain ────────────────────────────────────────────────────────────

#: Every step, in the only order they may run in. One definition: the state
#: file's keys are these names, `--from/--to/--only` name one of them, and the
#: mode chains below are slices of this tuple's membership rather than
#: independent lists that could drift apart.
CANONICAL_STEPS = (
    "write_sparse", "validate_sparse", "clean_images", "texture_workspace",
    "image_undistorter", "undistort_masks", "write_patch_match_cfg",
    "patch_match_stereo", "stereo_fusion", "merge",
    "poisson_mesher", "mesh_texturer", "glb",
)

#: What each mode actually runs. Texture-only meshes the laser cloud and paints
#: it with the photographs — minutes, no GPU, and it proves the text model,
#: the path mapping and the texture workspace, which is everything only a real
#: COLMAP can prove.
MODE_STEPS: dict[str, tuple[str, ...]] = {
    "texture-only": ("write_sparse", "validate_sparse", "texture_workspace",
                     "poisson_mesher", "mesh_texturer", "glb"),
    "dense": CANONICAL_STEPS,
}

#: The one mode-dependent path in the whole chain. Exactly two steps read it:
#: `texture_workspace` and (in dense mode) `image_undistorter`. Nothing else
#: does, and nothing ever writes to the other mode's directory.
MODE_IMAGE_PATH = {
    "texture-only": "colmap/images",        # verbatim  — written by write_sparse
    "dense":        "colmap/images_clean",  # inpainted — written by clean_images
}

#: What `poisson_mesher` meshes. In texture-only that is the laser cloud
#: itself; in dense it is the merged cloud, which is the laser surface with
#: dense filling only the holes it never saw.
MODE_POISSON_INPUT = {
    "texture-only": "laser.ply",
    "dense":        "colmap/dense/merged.ply",
}

#: Directories each step must create itself. COLMAP writes into the paths it is
#: given and creates no parents — the sibling pipeline does its own `mkdir -p`
#: (`pipeline.py:1117`) for exactly this reason.
DIRS_CREATED = {
    "write_sparse":      ("colmap", "colmap/images", "colmap/sparse", "mesh"),
    "clean_images":      ("colmap/images_clean", "colmap/_masks_raw", "clean"),
    "texture_workspace": ("colmap/texture",),      # image_undistorter's --output_path
    "image_undistorter": ("colmap/dense",),        # the reference's own `mkdir -p dense`
    "undistort_masks":   ("colmap/dense/masks",),
}

#: `texture_workspace` is the first step whose INPUT directory depends on the
#: mode, so a mode change invalidates it and everything after it — in both
#: directions. `write_sparse` and `validate_sparse` are mode-independent and
#: survive; `clean_images` needs no rule of its own because it is dense-only.
MODE_INVALIDATES_FROM = "texture_workspace"


# ── file names and defaults ──────────────────────────────────────────────

#: Every command and every line of its output, appended. The GUI shows the last
#: line; this file is what an operator reads afterwards, and it is
#: authoritative.
LOG_NAME = "recon.log"
#: `{mode, colmap_version, max_image_size, steps: {name: {done_utc, seconds}}}`.
STATE_NAME = "recon-state.json"
#: The confident cloud, board frame, millimetres. The app exports it here when
#: Reconstruct is pressed; the CLI expects to find it already written.
LASER_PLY = "laser.ply"

#: Below this many confident points there is no surface to mesh, and Poisson
#: over a handful of points produces a blob rather than an object.
MIN_CONFIDENT_POINTS = 1000
#: Below this many selected photographs there is no atlas worth building — and
#: in dense mode no baseline spread worth matching over.
MIN_SELECTED_PHOTOS = 20

#: Milestone 1 writes no depth maps, so its floor is flat: the mesh, the atlas
#: and the undistorted texture workspace together, with room to spare.
TEXTURE_ONLY_DISK_BYTES = 1 << 30
#: The dense estimate's fixed part, and the same 1 GiB for the same reason.
DISK_HEADROOM_BYTES = 1 << 30
#: A depth or normal sample is a float32.
BYTES_PER_SAMPLE = 4
#: Depth is one plane, a normal is three.
MAPS_PER_IMAGE = 4
#: Photometric first, then geometric — `geom_consistency true` keeps both.
PASSES_PER_IMAGE = 2
#: Slack over the arithmetic, as a fraction over ten: workspace copies, the
#: fused cloud, the mesh and COLMAP's own scratch.
DISK_SLACK_TENTHS = 13

#: The image the Docker backend runs, overridable for a locally built one.
COLMAP_IMAGE_ENV = "ORBITER_COLMAP_IMAGE"
DEFAULT_COLMAP_IMAGE = "colmap/colmap:latest"
#: A local `colmap` executable, used ONLY when this names one. No PATH search:
#: a silently discovered binary of unknown build is worse than a clear failure.
COLMAP_LOCAL_ENV = "ORBITER_COLMAP"
#: The named volume the driver's JIT cache lives in. The public image has no
#: sm_120 SASS, so an RTX 5060 Ti runs PatchMatch only by JIT-compiling the
#: compute_90 PTX — and inside `--rm` that work is thrown away unless it lands
#: somewhere that outlives the container.
JIT_VOLUME_ENV = "ORBITER_COLMAP_JIT_VOLUME"
DEFAULT_JIT_VOLUME = "orbiter-colmap-jit"
JIT_CACHE_BYTES = 1 << 30

#: Where the session is mounted inside the container. Every path in every argv
#: is written relative to this, by `container_path` and nothing else.
CONTAINER_ROOT = "/data"

#: `colmap -h` prints a CUDA banner from the base image before it says anything
#: about itself, so the version is found by pattern rather than by line number.
_VERSION_RE = re.compile(r"\bCOLMAP\s+(\d+\.\d+(?:\.\d+)?)")


# ── failures ─────────────────────────────────────────────────────────────


class ReconRefused(RuntimeError):
    """A precondition the operator can act on: too little disk, too thin a
    cloud, a step that is not in this mode's chain. The message is a full
    sentence naming the remedy, and it is printed as-is."""


class StepFailed(RuntimeError):
    """A step ran and did not succeed — a non-zero exit from COLMAP, or a
    post-condition that says the step did not do what it claimed."""


class ReconCancelled(RuntimeError):
    """The operator pressed Abort. Whatever finished stays recorded, so the
    next run resumes rather than starts over."""


# ── the backend seam ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class GpuSpec:
    """The card one step is to run on, by identity rather than by slot.

    `--gpus all` plus `CUDA_DEVICE_ORDER=PCI_BUS_ID` does not make a chosen card
    device 0 — slot order decides — so the runner passes a UUID and the
    container sees exactly one device, which is then its `gpu_index 0`.
    """

    #: `GPU-<uuid>`, exactly as `nvidia-smi -L` prints it.
    uuid: str
    #: The card's name, for the log and for `session.json`. Not passed to Docker.
    name: str = ""
    #: The named volume the JIT cache is kept in, so a successful PTX compile is
    #: paid for once across runs rather than once per container.
    jit_volume: str = ""
    cache_bytes: int = JIT_CACHE_BYTES

    def volume(self) -> str:
        return self.jit_volume or os.environ.get(JIT_VOLUME_ENV,
                                                 DEFAULT_JIT_VOLUME)


def container_path(session_dir: str | Path, host_path: str | Path) -> str:
    """A host path under the session as the container sees it: `/data/<posix
    relative>`.

    One helper, because path mapping is where a Windows host and a Linux
    container disagree in the most expensive way possible — an hour of compute
    writing into a directory nobody looks in. Everything the chain names lives
    under the session directory, which is the only thing mounted, so a path
    outside it is a bug and is refused here rather than mapped to something
    plausible.
    """
    root = Path(session_dir).resolve()
    target = Path(host_path)
    if not target.is_absolute():
        target = root / target
    try:
        rel = target.resolve().relative_to(root)
    except ValueError:
        raise ValueError(
            f"{target} is outside the session directory {root} — only the "
            "session is mounted into the container, so nothing else can be "
            "named in an argv") from None
    parts = [p for p in rel.parts if p not in (".", "")]
    return CONTAINER_ROOT + ("".join(f"/{p}" for p in parts))


class Backend(Protocol):
    """How a COLMAP invocation is run. Three implementations: Docker (the one
    that ships), a local executable (only when `ORBITER_COLMAP` names one), and
    a fake that records argv for the tests."""

    def available(self) -> str | None:
        """`None` when this backend can run, otherwise a full sentence saying
        why it cannot and what to do about it.

        The polarity is that way round because the caller's job with the answer
        is to print it: a truthy result is a problem, and `--check` and `run`
        both refuse with the sentence verbatim.
        """
        ...

    def run(self, argv: list[str], *, on_line: Callable[[str], None],
            gpu: GpuSpec | None = None,
            cancel: threading.Event | None = None) -> int:
        """Run one COLMAP command and return its exit code.

        `argv` starts with `colmap` and carries container paths. The command
        line itself is emitted through `on_line` before the process starts, so
        the log carries every command as well as every line of its output.
        `gpu` is passed by `patch_match_stereo` and by nothing else.
        """
        ...

    def close(self) -> None:
        """Release whatever the backend holds. Nothing, for the two that run one
        process per step — it exists so a future backend that keeps something
        warm can be dropped in without changing every caller."""
        ...


def _spawn(cmd: Sequence[str], on_line: Callable[[str], None],
           cancel: threading.Event | None) -> int:
    """Run a command, stream its merged output line by line, return its code.

    stdout and stderr are merged because COLMAP writes its progress to one and
    its glog lines to the other, and two streams read separately interleave by
    luck. `text=True` with an explicit encoding keeps a stray byte in a file
    name from killing an hour-long run.
    """
    kwargs: dict[str, Any] = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        # The app is a GUI; a console flashing up per step is not acceptable.
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace", **kwargs)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            on_line(line.rstrip("\r\n"))
            if cancel is not None and cancel.is_set():
                proc.terminate()
                break
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()


def _shown(cmd: Iterable[str]) -> str:
    """A command line as the log carries it — quoted only where a space would
    otherwise make two arguments look like one."""
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


class DockerBackend:
    """One `docker run --rm` per step, with the session bind-mounted at
    `/data`.

    The mount is the session directory and nothing else, which is why every path
    in every argv goes through `container_path`: what the container cannot see,
    it cannot be told about. The GPU flags and the JIT volume are attached only
    when a `GpuSpec` is passed, which only `patch_match_stereo` does — a chain
    that needs no GPU must not be able to die on a broken nvidia runtime.
    """

    def __init__(self, session_dir: str | Path, image: str | None = None,
                 docker: str = "docker") -> None:
        self.session_dir = Path(session_dir).resolve()
        self.image = image or os.environ.get(COLMAP_IMAGE_ENV,
                                             DEFAULT_COLMAP_IMAGE)
        self.docker = docker

    @property
    def mount(self) -> str:
        """The `-v` source. Forward slashes on every host: Docker Desktop takes
        `D:/sessions/...` and does not take `D:\\sessions\\...`."""
        return self.session_dir.as_posix()

    def command(self, argv: Sequence[str],
                gpu: GpuSpec | None = None) -> list[str]:
        """The full command line, so the log and the tests see the same thing
        the shell would."""
        cmd = [self.docker, "run", "--rm", "-v", f"{self.mount}:{CONTAINER_ROOT}"]
        if gpu is not None:
            cmd += ["--gpus", f"device={gpu.uuid}",
                    "-v", f"{gpu.volume()}:/jitcache",
                    "-e", "CUDA_CACHE_PATH=/jitcache",
                    "-e", f"CUDA_CACHE_MAXSIZE={int(gpu.cache_bytes)}"]
        return cmd + [self.image, *argv]

    def available(self) -> str | None:
        if shutil.which(self.docker) is None:
            return (f"{self.docker} is not on PATH — install Docker Desktop, or "
                    f"point {COLMAP_LOCAL_ENV} at a local colmap executable.")
        return None

    def run(self, argv: list[str], *, on_line: Callable[[str], None],
            gpu: GpuSpec | None = None,
            cancel: threading.Event | None = None) -> int:
        cmd = self.command(argv, gpu)
        on_line("$ " + _shown(cmd))
        return _spawn(cmd, on_line, cancel)

    def close(self) -> None:
        """Nothing is held: every step is its own `--rm` container."""


class LocalBackend:
    """A COLMAP executable on this machine, used only when `ORBITER_COLMAP`
    names one.

    No PATH search on purpose. A `colmap` picked up by accident is of unknown
    version and unknown CUDA architectures, and every flag in this chain was
    read off 4.2.0; a clear "set ORBITER_COLMAP" beats a run that fails on an
    option that moved.

    There is no bind mount to map to: a local binary reads the host filesystem
    directly, so the `/data/...` prefix every step writes is simply unwound back
    to the session directory on the way out. Every step therefore builds one
    argv and neither knows nor cares which backend runs it.
    """

    def __init__(self, session_dir: str | Path,
                 executable: str | None = None) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.executable = executable or os.environ.get(COLMAP_LOCAL_ENV, "")

    def command(self, argv: Sequence[str]) -> list[str]:
        out = [self.executable]
        for part in list(argv)[1:]:
            if part.startswith(CONTAINER_ROOT):
                rel = part[len(CONTAINER_ROOT):].lstrip("/")
                out.append(str(self.session_dir / rel) if rel
                           else str(self.session_dir))
            else:
                out.append(part)
        return out

    def available(self) -> str | None:
        if not self.executable:
            return (f"{COLMAP_LOCAL_ENV} is not set — it must name a colmap "
                    "executable for the local backend to be used at all.")
        if not Path(self.executable).is_file():
            return (f"{COLMAP_LOCAL_ENV} names {self.executable}, which is not "
                    "a file — point it at the colmap executable itself.")
        return None

    def run(self, argv: list[str], *, on_line: Callable[[str], None],
            gpu: GpuSpec | None = None,
            cancel: threading.Event | None = None) -> int:
        cmd = self.command(argv)
        on_line("$ " + _shown(cmd))
        return _spawn(cmd, on_line, cancel)

    def close(self) -> None:
        """Nothing is held: every step is its own process."""


class FakeBackend:
    """Records argv, replays canned output and exit codes, and can act.

    The whole runner is tested through this. `effects` is what makes the
    post-conditions testable: a real `image_undistorter` leaves a workspace
    behind, and a test that wants the post-condition to fire needs one that
    holds the wrong images. Each effect is called with the step's argv after its
    exit code is chosen, so a scripted failure leaves nothing behind, exactly
    like the real thing.
    """

    def __init__(self, exits: dict[str, int] | None = None,
                 output: dict[str, list[str]] | None = None,
                 effects: dict[str, Callable[[list[str]], None]] | None = None,
                 ) -> None:
        #: Every argv handed over, in order — `argv[1]` is the COLMAP tool.
        self.calls: list[list[str]] = []
        #: The `GpuSpec` (or None) each of those calls carried, same order.
        self.gpus: list[GpuSpec | None] = []
        self.exits = dict(exits or {})
        self.output = dict(output or {})
        self.effects = dict(effects or {})

    @property
    def tools(self) -> list[str]:
        """The COLMAP tool of each recorded call, in order."""
        return [argv[1] for argv in self.calls if len(argv) > 1]

    def argv_for(self, tool: str) -> list[str]:
        """The single recorded call to `tool`, or a clear failure when it ran a
        number of times other than once."""
        found = [argv for argv in self.calls if len(argv) > 1 and argv[1] == tool]
        if len(found) != 1:
            raise AssertionError(f"{tool} ran {len(found)} times, not once")
        return found[0]

    def available(self) -> str | None:
        return None

    def run(self, argv: list[str], *, on_line: Callable[[str], None],
            gpu: GpuSpec | None = None,
            cancel: threading.Event | None = None) -> int:
        argv = list(argv)
        self.calls.append(argv)
        self.gpus.append(gpu)
        on_line("$ " + _shown(["colmap-fake", *argv[1:]]))
        tool = argv[1] if len(argv) > 1 else ""
        for line in self.output.get(tool, ()):
            on_line(line)
        code = int(self.exits.get(tool, 0))
        effect = self.effects.get(tool)
        if effect is not None and code == 0:
            effect(argv)
        return code

    def close(self) -> None:
        """Nothing is held."""


# ── the log ──────────────────────────────────────────────────────────────


class RunLog:
    """`recon.log`, and the panel's line stream, written to as one.

    Appended rather than truncated: a resumed run belongs in the same file as
    the run it is resuming, and the sequence of attempts is most of what makes a
    failure readable afterwards. The file is reopened per line rather than held
    open, which costs microseconds against an hour and means an operator
    watching the run with `tail` sees it as it happens rather than in buffered
    lumps — and sees the last line before a crash rather than losing it.
    """

    def __init__(self, session_dir: str | Path,
                 on_line: Callable[[str], None] | None = None) -> None:
        self.path = Path(session_dir) / LOG_NAME
        self._on_line = on_line
        self._lock = threading.Lock()

    def line(self, text: str) -> None:
        """One line, timestamped, to the file and to the caller's sink."""
        stamped = f"{time.strftime('%H:%M:%S')} {text}"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(stamped + "\n")
        if self._on_line is not None:
            try:
                self._on_line(text)
            except Exception:
                # A panel callback that raises must not take an hour-long
                # reconstruction with it; the file already has the line.
                log.exception("recon log sink raised")


# ── disk ─────────────────────────────────────────────────────────────────


def free_space_bytes(path: str | Path) -> int:
    """Free bytes on the filesystem holding `path`. A thin wrapper so `--check`
    and the runner ask the same question of the same place."""
    return int(shutil.disk_usage(str(path)).free)


def undistorted_wh_for(source_wh: tuple[int, int],
                       max_image_size: int) -> tuple[int, int]:
    """The size `image_undistorter` writes at, given the raw sensor size and
    `--max_image_size`.

    COLMAP scales the longer side down to the limit and keeps the aspect; a
    limit of -1, or one above the source, means the image is copied at its own
    size. 1920x1080 at 1600 is 1600x900, which is the pair this rig's dense
    estimate is computed from.
    """
    w, h = int(source_wh[0]), int(source_wh[1])
    longest = max(w, h)
    if max_image_size is None or max_image_size <= 0 or longest <= max_image_size:
        return w, h
    scale = float(max_image_size) / float(longest)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def disk_estimate_bytes(mode: str, n_selected: int,
                        undistorted_wh: tuple[int, int]) -> int:
    """What a run of this shape needs free, in bytes.

    Texture-only is a flat gigabyte: it writes no depth maps, so there is
    nothing to scale. Dense is dominated by depth and normal maps, and both
    scale with the selection and the undistorted size, so it is computed —

        n_selected x (W x H x 4 B) x 4 (depth + normal) x 2 (photometric +
        geometric) x 1.3 + 1 GiB

    — rather than guessed. A flat figure is simultaneously far too much for a
    40-photograph capture and not nearly enough for a 150-photograph one, which
    is how a run fills a disk while its own check says it is fine.

    The third argument is the **undistorted** size, not the `--max_image_size`
    flag: the flag alone does not give a width and a height, and it is the pair
    that `session.json` records under `estimate_inputs.undistorted_wh`. Use
    `undistorted_wh_for` to get it from the flag and the sensor size.
    """
    if mode not in MODES:
        raise ValueError(f"no such mode: {mode!r} — one of {MODES}")
    if mode == "texture-only":
        return TEXTURE_ONLY_DISK_BYTES
    w, h = int(undistorted_wh[0]), int(undistorted_wh[1])
    per_image = w * h * BYTES_PER_SAMPLE * MAPS_PER_IMAGE * PASSES_PER_IMAGE
    return ((int(n_selected) * per_image * DISK_SLACK_TENTHS) // 10
            + DISK_HEADROOM_BYTES)


def _gb(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.1f} GB"


# ── the version ──────────────────────────────────────────────────────────


def colmap_version(backend: Backend) -> str | None:
    """Which COLMAP the backend runs, e.g. `"4.2.0"`, or None when its output
    does not say.

    Every flag in this chain was read off 4.2.0, so the version is recorded in
    `recon-state.json` and in `session.json` — a default that moved between
    versions is otherwise invisible until the result is wrong. The image prints
    a CUDA banner from its base before COLMAP says anything, so the line is
    found by pattern and not by position.
    """
    found: list[str] = []

    def sink(line: str) -> None:
        match = _VERSION_RE.search(line)
        if match and not found:
            found.append(match.group(1))

    backend.run(["colmap", "-h"], on_line=sink)
    return found[0] if found else None


# ── the state file ───────────────────────────────────────────────────────


@dataclass
class State:
    """`recon-state.json` — what has finished, and under what conditions.

    `mode` is the only field that drives invalidation. `colmap_version` and
    `max_image_size` are recorded and read back into the log for the same
    reason a run's parameters go into `session.json`: so a result that changed
    can be explained by something other than luck.
    """

    mode: str = ""
    colmap_version: str = ""
    max_image_size: int = 0
    #: `{step name: {"done_utc": ..., "seconds": ...}}`.
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def read(cls, session_dir: str | Path) -> State:
        path = Path(session_dir) / STATE_NAME
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mode=str(raw.get("mode", "")),
            colmap_version=str(raw.get("colmap_version", "")),
            max_image_size=int(raw.get("max_image_size", 0) or 0),
            steps={str(k): dict(v) for k, v in (raw.get("steps") or {}).items()},
        )

    def write(self, session_dir: str | Path) -> None:
        path = Path(session_dir) / STATE_NAME
        path.write_text(json.dumps({
            "mode": self.mode,
            "colmap_version": self.colmap_version,
            "max_image_size": self.max_image_size,
            "steps": self.steps,
        }, indent=2), encoding="utf-8")

    def done(self, step: str, seconds: float) -> None:
        self.steps[step] = {"done_utc": _utc_now(), "seconds": round(seconds, 3)}

    def finished(self, step: str) -> bool:
        return step in self.steps


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── parameters and result ────────────────────────────────────────────────


@dataclass(frozen=True)
class ReconParams:
    """Every number the chain passes to COLMAP or decides by, in one place, so
    a run is reproducible from `session.json` and a changed number is visible.
    """

    #: How the photographs are chosen, seeded and bucketed.
    select: SelectParams = SelectParams()
    #: The dense `image_undistorter`'s limit — the knob the GPU fallback lowers
    #: to 1000, and the one the disk estimate is computed at.
    max_image_size: int = 1600
    #: The texture workspace's limit. Unlimited on purpose: the atlas is the
    #: deliverable, and there is exactly one of it.
    texture_max_image_size: int = -1
    #: COLMAP's default is 10, which cut about 85 % of the object in the
    #: sibling repo.
    poisson_trim: int = 7
    #: `mesh_texturer`'s atlas knobs. Padding is 4 rather than the default 2
    #: because a q95 JPEG's ringing crosses a two-pixel gutter.
    texture_scale_factor: int = 1
    atlas_patch_padding: int = 4
    inpaint_radius: int = 5
    min_visible_vertices: int = 3
    apply_color_correction: int = 1
    #: The refusal floors, exposed so a test can drive them without a session
    #: of the size the real ones need.
    min_confident_points: int = MIN_CONFIDENT_POINTS
    min_selected_photos: int = MIN_SELECTED_PHOTOS
    #: A pass scoring below this fraction of the laser pass on silhouette
    #: agreement is reported: the object may have moved when the switch was
    #: flipped. A warning, never a refusal.
    pass_agree_frac: float = 0.6
    #: Photographs sampled per pass for that score.
    pass_samples: int = 5


@dataclass
class RunResult:
    """What the run did, for the CLI to print and the panel to show."""

    session_dir: Path
    mode: str
    #: The steps that actually ran, in order.
    ran: tuple[str, ...] = ()
    #: The steps skipped because the state file said they had finished.
    skipped: tuple[str, ...] = ()
    colmap_version: str = ""
    #: Everything the run wants an operator to read. A run with warnings
    #: finished; a run with `degraded` entries finished as something less than
    #: it set out to be, and that is not a pass.
    warnings: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    texture_source: str = ""
    n_selected: int = 0


# ── the run's shared state ───────────────────────────────────────────────


@dataclass
class Run:
    """One invocation, as the steps see it.

    Steps read the session, the backend and the log from here and write back
    what later steps and `session.json` need. It is deliberately a plain object
    rather than a closure: `--only mesh_texturer` on a resumed session must be
    able to run a step whose predecessors never ran in this process, so every
    step states what it needs and finds it on disk when it is not here.
    """

    session_dir: Path
    mode: str
    backend: Backend
    log: RunLog
    params: ReconParams
    state: State
    force: bool = False
    cancel: threading.Event | None = None

    #: Filled by `write_sparse`; absent on a run that resumed past it.
    session: SessionInfo | None = None
    photos: list[PhotoMeta] = field(default_factory=list)
    selection: Selection | None = None
    texture_names: list[str] = field(default_factory=list)
    texture_source: str = ""
    warnings: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    #: `session.json`'s `reconstruct` block as it is being built.
    report: dict[str, Any] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        """A session-relative path on the host."""
        return self.session_dir.joinpath(*parts)

    def inside(self, relative: str) -> str:
        """A session-relative path as the container sees it."""
        return container_path(self.session_dir, self.session_dir / relative)

    def colmap(self, argv: list[str], *, gpu: GpuSpec | None = None) -> int:
        """Run one COLMAP command through the backend, logging it and its
        output."""
        return self.backend.run(argv, on_line=self.log.line, gpu=gpu,
                                cancel=self.cancel)

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        self.log.line("warning: " + text)


# ── the step registry ────────────────────────────────────────────────────

#: Name → the function that runs it. Milestone 2's steps are named in
#: `CANONICAL_STEPS` and are absent from here until D1b registers them, which
#: is what makes `--mode dense` a named refusal rather than a stub that fails
#: somewhere harder to read.
STEPS: dict[str, Callable[[Run], None]] = {}


def _step(name: str) -> Callable[[Callable[[Run], None]], Callable[[Run], None]]:
    if name not in CANONICAL_STEPS:
        raise ValueError(f"{name!r} is not one of CANONICAL_STEPS")

    def register(fn: Callable[[Run], None]) -> Callable[[Run], None]:
        STEPS[name] = fn
        return fn
    return register


# ── (1) write_sparse ─────────────────────────────────────────────────────


@_step("write_sparse")
def _write_sparse(run: Run) -> None:
    """The whole text model, the photographs COLMAP reads, and the decisions
    every later step inherits.

    It is the only step that reads the session and the only one that writes
    `colmap/images/`, verbatim and in both modes. Everything that can refuse a
    run refuses here, before a container has started: a cloud too thin to mesh,
    a selection too small to texture, a disk that cannot hold the result.
    """
    session, photos = load_session(run.session_dir)
    run.session, run.photos = session, photos

    laser = run.path(LASER_PLY)
    if not laser.exists():
        raise ReconRefused(
            f"write_sparse: {laser} does not exist — the confident cloud is "
            "exported there when Reconstruct is pressed; export the cloud "
            "before running the CLI against this session.")
    xyz, rgb, normals = read_ply_full(str(laser))
    if len(xyz) < run.params.min_confident_points:
        raise ReconRefused(
            f"write_sparse: {laser.name} holds {len(xyz)} confident points, "
            f"below the {run.params.min_confident_points} a reconstruction "
            "needs — scan more of the object and export again.")

    # Poisson is undefined without normals, and PCA gives a line rather than a
    # direction. Orienting needs camera centres and the selection needs
    # normals, so the PCA is done once and ORIENTED twice: first toward every
    # camera the session recorded, which is what the front-facing test in
    # `select` reads, and then toward the centres that were actually selected,
    # which is what goes into `laser.ply`. Only the second is the story's
    # promise; the first exists because the selection has not been made yet.
    # An orientation is a KD query against a handful of camera centres.
    have_normals = normals is not None
    raw = np.asarray(normals, float) if have_normals else normals_pca(xyz)
    if have_normals:
        normals_all = raw
    else:
        normals_all = orient_normals(
            raw, xyz, np.array([p.centre_mm for p in photos], float)
            if photos else np.zeros((0, 3)))

    selection = select(session, photos, xyz, normals_all, run.params.select)
    run.selection = selection
    n_selected = len(selection.accepted)
    if n_selected < run.params.min_selected_photos:
        raise ReconRefused(
            f"write_sparse: {n_selected} of {len(photos)} photographs were "
            f"selected, below the {run.params.min_selected_photos} a "
            "reconstruction needs — photograph the object from more places, "
            "or loosen SelectParams.")

    if have_normals:
        cloud_normals = normals_all
    else:
        cloud_normals = orient_normals(
            raw, xyz,
            np.array([v.photo.centre_mm for v in selection.accepted], float))
        write_ply(str(laser), xyz, rgb, cloud_normals)
        run.log.line(f"write_sparse: {laser.name} carried no normals — wrote "
                     f"PCA normals for {len(xyz)} points, oriented toward the "
                     "nearest selected camera")

    source_wh = _source_wh(session, selection)
    wh = undistorted_wh_for(source_wh, run.params.max_image_size)
    estimate = disk_estimate_bytes(run.mode, n_selected, wh)
    free = _check_disk(run, estimate,
                       f"{run.mode} needs about {_gb(estimate)} for "
                       f"{n_selected} images at {wh[0]}x{wh[1]}"
                       + (" (depth and normal maps dominate)"
                          if run.mode == "dense" else ""))

    seeds, seed_rgb, seed_normals = seed_points(
        xyz, rgb, cloud_normals, run.params.select.seed_cap)
    points3d, points2d = tracks(selection, seeds, seed_normals, seed_rgb)
    images = [ImageRecord(image_id=v.image_id, name=v.photo.name,
                          camera_id=v.photo.colmap_camera_id,
                          R=v.photo.R, t=v.photo.t_mm,
                          points2d=points2d.get(v.image_id, []))
              for v in selection.accepted]
    eyes = session.cameras(selection)

    sparse = run.path("colmap", "sparse")
    colmapio.write_cameras(sparse / "cameras.txt", eyes)
    colmapio.write_images(sparse / "images.txt", images)
    colmapio.write_points3d(sparse / "points3D.txt", points3d)
    colmapio.write_rigs(sparse / "rigs.txt", eyes)
    colmapio.write_frames(sparse / "frames.txt", images)
    run.log.line(f"write_sparse: {len(eyes)} cameras, {len(images)} images, "
                 f"{len(points3d)} seed points of {len(seeds)} sampled")

    # Verbatim copies, under exactly the NAME `images.txt` carries. That single
    # rule is what makes masks, image lists and the cfg rewrite match by string
    # with no normalisation anywhere.
    for view in selection.accepted:
        shutil.copyfile(run.path(view.photo.file),
                        run.path("colmap", "images", view.photo.name))

    names, source, warning = texture_set(session, selection, run.mode)
    write_image_list(run.path("colmap", "texture_images.txt"), names)
    run.texture_names, run.texture_source = list(names), source
    if warning:
        run.warn(warning)
        run.degraded.append(f"texture_source={source}")
    run.log.line(f"write_sparse: texture set is {len(names)} images, "
                 f"texture_source={source}")

    pass_scores = _pass_agreement(run, session, photos, xyz)

    covered = buckets(session, selection)
    clean_ids = {v.image_id for v in selection.clean}
    clean_covered = {k for k, ids in covered.items() if clean_ids.intersection(ids)}
    run.report.update({
        "mode": run.mode,
        "colmap_version": run.state.colmap_version,
        "started_utc": run.report.get("started_utc", _utc_now()),
        "disk": {"free_bytes_at_start": free, "estimate_bytes": estimate,
                 "estimate_inputs": {"n_selected": n_selected,
                                     "undistorted_wh": list(wh)},
                 "forced": run.force},
        "selected": selection.counts,
        "buckets": {"grid": list(run.params.select.buckets),
                    "covered_by_selection": len(covered),
                    "covered_by_clean": len(clean_covered),
                    "clean_coverage": (len(clean_covered) / len(covered)
                                       if covered else 0.0),
                    "uncovered": sorted(list(k) for k in
                                        set(covered) - clean_covered)},
        "passes": pass_scores,
        "texture_source": source,
    })


def _source_wh(session: SessionInfo, selection: Selection) -> tuple[int, int]:
    """The raw sensor size the selected photographs were taken at.

    The largest of them, because this feeds an estimate and a rig whose eyes
    differ should be estimated by the more expensive one. A session with mixed
    sizes on ONE side is refused a few lines later by `write_cameras` — a camera
    matrix is only valid at the resolution it was solved at — so this is a
    number for the disk check, not a resolution guard of its own.
    """
    sizes = {v.photo.wh for v in selection.accepted}
    if not sizes:
        eye = session.eye("left") or session.eye("right")
        return (int(eye.wh[0]), int(eye.wh[1])) if eye else (0, 0)
    return max(sizes, key=lambda wh: wh[0] * wh[1])


def _check_disk(run: Run, estimate: int, need: str) -> int:
    """Refuse a run that cannot fit, unless `--force` says otherwise. Returns
    the free bytes so the report can record what it saw.

    `need` is the clause naming what wants the space, because the two call
    sites want different ones: before step 1 nobody yet knows how many
    photographs will be selected, so all that can be checked is the floor every
    run needs; at `write_sparse` the dense estimate is finally computable and
    the sentence names the shape it was computed from.
    """
    free = free_space_bytes(run.session_dir)
    if free >= estimate:
        run.log.line(f"disk: {_gb(free)} free, {need}")
        return free
    message = (f"refusing to start: {_gb(free)} free on {run.session_dir}, "
               f"{need}. Free space, move the root with ORBITER_SESSIONS_DIR, "
               "or pass --force.")
    if not run.force:
        raise ReconRefused(message)
    run.warn(message + " Forced.")
    return free


# ── the per-pass agreement warning (§2.7) ────────────────────────────────


def _pass_agreement(run: Run, session: SessionInfo, photos: list[PhotoMeta],
                    cloud_xyz: np.ndarray) -> list[dict[str, Any]]:
    """Score each pass on how well the laser cloud's silhouette lands on an
    edge in its photographs, and warn about a pass that scores badly.

    Flipping the physical laser switch can nudge the object relative to the
    board, and view selection *prefers* the clean photographs — so that
    corruption would land precisely where the atlas is most confident. The
    check is deliberately cheap: five photographs per pass, the cloud projected
    into each, the mean image gradient along the projected boundary. A pass
    below `pass_agree_frac` of the laser pass's score is reported and its
    photographs are still used. It is a warning, not a gate: a low score can
    also mean a dark object or a soft edge, and refusing on it would reject
    exactly the captures dense exists to rescue.
    """
    by_pass: dict[int, list[PhotoMeta]] = {}
    for photo in photos:
        by_pass.setdefault(photo.pass_id, []).append(photo)

    out: list[dict[str, Any]] = []
    for pass_id in sorted(by_pass):
        group = by_pass[pass_id]
        scores = []
        for photo in group[:: max(1, len(group) // run.params.pass_samples)]:
            if len(scores) >= run.params.pass_samples:
                break
            value = _silhouette_score(session, photo, cloud_xyz,
                                      run.path(photo.file))
            if value is not None:
                scores.append(value)
        out.append({"pass_id": pass_id,
                    "laser_on": bool(group[0].laser_on),
                    "photos": len(group),
                    "silhouette_score": (float(np.mean(scores)) if scores
                                         else None)})

    laser = next((p for p in out if p["laser_on"]
                  and p["silhouette_score"]), None)
    if laser is None:
        return out
    for entry in out:
        score = entry["silhouette_score"]
        if entry is laser or not score:
            continue
        ratio = score / laser["silhouette_score"]
        entry["ratio_to_laser_pass"] = round(ratio, 3)
        if ratio < run.params.pass_agree_frac:
            run.warn(
                f"pass {entry['pass_id']} "
                f"({'laser on' if entry['laser_on'] else 'laser off'}, "
                f"{entry['photos']} photos) scores {ratio:.2f} of the laser "
                "pass on silhouette agreement — the object may have moved "
                "between passes. Its photographs are still used.")
    return out


def _silhouette_score(session: SessionInfo, photo: PhotoMeta,
                      cloud_xyz: np.ndarray, image_path: Path) -> float | None:
    """The mean image gradient along the cloud's projected silhouette boundary,
    or None when the photograph cannot be read or projects to nothing.

    Cheap on purpose: the image is read grey, the cloud is projected through
    this photograph's own eye and pose into a coarse grid, and the boundary is
    that grid's mask minus its erosion. A real edge under the boundary scores
    high; a boundary sitting on flat background scores near zero.
    """
    import cv2

    eye = session.eye(photo.side)
    if eye is None or not image_path.exists():
        return None
    grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grey is None or grey.size == 0:
        return None

    xyz = np.asarray(cloud_xyz, float).reshape(-1, 3)
    cam = xyz @ np.asarray(photo.R, float).T + np.asarray(photo.t_mm, float)
    front = cam[:, 2] > 1e-6
    if not front.any():
        return None
    cam = cam[front]
    u = eye.fx * cam[:, 0] / cam[:, 2] + eye.cx
    v = eye.fy * cam[:, 1] / cam[:, 2] + eye.cy

    h, w = grey.shape[:2]
    step = max(1, min(h, w) // 128)          # a coarse grid; the boundary is thick
    gh, gw = h // step, w // step
    if gh < 3 or gw < 3:
        return None
    gu = np.rint(u / step).astype(int)
    gv = np.rint(v / step).astype(int)
    keep = (gu >= 0) & (gu < gw) & (gv >= 0) & (gv < gh)
    if not keep.any():
        return None
    mask = np.zeros((gh, gw), np.uint8)
    mask[gv[keep], gu[keep]] = 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    border = cv2.dilate(mask, kernel) - mask
    if not border.any():
        return None

    small = cv2.resize(grey, (gw, gh), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.hypot(gx, gy)[border > 0].mean())


# ── (2) validate_sparse ──────────────────────────────────────────────────


@_step("validate_sparse")
def _validate_sparse(run: Run) -> None:
    """Ask COLMAP whether it accepts the model we just wrote.

    Five seconds against an hour of compute, and the only thing in the chain
    that can tell us our `rigs.txt`/`frames.txt` topology is one COLMAP reads.
    The PLY is a smoke test rather than an artefact, so it is removed on
    success — leaving it would put a point cloud nobody asked for beside the two
    that were.
    """
    out = "colmap/_validate.ply"
    code = run.colmap(["colmap", "model_converter",
                       "--input_path", run.inside("colmap/sparse"),
                       "--output_path", run.inside(out),
                       "--output_type", "PLY"])
    if code != 0:
        raise StepFailed(
            f"validate_sparse: COLMAP refused the model in colmap/sparse "
            f"(model_converter exited {code}) — read the lines above in "
            f"{LOG_NAME}; nothing downstream can be trusted until it loads.")
    run.path(out).unlink(missing_ok=True)


# ── (4) texture_workspace ────────────────────────────────────────────────


@_step("texture_workspace")
def _texture_workspace(run: Run) -> None:
    """Undistort the texture set into `colmap/texture/`, then check that only
    the texture set is in it.

    `mesh_texturer` has no image or image-list flag, so which images the
    workspace holds is the only lever on the atlas — which makes
    `--image_list_path` the single flag the whole texture story rests on. A
    workspace that quietly contained every image would texture from laser
    stripes while `session.json` reported the clean set, and no later step could
    notice. So the step asserts what it just did, in the two ways that can
    disagree: how many files landed in the workspace, and which names its own
    model knows them by.

    `--image_path` is the only mode-dependent argument in the M1 chain, and it
    is why a `--mode` change invalidates this step and everything after it.
    """
    code = run.colmap([
        "colmap", "image_undistorter",
        "--image_path", run.inside(MODE_IMAGE_PATH[run.mode]),
        "--input_path", run.inside("colmap/sparse"),
        "--output_path", run.inside("colmap/texture"),
        "--output_type", "COLMAP",
        "--max_image_size", str(run.params.texture_max_image_size),
        "--image_list_path", run.inside("colmap/texture_images.txt"),
    ])
    if code != 0:
        raise StepFailed(
            f"texture_workspace: image_undistorter exited {code} — the texture "
            f"workspace was not built; read the lines above in {LOG_NAME}.")
    _check_texture_workspace(run)


def _check_texture_workspace(run: Run) -> None:
    """The post-condition of §2.3, on the host, immediately after the container
    exits."""
    listed = [line for line in run.path("colmap", "texture_images.txt")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    images = run.path("colmap", "texture", "images")
    on_disk = [p for p in images.rglob("*") if p.is_file()] if images.exists() else []
    if len(on_disk) != len(listed):
        raise StepFailed(
            f"texture_workspace: the workspace holds {len(on_disk)} images but "
            f"the list named {len(listed)} — --image_list_path did not restrict "
            "the model; see PHOTOGRAMMETRY.md for the texture_sparse fallback.")

    model = run.path("colmap", "texture", "sparse", "images.bin")
    if not model.exists():
        raise StepFailed(
            f"texture_workspace: {model} is missing — image_undistorter wrote "
            "no sparse model, so mesh_texturer has nothing to read.")
    known = set(read_images_bin(model))
    if known != set(listed):
        extra = sorted(known - set(listed))[:5]
        missing = sorted(set(listed) - known)[:5]
        raise StepFailed(
            f"texture_workspace: the workspace's model names {len(known)} "
            f"images, not the {len(listed)} that were listed"
            + (f" (unexpected: {', '.join(extra)})" if extra else "")
            + (f" (absent: {', '.join(missing)})" if missing else "")
            + " — --image_list_path did not restrict the model; see "
              "PHOTOGRAMMETRY.md for the texture_sparse fallback.")
    run.log.line(f"texture_workspace: {len(listed)} images, and the workspace "
                 "holds exactly those")


# ── (11) poisson_mesher ──────────────────────────────────────────────────


@_step("poisson_mesher")
def _poisson_mesher(run: Run) -> None:
    """Mesh the cloud this mode meshes, into `mesh/` and not into
    `colmap/dense/`.

    The dense tree is Milestone 2's alone and the GPU fallback deletes it
    wholesale, so a mesh written there would be either impossible in
    texture-only or destructible in dense. `trim` is 7 rather than COLMAP's 10,
    which cut about 85 % of the object in the sibling repo.
    """
    code = run.colmap([
        "colmap", "poisson_mesher",
        "--input_path", run.inside(MODE_POISSON_INPUT[run.mode]),
        "--output_path", run.inside("mesh/meshed-poisson.ply"),
        "--PoissonMeshing.trim", str(run.params.poisson_trim),
    ])
    if code != 0:
        raise StepFailed(
            f"poisson_mesher: exited {code} — no mesh was written to "
            f"mesh/meshed-poisson.ply; read the lines above in {LOG_NAME}.")


# ── (12) mesh_texturer ───────────────────────────────────────────────────


@_step("mesh_texturer")
def _mesh_texturer(run: Run) -> None:
    """Paint the mesh from the texture workspace — always that workspace, in
    both modes.

    There is no second workspace path and no image flag to reach for:
    `mesh_texturer`'s whole option set carries neither, which is why the
    restricted workspace exists at all. `--output_type` is left at its default
    BIN, so the GLB step reads a binary PLY.
    """
    code = run.colmap([
        "colmap", "mesh_texturer",
        "--workspace_path", run.inside("colmap/texture"),
        "--input_path", run.inside("mesh/meshed-poisson.ply"),
        "--output_path", run.inside("mesh"),
        "--MeshTextureMapping.texture_scale_factor",
        str(run.params.texture_scale_factor),
        "--MeshTextureMapping.atlas_patch_padding",
        str(run.params.atlas_patch_padding),
        "--MeshTextureMapping.inpaint_radius", str(run.params.inpaint_radius),
        "--MeshTextureMapping.min_visible_vertices",
        str(run.params.min_visible_vertices),
        "--MeshTextureMapping.apply_color_correction",
        str(run.params.apply_color_correction),
    ])
    if code != 0:
        raise StepFailed(
            f"mesh_texturer: exited {code} — mesh/mesh.ply and mesh/texture.png "
            f"were not written; read the lines above in {LOG_NAME}.")


# ── (13) glb ─────────────────────────────────────────────────────────────


@_step("glb")
def _glb(run: Run) -> None:
    """Pack the textured PLY and its atlas into one `mesh/model.glb`.

    COLMAP writes a `binary_little_endian` PLY with **per-face** `texcoord` and
    a sidecar PNG named in a `comment TextureFile` line. Almost nothing opens
    that cleanly — Blender's PLY importer drops per-face UVs and the mesh loads
    untextured — while a GLB is one self-contained file every viewer imports
    with the texture already on. trimesh understands the per-face layout and
    welds it into per-vertex UVs, and its GLB writer embeds the PNG without
    re-encoding it.

    **Best effort.** A failure here leaves `mesh.ply` and `texture.png` on disk,
    which are the deliverable an external tool can still open, and logs a
    warning. Losing an hour of reconstruction to a packaging library is not a
    trade worth making.
    """
    ply = run.path("mesh", "mesh.ply")
    out = run.path("mesh", "model.glb")
    try:
        texture = _atlas_for(ply)
        _pack_glb(ply, texture, out)
    except Exception as exc:                      # noqa: BLE001 — best effort
        run.warn(f"glb: {ply.name} was not packed into model.glb ({exc}) — "
                 "mesh/mesh.ply and mesh/texture.png are on disk and open in "
                 "MeshLab or Blender with the atlas attached by hand.")
        return
    run.log.line(f"glb: wrote {out.name}, "
                 f"{out.stat().st_size / 1e6:.1f} MB")


def _atlas_for(ply: Path) -> Path:
    """The atlas a textured PLY names, or the sibling `texture.png`.

    The header's `comment TextureFile` line is the authority when the file it
    names is there; a renamed or relocated atlas falls back to the sibling,
    because the UVs are valid either way and refusing over a file name would
    throw away a good mesh.
    """
    if not ply.is_file():
        raise FileNotFoundError(f"{ply} does not exist")
    head = ply.open("rb").read(4096)
    cut = head.find(b"end_header")
    text = head[: cut if cut >= 0 else len(head)].decode("ascii", "replace")
    match = re.search(r"comment\s+TextureFile\s+(.+)", text)
    if match:
        named = ply.parent / match.group(1).strip()
        if named.is_file():
            return named
    sibling = ply.with_name("texture.png")
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"no atlas beside {ply.name}: no readable TextureFile comment and no "
        "texture.png")


def _pack_glb(ply: Path, texture: Path, out: Path) -> None:
    """The packing itself, separated so it can be tested without a chain."""
    import trimesh
    from PIL import Image

    # The atlas is our own output, not untrusted input, and a dense scan's can
    # exceed Pillow's ~179 MP decompression-bomb guard (17k x 17k is 298 MP),
    # which would otherwise abort both trimesh's own load and the open below.
    Image.MAX_IMAGE_PIXELS = None

    loaded = trimesh.load(ply, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if len(geometries) != 1:
            raise ValueError(f"expected one mesh, found {len(geometries)}")
        mesh = geometries[0]
    else:
        mesh = loaded

    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None or len(uv) == 0:
        raise ValueError("the PLY carries no UVs — not a textured mesh")

    image = Image.open(texture)
    image.load()
    # Rebind the atlas explicitly: the header may name a file that moved, but
    # the UVs are the ones COLMAP solved and they are still right.
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out, file_type="glb")


# ── the chain ────────────────────────────────────────────────────────────


def _chain(mode: str, from_step: str | None, to_step: str | None,
           only: str | None) -> tuple[str, ...]:
    """The steps this invocation will consider, with every name checked against
    this mode's chain first.

    A name that belongs to the other mode is the failure this refusal exists
    for: `--mode texture-only --to patch_match_stereo` would otherwise run the
    short chain to its end and look like it had honoured the flag.
    """
    steps = MODE_STEPS[mode]
    for label, name in (("--from", from_step), ("--to", to_step),
                        ("--only", only)):
        if name is None or name in steps:
            continue
        why = (f"{name} is a step of the dense chain and this run is {mode}"
               if name in CANONICAL_STEPS else f"there is no step called {name}")
        raise ReconRefused(
            f"{label} {name}: {why}. The {mode} chain is: "
            f"{' -> '.join(steps)}.")
    if only is not None:
        return (only,)
    chosen = steps
    if from_step is not None:
        chosen = chosen[chosen.index(from_step):]
    if to_step is not None:
        if to_step not in chosen:
            raise ReconRefused(
                f"--to {to_step} comes before --from {from_step} in the {mode} "
                f"chain, which is: {' -> '.join(steps)}.")
        chosen = chosen[:chosen.index(to_step) + 1]
    return tuple(chosen)


def _apply_mode_change(run: Run, mode: str) -> None:
    """Drop everything a `--mode` change makes unreusable, and say so by name.

    `texture_workspace` is the first step whose `--image_path` differs between
    the modes — `colmap/images` holds verbatim pixels and `colmap/images_clean`
    inpainted ones — so nothing from it onward describes this run's chain any
    more, in either direction. `write_sparse` and `validate_sparse` are
    mode-independent and survive, which is most of the minutes.
    """
    was = run.state.mode
    if not was or was == mode:
        return
    tail = CANONICAL_STEPS[CANONICAL_STEPS.index(MODE_INVALIDATES_FROM):]
    dropped = [name for name in tail if run.state.finished(name)]
    for name in tail:
        run.state.steps.pop(name, None)
    after = [name for name in dropped if name != MODE_INVALIDATES_FROM]
    if dropped:
        run.log.line(
            f"mode changed {was} -> {mode}: invalidating "
            f"{MODE_INVALIDATES_FROM} and the {len(after)} steps after it"
            + (f" ({', '.join(after)})" if after else ""))
    else:
        run.log.line(f"mode changed {was} -> {mode}: nothing from "
                     f"{MODE_INVALIDATES_FROM} onward had finished")


def _check_clean_images_first(run: Run, from_step: str | None,
                              only: str | None) -> None:
    """Refuse a dense range that starts after `clean_images` without one on
    record.

    `colmap/images_clean/` is that step's output, and in dense mode both
    `texture_workspace` and `image_undistorter` read it. Starting after it means
    undistorting a directory that is missing or left over from a run whose
    selection was different — which produces a workspace nobody can tell is
    stale by looking at it.
    """
    if run.mode != "dense":
        return
    boundary = CANONICAL_STEPS.index("clean_images")
    for label, name in (("--from", from_step), ("--only", only)):
        if name is None or CANONICAL_STEPS.index(name) <= boundary:
            continue
        if run.state.finished("clean_images"):
            return
        raise ReconRefused(
            f"{label} {name}: colmap/images_clean/ is clean_images' output and "
            f"{name} reads it, but {STATE_NAME} has no finished clean_images. "
            "Start at --from clean_images, or run --mode texture-only, which "
            "reads the verbatim colmap/images/ instead.")


def run(session_dir: str | Path, mode: str = "texture-only", *,
        from_step: str | None = None, to_step: str | None = None,
        only: str | None = None, restart: bool = False, force: bool = False,
        backend: Backend | None = None,
        on_line: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
        params: ReconParams | None = None) -> RunResult:
    """Run the chain over one session and return what it did.

    Resumes by default: a step `recon-state.json` records as finished is
    skipped, which is what makes a five-hour dense run survivable. `--restart`
    clears the record; `--from`, `--to` and `--only` narrow the range and force
    the step they name to run, because an operator who types a step name means
    "run this one", not "skip it if the file says it is done".

    Raises `ReconRefused` for anything an operator can fix before a container
    starts, `StepFailed` when a step ran and did not succeed, and
    `ReconCancelled` when Abort was pressed. All three carry a full sentence.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise ReconRefused(f"{session_dir} is not a directory — pass the "
                           "session's own directory, the one holding "
                           "session.json.")
    if mode not in MODE_STEPS:
        raise ReconRefused(f"no such mode: {mode!r} — one of "
                           f"{', '.join(sorted(MODE_STEPS))}.")

    missing = [name for name in MODE_STEPS[mode] if name not in STEPS]
    if missing:
        raise ReconRefused(
            f"--mode {mode} is not built yet: {', '.join(missing)} "
            f"{'have' if len(missing) > 1 else 'has'} no implementation. Run "
            "--mode texture-only, which is the whole Milestone 1 chain.")

    params = params or ReconParams()
    backend = backend or DockerBackend(session_dir)
    unavailable = backend.available()
    if unavailable:
        raise ReconRefused(unavailable)

    logger = RunLog(session_dir, on_line)
    state = State.read(session_dir)
    if restart:
        state = State()
        logger.line("--restart: the state file was cleared; every step will run")

    ctx = Run(session_dir=session_dir, mode=mode, backend=backend, log=logger,
              params=params, state=state, force=force, cancel=cancel)
    ctx.report["started_utc"] = _utc_now()

    chosen = _chain(mode, from_step, to_step, only)
    _apply_mode_change(ctx, mode)
    _check_clean_images_first(ctx, from_step, only)
    for name in (from_step, only):
        if name is not None:
            state.steps.pop(name, None)

    version = colmap_version(backend) or ""
    if version:
        logger.line(f"COLMAP {version}")
    state.mode = mode
    state.colmap_version = version
    state.max_image_size = params.max_image_size

    # The floor every run needs, before a single container starts. Dense's own
    # estimate cannot be computed until `write_sparse` knows how many
    # photographs were selected, and it is re-checked there.
    _check_disk(ctx, TEXTURE_ONLY_DISK_BYTES,
                f"any run needs at least {_gb(TEXTURE_ONLY_DISK_BYTES)} free to "
                "start")

    logger.line(f"run: {mode}, steps {' -> '.join(chosen)}")
    ran: list[str] = []
    skipped: list[str] = []
    try:
        for name in chosen:
            if cancel is not None and cancel.is_set():
                raise ReconCancelled(
                    f"cancelled before {name}; {len(ran)} steps finished and "
                    "are recorded, so the next run resumes here.")
            if state.finished(name):
                skipped.append(name)
                logger.line(f"{name}: skipped, finished "
                            f"{state.steps[name].get('done_utc', 'earlier')}")
                continue
            for relative in DIRS_CREATED.get(name, ()):
                session_dir.joinpath(*relative.split("/")).mkdir(
                    parents=True, exist_ok=True)
            logger.line(f"{name}: running")
            started = time.monotonic()
            STEPS[name](ctx)
            seconds = time.monotonic() - started
            state.done(name, seconds)
            state.write(session_dir)
            ran.append(name)
            logger.line(f"{name}: done in {seconds:.1f} s")
    finally:
        state.write(session_dir)
        backend.close()

    ctx.report["finished_utc"] = _utc_now()
    ctx.report.setdefault("mode", mode)
    ctx.report["colmap_version"] = version
    ctx.report["degraded"] = list(ctx.degraded)
    ctx.report["warnings"] = list(ctx.warnings)
    _write_report(ctx)

    logger.line(f"run: finished — ran {len(ran)}, skipped {len(skipped)}"
                + (f", {len(ctx.warnings)} warnings" if ctx.warnings else ""))
    return RunResult(
        session_dir=session_dir, mode=mode, ran=tuple(ran),
        skipped=tuple(skipped), colmap_version=version,
        warnings=list(ctx.warnings), degraded=list(ctx.degraded),
        texture_source=ctx.texture_source,
        n_selected=len(ctx.selection.accepted) if ctx.selection else 0)


def _write_report(ctx: Run) -> None:
    """Add the `reconstruct` block to `session.json`, changing nothing above it.

    A run that resumed past `write_sparse` has no selection to report, so it
    merges what it does know into the block that is already there rather than
    replacing it with a thinner one.
    """
    path = ctx.session_dir / "session.json"
    if not path.exists():
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    block = dict(raw.get("reconstruct") or {})
    block.update(ctx.report)
    raw["reconstruct"] = block
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
