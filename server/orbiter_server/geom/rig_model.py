"""CAD rig as a backend-owned geometry source.

PROJECT RULE: any 3D-geometric query lives here, not on the frontend.

`Orbiter.glb` is the source of truth for the physical assembly: the v2
mechanics, exported from the Fusion model's Blender scene by
`cad/v2/export_glb.py` into `cad/v2/assembly/`. Its rig nodes:

- `ORB_ROTOR_base` — origin on the azimuth axis (world Z), parent of
  everything that turns with the platform.
- `ORB_ROTOR_elevation` — origin on the elevation axis (world X), parent of
  the shaft, its pulley, the bar insert and the orbit arm.
- Parts are named after their Fusion occurrence
  (`BottomBody:1 / BasePlatformInsert Taper:1`, `OrbitArm:1`, ...).

This module:

- Loads the GLB once (`trimesh.load`, cached at module level).
- Walks the scene graph by true parent → child edges, so a rotor's subtree
  is exactly the parts that turn with it.
- Bakes the glTF Y-up → our Z-up conversion (`(x,y,z)→(x,-z,y)`) and a
  uniform `× 1000` (glTF metres → mm) — the same space the front-end sees
  after loading the GLB with scale 1000 and `rotation.x = +π/2`.

Result: `RigModel.node_origin('ORB_ROTOR_elevation')` is a millimetre
point on the elevation axis as it appears in the running viewer. The server
can therefore compute pivot points, anchor frames, kinematic chain
lengths etc. without round-tripping through the browser.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh


# Mirrors three.js `PropertyBinding.sanitizeNodeName` (used by GLTFLoader when
# importing node names): spaces → underscore, then strip `[]./:`. We must hand
# the front-end already-sanitized names or its `getObjectByName(...)` lookups
# will silently miss every node — three.js drops the colons/dots that the GLB
# file actually contains.
_THREE_RESERVED = re.compile(r"[\[\].:/]")


def to_three_name(name: str) -> str:
    """`OrbitArm:1` → `OrbitArm1`, `BottomBody:1 / HexPlug:2` → `BottomBody1__HexPlug2`."""
    return _THREE_RESERVED.sub("", name.replace(" ", "_"))

# Y-up GLB → our Z-up world: (x, y, z) → (x, -z, y). Plus a scalar ×1000
# to go from glTF meters to scene millimetres.
_GLB_TO_WORLD_MM = np.array(
    [
        [1000.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1000.0, 0.0],
        [0.0, 1000.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PartAABB:
    """Axis-aligned bounding box in our Z-up millimetre world."""

    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @property
    def centre(self) -> tuple[float, float, float]:
        return (
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        )

    @property
    def extents(self) -> tuple[float, float, float]:
        return (
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        )

    @property
    def top_centre(self) -> tuple[float, float, float]:
        """(centre.x, centre.y, max.z) — anchor for things sitting on top."""
        cx, cy, _ = self.centre
        return (cx, cy, self.max[2])


class RigModel:
    """Loaded CAD rig with named-part queries. Holds the trimesh.Scene plus
    a precomputed child map so AABBs are O(descendant-meshes)."""

    def __init__(self, scene: trimesh.Scene) -> None:
        self.scene = scene
        # parent → list[child]
        self._children: dict[str, list[str]] = {}
        for edge in scene.graph.to_edgelist():
            # trimesh emits [parent, child, attrs]
            parent, child = edge[0], edge[1]
            self._children.setdefault(parent, []).append(child)
        # The rig is static (rest pose), so per-node vertex sets never change;
        # the scene builder queries the same few nodes on every rebuild.
        self._vertex_cache: dict[str, np.ndarray | None] = {}

    # ── public API ──────────────────────────────────────────────────────

    def part_names(self) -> list[str]:
        """All named scene-graph nodes (component instances + intermediates)."""
        return sorted(self.scene.graph.nodes)

    def node_origin(self, name: str) -> tuple[float, float, float] | None:
        """World-mm position of node `name`'s own origin — for a rotor empty,
        a point on its axis. None if the node doesn't exist."""
        if name not in self.scene.graph.nodes:
            return None
        world_T, _ = self.scene.graph[name]
        origin = _GLB_TO_WORLD_MM @ np.asarray(world_T, dtype=np.float64)
        return tuple(origin[:3, 3].tolist())

    def part_aabb(self, name: str) -> PartAABB | None:
        """World-mm AABB of every mesh descendant of `name`, unioned.
        Returns None if `name` doesn't exist or has no mesh leaves."""
        world = self._world_vertices(name)
        if world is None:
            return None
        return PartAABB(min=tuple(world.min(axis=0).tolist()),
                        max=tuple(world.max(axis=0).tolist()))

    def part_centre(self, name: str) -> tuple[float, float, float] | None:
        aabb = self.part_aabb(name)
        return None if aabb is None else aabb.centre

    def part_top_centre(self, name: str) -> tuple[float, float, float] | None:
        aabb = self.part_aabb(name)
        return None if aabb is None else aabb.top_centre

    def part_end_centre(
        self, name: str, axis: int, sign: int = 1, depth_mm: float = 1.0,
    ) -> tuple[float, float, float] | None:
        """Centre of `name`'s end face along `axis` (0=x, 1=y, 2=z): the AABB
        centre of the vertices within `depth_mm` of the max (sign=+1) or min
        (sign=-1) coordinate, in world-mm coords.

        For a tube such as `OrbitArm:1` this is the point on its centreline
        at the tip. The extreme vertex is not: it is a corner of the tube,
        12.5 mm off the centreline, and tilts the arm's Y-Z direction by ~3°."""
        world = self._world_vertices(name)
        if world is None:
            return None
        coord = world[:, axis] * sign
        end = world[coord >= coord.max() - depth_mm]
        return tuple(((end.min(axis=0) + end.max(axis=0)) * 0.5).tolist())

    # ── internals ───────────────────────────────────────────────────────

    def _world_vertices(self, name: str) -> np.ndarray | None:
        """(N, 3) world-mm vertices of every mesh descendant of `name`, or
        None if `name` doesn't exist or has no mesh leaves. Cached."""
        if name in self._vertex_cache:
            return self._vertex_cache[name]
        chunks: list[np.ndarray] = []
        if name in self.scene.graph.nodes:
            for mesh_node, mesh_world_T in self._mesh_descendants(name):
                geom_name = self.scene.graph[mesh_node][1]
                if geom_name is None:
                    continue
                verts = np.asarray(self.scene.geometry[geom_name].vertices, dtype=np.float64)
                # mesh local → GLB world → our Z-up mm world
                full = _GLB_TO_WORLD_MM @ mesh_world_T
                chunks.append(verts @ full[:3, :3].T + full[:3, 3])
        world = np.vstack(chunks) if chunks else None
        self._vertex_cache[name] = world
        return world

    def _mesh_descendants(self, root: str):
        """Yield (node_name, world_matrix) for every descendant that has
        geometry. Uses the scene graph's own cumulative world matrix —
        we do NOT compose ourselves."""
        stack = [root]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            try:
                world_T, geom = self.scene.graph[node]
            except KeyError:
                continue
            if geom is not None:
                yield node, np.asarray(world_T, dtype=np.float64)
            stack.extend(self._children.get(node, ()))


# ── module-level cache ──────────────────────────────────────────────────

# Candidate locations for the CAD export, first hit wins. Override with the
# `ORBITER_RIG_GLB` env var when running against a different export.
_PKG_DIR = Path(__file__).resolve().parent.parent
_GLB_CANDIDATES = (
    _PKG_DIR / "assets" / "Orbiter.glb",                       # packaged asset
    _PKG_DIR.parent / "Orbiter.glb",                           # server/ dev copy
    _PKG_DIR.parent.parent / "cad" / "v2" / "assembly" / "Orbiter.glb",  # repo checkout
)


def _default_glb() -> Path:
    for p in _GLB_CANDIDATES:
        if p.is_file():
            return p
    # Report the canonical location in the error — it exists in any checkout.
    return _GLB_CANDIDATES[-1]


def rig_glb_path() -> Path:
    """The rig GLB this server uses: `ORBITER_RIG_GLB`, else the first
    candidate that exists."""
    return Path(os.environ.get("ORBITER_RIG_GLB") or _default_glb()).resolve()


@lru_cache(maxsize=2)
def rig_identity(glb_path: str | Path | None = None) -> str:
    """Short content hash of the rig GLB — the mechanics identity a geometry
    calibration is stamped with (`model.calib_rig_id`).

    A calibration solves offsets relative to this CAD model, so one made
    against another GLB (other mechanics, or a re-export after a CAD change)
    no longer describes the rig the server draws and computes poses for."""
    path = rig_glb_path() if glb_path is None else Path(glb_path).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@lru_cache(maxsize=2)
def load_rig(glb_path: str | Path | None = None) -> RigModel:
    """Cached loader. Call without args to use the bundled `Orbiter.glb`."""
    path = rig_glb_path() if glb_path is None else Path(glb_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rig GLB not found: {path}")
    scene = trimesh.load(str(path), force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise TypeError(f"expected scene, got {type(scene).__name__}")
    return RigModel(scene)
