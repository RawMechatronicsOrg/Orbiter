"""CAD rig as a backend-owned geometry source.

PROJECT RULE: any 3D-geometric query lives here, not on the frontend.

`Orbiter.glb` (generated once from `Orbiter.fbx` via `tools/fbx_to_glb.py`)
is the source of truth for the physical assembly. This module:

- Loads the GLB once (`trimesh.load`, cached at module level).
- Walks the scene graph by true parent → child edges (so 'BasePulley:1'
  and 'BasePulley:2' stay disjoint; an earlier `startswith` shortcut on
  the frontend lumped both pulleys into one giant AABB).
- Bakes the GLB-side Blender→glTF transform (`× 0.001 × rotX(+π/2)`,
  meters + Y-up) AND the viewer-side Y-up→Z-up conversion
  (`(x,y,z)→(x,-z,y)`) AND a uniform `× 1000` so the returned numbers
  are millimetres in our Z-up world — the same unit space the front-end
  sees after loading Orbiter.fbx with scale 1.0 and `rotation.x = +π/2`.

Result: `RigModel.part_aabb('BasePulley:2')` gives the millimetre AABB
of the azimuthal pulley as it appears in the running viewer. The server
can therefore compute pivot points, anchor frames, kinematic chain
lengths etc. without round-tripping through the browser.
"""

from __future__ import annotations

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
    """`BasePulley:2` → `BasePulley2`, `Orbiter v38` → `Orbiter_v38`."""
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

    # ── public API ──────────────────────────────────────────────────────

    def part_names(self) -> list[str]:
        """All named scene-graph nodes (component instances + intermediates)."""
        return sorted(self.scene.graph.nodes)

    def part_aabb(self, name: str) -> PartAABB | None:
        """World-mm AABB of every mesh descendant of `name`, unioned.
        Returns None if `name` doesn't exist or has no mesh leaves."""
        if name not in self.scene.graph.nodes:
            return None
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for mesh_node, mesh_world_T in self._mesh_descendants(name):
            geom_name = self.scene.graph[mesh_node][1]
            if geom_name is None:
                continue
            mesh = self.scene.geometry[geom_name]
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            # mesh local → GLB world → our Z-up mm world
            full = _GLB_TO_WORLD_MM @ mesh_world_T
            ones = np.ones((len(verts), 1), dtype=np.float64)
            world = (full @ np.hstack([verts, ones]).T).T[:, :3]
            lo = np.minimum(lo, world.min(axis=0))
            hi = np.maximum(hi, world.max(axis=0))
        if not np.isfinite(lo).all():
            return None
        return PartAABB(min=tuple(lo.tolist()), max=tuple(hi.tolist()))

    def part_centre(self, name: str) -> tuple[float, float, float] | None:
        aabb = self.part_aabb(name)
        return None if aabb is None else aabb.centre

    def part_top_centre(self, name: str) -> tuple[float, float, float] | None:
        aabb = self.part_aabb(name)
        return None if aabb is None else aabb.top_centre

    def part_extreme_vertex(
        self, name: str, axis: int, sign: int = 1,
    ) -> tuple[float, float, float] | None:
        """Vertex of `name`'s mesh with the max (sign=+1) or min (sign=-1)
        coordinate along `axis` (0=x, 1=y, 2=z), in world-mm coords.

        Use this when you need an actual mesh point, not the AABB centre:
        the AABB extends out to the bounding box face, but `aabb.centre`
        XY/XZ at the max-Y face is a fictitious point inside the bounding
        box that may sit far from any real vertex. We hit this with
        `OrbitArm:1` — its max-Y tip is at z≈-1 mm in world, while the
        AABB centre Z is +55 mm. Using the centre Z for the tip's Y-Z
        direction biased the alignment angle by ~18°."""
        if name not in self.scene.graph.nodes:
            return None
        best_val = -np.inf if sign > 0 else np.inf
        best_v: np.ndarray | None = None
        for mesh_node, mesh_world_T in self._mesh_descendants(name):
            geom_name = self.scene.graph[mesh_node][1]
            if geom_name is None:
                continue
            mesh = self.scene.geometry[geom_name]
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            ones = np.ones((len(verts), 1))
            full = _GLB_TO_WORLD_MM @ mesh_world_T
            world = (full @ np.hstack([verts, ones]).T).T[:, :3]
            if sign > 0:
                i = int(np.argmax(world[:, axis]))
                if world[i, axis] > best_val:
                    best_val = float(world[i, axis])
                    best_v = world[i]
            else:
                i = int(np.argmin(world[:, axis]))
                if world[i, axis] < best_val:
                    best_val = float(world[i, axis])
                    best_v = world[i]
        return None if best_v is None else tuple(best_v.tolist())

    # ── internals ───────────────────────────────────────────────────────

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

# Default path — sibling of the storage-api directory; override with
# `ORBITER_RIG_GLB` env var if running from elsewhere.
_DEFAULT_GLB = Path(__file__).resolve().parent.parent.parent / "Orbiter.glb"


@lru_cache(maxsize=2)
def load_rig(glb_path: str | Path = _DEFAULT_GLB) -> RigModel:
    """Cached loader. Call without args to use the bundled `Orbiter.glb`."""
    path = Path(glb_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rig GLB not found: {path}")
    scene = trimesh.load(str(path), force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise TypeError(f"expected scene, got {type(scene).__name__}")
    return RigModel(scene)
