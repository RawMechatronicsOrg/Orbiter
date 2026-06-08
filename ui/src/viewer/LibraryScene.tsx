/**
 * Static 3D scene for the Library tab.
 *
 * A thin R3F canvas that renders a local `SceneNode[]` through the *same*
 * `nodeRegistry` factories the Scaner's `SceneRenderer` uses — point clouds,
 * meshes, frames, line segments. Unlike `SceneRenderer` there is no WS
 * reconciliation: the node list is rebuilt wholesale whenever the inspected
 * entity changes, so a plain mount/unmount effect is enough.
 *
 * The node-builder helpers below turn Library data (a point cloud, a
 * calibration plane, a pose) into those abstract nodes.
 */

import { useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, GizmoHelper, GizmoViewport } from '@react-three/drei';
import * as THREE from 'three';
import type { SceneNode, Transform } from './protocol';
import { createObject, disposeObject } from './nodeRegistry';

export type Vec3 = [number, number, number];

const IDENTITY: Transform = {
  position: [0, 0, 0],
  quaternion: [0, 0, 0, 1],
  scale: [1, 1, 1],
};

// ── node builders ───────────────────────────────────────────────────────────

/** A point cloud from flat XYZ (and optional flat 0..1 RGB) buffers. */
export function pointCloudNode(positions: number[], colors?: number[]): SceneNode {
  return {
    id: 'cloud',
    parent: null,
    type: 'point_cloud',
    transform: IDENTITY,
    visible: true,
    pickable: false,
    props: { positions, colors, pointSize: 1.6, color: '#22d3ee' },
  };
}

/** A calibration plane `normal·x + d = 0` — a translucent quad plus its
 *  normal drawn as a line segment from the plane's closest point to origin. */
export function planeNodes(normal: Vec3, d: number, size = 240): SceneNode[] {
  const n = new THREE.Vector3(normal[0], normal[1], normal[2]);
  const len = n.length() || 1;
  n.divideScalar(len);
  const p0 = n.clone().multiplyScalar(-d / len);
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    n,
  );
  return [
    {
      id: 'cal-plane',
      parent: null,
      type: 'mesh',
      transform: {
        position: [p0.x, p0.y, p0.z],
        quaternion: [q.x, q.y, q.z, q.w],
        scale: [1, 1, 1],
      },
      visible: true,
      pickable: false,
      props: {
        primitive: { kind: 'plane', width: size, height: size },
        material: {
          type: 'basic',
          color: '#38bdf8',
          transparent: true,
          opacity: 0.16,
          side: 'double',
        },
      },
    },
    {
      id: 'cal-normal',
      parent: null,
      type: 'line_segments',
      transform: IDENTITY,
      visible: true,
      pickable: false,
      props: {
        points: [
          [p0.x, p0.y, p0.z],
          [p0.x + n.x * 90, p0.y + n.y * 90, p0.z + n.z * 90],
        ],
        color: '#7dd3fc',
      },
    },
  ];
}

/** An axis triad at a pose — translation + Rodrigues `rvec` → RGB axes. */
export function triadNode(
  id: string,
  t: Vec3,
  rvec?: Vec3,
  size = 55,
): SceneNode {
  const q = new THREE.Quaternion();
  if (rvec) {
    const angle = Math.hypot(rvec[0], rvec[1], rvec[2]);
    if (angle > 1e-9) {
      q.setFromAxisAngle(
        new THREE.Vector3(rvec[0] / angle, rvec[1] / angle, rvec[2] / angle),
        angle,
      );
    }
  }
  return {
    id,
    parent: null,
    type: 'frame',
    transform: {
      position: t,
      quaternion: [q.x, q.y, q.z, q.w],
      scale: [1, 1, 1],
    },
    visible: true,
    pickable: false,
    props: { axesLength: size },
  };
}

// ── canvas ──────────────────────────────────────────────────────────────────

function SceneRoot({ nodes }: { nodes: SceneNode[] }) {
  const rootRef = useRef<THREE.Group>(null);
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const built: THREE.Object3D[] = [];
    for (const node of nodes) {
      const obj = createObject(node);
      const t = node.transform;
      obj.position.set(t.position[0], t.position[1], t.position[2]);
      obj.quaternion.set(
        t.quaternion[0], t.quaternion[1], t.quaternion[2], t.quaternion[3],
      );
      obj.scale.set(t.scale[0], t.scale[1], t.scale[2]);
      obj.visible = node.visible;
      root.add(obj);
      built.push(obj);
    }
    return () => {
      for (const obj of built) {
        root.remove(obj);
        disposeObject(obj);
      }
    };
  }, [nodes]);
  return <group ref={rootRef} />;
}

/** Library 3D viewer — world axes + grid, orbit controls, and the given
 *  nodes rendered through `nodeRegistry` (same path as the Scaner). */
export function LibraryScene({
  nodes,
  target = [0, 0, 0],
}: {
  nodes: SceneNode[];
  target?: Vec3;
}) {
  return (
    <Canvas
      camera={{
        position: [280, -280, 220],
        fov: 50,
        up: [0, 0, 1],
        near: 0.5,
        far: 20000,
      }}
      className="h-full w-full bg-[#0c1424]"
    >
      <ambientLight intensity={1.1} />
      <directionalLight position={[300, -300, 500]} intensity={1.0} />
      {/* world frame: +Z up, origin at the rig axis intersection */}
      <axesHelper args={[80]} />
      {/* dimensional grid — 10 mm cells, 100 mm sections (the world is mm) */}
      <Grid
        infiniteGrid
        cellSize={10}
        cellThickness={0.5}
        cellColor="#27406a"
        sectionSize={100}
        sectionThickness={1.1}
        sectionColor="#3f6ea5"
        fadeDistance={1800}
        fadeStrength={1.2}
        rotation={[Math.PI / 2, 0, 0]}
      />
      <OrbitControls makeDefault enableDamping target={target} />
      {/* orientation gizmo — click an axis to snap the view */}
      <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
        <GizmoViewport
          axisColors={['#ef4444', '#22c55e', '#3b82f6']}
          labelColor="#e2e8f0"
        />
      </GizmoHelper>
      <SceneRoot nodes={nodes} />
    </Canvas>
  );
}
