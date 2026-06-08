/**
 * Generic scene reconciler — the domain-agnostic three.js viewer.
 *
 * Holds a `Map<nodeId, THREE.Object3D>` and applies `scene_snapshot` /
 * `scene_update` messages from the server. Effective visibility combines the
 * server's `node.visible` with the Scene Explorer's per-node toggles; a hidden
 * parent hides its whole subtree (three.js propagates `visible`).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import * as THREE from 'three';
import type { WsClient } from './WsClient';
import type { NodePatch, SceneNode, SceneUpdate, Transform } from './protocol';
import { createObject, disposeObject } from './nodeRegistry';
import { raycastPick } from './picking';
import { useViewerStore } from './modelStore';
import { CaptureModal, type CaptureView } from './CaptureModal';
import { makeCommands } from './commands';

/** Capture node ids the picker maps to a photo modal: a frustum / photo card
 *  for an ACTIVE capture (`capture_{i}` / `capture_card_{i}`, in
 *  `model.captures`) OR a LOADED-for-review capture (`loaded_{i}` /
 *  `loaded_card_{i}`, in `model.loaded_captures`). Both carry the capture's
 *  `index`. */
const CAPTURE_ID_RE = /^(capture|loaded)(?:_card)?_(\d+)$/;

/** Resolve a picked node id to its capture record + whether it's a loaded
 *  (saved-scan review) capture. Null when the id isn't a capture node. */
function captureForNodeId(
  nodeId: string,
): { capture: CaptureView; isLoaded: boolean } | null {
  const m = CAPTURE_ID_RE.exec(nodeId);
  if (!m) return null;
  const isLoaded = m[1] === 'loaded';
  const idx = Number(m[2]);
  const model = useViewerStore.getState().model;
  const arr = isLoaded ? model.loaded_captures : model.captures;
  if (!Array.isArray(arr)) return null;
  const found = (arr as CaptureView[]).find((c) => c?.index === idx);
  return found ? { capture: found, isLoaded } : null;
}

function applyTransform(obj: THREE.Object3D, t: Transform): void {
  obj.position.set(t.position[0], t.position[1], t.position[2]);
  obj.quaternion.set(t.quaternion[0], t.quaternion[1], t.quaternion[2], t.quaternion[3]);
  obj.scale.set(t.scale[0], t.scale[1], t.scale[2]);
}

/** Imperative reconciler mounted inside the R3F canvas. */
function SceneRoot({ client }: { client: WsClient }) {
  const rootRef = useRef<THREE.Group>(null);
  const objects = useRef(new Map<string, THREE.Object3D>());
  const data = useRef(new Map<string, SceneNode>());
  const hiddenNodes = useViewerStore((s) => s.hiddenNodes);
  const hiddenRef = useRef<string[]>(hiddenNodes);
  hiddenRef.current = hiddenNodes;

  /** Server visibility AND not hidden by the Scene Explorer. */
  function effectiveVisible(node: SceneNode): boolean {
    return node.visible && !hiddenRef.current.includes(node.id);
  }

  /** Publish the node tree to the store so the Scene Explorer can render it. */
  function syncStore(): void {
    useViewerStore.getState().setSceneNodes(
      [...data.current.values()].map((n) => ({
        id: n.id,
        parent: n.parent,
        type: n.type,
      })),
    );
  }

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const objMap = objects.current;
    const dataMap = data.current;

    const parentObj = (parentId: string | null): THREE.Object3D =>
      (parentId && objMap.get(parentId)) || root;

    function buildOne(node: SceneNode): void {
      const obj = createObject(node);
      applyTransform(obj, node.transform);
      obj.visible = effectiveVisible(node);
      obj.userData.nodeId = node.id;
      obj.userData.pickable = node.pickable;
      objMap.set(node.id, obj);
      dataMap.set(node.id, node);
    }

    function addNodes(nodes: SceneNode[]): void {
      for (const n of nodes) buildOne(n);
      for (const n of nodes) parentObj(n.parent).add(objMap.get(n.id)!);
    }

    function removeNode(id: string): void {
      const obj = objMap.get(id);
      if (!obj) return;
      obj.parent?.remove(obj);
      disposeObject(obj);
      objMap.delete(id);
      dataMap.delete(id);
    }

    function clearAll(): void {
      for (const id of [...objMap.keys()]) removeNode(id);
    }

    function recreate(node: SceneNode): void {
      removeNode(node.id);
      buildOne(node);
      parentObj(node.parent).add(objMap.get(node.id)!);
    }

    function patchNode(patch: NodePatch): void {
      const prev = dataMap.get(patch.id);
      const obj = objMap.get(patch.id);
      if (!prev || !obj) return;
      const merged: SceneNode = {
        ...prev,
        ...(patch.parent !== undefined ? { parent: patch.parent } : {}),
        ...(patch.type ? { type: patch.type } : {}),
        ...(patch.transform ? { transform: patch.transform } : {}),
        ...(patch.props ? { props: patch.props } : {}),
        ...(patch.visible !== undefined ? { visible: patch.visible } : {}),
        ...(patch.pickable !== undefined ? { pickable: patch.pickable } : {}),
      };
      if (patch.type || patch.props || patch.parent !== undefined) {
        recreate(merged);
        return;
      }
      dataMap.set(patch.id, merged);
      if (patch.transform) applyTransform(obj, patch.transform);
      obj.visible = effectiveVisible(merged);
      if (patch.pickable !== undefined) obj.userData.pickable = patch.pickable;
    }

    client.handlers.onSceneSnapshot = (nodes) => {
      clearAll();
      addNodes(nodes);
      syncStore();
    };
    client.handlers.onSceneUpdate = (update: SceneUpdate) => {
      for (const id of update.removed) removeNode(id);
      if (update.added.length) addNodes(update.added);
      for (const p of update.updated) patchNode(p);
      if (update.added.length || update.removed.length) syncStore();
    };

    // The reconciler just (re)mounted with an EMPTY object map — pull a fresh
    // full snapshot so the scene repaints. Critical for HMR / StrictMode
    // remounts where the WS is ALREADY open (so its onopen won't fire again),
    // and where incremental scene_update diffs — computed against the server's
    // last-sent baseline — would add nothing to our just-cleared map, leaving
    // the scene blank ("scene doesn't load"). No-op if the socket isn't open
    // yet; the WS onopen requests one too, so initial load isn't double-sent.
    client.requestSnapshot();

    return () => {
      client.handlers.onSceneSnapshot = undefined;
      client.handlers.onSceneUpdate = undefined;
      clearAll();
    };
  }, [client]);

  // Re-apply visibility whenever the Scene Explorer toggles a node.
  useEffect(() => {
    for (const [id, obj] of objects.current) {
      const node = data.current.get(id);
      if (node) obj.visible = effectiveVisible(node);
    }
  }, [hiddenNodes]);

  return <group ref={rootRef} />;
}

/** Raycasts pointer-downs against the scene and reports picks to the server.
 *  Picking a capture frustum/card also opens the photo modal client-side via
 *  `onPickCapture` — no server round-trip needed. */
function PickController({
  client,
  onPickCapture,
}: {
  client: WsClient;
  onPickCapture: (capture: CaptureView, isLoaded: boolean) => void;
}) {
  const { camera, gl, scene } = useThree();
  const raycaster = useMemo(() => new THREE.Raycaster(), []);

  useEffect(() => {
    const el = gl.domElement;
    function onPointerDown(ev: PointerEvent): void {
      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((ev.clientX - rect.left) / rect.width) * 2 - 1,
        -((ev.clientY - rect.top) / rect.height) * 2 + 1,
      );
      const hit = raycastPick(raycaster, camera, ndc, scene);
      if (!hit) return;
      // Still notify the server (harmless logging) for every pick.
      client.sendPick(hit.nodeId, hit.point, ev.button);
      // Captures (active OR loaded-review) additionally open a local photo modal.
      const picked = captureForNodeId(hit.nodeId);
      if (picked) onPickCapture(picked.capture, picked.isLoaded);
    }
    el.addEventListener('pointerdown', onPointerDown);
    return () => el.removeEventListener('pointerdown', onPointerDown);
  }, [camera, gl, scene, raycaster, client, onPickCapture]);

  return null;
}

export function SceneRenderer({ client }: { client: WsClient }) {
  // Capture picked in the scene → opens the photo modal (a Radix Dialog, which
  // is HTML and must live OUTSIDE the <Canvas>, so it's a sibling below).
  const [selectedCapture, setSelectedCapture] = useState<CaptureView | null>(null);
  // Loaded (saved-scan review) captures open the same modal but WITHOUT the
  // delete action — per-frame delete of a stored scan is out of v0.1 scope
  // (delete the whole scan from the Library instead).
  const [selectedIsLoaded, setSelectedIsLoaded] = useState(false);
  const onPickCapture = useCallback((c: CaptureView, isLoaded: boolean) => {
    setSelectedCapture(c);
    setSelectedIsLoaded(isLoaded);
  }, []);
  // Commands is a thin pure factory over the WS client we already hold — derive
  // it here so the capture modal can offer "Delete photo" without threading a
  // new prop down from ViewerApp.
  const commands = useMemo(() => makeCommands(client), [client]);

  return (
    <>
      <Canvas
        camera={{ position: [320, -320, 240], fov: 50, up: [0, 0, 1], near: 1, far: 10000 }}
        className="bg-[#0c1424]"
      >
        <ambientLight intensity={1.0} />
        <hemisphereLight args={['#e0f2fe', '#1e293b', 0.6]} />
        <directionalLight position={[300, -300, 500]} intensity={1.1} />
        <directionalLight position={[-250, 250, 200]} intensity={0.4} />
        <OrbitControls makeDefault enableDamping target={[0, 0, 0]} />
        <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
          <GizmoViewport
            axisColors={['#ef4444', '#22c55e', '#3b82f6']}
            labelColor="#e2e8f0"
          />
        </GizmoHelper>
        <SceneRoot client={client} />
        <PickController client={client} onPickCapture={onPickCapture} />
      </Canvas>
      <CaptureModal
        capture={selectedCapture}
        open={selectedCapture !== null}
        onClose={() => setSelectedCapture(null)}
        commands={selectedIsLoaded ? undefined : commands}
      />
    </>
  );
}
