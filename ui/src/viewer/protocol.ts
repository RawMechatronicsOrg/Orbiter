/**
 * Wire protocol for the `/ws/scene` channel — must mirror the Python side
 * (`storage-api/scene_graph.py`, `ws_hub.py`).
 *
 * The server holds the authoritative scene as a flat list of typed nodes and
 * pushes a `scene_snapshot` on connect, then `scene_update` diffs. The frontend
 * is domain-agnostic: it just maps node `type` → a three.js object.
 */

/** Abstract node kinds. Mirrors Viser's taxonomy. */
export type NodeType =
  | 'frame'
  | 'grid'
  | 'mesh'
  | 'line_segments'
  | 'point_cloud'
  | 'image_plane'
  | 'camera_frustum'
  | 'label'
  | 'cad_model'
  | 'cad_part'
  | 'disc_dial';

/** Local transform of a node, relative to its parent. */
export interface Transform {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  scale: [number, number, number];
}

/** One scene node. `props` is type-specific and intentionally untyped. */
export interface SceneNode {
  id: string;
  parent: string | null;
  type: NodeType;
  transform: Transform;
  visible: boolean;
  pickable: boolean;
  props: Record<string, unknown>;
}

/** A partial node update — only the changed fields are present. */
export interface NodePatch {
  id: string;
  parent?: string | null;
  type?: NodeType;
  transform?: Transform;
  props?: Record<string, unknown>;
  visible?: boolean;
  pickable?: boolean;
}

export interface SceneUpdate {
  added: SceneNode[];
  updated: NodePatch[];
  removed: string[];
}

/** Every WebSocket frame is one of these. */
export interface Envelope<T = unknown> {
  t: string;
  seq: number;
  ts: number;
  data: T;
}
