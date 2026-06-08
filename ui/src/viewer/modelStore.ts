/**
 * Thin zustand store mirroring the server's authoritative model state plus
 * viewer-only UI state (the scene-node tree and per-node visibility).
 *
 * No mutating actions for model data — the server owns it. Components read
 * from here and dispatch named commands through the WsClient.
 */

import { create } from 'zustand';
import type { ConnState } from './WsClient';

export interface LogEntry {
  level?: string;
  source?: string;
  tag?: string;
  msg?: string;
  ts?: number;
}

/** Lightweight scene-node summary for the Scene Explorer tree. */
export interface SceneNodeInfo {
  id: string;
  parent: string | null;
  type: string;
}

interface ViewerState {
  /** WebSocket lifecycle state, for the connection indicator. */
  connState: ConnState;
  /** Convenience boolean — true only while `connState === 'online'`. */
  connected: boolean;
  /** Full mirror of the server `model` message. */
  model: Record<string, unknown>;
  log: LogEntry[];
  /** Current scene graph (flat list with parent links). */
  sceneNodes: SceneNodeInfo[];
  /** Node ids hidden by the Scene Explorer (client-side visibility). */
  hiddenNodes: string[];

  setConnState: (state: ConnState) => void;
  setModel: (model: Record<string, unknown>) => void;
  patchModel: (patch: Record<string, unknown>) => void;
  appendLog: (entry: LogEntry) => void;
  setSceneNodes: (nodes: SceneNodeInfo[]) => void;
  toggleNode: (id: string) => void;
}

const LOG_CAP = 500;

export const useViewerStore = create<ViewerState>((set) => ({
  connState: 'connecting',
  connected: false,
  model: {},
  log: [],
  sceneNodes: [],
  hiddenNodes: [],

  setConnState: (connState) =>
    set({ connState, connected: connState === 'online' }),
  setModel: (model) => set({ model }),
  patchModel: (patch) => set((s) => {
    const next: Record<string, unknown> = { ...s.model, ...patch };
    const incoming = patch.live_preview;
    const prev = s.model.live_preview;
    if (
      incoming != null
      && typeof incoming === 'object'
      && prev != null
      && typeof prev === 'object'
    ) {
      const inc = incoming as Record<string, unknown>;
      const old = prev as Record<string, unknown>;
      const incImg = inc.images;
      const oldImg = old.images;
      next.live_preview = {
        ...old,
        ...inc,
        images: {
          ...(typeof oldImg === 'object' && oldImg != null ? oldImg : {}),
          ...(typeof incImg === 'object' && incImg != null ? incImg : {}),
        },
      };
    }
    return { model: next };
  }),
  appendLog: (entry) => set((s) => ({ log: [...s.log, entry].slice(-LOG_CAP) })),
  setSceneNodes: (sceneNodes) => set({ sceneNodes }),
  toggleNode: (id) =>
    set((s) => ({
      hiddenNodes: s.hiddenNodes.includes(id)
        ? s.hiddenNodes.filter((k) => k !== id)
        : [...s.hiddenNodes, id],
    })),
}));

/** Wire a WsClient's handlers into the store. Returns nothing — call once. */
export function bindClientToStore(handlers: {
  onConnState?: (state: ConnState) => void;
  onModel?: (m: Record<string, unknown>) => void;
  onModelPatch?: (p: Record<string, unknown>) => void;
  onLog?: (e: Record<string, unknown>) => void;
  onCommandResult?: (p: {
    name: string;
    ok: boolean;
    result?: Record<string, unknown>;
    error?: string;
  }) => void;
}): void {
  const s = useViewerStore.getState();
  handlers.onConnState = s.setConnState;
  handlers.onModel = s.setModel;
  handlers.onModelPatch = s.patchModel;
  handlers.onLog = (e) => s.appendLog(e as LogEntry);
  // Surface FAILED commands in the log panel — otherwise a rejected command
  // (e.g. "Start scan" with no geometry) fails silently and the button looks
  // dead. Successful commands stay quiet; their effect shows via model updates.
  handlers.onCommandResult = (p) => {
    if (!p.ok) {
      s.appendLog({
        level: 'E', source: 'cmd', tag: p.name,
        msg: p.error ?? 'command failed',
      });
    }
  };
}
