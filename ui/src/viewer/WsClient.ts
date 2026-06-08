/**
 * WebSocket client for the `/ws/scene` channel.
 *
 * Reconnects with exponential backoff and runs an application-level
 * heartbeat: a connection that dies silently (network drop, laptop sleep,
 * NAT timeout — cases where the browser never fires `onclose`) is detected
 * by a missed-traffic watchdog and forced to reconnect. The current
 * connection state is reported through `onConnState`.
 */

import type { Envelope, SceneNode, SceneUpdate } from './protocol';

/** Connection lifecycle as surfaced to the UI. */
export type ConnState = 'connecting' | 'online' | 'offline';

export interface WsHandlers {
  onConnState?: (state: ConnState) => void;
  onSceneSnapshot?: (nodes: SceneNode[]) => void;
  onSceneUpdate?: (update: SceneUpdate) => void;
  onModel?: (model: Record<string, unknown>) => void;
  onModelPatch?: (patch: Record<string, unknown>) => void;
  onLog?: (entry: Record<string, unknown>) => void;
  onTask?: (task: Record<string, unknown>) => void;
  onCommandResult?: (payload: {
    name: string;
    ok: boolean;
    result?: Record<string, unknown>;
    error?: string;
  }) => void;
}

/** Heartbeat ping cadence. */
const PING_INTERVAL_MS = 10_000;
/** No inbound traffic for this long ⇒ treat the socket as dead. */
const DEAD_AFTER_MS = 28_000;
const BACKOFF_MIN_MS = 1_000;
const BACKOFF_MAX_MS = 8_000;

export class WsClient {
  /** Mutable — each consumer component assigns the handlers it cares about. */
  handlers: WsHandlers = {};

  private ws: WebSocket | null = null;
  private backoffMs = BACKOFF_MIN_MS;
  private closedByUser = false;
  /** Bumped on every open()/close(); stale socket callbacks compare against it. */
  private gen = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  /** Wall-clock of the last frame received — drives the dead-socket watchdog. */
  private lastRxAt = 0;

  constructor(private readonly url: string) {}

  connect(): void {
    this.closedByUser = false;
    this.open();
  }

  close(): void {
    this.closedByUser = true;
    this.clearReconnect();
    this.stopHeartbeat();
    this.gen++; // invalidate any in-flight socket callbacks
    const ws = this.ws;
    this.ws = null;
    ws?.close();
    this.handlers.onConnState?.('offline');
  }

  /** Send a named command — the only channel for mutating server state. */
  sendCommand(name: string, args: Record<string, unknown> = {}): void {
    this.send('command', { name, args });
  }

  /** Report a raycast pick back to the server. */
  sendPick(nodeId: string, point: [number, number, number], button = 0): void {
    this.send('pick', { nodeId, point, button });
  }

  /** Ask the server to re-send the scene snapshot. Cheap on the server
   *  (one `build_scene`) and useful when the client suspects its local
   *  scene is stale — e.g. just after a reconnect or when no scene-update
   *  has arrived in a while. */
  requestSnapshot(): void {
    this.send('snapshot', {});
  }

  private send(t: string, data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ t, data }));
    }
  }

  private open(): void {
    this.clearReconnect();
    this.stopHeartbeat();
    const gen = ++this.gen;
    this.handlers.onConnState?.('connecting');

    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      // Constructor only throws on a malformed URL — still retry so a
      // transient environment issue doesn't wedge the client forever.
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      if (gen !== this.gen) {
        ws.close();
        return;
      }
      this.backoffMs = BACKOFF_MIN_MS;
      this.lastRxAt = Date.now();
      this.handlers.onConnState?.('online');
      this.startHeartbeat(gen);
      // Belt-and-braces: even though the server pushes a snapshot on
      // accept(), explicitly ask for one. Cheap (one `build_scene` call)
      // and covers the case where the initial snapshot was lost — e.g.
      // a race with the reconciler's effect cleanup, or a server-side
      // exception during connect() that we'd otherwise never recover
      // from without another reconnect.
      this.send('snapshot', {});
    };
    ws.onmessage = (ev) => {
      if (gen !== this.gen) return;
      this.lastRxAt = Date.now();
      try {
        this.dispatch(JSON.parse(ev.data as string) as Envelope);
      } catch {
        /* ignore malformed frames */
      }
    };
    ws.onerror = () => ws.close();
    ws.onclose = () => {
      if (gen !== this.gen) return; // superseded by a newer socket / close()
      this.ws = null;
      this.stopHeartbeat();
      if (this.closedByUser) {
        this.handlers.onConnState?.('offline');
        return;
      }
      this.handlers.onConnState?.('connecting');
      this.scheduleReconnect();
    };
  }

  /** Periodic ping + dead-socket watchdog for the socket of generation `gen`. */
  private startHeartbeat(gen: number): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (gen !== this.gen) return;
      const ws = this.ws;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      // No traffic for too long ⇒ the socket is silently dead. Force a
      // close so `onclose` runs and the reconnect loop kicks in.
      if (Date.now() - this.lastRxAt > DEAD_AFTER_MS) {
        ws.close();
        return;
      }
      this.send('ping', {}); // server replies `pong` → refreshes lastRxAt
    }, PING_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect(): void {
    this.clearReconnect();
    const delay = this.backoffMs;
    this.backoffMs = Math.min(Math.round(this.backoffMs * 1.6), BACKOFF_MAX_MS);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.closedByUser) this.open();
    }, delay);
  }

  private clearReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private dispatch(msg: Envelope): void {
    switch (msg.t) {
      case 'scene_snapshot':
        this.handlers.onSceneSnapshot?.((msg.data as { nodes: SceneNode[] }).nodes);
        break;
      case 'scene_update':
        this.handlers.onSceneUpdate?.(msg.data as SceneUpdate);
        break;
      case 'model':
        this.handlers.onModel?.(msg.data as Record<string, unknown>);
        break;
      case 'model_patch':
        this.handlers.onModelPatch?.(msg.data as Record<string, unknown>);
        break;
      case 'log':
        this.handlers.onLog?.(msg.data as Record<string, unknown>);
        break;
      case 'task':
        this.handlers.onTask?.(msg.data as Record<string, unknown>);
        break;
      case 'command_result':
        this.handlers.onCommandResult?.(
          msg.data as {
            name: string;
            ok: boolean;
            result?: Record<string, unknown>;
            error?: string;
          },
        );
        break;
      case 'pong': // traffic already refreshed lastRxAt in onmessage
      case 'error':
      default:
        break;
    }
  }
}
