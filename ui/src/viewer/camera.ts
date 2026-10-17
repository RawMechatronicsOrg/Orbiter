/**
 * The phone camera as the server's camera monitor publishes it
 * (`server/orbiter_server/camera_io.py::CameraMonitor`):
 *
 *   * `model.camera_kind` — which app answers on the camera URL,
 *   * `model.camera_node` — the CameraStreamer capture node's state,
 *   * `model.camera_quarter_turns` — clockwise quarter-turns the capture
 *     pipeline applies to this camera's frames (the preview turns the same).
 *
 * The UI only reads these; it never probes the phone itself.
 */

export type CameraKind = 'none' | 'offline' | 'ipwebcam' | 'capture_node';

/** `camera_io._node_summary` — the slow-moving part of the node's /status. */
export interface CameraNode {
  version: string | null;
  state: string | null;          // opening | ready | error | …
  mode: string | null;           // locked | metering
  settled: boolean | null;
  settle_issues: string[];
  control_op: string | null;     // a running lock / profile / sweep, if any
  last_error: string | null;
  camera_id: string | null;
  still_size: [number, number] | null;
  sensor_orientation: number | null;
  profile_fingerprint: string | null;
  profile: {
    exposure_ns: number | null;
    iso: number | null;
    focus_diopters: number | null;
    wb_mode: string | null;
    torch: boolean | null;
  };
  shots: Partial<Record<'queued' | 'exposing' | 'exposed' | 'stored' | 'failed', number>> & {
    stored_bytes?: number;
  };
  storage_free_bytes: number | null;
  battery_pct: number | null;
  charging: boolean | null;
  thermal: string | null;
}

const KINDS: ReadonlySet<string> = new Set(['none', 'offline', 'ipwebcam', 'capture_node']);

export function cameraKind(model: Record<string, unknown>): CameraKind {
  const k = model.camera_kind;
  return typeof k === 'string' && KINDS.has(k) ? (k as CameraKind) : 'none';
}

export function cameraNode(model: Record<string, unknown>): CameraNode | null {
  const n = model.camera_node;
  return n && typeof n === 'object' ? (n as CameraNode) : null;
}

export const CAMERA_KIND_LABEL: Record<CameraKind, string> = {
  none: 'not configured',
  offline: 'not answering',
  ipwebcam: 'IP Webcam',
  capture_node: 'capture node',
};

/** Host part of a URL or a bare address, lower-cased:
 *  `http://192.168.0.248:8080/` and `192.168.0.248:80` both give the IP;
 *  '' when it does not parse. */
export function urlHost(url: string): string {
  const s = url.trim();
  if (!s) return '';
  try {
    return new URL(/^[a-z]+:\/\//i.test(s) ? s : `http://${s}`).hostname.toLowerCase();
  } catch {
    return '';
  }
}
