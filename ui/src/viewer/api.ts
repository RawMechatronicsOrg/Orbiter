/**
 * Shared HTTP client for the storage-api. WS commands handle mutating state
 * the model exposes; HTTP is for everything else (calibration sessions,
 * scan storage, verify, config). Single source of truth for `API_BASE`.
 */

/** True when the UI is served by the Vite dev server (proxied API + WS). */
export function isViteDev(): boolean {
  return new Set(['5173', '5174']).has(window.location.port);
}

/** HTTP base URL — same origin in dev (Vite proxy), direct :8000 in production. */
export const API_BASE = isViteDev()
  ? ''
  : `http://${window.location.hostname}:8000`;

/** WebSocket URL for `/ws/scene` — must match `API_BASE` routing in dev. */
export function wsSceneUrl(): string {
  const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  if (isViteDev()) {
    return `${wsProto}//${window.location.host}/ws/scene`;
  }
  return `${wsProto}//${window.location.hostname}:8000/ws/scene`;
}

/**
 * URL for the live MJPEG camera stream — SAME-ORIGIN (`API_BASE`): in dev that
 * is the Vite proxy (`/camera/stream.mjpeg` → :8000), in prod the direct server
 * origin. The Vite proxy streams MJPEG fine once frames flow — it only withholds
 * the response until the first body byte, so while the camera is idle the
 * `<img>` simply shows nothing (correct). Going through the proxy keeps the
 * `<img>` same-origin, avoiding the cross-origin / LAN-firewall failures a
 * direct `http://<host>:8000` hit runs into (e.g. when the UI is opened on the
 * LAN IP but the browser can't reach the phone-facing :8000 directly).
 */
export function cameraStreamUrl(): string {
  return `${API_BASE}/camera/stream.mjpeg`;
}

export async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(API_BASE + path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export async function postJson<T>(path: string, body: unknown = {}): Promise<T> {
  const r = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = '';
    try { detail = ` — ${(await r.json()).detail ?? ''}`; } catch { /* ignore */ }
    throw new Error(`POST ${path}: HTTP ${r.status}${detail}`);
  }
  return (await r.json()) as T;
}

export async function del(path: string): Promise<void> {
  const r = await fetch(API_BASE + path, { method: 'DELETE' });
  if (!r.ok) throw new Error(`DELETE ${path}: HTTP ${r.status}`);
}
