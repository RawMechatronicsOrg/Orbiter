/**
 * Stereo tab — the binocular camserver pair.
 *
 * Two live previews side by side in LEFT | RIGHT order, plus the per-run
 * baseline the operator sets before anything else happens: which upstream
 * camera is which eye, and how each one is oriented.
 *
 * ─── Where the bytes come from ────────────────────────────────────────────
 * The `<img>` tags point **straight at camserver**, not at the Orbiter
 * server. Two 1080p MJPEG streams are ~30-90 Mbit/s each; proxying them
 * would burn CPU and add latency for nothing. An `<img>` is not subject to
 * CORS as long as nobody reads its pixels, and a preview never needs to.
 *
 * The small JSON API *is* cross-origin, so it comes through
 * `/stereo/upstream/*` on our server (see `routes/stereo.py`).
 *
 * ─── ORIENTATION ORDER — the one thing that must not drift ────────────────
 * An eye is **flipped first, then rotated**: `flip_h` / `flip_v` act in the
 * sensor's own frame and `quarter_turns_cw` is applied to the already
 * flipped image. `eyeTransform` below emits
 *
 *     rotate(<90*n>deg) scaleX(±1) scaleY(±1)
 *
 * and CSS composes right-to-left, so that reads flip-then-rotate.
 *
 * Any future server-side path — folding these into the `cv2.remap` map next
 * to undistort and stereo rectification, which costs nothing extra there —
 * MUST use the same order. If preview and CV disagree, the operator aligns
 * the rig against a picture the solver never sees, and it surfaces later as
 * an inexplicable calibration error instead of an obvious mirror.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE } from './api';
import type { Commands, StereoEye } from './commands';
import { useViewerStore } from './modelStore';
import { cls } from './ui';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Card, CardContent, CardTitle } from '../components/ui/card';

type Side = 'left' | 'right';

interface StereoRig {
  host: string;
  token: string;
  baseline_mm: number;
  left: StereoEye;
  right: StereoEye;
}

/** One camera's entry in camserver's `/api/state`. */
interface UpstreamCamera {
  id: string;
  path?: string;
  state?: string;
  error?: string | null;
  info?: { card?: string };
  actual?: { width?: number; height?: number; fourcc?: string; fps?: number };
  stats?: {
    fps?: number;
    kbytes?: number;
    mbit_s?: number;
    jitter_ms?: number;
    dropped?: number;
  };
}

interface UpstreamState {
  cameras?: UpstreamCamera[];
  sync?: { holding?: boolean; spread_ms?: number; skew_ms?: Record<string, number> };
  server?: { version?: string; uptime_s?: number };
}

const EMPTY_EYE: StereoEye = {
  camera_id: '',
  quarter_turns_cw: 0,
  flip_h: false,
  flip_v: false,
};

const DEFAULT_RIG: StereoRig = {
  host: '',
  token: '',
  baseline_mm: 200,
  left: EMPTY_EYE,
  right: EMPTY_EYE,
};

/** How often to poll camserver's stats through our proxy. */
const POLL_MS = 1000;

const ROTATIONS: ReadonlyArray<readonly [number, string]> = [
  [0, '0°'],
  [1, '90° CW'],
  [2, '180°'],
  [3, '270° CW'],
];

/** Read the rig out of the mirrored server model, filling any gaps. */
function readRig(model: Record<string, unknown>): StereoRig {
  const raw = model.stereo_rig;
  if (!raw || typeof raw !== 'object') return DEFAULT_RIG;
  const r = raw as Partial<StereoRig>;
  const eye = (e: Partial<StereoEye> | undefined): StereoEye => ({ ...EMPTY_EYE, ...(e ?? {}) });
  return {
    host: typeof r.host === 'string' ? r.host : '',
    token: typeof r.token === 'string' ? r.token : '',
    baseline_mm: typeof r.baseline_mm === 'number' ? r.baseline_mm : 200,
    left: eye(r.left),
    right: eye(r.right),
  };
}

/**
 * The CSS that renders one eye's orientation. Flip-then-rotate — see the
 * module docstring; this must stay in lockstep with the server-side order.
 */
export function eyeTransform(eye: StereoEye): string {
  const deg = (eye.quarter_turns_cw % 4) * 90;
  return `rotate(${deg}deg) scaleX(${eye.flip_h ? -1 : 1}) scaleY(${eye.flip_v ? -1 : 1})`;
}

/** MJPEG URL on camserver itself. `sync=1` asks for the pair-aligned stream. */
function streamUrl(rig: StereoRig, eye: StereoEye, nonce: number): string | null {
  const host = rig.host.trim().replace(/\/+$/, '');
  if (!host || !eye.camera_id) return null;
  const u = new URL(`${host}/stream/${eye.camera_id}`);
  u.searchParams.set('sync', '1');
  if (rig.token) u.searchParams.set('token', rig.token);
  // camserver's own UI does this: without it the browser image cache can
  // hand back a dead stream after a reconnect.
  u.searchParams.set('_', String(nonce));
  return u.toString();
}

const fmt = (v: number | undefined, d = 1) =>
  v === undefined || v === null ? '—' : v.toFixed(d);

// ── one eye ────────────────────────────────────────────────────────────────

function EyePanel({
  side,
  eye,
  rig,
  cameras,
  upstream,
  nonce,
  onChange,
}: {
  side: Side;
  eye: StereoEye;
  rig: StereoRig;
  cameras: UpstreamCamera[];
  upstream: UpstreamCamera | undefined;
  nonce: number;
  onChange: (patch: Partial<StereoEye>) => void;
}) {
  const src = streamUrl(rig, eye, nonce);
  const rotated = eye.quarter_turns_cw % 2 === 1;
  const stats = upstream?.stats;

  return (
    <div className="flex min-w-0 flex-1 flex-col gap-2">
      <div className="flex items-baseline gap-2">
        <span className="text-[13px] font-semibold uppercase tracking-[0.18em] text-accent">
          {side}
        </span>
        <span className="font-mono text-[11px] text-inkmute">
          {upstream?.path ?? '—'} · {upstream?.info?.card ?? 'not connected'}
        </span>
      </div>

      {/* The stage keeps a fixed box and letterboxes inside it. A 90°/270°
          eye is taller than it is wide once rotated, so the rotated image is
          scaled to fit rather than cropped. */}
      <div className="relative flex aspect-video w-full items-center justify-center overflow-hidden rounded-lg border border-cardline bg-black">
        {src ? (
          <img
            key={src}
            src={src}
            alt={`${side} camera preview`}
            className="max-h-full max-w-full object-contain"
            style={{
              transform: eyeTransform(eye),
              // Rotating a 16:9 frame into a 16:9 box would overflow; cap the
              // long edge by the box's SHORT edge instead.
              maxWidth: rotated ? '56.25%' : '100%',
              maxHeight: rotated ? '177%' : '100%',
            }}
          />
        ) : (
          <span className="px-6 text-center text-[13px] text-inkmute">
            {rig.host.trim()
              ? 'Pick a camera for this eye'
              : 'Set the camserver host above'}
          </span>
        )}
        {stats && (
          <div className="pointer-events-none absolute left-2 top-2 rounded-md bg-black/60 px-2 py-1 font-mono text-[11px] leading-5 text-zinc-100">
            <div>
              <span className="text-inkmute">fps </span>
              {fmt(stats.fps, 2)}
            </div>
            <div>
              <span className="text-inkmute">bw  </span>
              {fmt(stats.kbytes, 0)} kB · {fmt(stats.mbit_s)} Mbit/s
            </div>
            <div>
              <span className="text-inkmute">drop </span>
              {stats.dropped ?? 0}
            </div>
          </div>
        )}
      </div>

      <div className={cls.col}>
        <div className={cls.row}>
          <span className={cls.fieldName}>camera</span>
          <select
            className={cls.input}
            value={eye.camera_id}
            onChange={(e) => onChange({ camera_id: e.target.value })}
          >
            <option value="">— none —</option>
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.id} · {c.path ?? ''}
              </option>
            ))}
            {/* Keep a configured-but-absent id selectable so a camserver
                restart that renumbers devices doesn't silently blank it. */}
            {eye.camera_id && !cameras.some((c) => c.id === eye.camera_id) && (
              <option value={eye.camera_id}>{eye.camera_id} (offline)</option>
            )}
          </select>
        </div>

        <div className={cls.row}>
          <span className={cls.fieldName}>rotate</span>
          <select
            className={cls.input}
            value={eye.quarter_turns_cw}
            onChange={(e) => onChange({ quarter_turns_cw: Number(e.target.value) })}
          >
            {ROTATIONS.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
        </div>

        <div className={cls.row}>
          <span className={cls.fieldName}>mirror</span>
          <label className="flex items-center gap-1.5 text-[12px] text-zinc-300">
            <input
              type="checkbox"
              className={cls.check}
              checked={eye.flip_h}
              onChange={(e) => onChange({ flip_h: e.target.checked })}
            />
            horizontal
          </label>
          <label className="flex items-center gap-1.5 text-[12px] text-zinc-300">
            <input
              type="checkbox"
              className={cls.check}
              checked={eye.flip_v}
              onChange={(e) => onChange({ flip_v: e.target.checked })}
            />
            vertical
          </label>
        </div>
      </div>
    </div>
  );
}

// ── the tab ────────────────────────────────────────────────────────────────

export function StereoView({ commands }: { commands: Commands }) {
  const model = useViewerStore((s) => s.model);
  const serverRig = useMemo(() => readRig(model), [model]);

  // Local draft: edits are live in the preview but reach the server only on
  // Apply, so a half-typed host never becomes the run's baseline.
  const [draft, setDraft] = useState<StereoRig>(serverRig);
  const [upstream, setUpstream] = useState<UpstreamState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(() => 1);
  const dirtyRef = useRef(false);

  // Adopt server state whenever it changes and we have no pending edits.
  useEffect(() => {
    if (!dirtyRef.current) setDraft(serverRig);
  }, [serverRig]);

  const edit = useCallback((patch: Partial<StereoRig>) => {
    dirtyRef.current = true;
    setDraft((d) => ({ ...d, ...patch }));
  }, []);

  const editEye = useCallback(
    (side: Side, patch: Partial<StereoEye>) => {
      dirtyRef.current = true;
      setDraft((d) => ({ ...d, [side]: { ...d[side], ...patch } }));
    },
    [],
  );

  // Poll camserver through our proxy for the stats overlays and camera list.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(`${API_BASE}/stereo/upstream/state`);
        if (!r.ok) {
          let detail = `HTTP ${r.status}`;
          try {
            detail = (await r.json()).detail ?? detail;
          } catch {
            /* keep the status line */
          }
          throw new Error(detail);
        }
        const data = (await r.json()) as UpstreamState;
        if (!alive) return;
        setUpstream(data);
        setError(null);
      } catch (e) {
        if (!alive) return;
        setUpstream(null);
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [draft.host]);

  const cameras = upstream?.cameras ?? [];
  const byId = (id: string) => cameras.find((c) => c.id === id);

  const duplicate =
    !!draft.left.camera_id && draft.left.camera_id === draft.right.camera_id;
  const dirty = JSON.stringify(draft) !== JSON.stringify(serverRig);

  const apply = () => {
    if (duplicate) return;
    commands.setStereoRig({
      host: draft.host,
      token: draft.token,
      baseline_mm: draft.baseline_mm,
      left: draft.left,
      right: draft.right,
    });
    dirtyRef.current = false;
  };

  const revert = () => {
    dirtyRef.current = false;
    setDraft(serverRig);
  };

  /** Swap the eyes wholesale — the usual fix after the first look. */
  const swap = () => {
    dirtyRef.current = true;
    setDraft((d) => ({ ...d, left: d.right, right: d.left }));
  };

  const sync = upstream?.sync;
  const skew = sync?.skew_ms ? Object.entries(sync.skew_ms)[0] : undefined;

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-y-auto p-4">
      <Card>
        <CardContent className="flex flex-col gap-2.5">
          <CardTitle>Camserver</CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <span className={cls.fieldName}>host</span>
            <Input
              className="w-72"
              value={draft.host}
              placeholder="http://192.168.0.222:8088"
              onChange={(e) => edit({ host: e.target.value })}
            />
            <span className={cls.fieldName}>token</span>
            <Input
              className="w-40"
              value={draft.token}
              placeholder="(none)"
              onChange={(e) => edit({ token: e.target.value })}
            />
            <span className={cls.fieldName}>baseline</span>
            <Input
              className="w-24"
              type="number"
              min={1}
              step={1}
              value={draft.baseline_mm}
              onChange={(e) => edit({ baseline_mm: Number(e.target.value) })}
            />
            <span className="text-[12px] text-inkmute">
              mm — nominal, superseded by stereo calibration
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-[12px] text-inkmute">
            {error ? (
              <span className="rounded-md border border-red-900/80 bg-red-950/60 px-2 py-1 font-mono text-red-200">
                camserver unreachable — {error}
              </span>
            ) : (
              <>
                <span className="font-mono">
                  {upstream?.server?.version ?? '—'} · {cameras.length} camera
                  {cameras.length === 1 ? '' : 's'}
                </span>
                {sync && (
                  <span className="font-mono">
                    pair {sync.holding ? 'holding' : 'free-running'}
                    {skew ? ` · skew ${skew[0]} ${fmt(skew[1], 2)} ms` : ''}
                  </span>
                )}
              </>
            )}
            {draft.host.trim() && (
              <a
                className="text-accent underline-offset-2 hover:underline"
                href={draft.host.trim()}
                target="_blank"
                rel="noreferrer"
              >
                open raw camserver UI
              </a>
            )}
            <Button variant="ghost" size="xs" onClick={() => setNonce((n) => n + 1)}>
              Reconnect streams
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <CardTitle>Eyes</CardTitle>
            <Button variant="secondary" size="sm" onClick={swap}>
              Swap L ↔ R
            </Button>
            <span className="flex-1" />
            {dirty && <span className="text-[12px] text-amber-300">unapplied changes</span>}
            <Button variant="outline" size="sm" onClick={revert} disabled={!dirty}>
              Revert
            </Button>
            <Button size="sm" onClick={apply} disabled={!dirty || duplicate}>
              Apply
            </Button>
          </div>

          {duplicate && (
            <div className="rounded-md border border-red-900/80 bg-red-950/60 px-2.5 py-1.5 text-[12px] text-red-200">
              Both eyes are set to <span className="font-mono">{draft.left.camera_id}</span>.
              Pick two different cameras — the server refuses this.
            </div>
          )}

          <div className="flex min-w-0 flex-col gap-4 xl:flex-row">
            <EyePanel
              side="left"
              eye={draft.left}
              rig={draft}
              cameras={cameras}
              upstream={byId(draft.left.camera_id)}
              nonce={nonce}
              onChange={(p) => editEye('left', p)}
            />
            <EyePanel
              side="right"
              eye={draft.right}
              rig={draft}
              cameras={cameras}
              upstream={byId(draft.right.camera_id)}
              nonce={nonce}
              onChange={(p) => editEye('right', p)}
            />
          </div>

          <p className="text-[12px] leading-5 text-inkmute">
            Orientation is <b>flip first, then rotate</b>. Nothing here is sent to the
            cameras — these UVC devices expose no hflip/vflip, so the pair is oriented
            on our side and this config is the run's baseline for it. Confirm L/R
            physically: put a hand at the left edge of the real scene and check it
            shows up in the LEFT pane.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
