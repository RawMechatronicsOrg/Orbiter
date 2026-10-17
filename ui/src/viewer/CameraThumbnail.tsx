/**
 * Live camera monitor — a small, always-on MJPEG thumbnail pinned to the
 * top-right of the 3D viewer for at-a-glance control of what the phone sees.
 * Click it to open a full-size modal of the same stream.
 *
 * The server re-multiplexes the phone's MJPEG at `GET /camera/stream.mjpeg`
 * (a multipart/x-mixed-replace response that is a drop-in for `<img src>` —
 * no JS decoder needed). `GET /camera/stream/status` drives the live dot and
 * says which URL it reads and why it last failed. Each `<img>` opens its own
 * stream connection, but the server shares ONE upstream phone connection
 * across all viewers, so thumbnail + modal is fine.
 *
 * Which app the phone runs, the capture node's state and the preview turn
 * come from the server's camera monitor (`camera.ts`). On a capture node the
 * modal also locks / unlocks its manual profile — a scan refuses to start on
 * an unlocked node.
 */

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useViewerStore } from './modelStore';
import { cameraStreamUrl, getJson, postJson } from './api';
import { CAMERA_KIND_LABEL, cameraKind, cameraNode, type CameraKind, type CameraNode } from './camera';
import { num } from './ui';
import { Button } from '../components/ui/button';
import { Dialog, DialogClose, DialogContent, DialogTitle } from '../components/ui/dialog';

const STREAM_URL = cameraStreamUrl();
const STATUS_PATH = '/camera/stream/status';
const STATUS_POLL_MS = 2000;

/**
 * MJPEG `<img>` rotated to match the capture orientation. For an odd number of
 * quarter-turns the box is portrait and the (landscape) stream is rotated 90°
 * about its centre; we swap the rotated element's layout width/height to the
 * measured container so `object-contain` fits it exactly. Even turns render
 * the stream straight.
 *
 * `variant`: `thumb` is a width-driven aspect box (the pinned monitor),
 * `full` is height-driven for the modal. `aspect` is the stream's own
 * width / height (4:3 on the capture node, often 16:9 on IP Webcam).
 */
function StreamView({
  turns,
  variant,
  aspect,
}: {
  turns: number;
  variant: 'thumb' | 'full';
  aspect: number;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const portrait = turns % 2 !== 0;

  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el || !portrait) return;
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [portrait]);

  // Landscape — no quarter turn; a half turn needs no box swap.
  if (!portrait) {
    return (
      <div className="relative flex w-full items-center justify-center overflow-hidden bg-black">
        <img
          src={STREAM_URL}
          alt="camera stream"
          draggable={false}
          className={
            variant === 'thumb'
              ? 'w-full object-contain'
              : 'max-h-[70vh] w-auto max-w-full object-contain'
          }
          style={{
            aspectRatio: variant === 'thumb' ? aspect : undefined,
            transform: turns % 4 ? `rotate(${turns * 90}deg)` : undefined,
          }}
        />
      </div>
    );
  }

  // Portrait box; the landscape stream is rotated to fill it.
  const boxCls =
    variant === 'thumb'
      ? 'relative w-full overflow-hidden bg-black'
      : 'relative h-[70vh] overflow-hidden bg-black';
  return (
    <div ref={boxRef} className={boxCls} style={{ aspectRatio: 1 / aspect }}>
      <img
        src={STREAM_URL}
        alt="camera stream"
        draggable={false}
        className="object-contain"
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: box.h,
          height: box.w,
          // Preflight's `img { max-width: 100% }` would clamp the pre-rotation
          // width (the box's height) to the box's width and squash the frame.
          maxWidth: 'none',
          transform: `translate(-50%, -50%) rotate(${turns * 90}deg)`,
          transformOrigin: 'center',
        }}
      />
    </div>
  );
}

/** `camera_stream.CameraStreamReader.status()`. */
interface StreamStatus {
  connected: boolean;
  have_frame: boolean;
  seq: number;
  url: string | null;
  frame_wh: [number, number] | null;
  last_error: string | null;
}

function StatusDot({ live, label }: { live: boolean; label: string }) {
  return (
    <span
      className="flex items-center gap-1.5 text-[11px] font-semibold uppercase
                 tracking-[0.16em] text-inkmute"
    >
      <span
        className={
          'inline-block h-2 w-2 rounded-full ' +
          (live
            ? 'bg-emerald-400 shadow-[0_0_6px] shadow-emerald-400/70'
            : 'bg-rose-500/70')
        }
      />
      {label}
    </span>
  );
}

/** Short kind/mode tag for the thumbnail header: green when a node is
 *  locked (ready to scan), amber when it is not. */
function KindTag({ kind, node }: { kind: CameraKind; node: CameraNode | null }) {
  if (kind === 'capture_node' && node) {
    const locked = node.mode === 'locked' && node.state === 'ready';
    return (
      <span
        className={'text-[10px] uppercase tracking-wider ' +
          (locked ? 'text-emerald-300' : 'text-amber-300')}
        title={locked ? 'capture node, locked on its manual profile'
                      : `capture node: ${node.state ?? '?'} / ${node.mode ?? '?'} — scans need it locked`}
      >
        node · {node.state === 'ready' ? node.mode : node.state}
      </span>
    );
  }
  return (
    <span className="text-[10px] uppercase tracking-wider text-inkmute">
      {CAMERA_KIND_LABEL[kind]}
    </span>
  );
}

/** Where the stream is read from and why it failed — shown while not live. */
function StreamProblem({ status }: { status: StreamStatus | null }) {
  if (!status) {
    return <div className="text-rose-300/90">server stream status unavailable</div>;
  }
  return (
    <>
      <div className="truncate font-mono text-inkdim" title={status.url ?? ''}>
        {status.url ?? '—'}
      </div>
      {status.last_error ? (
        <div className="line-clamp-2 break-all text-rose-300/90" title={status.last_error}>
          {status.last_error}
        </div>
      ) : (
        <div>{status.connected ? 'connected, no frame yet' : 'connecting…'}</div>
      )}
    </>
  );
}

const fmt = (v: number | null | undefined, digits: number, unit = ''): string =>
  v == null ? '—' : `${v.toFixed(digits)}${unit}`;

function focusText(d: number | null | undefined): string {
  if (d == null) return '—';
  return d > 0 ? `${d.toFixed(2)} D (${(100 / d).toFixed(0)} cm)` : '∞';
}

function NodeRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[10px] uppercase tracking-[0.14em] text-inkmute">{label}</span>
      <span className={'truncate font-mono text-[12px] ' + (warn ? 'text-amber-300' : 'text-zinc-100')}>
        {value}
      </span>
    </div>
  );
}

/** The capture node's state and its profile operations (`POST /camera/node/{op}`). */
function NodePanel({ node }: { node: CameraNode }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (op: 'lock' | 'profile' | 'unlock') => {
    setBusy(op);
    setError(null);
    try {
      await postJson(`/camera/node/${op}`, {});
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const locked = node.mode === 'locked';
  const p = node.profile;
  const stored = node.shots.stored ?? 0;
  const failed = node.shots.failed ?? 0;
  const freeGb = node.storage_free_bytes == null ? null : node.storage_free_bytes / 1e9;
  return (
    <div className="flex w-[260px] shrink-0 flex-col gap-2 border-l border-cardline p-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-inkmute">
        Capture node {node.version ? `v${node.version}` : ''}
      </div>
      <div className="flex flex-col gap-0.5 rounded-md border border-cardline bg-black/25 px-2.5 py-1.5">
        <NodeRow label="state" value={node.state ?? '—'} warn={node.state !== 'ready'} />
        <NodeRow label="mode" value={node.mode ?? '—'} warn={!locked} />
        <NodeRow
          label="settled"
          value={node.settled ? 'yes' : node.settle_issues.join(', ') || 'no'}
          warn={locked && !node.settled}
        />
        <NodeRow label="exposure" value={fmt(p.exposure_ns == null ? null : p.exposure_ns / 1e6, 2, ' ms')} />
        <NodeRow label="iso" value={fmt(p.iso, 0)} />
        <NodeRow label="focus" value={focusText(p.focus_diopters)} />
        <NodeRow label="white bal." value={p.wb_mode ?? '—'} />
        <NodeRow label="profile" value={node.profile_fingerprint ?? '—'} />
        <NodeRow
          label="still"
          value={node.still_size ? `${node.still_size[0]}×${node.still_size[1]}` : '—'}
        />
        <NodeRow label="on phone" value={`${stored} stored${failed ? `, ${failed} failed` : ''}`}
                 warn={failed > 0} />
        <NodeRow label="free" value={fmt(freeGb, 1, ' GB')} warn={freeGb != null && freeGb < 2} />
        <NodeRow
          label="battery"
          value={`${fmt(node.battery_pct, 0, '%')}${node.charging ? ' ⚡' : ''} · ${node.thermal ?? '—'}`}
          warn={(node.battery_pct ?? 100) < 20 && !node.charging}
        />
      </div>
      {node.control_op && (
        <div className="text-[11px] text-sky-300">phone busy: {node.control_op}</div>
      )}
      {node.last_error && (
        <div className="break-all text-[11px] text-rose-300/90">{node.last_error}</div>
      )}
      <div className="grid grid-cols-2 gap-1.5">
        <Button
          size="sm"
          variant={locked ? 'secondary' : 'default'}
          disabled={busy !== null}
          onClick={() => void run('lock')}
          title="Meter once with auto exposure, focus (centre) and white balance, then freeze exactly those values as the manual profile"
        >
          {busy === 'lock' ? 'Metering…' : 'Lock (meter)'}
        </Button>
        <Button
          size="sm"
          variant="secondary"
          disabled={busy !== null}
          onClick={() => void run('profile')}
          title="Lock the profile saved on the phone again, without metering"
        >
          {busy === 'profile' ? 'Locking…' : 'Re-lock saved'}
        </Button>
        <Button
          size="sm"
          variant="secondary"
          className="col-span-2"
          disabled={busy !== null || !locked}
          onClick={() => void run('unlock')}
          title="Back to auto exposure / focus / white balance; the saved profile stays"
        >
          {busy === 'unlock' ? 'Unlocking…' : 'Unlock (auto)'}
        </Button>
      </div>
      {error && <div className="break-all text-[11px] text-rose-300/90">{error}</div>}
      {!locked && (
        <div className="text-[11px] leading-snug text-amber-300/90">
          Not locked: scans refuse to start and shots are refused until the
          profile is locked.
        </div>
      )}
    </div>
  );
}

export function CameraThumbnail() {
  const cameraUrl = useViewerStore((s) =>
    typeof s.model.camera_url === 'string' ? (s.model.camera_url as string) : '',
  );
  const kind = useViewerStore((s) => cameraKind(s.model));
  const node = useViewerStore((s) => cameraNode(s.model));
  // The same quarter-turns the server applies to stored photos of this camera.
  const turns = useViewerStore((s) => num(s.model, 'camera_quarter_turns', 0));
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<StreamStatus | null>(null);

  // Poll the upstream connectivity snapshot for the live/offline dot.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await getJson<StreamStatus>(STATUS_PATH);
        if (alive) setStatus(s);
      } catch {
        if (alive) setStatus(null);
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), STATUS_POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  // No camera configured — show a hint instead of a dead stream connection.
  if (!cameraUrl) {
    return (
      <div
        className="absolute right-3 top-3 z-20 w-52 rounded-lg border border-cardline
                   bg-[#131c2e]/90 px-3 py-2 text-[11px] text-inkmute backdrop-blur-sm"
      >
        <StatusDot live={false} label="No camera" />
        <div className="mt-1 leading-snug">
          Set the camera URL in <span className="text-ink">Machine&nbsp;config</span> to
          see the live stream.
        </div>
      </div>
    );
  }

  const live = status?.connected === true && status?.have_frame === true;
  const wh = status?.frame_wh;
  const aspect = wh && wh[1] > 0 ? wh[0] / wh[1] : 16 / 9;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Open full-size camera stream"
        className="group absolute right-3 top-3 z-20 w-52 overflow-hidden rounded-lg
                   border border-cardline bg-[#131c2e]/90 text-left shadow-lg
                   backdrop-blur-sm transition-colors hover:border-sky-400/60
                   focus:outline-none focus:ring-2 focus:ring-sky-400/50"
      >
        <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
          <StatusDot live={live} label="Camera" />
          <KindTag kind={kind} node={node} />
        </div>
        <div className="relative w-full">
          <StreamView turns={turns} variant="thumb" aspect={aspect} />
          {!live && (
            <div className="absolute inset-0 flex flex-col justify-center gap-1 px-2.5
                            text-[11px] text-inkmute">
              <StreamProblem status={status} />
            </div>
          )}
        </div>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          aria-describedby={undefined}
          className="w-[92vw] max-w-[1280px] p-0"
        >
          <div className="flex items-center justify-between border-b border-cardline px-4 py-2.5">
            <DialogTitle>Camera stream</DialogTitle>
            <div className="flex items-center gap-4">
              <span className="max-w-[360px] truncate font-mono text-[11px] text-inkmute"
                    title={status?.url ?? cameraUrl}>
                {CAMERA_KIND_LABEL[kind]} · {status?.url ?? cameraUrl}
              </span>
              <StatusDot live={live} label={live ? 'live' : 'offline'} />
              <DialogClose
                aria-label="Close"
                className="rounded px-2 py-0.5 text-[15px] leading-none text-inkmute
                           transition-colors hover:text-ink focus:outline-none
                           focus:ring-2 focus:ring-sky-400/50"
              >
                ✕
              </DialogClose>
            </div>
          </div>
          <div className="flex max-h-[80vh]">
            <div className="flex min-w-0 flex-1 flex-col items-center justify-center bg-black">
              {/* Mount the large stream only while open so the second upstream
                  connection is opened on demand and closed when the modal does. */}
              {open && <StreamView turns={turns} variant="full" aspect={aspect} />}
              {!live && (
                <div className="flex w-full flex-col gap-1 px-4 py-2 text-[12px] text-inkmute">
                  <StreamProblem status={status} />
                </div>
              )}
            </div>
            {kind === 'capture_node' && node && <NodePanel node={node} />}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
