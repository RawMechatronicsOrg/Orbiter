/**
 * Library tab — a browser over stored scans.
 *
 *   Each scan row exposes "Export SfM priors" (POST /scans/{sid}/sfm_priors);
 *   the per-scan archive download bundles those priors in too. Photo preview is
 *   a Radix Dialog. The scan-list supports sort, flags empty / suspicious
 *   entries and offers per-entity deletion + downloads. (The machine-config
 *   JSON readout used to live here as a second tab — removed; config lives in
 *   the right bar.)
 */

import { useMemo, useState } from 'react';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { cls } from './ui';
import { Button } from '../components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '../components/ui/dialog';
import { API_BASE, del, postJson } from './api';

const LABEL = 'text-[13px] uppercase tracking-[0.14em] text-ink font-semibold';
const SORT_SEL =
  'shrink-0 bg-field border border-fieldline rounded-md px-1.5 py-1 ' +
  'text-[12px] text-inkdim focus:outline-2 focus:outline-accent';
/** Library list rail — twice the width of the Scaner bars, so rows fit on
 *  a single compact line. */
const LIB_BAR =
  'w-[540px] shrink-0 h-full overflow-y-auto p-3 bg-stage flex flex-col gap-3';



/** Save a blob to disk under `filename` via a transient <a download>. */
function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Fetch a file and save it (cross-origin friendly — controls the filename). */
async function downloadFile(url: string, filename: string): Promise<void> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download: HTTP ${r.status}`);
  triggerDownload(await r.blob(), filename);
}


/** Server timestamps are UTC ISO-8601. Render uniformly in local time. */
function formatTime(iso: unknown): string {
  if (typeof iso !== 'string' || !iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}`
  );
}


/** Sort `<select>` shared by the list sections. */
function SortSelect({
  sort,
  onSort,
  sorts,
}: {
  sort: string;
  onSort: (s: string) => void;
  sorts: ReadonlyArray<readonly [string, string]>;
}) {
  return (
    <select
      value={sort}
      onChange={(e) => onSort(e.target.value)}
      className={SORT_SEL}
    >
      {sorts.map(([k, l]) => (
        <option key={k} value={k}>
          {l}
        </option>
      ))}
    </select>
  );
}

/** One compact, single-line list row — selectable + deletable, with an
 *  amber flag for empty / suspicious entries. */
function EntityRow({
  id,
  sub,
  selected,
  suspicious,
  onSelect,
  onDelete,
}: {
  id: string;
  sub: string;
  selected: boolean;
  suspicious: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const border = selected
    ? 'border-accent bg-accent/15'
    : suspicious
      ? 'border-amber-500/80 bg-amber-500/10'
      : 'border-cardline bg-field hover:border-[#3a4862]';
  return (
    <div
      className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 transition-colors ${border}`}
    >
      {suspicious && (
        <span title="empty" className="shrink-0 text-[13px] text-amber-400">
          ⚠
        </span>
      )}
      <button
        onClick={onSelect}
        className="flex min-w-0 flex-1 items-baseline gap-2.5 text-left"
      >
        <span
          className={
            'shrink-0 font-mono text-[13px] ' +
            (selected ? 'text-sky-100' : 'text-inkdim')
          }
        >
          {id}
        </span>
        <span className="truncate text-[12px] text-inkmute">{sub}</span>
      </button>
      <Button
        variant="ghost"
        size="icon"
        onClick={onDelete}
        title="delete"
        className="h-6 w-6 hover:text-red-300"
      >
        ×
      </Button>
    </div>
  );
}


// ── scans ───────────────────────────────────────────────────────────────────

interface ScanSummary {
  scan_id: string;
  created: string;
  captures_count: number;
  archived?: boolean;
}

interface LoadedCapture {
  capture_id: string;
  thumb_small_url?: string;
  thumb_url?: string;
  full_url?: string;
  az_deg?: number;
  el_deg?: number;
  index?: number;
  timestamp?: string;
  camera_preset?: string;
  stored_width?: number;
  stored_height?: number;
}

/** Full-size photo preview — metadata + download original.
 *
 *  v0.1 SCOPE: a SAVED scan has no per-frame delete — delete the WHOLE scan
 *  from the list instead. (Per-frame delete exists only for the ACTIVE
 *  recording session, via the in-scene `CaptureModal` → WS `delete_capture`,
 *  which mutates `model.captures`. These are LOADED captures from a stored
 *  manifest — a separate path — so per-frame delete here is intentionally
 *  left out of v0.1.) */
function PhotoModal({
  capture,
  open,
  onClose,
}: {
  capture: LoadedCapture | null;
  open: boolean;
  onClose: () => void;
}) {
  if (!capture) return null;
  const fullUrl = API_BASE + (capture.full_url ?? capture.thumb_url ?? '');
  const meta: ReadonlyArray<readonly [string, string]> = [
    ['index', String(capture.index ?? '—')],
    [
      'az / el',
      `${(capture.az_deg ?? 0).toFixed(1)}° / ${(capture.el_deg ?? 0).toFixed(1)}°`,
    ],
    ['timestamp', formatTime(capture.timestamp)],
    ['camera', String(capture.camera_preset ?? '—')],
    [
      'stored size',
      capture.stored_width && capture.stored_height
        ? `${capture.stored_width} × ${capture.stored_height}`
        : '—',
    ],
    ['capture id', capture.capture_id],
  ];

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="flex max-h-[88vh] max-w-none p-0">
        <div className="flex items-center justify-center bg-app p-3">
          <img
            src={fullUrl}
            alt={capture.capture_id}
            className="max-h-[84vh] max-w-[66vw] object-contain"
          />
        </div>
        <div className="flex w-[300px] shrink-0 flex-col gap-3 overflow-y-auto p-4">
          <DialogTitle>Capture</DialogTitle>
          <div className="flex flex-col gap-1.5">
            {meta.map(([k, v]) => (
              <div key={k} className="flex gap-2 text-[13px]">
                <span className="w-28 shrink-0 text-inkmute">{k}</span>
                <span className="break-all font-mono text-inkdim">{v}</span>
              </div>
            ))}
          </div>
          <div className="mt-auto flex gap-2 pt-2">
            <Button
              onClick={() =>
                downloadFile(fullUrl, `${capture.capture_id}.jpg`).catch((e) =>
                  window.alert(String(e)),
                )
              }
            >
              Download original
            </Button>
            <Button variant="secondary" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

const SCAN_SORTS = [
  ['new', 'newest'],
  ['old', 'oldest'],
  ['shots-hi', 'most shots'],
  ['shots-lo', 'fewest shots'],
  ['id', 'name'],
] as const;

function scanCmp(key: string): (a: ScanSummary, b: ScanSummary) => number {
  return (a, b) => {
    switch (key) {
      case 'old':
        return a.created.localeCompare(b.created);
      case 'shots-hi':
        return b.captures_count - a.captures_count;
      case 'shots-lo':
        return a.captures_count - b.captures_count;
      case 'id':
        return a.scan_id.localeCompare(b.scan_id);
      default:
        return b.created.localeCompare(a.created);
    }
  };
}

/** Server response from POST /scans/{sid}/sfm_priors. Shape mirrors what
 *  the storage-api returns; we only surface `path` to the operator. */
interface SfmPriorsResp {
  path?: string;
  ok?: boolean;
  detail?: string;
}

function ScansSection({ commands }: { commands: Commands }) {
  const model = useViewerStore((s) => s.model);
  const scans = (Array.isArray(model.scans) ? model.scans : []) as ScanSummary[];
  const loadedId =
    typeof model.loaded_scan_id === 'string' ? model.loaded_scan_id : null;
  const captures = (
    Array.isArray(model.loaded_captures) ? model.loaded_captures : []
  ) as LoadedCapture[];

  const [sort, setSort] = useState('new');
  const [modal, setModal] = useState<LoadedCapture | null>(null);
  // Last SfM-priors export per scan_id — surfaces the path the server wrote
  // to so the operator can find / copy it.
  const [priorsBusy, setPriorsBusy] = useState<string | null>(null);
  const [priorsResult, setPriorsResult] = useState<Record<string, string>>({});

  const shown = useMemo(() => [...scans].sort(scanCmp(sort)), [scans, sort]);

  const onDelete = (id: string) => {
    if (!window.confirm(`Delete scan ${id}? The manifest is removed (photos kept).`)) {
      return;
    }
    del(`/scans/${id}`)
      .then(() => {
        if (id === loadedId) {
          commands.raw('set_active_session', { scan_id: null });
        }
      })
      .catch((e) => window.alert(String(e)));
  };

  const onExportPriors = (id: string) => {
    setPriorsBusy(id);
    postJson<SfmPriorsResp>(`/scans/${id}/sfm_priors`, {})
      .then((r) => {
        const path = r.path ?? 'priors written';
        setPriorsResult((prev) => ({ ...prev, [id]: path }));
      })
      .catch((e) => window.alert(String(e)))
      .finally(() => setPriorsBusy(null));
  };

  return (
    <>
      <div className={LIB_BAR}>
        <div className={cls.card}>
          <div className="mb-2 flex items-center gap-2">
            <span className={`${LABEL} flex-1`}>Scans · {scans.length}</span>
            <SortSelect sort={sort} onSort={setSort} sorts={SCAN_SORTS} />
          </div>
          <div className="flex flex-col gap-1.5">
            {shown.length === 0 ? (
              <span className="text-[13px] text-inkmute">no scans</span>
            ) : (
              shown.map((s) => (
                <div key={s.scan_id} className="flex flex-col gap-1">
                  <EntityRow
                    id={s.scan_id}
                    selected={s.scan_id === loadedId}
                    suspicious={s.captures_count === 0}
                    onSelect={() =>
                      // Toggle: click a loaded scan again to UNLOAD it (clears
                      // the yellow review frustums from the 3D scene).
                      commands.raw('set_active_session', {
                        scan_id: s.scan_id === loadedId ? null : s.scan_id,
                      })
                    }
                    onDelete={() => onDelete(s.scan_id)}
                    sub={
                      `${formatTime(s.created)} · ${s.captures_count} shots` +
                      (s.archived ? ' · archived' : '')
                    }
                  />
                  {/* Pipeline buttons — surface the path on success so the
                      operator can copy/paste it into their COLMAP CLI. */}
                  <div className="flex items-center gap-2 pl-2">
                    <Button
                      variant="outline"
                      size="xs"
                      disabled={priorsBusy === s.scan_id || s.captures_count === 0}
                      onClick={() => onExportPriors(s.scan_id)}
                      title="POST /scans/{sid}/sfm_priors — writes a JSON sidecar with per-capture poses for COLMAP"
                    >
                      {priorsBusy === s.scan_id ? 'Exporting…' : 'Export SfM priors'}
                    </Button>
                    {priorsResult[s.scan_id] && (
                      <span
                        className="truncate font-mono text-[11px] text-emerald-300"
                        title={priorsResult[s.scan_id]}
                      >
                        → {priorsResult[s.scan_id]}
                      </span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="min-w-0 flex-1 overflow-y-auto bg-stage p-3">
        <div className={cls.card}>
          <div className="mb-3 flex items-center justify-between">
            <span className={LABEL}>
              {loadedId ? `Captures · ${loadedId}` : 'Captures'}
            </span>
            {loadedId && (
              <div className="flex gap-2">
                <Button variant="outline" size="xs" asChild>
                  <a href={`${API_BASE}/scans/${loadedId}/download`}>
                    Download archive (+SfM)
                  </a>
                </Button>
                <Button
                  variant="outline"
                  size="xs"
                  onClick={() =>
                    commands.raw('set_active_session', { scan_id: null })
                  }
                >
                  Unload
                </Button>
              </div>
            )}
          </div>
          {!loadedId ? (
            <span className="text-[13px] text-inkmute">
              select a scan to load its captures
            </span>
          ) : captures.length === 0 ? (
            <span className="text-[13px] text-amber-300">
              ⚠ this scan has no captures
            </span>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-3">
              {captures.map((c) => {
                const thumb = c.thumb_url ?? c.thumb_small_url;
                return (
                  <button
                    key={c.capture_id}
                    onClick={() => setModal(c)}
                    className="overflow-hidden rounded-lg border border-cardline bg-field text-left transition-colors hover:border-accent"
                  >
                    {thumb ? (
                      <img
                        src={API_BASE + thumb}
                        alt={c.capture_id}
                        loading="lazy"
                        decoding="async"
                        className="block aspect-square w-full bg-app object-cover"
                      />
                    ) : (
                      <div className="aspect-square w-full bg-app" />
                    )}
                    <div className="px-2 py-1.5 font-mono text-[12px] text-inkdim">
                      az {(c.az_deg ?? 0).toFixed(0)}° · el{' '}
                      {(c.el_deg ?? 0).toFixed(0)}°
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <PhotoModal
        capture={modal}
        open={modal !== null}
        onClose={() => setModal(null)}
      />
    </>
  );
}

// ── shell ───────────────────────────────────────────────────────────────────

export function LibraryView({ commands }: { commands: Commands }) {
  // Single Scans browser now — the machine-config JSON tab was removed.
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <ScansSection commands={commands} />
    </div>
  );
}
