/**
 * Floating live-preview panel — does not block the 3D viewer.
 * Draggable, closable, reopenable while capture runs.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { useViewerStore } from './modelStore';
import { API_BASE } from './api';
import { Button } from '../components/ui/button';

interface LiveImages {
  off?: string | null;
  on?: string | null;
  line?: string | null;
  plane?: string | null;
}

interface LivePreview {
  active?: boolean;
  source?: string;
  phase?: string;
  title?: string;
  pair_index?: number;
  frame_id?: string;
  capture_id?: string;
  images?: LiveImages;
}

interface PanelPos {
  x: number;
  y: number;
}

const PREVIEW_LABELS: { key: keyof LiveImages; label: string }[] = [
  { key: 'off', label: 'Photo' },
];

const POS_KEY = 'orbiter.capturePreview.pos';
const PANEL_W = 320;

function loadSavedPos(): PanelPos {
  if (typeof window === 'undefined') return { x: 16, y: 72 };
  try {
    const raw = sessionStorage.getItem(POS_KEY);
    if (raw) {
      const p = JSON.parse(raw) as PanelPos;
      if (Number.isFinite(p.x) && Number.isFinite(p.y)) return p;
    }
  } catch {
    /* ignore */
  }
  return { x: Math.max(16, window.innerWidth - PANEL_W - 24), y: 72 };
}

function clampPos(pos: PanelPos): PanelPos {
  const maxX = Math.max(8, window.innerWidth - PANEL_W - 8);
  const maxY = Math.max(8, window.innerHeight - 120);
  return {
    x: Math.min(Math.max(8, pos.x), maxX),
    y: Math.min(Math.max(8, pos.y), maxY),
  };
}

function imageUrl(path: string | null | undefined, rev: string | number): string | null {
  if (!path) return null;
  const base = path.startsWith('http') ? path : `${API_BASE}${path}`;
  return `${base}${base.includes('?') ? '&' : '?'}v=${encodeURIComponent(String(rev))}`;
}

function mergeImages(prev: LiveImages, incoming: LiveImages | undefined): LiveImages {
  if (!incoming) return prev;
  return {
    off: incoming.off ?? prev.off,
    on: incoming.on ?? prev.on,
    line: incoming.line ?? prev.line,
    plane: incoming.plane ?? prev.plane,
  };
}

function PreviewThumb({
  label,
  src,
  imgKey,
  onExpand,
}: {
  label: string;
  src: string;
  imgKey: string;
  onExpand: () => void;
}) {
  const [attempt, setAttempt] = useState(0);
  const srcWithRetry = `${src}${src.includes('?') ? '&' : '?'}r=${attempt}`;

  return (
    <button
      type="button"
      onClick={onExpand}
      className="group flex flex-col items-stretch overflow-hidden rounded border border-white/10 bg-black/40 text-left hover:border-sky-500/50"
      title={`${label} — click to enlarge`}
    >
      <div className="px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-inkmute">
        {label}
      </div>
      <img
        key={`${imgKey}-${attempt}`}
        src={srcWithRetry}
        alt={label}
        className="h-20 w-full object-contain bg-black/60"
        onError={() => {
          if (attempt < 3) {
            window.setTimeout(() => setAttempt((a) => a + 1), 280);
          }
        }}
      />
    </button>
  );
}

export function CaptureProgressModal() {
  const livePreview = useViewerStore((s) => s.model.live_preview) as LivePreview | null | undefined;
  const scanRunning = useViewerStore((s) => s.model.scan_running === true);

  const live = livePreview ?? {};

  const [panelOpen, setPanelOpen] = useState(true);
  const [pos, setPos] = useState<PanelPos>(loadSavedPos);
  const [expanded, setExpanded] = useState<{ label: string; src: string } | null>(null);
  const [stickyImages, setStickyImages] = useState<LiveImages>({});

  const frameRev =
    live.frame_id
    ?? (live.pair_index != null ? `p${live.pair_index}` : null)
    ?? live.capture_id
    ?? live.title
    ?? '0';

  const displayImages = useMemo(
    () => mergeImages(stickyImages, live.images),
    [stickyImages, live.images],
  );

  useEffect(() => {
    if (live.images && (live.images.off || live.images.on || live.images.line || live.images.plane)) {
      setStickyImages((prev) => mergeImages(prev, live.images));
    }
  }, [live.images, frameRev]);

  useEffect(() => {
    if (!scanRunning) {
      setStickyImages({});
    }
  }, [scanRunning]);

  const isDonePhase = live.phase === 'done' || live.phase === 'failed';
  const isRunning = scanRunning;

  const [showDoneHint, setShowDoneHint] = useState(false);
  useEffect(() => {
    if (!isDonePhase) {
      setShowDoneHint(false);
      return undefined;
    }
    setShowDoneHint(true);
    const t = window.setTimeout(() => setShowDoneHint(false), 4500);
    return () => window.clearTimeout(t);
  }, [isDonePhase, live.phase, live.title]);

  const hasPreviewData = Boolean(
    displayImages.off || displayImages.on || displayImages.line || displayImages.plane,
  );

  const canShowReopen =
    (isRunning || hasPreviewData || showDoneHint) && !panelOpen;

  const prevRunning = useRef(false);
  useEffect(() => {
    if (isRunning && !prevRunning.current) {
      setPanelOpen(true);
    }
    prevRunning.current = isRunning;
  }, [isRunning]);

  useEffect(() => {
    if (showDoneHint) setPanelOpen(true);
  }, [showDoneHint]);

  useEffect(() => {
    sessionStorage.setItem(POS_KEY, JSON.stringify(pos));
  }, [pos]);

  const drag = useRef({
    active: false,
    startX: 0,
    startY: 0,
    origX: 0,
    origY: 0,
  });

  const onDragStart = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('button')) return;
    drag.current = {
      active: true,
      startX: e.clientX,
      startY: e.clientY,
      origX: pos.x,
      origY: pos.y,
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  }, [pos.x, pos.y]);

  const onDragMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current.active) return;
    setPos(clampPos({
      x: drag.current.origX + (e.clientX - drag.current.startX),
      y: drag.current.origY + (e.clientY - drag.current.startY),
    }));
  }, []);

  const onDragEnd = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    drag.current.active = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
  }, []);

  const title = live.title
    || (scanRunning ? 'Scan capture' : 'Live preview');

  const subtitle = 'Click a thumbnail to enlarge';

  if (!isRunning && !hasPreviewData && !showDoneHint && !panelOpen) {
    return null;
  }

  return (
    <>
      {canShowReopen && (
        <button
          type="button"
          onClick={() => setPanelOpen(true)}
          className="fixed bottom-20 right-4 z-[45] rounded-full border border-sky-500/40 bg-stage/95 px-3 py-1.5 text-[11px] font-medium text-sky-200 shadow-lg backdrop-blur-sm hover:border-sky-400"
          title="Open live capture preview"
        >
          Live preview
        </button>
      )}

      {panelOpen && (isRunning || hasPreviewData || showDoneHint) && (
        <div
          className="fixed z-[45] w-[320px] rounded-lg border border-white/15 bg-stage/95 shadow-2xl backdrop-blur-sm"
          style={{ left: pos.x, top: pos.y }}
          role="dialog"
          aria-label="Live capture preview"
        >
          <div
            className="flex cursor-grab touch-none items-center gap-2 border-b border-white/10 px-2 py-1.5 active:cursor-grabbing"
            onPointerDown={onDragStart}
            onPointerMove={onDragMove}
            onPointerUp={onDragEnd}
            onPointerCancel={onDragEnd}
          >
            <span className="min-w-0 flex-1 truncate text-[12px] font-medium text-ink">
              {title}
            </span>
            <Button
              type="button"
              variant="ghost"
              size="xs"
              className="shrink-0 px-1.5 text-inkmute hover:text-ink"
              onClick={() => setPanelOpen(false)}
              title="Close panel"
            >
              ×
            </Button>
          </div>

          <div className="p-3">
            {isDonePhase && (
              <div
                className={`mb-2 text-[11px] ${
                  live.phase === 'failed' ? 'text-rose-300' : 'text-emerald-300'
                }`}
              >
                {live.phase === 'failed'
                  ? 'Job failed — see firmware log'
                  : 'Done'}
              </div>
            )}
            <p className="mb-2 text-[11px] text-inkmute">{subtitle}</p>
            <div className="grid grid-cols-2 gap-2">
              {PREVIEW_LABELS.map(({ key, label }) => {
                const src = imageUrl(displayImages[key], frameRev);
                if (!src) {
                  return (
                    <div
                      key={key}
                      className="flex h-[6.5rem] items-center justify-center rounded border border-dashed border-white/10 text-[10px] text-inkmute"
                    >
                      {label}
                      <span className="ml-1 opacity-50">—</span>
                    </div>
                  );
                }
                return (
                  <PreviewThumb
                    key={key}
                    label={label}
                    src={src}
                    imgKey={`${key}-${frameRev}`}
                    onExpand={() => setExpanded({ label, src })}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}

      {expanded && (
        <div className="pointer-events-none fixed inset-0 z-[46] flex items-center justify-center p-6">
          <div className="pointer-events-auto max-h-[85vh] max-w-4xl rounded-lg border border-white/15 bg-stage/98 p-3 shadow-2xl">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-sm text-ink">{expanded.label}</span>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                onClick={() => setExpanded(null)}
              >
                Close
              </Button>
            </div>
            <img
              src={expanded.src}
              alt={expanded.label}
              className="max-h-[75vh] w-full object-contain"
            />
          </div>
        </div>
      )}
    </>
  );
}
