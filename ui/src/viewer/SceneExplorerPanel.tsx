/**
 * Floating wrapper around <SceneExplorer/>. Lives over the 3D viewer area
 * (sibling of PositionOverlay) so the right column is free for other
 * panels.
 *
 *   - resizable: drag the bottom-right corner handle
 *   - collapsible: header chevron collapses the panel into a small tab
 *     pinned to the top-left of the viewer; same button re-expands.
 *   - layout persisted to localStorage so width/height/collapsed state
 *     stick between sessions.
 */

import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { SceneExplorer } from './SceneExplorer';

const STORAGE_KEY = 'orbiter.scene-explorer.layout';

interface Layout {
  width: number;
  height: number;
  collapsed: boolean;
}

const DEFAULTS: Layout = { width: 340, height: 480, collapsed: false };
const MIN_W = 220;
const MIN_H = 180;
const MAX_W = 900;
const MAX_H = 1200;

function loadLayout(): Layout {
  if (typeof window === 'undefined') return DEFAULTS;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Partial<Layout>;
    return {
      width: clamp(parsed.width ?? DEFAULTS.width, MIN_W, MAX_W),
      height: clamp(parsed.height ?? DEFAULTS.height, MIN_H, MAX_H),
      collapsed: parsed.collapsed === true,
    };
  } catch {
    return DEFAULTS;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}

export function SceneExplorerPanel() {
  const [layout, setLayout] = useState<Layout>(loadLayout);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
    } catch {
      /* ignore quota errors */
    }
  }, [layout]);

  const dragRef = useRef<{ startX: number; startY: number; startW: number; startH: number } | null>(null);

  const onResizeStart = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startW: layout.width,
        startH: layout.height,
      };
      const target = e.currentTarget;
      try {
        target.setPointerCapture(e.pointerId);
      } catch {
        /* not supported — window listener still runs */
      }

      const onMove = (ev: PointerEvent) => {
        const d = dragRef.current;
        if (!d) return;
        setLayout((prev) => ({
          ...prev,
          width: clamp(d.startW + (ev.clientX - d.startX), MIN_W, MAX_W),
          height: clamp(d.startH + (ev.clientY - d.startY), MIN_H, MAX_H),
        }));
      };
      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    },
    [layout.width, layout.height],
  );

  const toggleCollapsed = useCallback(() => {
    setLayout((p) => ({ ...p, collapsed: !p.collapsed }));
  }, []);

  if (layout.collapsed) {
    return (
      <button
        onClick={toggleCollapsed}
        title="Expand scene explorer"
        className="absolute left-3 top-3 z-10 flex items-center gap-1.5
                   rounded-md border border-cardline bg-[#131c2e]/90
                   px-2.5 py-1.5 text-[11px] uppercase tracking-[0.12em]
                   text-inkdim backdrop-blur-sm hover:text-ink"
      >
        <span>▸</span>
        <span>Scene</span>
      </button>
    );
  }

  return (
    <div
      className="absolute left-3 top-3 z-10 flex flex-col overflow-hidden
                 rounded-lg border border-cardline bg-[#131c2e]/95
                 shadow-xl backdrop-blur-sm"
      style={{ width: layout.width, height: layout.height }}
    >
      <div className="flex items-center gap-2 border-b border-cardline px-2.5 py-1.5">
        <button
          onClick={toggleCollapsed}
          title="Collapse"
          className="text-inkmute hover:text-ink"
        >
          ▾
        </button>
        <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-inkdim">
          Scene explorer
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-2">
        <SceneExplorer />
      </div>
      {/* Resize handle — bottom-right corner */}
      <div
        role="separator"
        aria-label="Resize scene explorer"
        title="Drag to resize"
        onPointerDown={onResizeStart}
        className="absolute bottom-0 right-0 h-3.5 w-3.5 cursor-nwse-resize
                   text-inkmute hover:text-ink"
        style={{
          background:
            'linear-gradient(135deg, transparent 0 50%, currentColor 50% 60%, transparent 60% 70%, currentColor 70% 80%, transparent 80%)',
        }}
      />
    </div>
  );
}
