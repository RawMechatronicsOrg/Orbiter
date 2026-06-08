/**
 * Right bar — Machine Config. Scene Explorer moved to a floating overlay
 * (SceneExplorerPanel) so the right column is free for upcoming panels.
 *
 * Resizable: drag the left edge to widen/narrow. Width is persisted to
 * localStorage so the layout sticks between sessions.
 */

import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import type { Commands } from './commands';
import { MachineConfig } from './MachineConfig';
import { cls } from './ui';

const WIDTH_KEY = 'orbiter.rightbar.width';
const DEFAULT_WIDTH = 380;
const MIN_WIDTH = 280;
const MAX_WIDTH = 1200;

function loadSavedWidth(): number {
  if (typeof window === 'undefined') return DEFAULT_WIDTH;
  try {
    const raw = window.localStorage.getItem(WIDTH_KEY);
    if (raw) {
      const n = parseInt(raw, 10);
      if (Number.isFinite(n) && n >= MIN_WIDTH && n <= MAX_WIDTH) return n;
    }
  } catch {
    /* ignore — fall back to default */
  }
  return DEFAULT_WIDTH;
}

export function RightBar({ commands }: { commands: Commands }) {
  const [width, setWidth] = useState<number>(loadSavedWidth);

  // Persist on every commit — cheap (<1µs to localStorage) and means a
  // mid-drag reload (e.g. HMR) keeps the user's intended width.
  useEffect(() => {
    try {
      window.localStorage.setItem(WIDTH_KEY, String(width));
    } catch {
      /* ignore quota errors */
    }
  }, [width]);

  // Drag handle — captures pointer + listens on window so the drag keeps
  // working when the cursor briefly leaves the 8 px handle strip.
  const dragRef = useRef<{ startX: number; startW: number } | null>(null);

  const onDragStart = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      dragRef.current = { startX: e.clientX, startW: width };
      const target = e.currentTarget;
      try {
        target.setPointerCapture(e.pointerId);
      } catch {
        /* old browsers — fall through; window listener still works */
      }

      const onMove = (ev: PointerEvent) => {
        const d = dragRef.current;
        if (!d) return;
        // Bar is on the RIGHT edge — dragging left (negative dx) widens it.
        const next = d.startW + (d.startX - ev.clientX);
        setWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, Math.round(next))));
      };
      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    },
    [width],
  );

  // Double-click handle to reset to default — quick escape if the user
  // dragged it past the visible area on a smaller screen.
  const onHandleDoubleClick = useCallback(() => {
    setWidth(DEFAULT_WIDTH);
  }, []);

  return (
    <div className="relative h-full shrink-0" style={{ width }}>
      <div
        role="separator"
        aria-label="Resize right sidebar"
        aria-orientation="vertical"
        title="Drag to resize · double-click to reset"
        onPointerDown={onDragStart}
        onDoubleClick={onHandleDoubleClick}
        className={
          'absolute left-0 top-0 z-10 h-full w-1.5 cursor-ew-resize ' +
          'bg-transparent hover:bg-accent/40 active:bg-accent/60 ' +
          'transition-colors'
        }
      />
      <div className={cls.bar}>
        <MachineConfig commands={commands} />
      </div>
    </div>
  );
}
