/**
 * Collapsible bottom log strip. Shows the firmware log stream — lines arrive
 * over /ws/scene as `log` messages (firmware → storage-api → browser) and land
 * in `modelStore.log`.
 */

import { useState } from 'react';
import { useViewerStore } from './modelStore';

const LEVEL_COLOR: Record<string, string> = {
  I: 'text-inkdim',
  W: 'text-amber-400',
  E: 'text-red-400',
};

export function LogPanel() {
  const entries = useViewerStore((s) => s.log);
  const [open, setOpen] = useState(false);

  return (
    <div className="shrink-0 border-t border-barline bg-app font-mono text-[12px]">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-1 text-inkmute hover:text-inkdim"
      >
        <span className="w-3">{open ? '▾' : '▸'}</span>
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-inkdim">
          Log
        </span>
        <span className="ml-auto text-[10px]">{entries.length}</span>
      </button>
      {open && (
        <div className="h-[140px] overflow-y-auto px-3 pb-2">
          {entries.length === 0 ? (
            <div className="text-inkmute">no log entries</div>
          ) : (
            entries.slice(-300).map((e, i) => (
              <div key={i} className="flex gap-2 leading-5">
                <span className={LEVEL_COLOR[String(e.level)] ?? 'text-inkdim'}>
                  {String(e.level ?? 'I')}
                </span>
                {e.tag ? <span className="shrink-0 text-inkmute">{e.tag}</span> : null}
                <span className="text-ink">{e.msg}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
