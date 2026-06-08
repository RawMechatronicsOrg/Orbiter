/** Shared Tailwind class strings + the model-field helper for the viewer bars. */

/** Read a numeric field from the mirrored server model. */
export const num = (m: Record<string, unknown>, k: string, d = 0): number => {
  const v = m[k];
  return typeof v === 'number' ? v : d;
};

/** Tailwind class strings for the recurring viewer UI pieces.
 *  Buttons live in `components/ui/button.tsx` now — keep this file to shared
 *  bar / card / field utilities. */
export const cls = {
  // Width is now set by callers (LeftBar = fixed, RightBar = resizable),
  // so the sidebar template stops imposing one. Keep the flex/scroll bits.
  bar: 'shrink-0 h-full overflow-y-auto p-3 bg-stage flex flex-col gap-2.5',
  topbar:
    'h-10 shrink-0 flex items-center gap-4 px-4 bg-app border-b border-barline ' +
    'text-[13px] text-inkdim',
  card: 'bg-card border border-cardline rounded-lg px-3.5 py-3',
  label:
    'text-[11px] uppercase tracking-[0.18em] text-inkmute font-semibold mb-2',
  col: 'flex flex-col gap-2',
  row: 'flex gap-2 items-center',
  fieldName: 'shrink-0 w-24 text-[12px] text-inkmute',
  input:
    'w-full bg-field border border-fieldline rounded-md px-2.5 py-1 ' +
    'text-[13px] text-zinc-100 font-mono ' +
    'focus:outline-1 focus:outline-accent focus:border-accent/40',
  check: 'accent-accent',
} as const;
