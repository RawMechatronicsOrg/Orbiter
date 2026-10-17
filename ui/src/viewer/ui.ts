/** Shared Tailwind class strings + the model-field helper for the viewer bars. */

/** Read a numeric field from the mirrored server model. */
export const num = (m: Record<string, unknown>, k: string, d = 0): number => {
  const v = m[k];
  return typeof v === 'number' ? v : d;
};

/** Per-axis drive facts the server publishes as `model.axes` (drive.py):
 *  steps per degree for °/s readouts (measured by the firmware's motor
 *  calibration when the axis has one), the speed-knob band (Hz), and for EL
 *  the travel limits. */
export interface AxisInfo {
  steps_per_deg: number;
  steps_per_deg_nominal: number;
  steps_per_deg_source: string;
  /** null = the firmware reports no motor calibration at all. */
  motor_calibrated: boolean | null;
  /** Measured top speed (°/s), when motor-calibrated. */
  max_speed_dps: number | null;
  hz_min: number;
  hz_max: number;
  hz_default: number;
  min_deg?: number;
  max_deg?: number;
}

/** Why closed-loop moves on `axis` are refused right now (the server's
 *  esp_proxy.move rule), or null when they are allowed. */
export function motorBlock(m: Record<string, unknown>, axis: 'az' | 'el'): string | null {
  if ((m.motor_cal as { active?: boolean } | null | undefined)?.active === true) {
    return 'a motor calibration is running';
  }
  return axisInfo(m, axis)?.motor_calibrated === false
    ? `${axis.toUpperCase()} is not motor-calibrated (Machine config → Step 1)`
    : null;
}

/** `model.axes[axis]`, or null before the server has sent it. */
export function axisInfo(m: Record<string, unknown>, axis: 'az' | 'el'): AxisInfo | null {
  const axes = m.axes as Record<string, AxisInfo> | undefined;
  const a = axes?.[axis];
  return a && typeof a.steps_per_deg === 'number' ? a : null;
}

/** EL travel from `model.axes.el` (the firmware's limits, geom/angles.py),
 *  or null before the server has sent it. AZ turns without limit. */
export function elTravel(m: Record<string, unknown>): { min: number; max: number } | null {
  const el = axisInfo(m, 'el');
  return el && typeof el.min_deg === 'number' && typeof el.max_deg === 'number'
    ? { min: el.min_deg, max: el.max_deg }
    : null;
}

/** `−25…+90°` — a travel range with typographic signs. */
export function travelText(t: { min: number; max: number }): string {
  const s = (v: number) => (v < 0 ? `−${-v}` : `+${v}`);
  return `${s(t.min)}…${s(t.max)}°`;
}

/** Strict parse of a numeric text field: blank or junk → null (never NaN). */
export function parseNum(s: string): number | null {
  const t = s.trim().replace('−', '-');
  if (t === '') return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Output speed for a driver step rate, e.g. `34.4 °/s`. */
export function degPerS(hz: number, a: AxisInfo): string {
  const v = hz / a.steps_per_deg;
  return `${v < 10 ? v.toFixed(2) : v.toFixed(1)} °/s`;
}

/** Geometry-calibration state, derived by the server
 *  (`model.calibration_status`, reason in `calibration_stale_reason`):
 *  `stale` = solved on other mechanics (another rig GLB), before calibrations
 *  were stamped, or with other encoder zeros. */
export type CalibrationStatus = 'none' | 'current' | 'stale';

export function calibrationStatus(m: Record<string, unknown>): CalibrationStatus {
  const s = m.calibration_status;
  return s === 'current' || s === 'stale' ? s : 'none';
}

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
