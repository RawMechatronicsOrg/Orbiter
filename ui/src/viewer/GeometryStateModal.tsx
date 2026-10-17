/**
 * Geometry state modal — one screen with the geometry calibration (read-only
 * solver output + Reset), the overrides of the CAD-derived geometry, the
 * kinematic offsets and the render preferences.
 *
 * Commands: reset_calibration, set_machine_config, set_render_pref.
 */
import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '../components/ui/dialog';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { calibrationStatus } from './ui';

type ModelLike = Record<string, unknown>;

/** Compact readout of a solved value: numbers to 4 places, `{rvec, t}`
 *  transforms as two vectors, absent values as a dash. */
function fmtSolved(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'number') return v.toFixed(4);
  if (Array.isArray(v)) return `[${v.map((x) => fmtSolved(x)).join(', ')}]`;
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>;
    if ('rvec' in o || 't' in o) {
      return `rvec ${fmtSolved(o.rvec)} · t ${fmtSolved(o.t)}`;
    }
  }
  return String(v);
}

function ReadRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2 text-[12px]">
      <span className="w-44 shrink-0 font-mono text-inkdim">{label}</span>
      <span className="break-all font-mono text-[11px] text-zinc-200">{value}</span>
    </div>
  );
}

function Field({
  label,
  value,
  onCommit,
  step = '1',
  width = 'w-24',
  hint,
}: {
  label: string;
  value: number;
  onCommit: (v: number) => void;
  step?: string;
  width?: string;
  hint?: string;
}) {
  const [s, setS] = useState<string>(value.toString());
  useEffect(() => {
    setS(value.toString());
  }, [value]);
  return (
    <div className="flex items-center gap-2 text-[12px]">
      <span className="w-44 shrink-0 font-mono text-inkdim">{label}</span>
      <Input
        type="number"
        step={step}
        value={s}
        onChange={(e) => setS(e.target.value)}
        onBlur={() => {
          const v = parseFloat(s);
          if (Number.isFinite(v) && v !== value) onCommit(v);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
        }}
        className={`h-7 ${width} text-[11px]`}
      />
      {hint && <span className="text-[10px] text-inkmute">{hint}</span>}
    </div>
  );
}

function ToggleField({
  label,
  value,
  onCommit,
  hint,
}: {
  label: string;
  value: boolean;
  onCommit: (v: boolean) => void;
  hint?: string;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-[12px]">
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onCommit(e.target.checked)}
      />
      <span className="font-mono text-inkdim">{label}</span>
      {hint && <span className="text-[10px] text-inkmute">{hint}</span>}
    </label>
  );
}

function Section({
  title,
  children,
  hint,
}: {
  title: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="space-y-1.5 rounded border border-white/10 bg-black/20 p-3">
      <div className="flex items-baseline justify-between">
        <h3 className="text-[11px] uppercase tracking-wider text-inkmute">
          {title}
        </h3>
        {hint && <span className="text-[10px] text-inkmute">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

export function GeometryStateModal({
  open,
  onClose,
  commands,
}: {
  open: boolean;
  onClose: () => void;
  commands: Commands;
}) {
  const model = useViewerStore((s) => s.model) as ModelLike;

  const num = (k: string, d = 0) => {
    const v = model[k];
    return typeof v === 'number' ? v : d;
  };
  const bool = (k: string, d = false) => {
    const v = model[k];
    return typeof v === 'boolean' ? v : d;
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto p-0">
        <div className="sticky top-0 z-10 border-b border-white/10 bg-card px-4 py-3">
          <DialogTitle>Geometry state</DialogTitle>
          <p className="mt-1 text-[11px] text-inkdim">
            All live tunables in one place. Edit values commit on blur / Enter.
          </p>
        </div>

        <div className="space-y-3 p-4">
          {/* ── geometry calibration (read-only solver output) ── */}
          <Section
            title="Geometry calibration"
            hint={
              {
                none: 'not calibrated',
                current: 'matches this rig',
                stale: `stale: ${String(model.calibration_stale_reason ?? '')}`,
              }[calibrationStatus(model)]
            }
          >
            <ReadRow label="calibrated" value={String(model.calibrated === true)} />
            <ReadRow label="calib_rig_id" value={fmtSolved(model.calib_rig_id)} />
            <ReadRow label="rig_id (this server)" value={fmtSolved(model.rig_id)} />
            <ReadRow
              label="calib_encoder_zero"
              value={fmtSolved(
                model.calib_encoder_zero &&
                  Object.values(model.calib_encoder_zero as Record<string, number>),
              )}
            />
            <ReadRow
              label="encoder_zero (now)"
              value={fmtSolved(
                model.encoder_zero &&
                  Object.values(model.encoder_zero as Record<string, number>),
              )}
            />
            <ReadRow label="calib_extrinsic" value={fmtSolved(model.calib_extrinsic)} />
            <ReadRow label="calib_board_world" value={fmtSolved(model.calib_board_world)} />
            <ReadRow label="rocker_correction" value={fmtSolved(model.rocker_correction)} />
            <ReadRow label="el_encoder_scale" value={fmtSolved(model.el_encoder_scale)} />
            <ReadRow
              label="az_encoder_correction"
              value={fmtSolved(model.az_encoder_correction)}
            />
            <div className="pt-1">
              <Button
                variant="secondary"
                size="sm"
                disabled={model.calibrated !== true}
                onClick={() => {
                  if (
                    window.confirm(
                      'Reset the geometry calibration? The rig falls back to the ' +
                        'CAD model; camera intrinsics are kept.',
                    )
                  ) {
                    commands.resetCalibration();
                  }
                }}
              >
                Reset calibration
              </Button>
            </div>
          </Section>

          {/* ── overrides of the CAD-derived geometry ── */}
          <Section
            title="Geometry overrides"
            hint="Over the rig GLB; a calibration writes these"
          >
            <Field
              label="arm_radius_mm"
              value={num('arm_radius_mm')}
              onCommit={(v) => commands.setMachineConfig({ arm_radius_mm: v })}
              step="1"
              hint="0 = the CAD arm's reach"
            />
            <Field
              label="camera_offset_mm"
              value={num('camera_offset_mm')}
              onCommit={(v) => commands.setMachineConfig({ camera_offset_mm: v })}
              step="1"
            />
          </Section>

          {/* ── kinematic offsets ── */}
          <Section
            title="Kinematic offsets"
            hint="Operator-tunable corrections (degrees)"
          >
            <Field
              label="el_kinematic_offset_deg"
              value={num('el_kinematic_offset_deg')}
              onCommit={(v) =>
                commands.setMachineConfig({ el_kinematic_offset_deg: v })
              }
              step="0.5"
            />
            <Field
              label="az_kinematic_offset_deg"
              value={num('az_kinematic_offset_deg')}
              onCommit={(v) =>
                commands.setMachineConfig({ az_kinematic_offset_deg: v })
              }
              step="0.5"
            />
          </Section>

          {/* ── render preferences ── */}
          <Section title="Render preferences">
            <ToggleField
              label="show_axes"
              value={bool('show_axes', true)}
              onCommit={(v) => commands.setRenderPref({ show_axes: v })}
            />
            <ToggleField
              label="scan_preview"
              value={bool('scan_preview', false)}
              onCommit={(v) =>
                commands.setRenderPref({ scan_preview: v })
              }
            />
            <ToggleField
              label="hide_back_facing"
              value={bool('hide_back_facing', false)}
              onCommit={(v) =>
                commands.setRenderPref({ hide_back_facing: v })
              }
            />
            <ToggleField
              label="mirror_photo_on_frustum"
              value={bool('mirror_photo_on_frustum', false)}
              onCommit={(v) =>
                commands.setRenderPref({ mirror_photo_on_frustum: v })
              }
            />
          </Section>

          <div className="sticky bottom-0 -mx-4 border-t border-white/10 bg-card px-4 py-2 text-right">
            <Button variant="secondary" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
