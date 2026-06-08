/**
 * Geometry state modal — one screen with the live tunables (manual rig
 * geometry, kinematic offsets, render preferences). Read-only solver-output
 * sections (active_calibration, board_placement, etc.) were removed in v0.1
 * because the calibration pipeline isn't part of this slice yet.
 *
 * No new commands — uses the existing set_geometry / set_render_pref
 * dispatch.
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

type ModelLike = Record<string, unknown>;

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
          {/* ── manual rig geometry ── */}
          <Section title="Manual rig geometry">
            <Field
              label="arm_radius_mm"
              value={num('arm_radius_mm')}
              onCommit={(v) => commands.setGeometry({ arm_radius_mm: v })}
              step="1"
            />
            <Field
              label="camera_offset_mm"
              value={num('camera_offset_mm')}
              onCommit={(v) => commands.setGeometry({ camera_offset_mm: v })}
              step="1"
            />
            <Field
              label="base_height_mm"
              value={num('base_height_mm')}
              onCommit={(v) => commands.setGeometry({ base_height_mm: v })}
              step="1"
              hint="Disc top to AZ-EL axis (mm)"
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
                commands.setGeometry({ el_kinematic_offset_deg: v })
              }
              step="0.5"
            />
            <Field
              label="az_kinematic_offset_deg"
              value={num('az_kinematic_offset_deg')}
              onCommit={(v) =>
                commands.setGeometry({ az_kinematic_offset_deg: v })
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
