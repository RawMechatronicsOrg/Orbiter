/** Left bar — scan session + MotionPlanner. */

import { useState, useEffect, useCallback } from 'react';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { cls, elTravel, parseNum, travelText } from './ui';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';

/** Action checkboxes — the two camera-only actions: [key, label, tooltip].
 *  `photo_flash` turns the phone LED on for the shot: IP Webcam's torch, or
 *  on the capture node a torch-on step of its locked profile (exposure, ISO
 *  and white balance stay as locked, so the lit frames come out brighter). */
const ACTIONS: ReadonlyArray<readonly [string, string, string]> = [
  ['photo', 'plain photo', 'one photo at each point'],
  ['photo_flash', 'photo + flash',
   'one photo with the phone LED on — the torch is switched on and settled ' +
   'before the shot and off after it; exposure stays as locked'],
];

const FIELD_LABELS: Record<string, string> = {
  el_start_deg: 'el start',
  el_max_deg: 'el max',
  el_steps: 'el steps',
  az_step_deg: 'az step',
};

export function LeftBar({ commands }: { commands: Commands }) {
  const model = useViewerStore((s) => s.model);

  // Form state for the MotionPlanner — initialised from the server plan once;
  // edits stay local until "Apply plan" sends them back.
  const initial = (
    useViewerStore.getState().model.motion_plan ?? {}
  ) as {
    discrete?: Record<string, unknown>;
  };
  const initialD = (initial.discrete ?? {}) as Record<string, unknown>;

  const [discrete, setDiscrete] = useState({
    el_start_deg: String(initialD.el_start_deg ?? 0),
    el_max_deg: String(initialD.el_max_deg ?? 60),
    el_steps: String(initialD.el_steps ?? 4),
    az_step_deg: String(initialD.az_step_deg ?? 20),
  });
  const [actions, setActions] = useState<string[]>(
    Array.isArray(initialD.actions)
      ? (initialD.actions as string[])
      : ['photo'],
  );

  const captureCount = Array.isArray(model.captures) ? model.captures.length : 0;

  const scanId =
    typeof model.current_scan_id === 'string' ? model.current_scan_id : null;
  const dirty = model.scan_dirty === true;
  const machine = model.machine_captured === true;
  const scanRunning = model.scan_running === true;
  const savedAtRaw =
    typeof model.scan_saved_at === 'string' ? model.scan_saved_at : '';
  const savedAt = savedAtRaw.length >= 19 ? savedAtRaw.slice(11, 19) : '';

  const onNew = () => {
    if (dirty && !window.confirm('Discard unsaved changes and start a new scan?')) {
      return;
    }
    commands.newScan();
  };

  const toggleAction = (key: string, on: boolean) => {
    setActions((prev) =>
      on
        ? prev.includes(key)
          ? prev
          : [...prev, key]
        : prev.filter((x) => x !== key),
    );
  };

  // Ring elevations must lie in the EL travel the server publishes — it
  // rejects a plan that reaches past it (models.DiscretePlan).
  const travel = elTravel(model);
  const fieldError = (k: keyof typeof discrete): string | null => {
    const v = parseNum(discrete[k]);
    if (v == null) return 'not a number';
    if (k === 'el_steps') return Number.isInteger(v) && v >= 0 ? null : 'whole number ≥ 0';
    if (k === 'az_step_deg') return v > 0 ? null : 'must be > 0';
    if (travel && (v < travel.min || v > travel.max)) return `outside ${travelText(travel)}`;
    return null;
  };
  const planValid = (Object.keys(discrete) as (keyof typeof discrete)[]).every(
    (k) => fieldError(k) == null,
  );

  /** The plan from the form, or null while a field is invalid. */
  const planFromForm = useCallback(() => {
    const v = (k: keyof typeof discrete) => parseNum(discrete[k]);
    const elStart = v('el_start_deg');
    const elMax = v('el_max_deg');
    const elSteps = v('el_steps');
    const azStep = v('az_step_deg');
    if (!planValid || elStart == null || elMax == null || elSteps == null || azStep == null) {
      return null;
    }
    return {
      mode: 'discrete' as const,
      discrete: {
        el_start_deg: elStart,
        el_max_deg: elMax,
        el_steps: elSteps,
        az_step_deg: azStep,
        actions: actions.length ? actions : ['photo'],
      },
    };
  }, [discrete, actions, planValid]);

  // Auto-apply the plan whenever a field changes (debounced) so the scan
  // preview rebuilds live — no manual "Apply" step. Start scan also carries
  // the plan, so a sweep always runs the latest values regardless of timing.
  // An invalid form is not sent; the field shows why.
  useEffect(() => {
    const plan = planFromForm();
    if (!plan) return undefined;
    const id = setTimeout(() => commands.setMotionPlan(plan), 400);
    return () => clearTimeout(id);
  }, [planFromForm, commands]);

  const onStartScan = () => {
    const plan = planFromForm();
    if (plan) commands.startScan(plan);
  };

  return (
    <div className={`${cls.bar} w-[270px]`}>
      {/* ── scan session ── */}
      <Card>
        <CardHeader>
          <CardTitle>Scan session</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="truncate font-mono text-[12px] text-inkdim">
            {scanId ?? <span className="text-inkmute">no active scan</span>}
          </div>
          <div className="flex items-center gap-2 text-[12px]">
            {dirty ? (
              <span className="text-amber-300">● unsaved</span>
            ) : scanId ? (
              <span className="text-emerald-400">
                ✓ saved{savedAt && ` ${savedAt}`}
              </span>
            ) : null}
            {scanId && (
              <span className="text-inkmute">{machine ? 'machine' : 'manual'}</span>
            )}
          </div>

          {/* captures */}
          <div className="flex items-center gap-2">
            <span className="font-mono text-[12px] text-inkmute">
              captures: {captureCount}
            </span>
            <Button variant="secondary" onClick={() => commands.takeShot()}>
              Take shot
            </Button>
          </div>

          {/* run actions */}
          <div className="flex gap-2">
            <Button
              variant={dirty ? 'default' : 'secondary'}
              onClick={() => commands.saveScan()}
              disabled={!scanId}
            >
              Save
            </Button>
            <Button variant="secondary" onClick={onNew}>
              New
            </Button>
          </div>
          <Button
            variant="secondary"
            onClick={() => commands.recreateScan()}
            disabled={!scanId}
          >
            Recreate &amp; Save
          </Button>
        </CardContent>
      </Card>

      {/* ── motion planner ── */}
      <Card>
        <CardTitle className="mb-2">Motion planner</CardTitle>

        <CardContent>
          {(['el_start_deg', 'el_max_deg', 'el_steps', 'az_step_deg'] as const).map(
            (k) => {
              const err = fieldError(k);
              return (
                <div key={k}>
                  <label className="flex items-center gap-2">
                    <span className="shrink-0 w-24 text-[13px] text-inkdim">
                      {FIELD_LABELS[k]}
                    </span>
                    <Input
                      value={discrete[k]}
                      aria-invalid={err != null}
                      className={err ? 'border-red-500/70' : undefined}
                      onChange={(e) => setDiscrete({ ...discrete, [k]: e.target.value })}
                    />
                  </label>
                  {err && <div className="pl-[6.5rem] text-[11px] text-red-300">{err}</div>}
                </div>
              );
            },
          )}
          {travel && (
            <div className="text-[11px] text-inkmute">EL travel {travelText(travel)}</div>
          )}

          <div className="mt-1 text-[12px] uppercase tracking-[0.12em] text-inkdim">
            Actions per point
          </div>
          {ACTIONS.map(([key, label, title]) => (
            <label key={key} className="flex items-center gap-2 text-[13px]" title={title}>
              <input
                type="checkbox"
                className={cls.check}
                checked={actions.includes(key)}
                onChange={(e) => toggleAction(key, e.target.checked)}
              />
              {label}
            </label>
          ))}
        </CardContent>

        {scanRunning ? (
          <Button
            variant="destructive"
            className="mt-2 w-full"
            onClick={() => commands.stopScan()}
          >
            ■ Stop scan
          </Button>
        ) : (
          <Button
            variant="success"
            className="mt-2 w-full"
            onClick={onStartScan}
            disabled={!actions.length || !planValid}
            title={planValid ? 'Apply the plan above and run the sweep' : 'Fix the plan fields first'}
          >
            ▶ Start scan
          </Button>
        )}

        <label className="mt-2 flex items-center gap-2 text-[12px]">
          <input
            type="checkbox"
            className={cls.check}
            checked={model.scan_preview === true}
            onChange={(e) => commands.setRenderPref({ scan_preview: e.target.checked })}
          />
          compute scan preview
        </label>
      </Card>
    </div>
  );
}
