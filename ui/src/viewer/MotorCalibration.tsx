/**
 * Step 1 · Motor calibration — the firmware's self-calibration of each axis
 * (`POST /motor/calibrate`, progress on `GET /motor/calibration`, through the
 * server's `/firmware/motor`). It measures steps per degree, which direction
 * raises the angle, the top speed and an angle-deviation profile, turning the
 * axis full circle in every phase — elevation therefore only with the arm off.
 *
 * Until an axis has a result the firmware drives closed-loop moves with a
 * direction assumption the bench measured wrong; the server refuses moves on
 * such an axis, and the position controls say so.
 */

import { useCallback, useEffect, useState } from 'react';
import { getJson } from './api';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardTitle } from '../components/ui/card';

interface MotorAxis {
  valid: boolean;
  steps_per_deg: number;
  steps_per_deg_coarse?: number;
  max_step_hz?: number;
  max_speed_dps: number;
  dir_sign: number;
  revolutions?: number;
  compensation?: boolean;
  has_profile?: boolean;
  profile_peak_deg?: number;
  profile_rms_deg?: number;
}

interface MotorReport {
  run: {
    active: boolean;
    axis: 'az' | 'el';
    phase: string;
    progress: number;
    speed_dps: number;
    message: string;
  };
  az: MotorAxis;
  el: MotorAxis;
}

const AXES = [
  ['az', 'AZ'],
  ['el', 'EL'],
] as const;

export function MotorCalibration({ commands }: { commands: Commands }) {
  const espOnline = useViewerStore((s) => s.model.esp_online === true);
  const scanRunning = useViewerStore((s) => s.model.scan_running === true);
  // The /state summary (model.motor_cal) flips `active` even before the next
  // report arrives, so polling speeds up as soon as a run starts anywhere.
  const summaryActive = useViewerStore(
    (s) => (s.model.motor_cal as { active?: boolean } | null | undefined)?.active === true,
  );

  const [report, setReport] = useState<MotorReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [armRemoved, setArmRemoved] = useState(false);

  const read = useCallback(async () => {
    try {
      setReport(await getJson<MotorReport>('/firmware/motor'));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const active = report?.run.active === true || summaryActive;
  useEffect(() => {
    void read();
    const id = window.setInterval(read, active ? 1000 : 15_000);
    return () => window.clearInterval(id);
  }, [read, active]);

  const start = (axis: 'az' | 'el') => {
    const what =
      axis === 'el'
        ? 'EL turns full circles — the arm must be off.'
        : 'AZ turns full circles; keep the platform clear.';
    if (
      !window.confirm(
        `Calibrate the ${axis.toUpperCase()} motor? ${what} It takes 2–4 minutes; ` +
          'a failed or cancelled run keeps the previous result.',
      )
    ) {
      return;
    }
    commands.motorCalibrate(axis, axis === 'el' ? armRemoved : false);
    window.setTimeout(read, 600);
  };

  const run = report?.run;
  const blocked = !espOnline || scanRunning || active;

  return (
    <Card className="my-3 border-cardline/80 bg-black/20">
      <CardTitle className="mb-2">Step 1 · Motor calibration</CardTitle>
      <CardContent>
        <div className="text-[12px] text-inkdim">
          The firmware measures each axis: steps per degree, which way it
          turns, its top speed and an angle-deviation profile. Moves on an
          axis are refused until it has a result.
        </div>
        {error && <div className="text-[11px] text-red-300">{error}</div>}
        {report &&
          AXES.map(([key, label]) => {
            const a = report[key];
            return (
              <div
                key={key}
                className="rounded-md border border-cardline bg-black/20 px-2.5 py-1.5 text-[12px]"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-semibold text-ink">{label}</span>
                  <span className={a.valid ? 'text-emerald-300' : 'text-amber-300'}>
                    {a.valid ? 'calibrated' : 'not calibrated'}
                  </span>
                </div>
                {a.valid && (
                  <div className="font-mono text-[12px] text-zinc-100">
                    {a.steps_per_deg.toFixed(2)} µsteps/° · max {a.max_speed_dps.toFixed(1)} °/s ·
                    dir {a.dir_sign > 0 ? '+' : '−'}
                    {a.has_profile && (
                      <span className="text-inkdim">
                        {' '}
                        · profile peak {a.profile_peak_deg?.toFixed(2)}° rms{' '}
                        {a.profile_rms_deg?.toFixed(2)}°
                      </span>
                    )}
                  </div>
                )}
                {a.valid && a.has_profile && (
                  <label className="mt-1 flex cursor-pointer items-center gap-1.5 text-[12px] text-inkdim">
                    <input
                      type="checkbox"
                      checked={a.compensation === true}
                      disabled={!espOnline || active}
                      onChange={(e) => {
                        commands.motorCompensation(key, e.target.checked);
                        window.setTimeout(read, 400);
                      }}
                      className="h-3.5 w-3.5 cursor-pointer"
                    />
                    apply the deviation profile
                  </label>
                )}
              </div>
            );
          })}

        {run && (active || run.phase === 'failed' || run.phase === 'done') && (
          <div className="rounded-md border border-cardline bg-black/20 px-2.5 py-1.5 text-[12px]">
            <div className="flex justify-between">
              <span className="text-inkdim">
                {run.axis.toUpperCase()} · {run.phase}
              </span>
              <span className="font-mono text-zinc-100">
                {Math.round(run.progress * 100)}%
                {active && ` · ${run.speed_dps.toFixed(1)} °/s`}
              </span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded bg-zinc-800">
              <div
                className={run.phase === 'failed' ? 'h-full bg-red-500' : 'h-full bg-accent'}
                style={{ width: `${Math.round(run.progress * 100)}%` }}
              />
            </div>
            {run.message && (
              <div
                className={`mt-1 text-[11px] ${run.phase === 'failed' ? 'text-red-300' : 'text-inkdim'}`}
              >
                {run.message}
              </div>
            )}
          </div>
        )}

        <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-inkdim">
          <input
            type="checkbox"
            checked={armRemoved}
            onChange={(e) => setArmRemoved(e.target.checked)}
            className="h-3.5 w-3.5 cursor-pointer"
          />
          the arm is off (EL turns full circles)
        </label>
        <div className="flex gap-2">
          <Button size="sm" disabled={blocked} onClick={() => start('az')}>
            Calibrate AZ
          </Button>
          <Button
            size="sm"
            disabled={blocked || !armRemoved}
            title={armRemoved ? undefined : 'Take the arm off and tick the box first'}
            onClick={() => start('el')}
          >
            Calibrate EL
          </Button>
          <Button
            size="sm"
            variant="destructive"
            disabled={!active}
            onClick={() => {
              commands.motorCalibrateCancel();
              window.setTimeout(read, 600);
            }}
          >
            Cancel
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
