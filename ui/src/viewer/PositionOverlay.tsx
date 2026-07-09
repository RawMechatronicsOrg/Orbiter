/**
 * Position overlay — navball as a passive pose indicator, a readout tile on
 * the left, and a plain control column on the right: Motors toggle, degree
 * jog (arrow cross around the step field), absolute move.
 *
 * Jog is in **degrees** (no step-Hz) — arrows add a signed `step` to the
 * live pose and send a one-axis `move`.
 *
 * The phone-IMU elevation estimate comes ready-made from the server
 * (`model.phone_el_deg`, single mount convention shared with the encoder
 * auto-zero) — do NOT re-derive it here.
 */

import { useState } from 'react';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { NavBall } from './NavBall';
import { cls, num } from './ui';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';

/** One row of the readout tile: small uppercase label, monospaced value. */
function ReadoutRow({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="text-[10px] uppercase tracking-[0.14em] text-inkmute">
        {label}
      </span>
      <span
        className="font-mono text-[13px] tabular-nums text-zinc-100"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </span>
    </div>
  );
}

export function PositionOverlay({ commands }: { commands: Commands }) {
  const model = useViewerStore((s) => s.model);
  const [az, setAz] = useState('0');
  const [el, setEl] = useState('0');
  const [step, setStep] = useState('5');
  // Show/hide the purple phone-IMU marker on the navball.
  const [showPhoneMarker, setShowPhoneMarker] = useState(true);

  const stepN = () => parseFloat(step) || 5;
  const motorsOn = model.motors_on === true;
  const targetAz = model.move_target_az as number | null | undefined;
  const targetEl = model.move_target_el as number | null | undefined;
  const hasTarget = targetAz != null && targetEl != null;
  // Phone IMU — surfaced by the server's phone_sensor poll. Null while the
  // camera URL is empty or the IP Webcam sensors endpoint is unreachable.
  const phoneOnline = model.phone_sensor_online === true;
  const phoneEl = phoneOnline
    ? (model.phone_el_deg as number | null | undefined) ?? null
    : null;

  // Degree-jog: read live pose, add a signed step, send a one-axis `move`.
  const jogAz = (dir: 1 | -1) => {
    commands.move(num(model, 'az') + dir * stepN(), undefined);
  };
  const jogEl = (dir: 1 | -1) => {
    commands.move(undefined, num(model, 'el') + dir * stepN());
  };

  return (
    <div
      className="absolute bottom-5 left-1/2 flex -translate-x-1/2 items-stretch gap-5
                 rounded-2xl border border-cardline bg-[#131c2e]/90 px-5 py-4
                 backdrop-blur-sm"
    >
      {/* left column — pose readout + phone-marker toggle */}
      <div className="flex w-[150px] flex-col justify-center gap-2 self-stretch">
        <div className="rounded-lg border border-cardline bg-black/25 px-3 py-2">
          <ReadoutRow label="az" value={`${num(model, 'az').toFixed(1)}°`} />
          <ReadoutRow label="el" value={`${num(model, 'el').toFixed(1)}°`} />
          {hasTarget && (
            <ReadoutRow
              label="target"
              value={`${targetAz.toFixed(1)}° · ${targetEl.toFixed(1)}°`}
              accent="#7dd3fc"
            />
          )}
          {phoneEl != null && (
            <ReadoutRow
              label="phone el"
              value={`${phoneEl.toFixed(1)}°`}
              accent="#c084fc"
            />
          )}
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-[12px] text-inkmute">
          <input
            type="checkbox"
            className={cls.check}
            checked={showPhoneMarker}
            onChange={(e) => setShowPhoneMarker(e.target.checked)}
          />
          <span>phone marker</span>
        </label>
      </div>

      {/* navball — pure indicator, no controls on it */}
      <div className="flex flex-col items-center justify-center">
        <NavBall
          az={num(model, 'az')}
          el={num(model, 'el')}
          targetAz={hasTarget ? targetAz : null}
          targetEl={hasTarget ? targetEl : null}
          phoneEl={showPhoneMarker ? phoneEl : null}
          size={264}
        />
      </div>

      {/* control column — Motors, jog cross, absolute move */}
      <div className="flex w-[180px] flex-col gap-3 self-stretch border-l border-cardline pl-5">
        <Button
          variant={motorsOn ? 'success' : 'secondary'}
          onClick={() => commands.motors(!motorsOn)}
        >
          Motors {motorsOn ? 'ON' : 'off'}
        </Button>

        {/* jog cross: 4 directions around the step field (degrees) */}
        <div className="grid grid-cols-3 items-center justify-items-center gap-1
                        rounded-xl border border-cardline bg-black/20 p-2">
          <div />
          <Button variant="secondary" size="icon" onClick={() => jogEl(+1)}
                  title={`+${stepN()}° EL`}>
            ▲
          </Button>
          <div />

          <Button variant="secondary" size="icon" onClick={() => jogAz(-1)}
                  title={`−${stepN()}° AZ`}>
            ◀
          </Button>
          <Input
            className="h-9 text-center font-mono text-[12px] tabular-nums"
            style={{ width: 48, padding: '0 4px' }}
            value={step}
            onChange={(e) => setStep(e.target.value)}
            title="degrees per jog"
          />
          <Button variant="secondary" size="icon" onClick={() => jogAz(+1)}
                  title={`+${stepN()}° AZ`}>
            ▶
          </Button>

          <div />
          <Button variant="secondary" size="icon" onClick={() => jogEl(-1)}
                  title={`−${stepN()}° EL`}>
            ▼
          </Button>
          <div />
        </div>

        {/* absolute move */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <span className="w-6 text-[10px] uppercase tracking-[0.14em] text-inkmute">
              az
            </span>
            <Input className="h-8 font-mono text-[12px] tabular-nums" value={az}
                   onChange={(e) => setAz(e.target.value)} />
          </div>
          <div className="flex items-center gap-2">
            <span className="w-6 text-[10px] uppercase tracking-[0.14em] text-inkmute">
              el
            </span>
            <Input className="h-8 font-mono text-[12px] tabular-nums" value={el}
                   onChange={(e) => setEl(e.target.value)} />
          </div>
          <Button size="sm" onClick={() => commands.move(parseFloat(az), parseFloat(el))}>
            Move
          </Button>
        </div>
      </div>
    </div>
  );
}
