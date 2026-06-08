/**
 * Position overlay — navball as a passive pose indicator on the left, a
 * vertical control strip on the right with Motors → jog wheel (штурвал) →
 * absolute-move block, all stacked under the master Motors ON/OFF toggle.
 *
 * Jog is in **degrees** (no step-Hz) — arrows add a signed `step` to the
 * live pose and send a one-axis `move`. The same step input lives in the
 * centre of the wheel so the operator can adjust the bump without leaving
 * the cluster.
 */

import { useState } from 'react';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { NavBall } from './NavBall';
import { cls, num } from './ui';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';

/** Mount calibration: the phone's raw device-pitch is ANTI-correlated with rig
 *  el (slope ≈ −1) plus a fixed offset, measured from an el sweep
 *  (phone_pitch + el ≈ 6.5 across el 28→62°). So the phone's elevation estimate
 *  is `OFFSET − phone_pitch ≈ el`. Re-tune if the phone mount changes. */
const PHONE_EL_OFFSET_DEG = 6.5;

export function PositionOverlay({ commands }: { commands: Commands }) {
  const model = useViewerStore((s) => s.model);
  const [az, setAz] = useState('0');
  const [el, setEl] = useState('0');
  const [step, setStep] = useState('5');
  // Show/hide the purple phone-IMU marker on the navball. When off we pass
  // phoneEl={null} so <NavBall> skips rendering it.
  const [showPhoneMarker, setShowPhoneMarker] = useState(true);

  const stepN = () => parseFloat(step) || 5;
  const motorsOn = model.motors_on === true;
  const targetAz = model.move_target_az as number | null | undefined;
  const targetEl = model.move_target_el as number | null | undefined;
  const hasTarget = targetAz != null && targetEl != null;
  // Phone IMU — surfaced by the server's phone_sensor poll. Null while the
  // camera URL is empty or the IP Webcam sensors endpoint is unreachable.
  const phonePitch = model.phone_pitch_deg as number | null | undefined;
  const phoneRoll  = model.phone_roll_deg  as number | null | undefined;
  const phoneOnline = model.phone_sensor_online === true;
  // Phone's elevation estimate, calibrated to track rig el (see the const) so
  // the navball marker confirms el rather than drifting with the raw pitch.
  const phoneEl = phonePitch != null ? PHONE_EL_OFFSET_DEG - phonePitch : null;

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
      {/* left info column — pose readout in a tidy tile beside the navball */}
      <div className="flex w-[150px] flex-col justify-center gap-2 self-stretch">
        <div className="rounded-lg border border-cardline bg-black/25 px-3 py-2 font-mono text-[13px]">
          <div className="flex items-baseline justify-between gap-2 py-0.5">
            <span className="text-[11px] uppercase tracking-wide text-inkmute">
              <span className="text-amber-300/90">●</span> az
            </span>
            <span className="text-zinc-100">{num(model, 'az').toFixed(1)}°</span>
          </div>
          <div className="flex items-baseline justify-between gap-2 py-0.5">
            <span className="text-[11px] uppercase tracking-wide text-inkmute">el</span>
            <span className="text-zinc-100">{num(model, 'el').toFixed(1)}°</span>
          </div>
          {hasTarget && (
            <div className="flex items-baseline justify-between gap-2 py-0.5 text-sky-300">
              <span className="text-[11px] uppercase tracking-wide">target</span>
              <span>{targetAz.toFixed(1)}° · {targetEl.toFixed(1)}°</span>
            </div>
          )}
          {phoneOnline && phoneEl != null && (
            <div className="flex items-baseline justify-between gap-2 py-0.5 text-[#c084fc]">
              <span className="text-[11px] uppercase tracking-wide">phone</span>
              <span>{phoneEl.toFixed(1)}°</span>
            </div>
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

      {/* navball — pure indicator, no controls wrap it any more */}
      <div className="flex flex-col items-center gap-2">
        <NavBall
          az={num(model, 'az')}
          el={num(model, 'el')}
          targetAz={hasTarget ? targetAz : null}
          targetEl={hasTarget ? targetEl : null}
          phoneEl={showPhoneMarker && phoneOnline ? phoneEl : null}
          phoneRoll={phoneOnline ? (phoneRoll ?? null) : null}
          size={264}
        />
      </div>

      {/* control strip — Motors → jog wheel → absolute move */}
      <div className="flex w-[180px] flex-col gap-3 self-stretch border-l border-cardline pl-5">
        <Button
          variant={motorsOn ? 'success' : 'secondary'}
          onClick={() => commands.motors(!motorsOn)}
        >
          Motors {motorsOn ? 'ON' : 'off'}
        </Button>

        {/* штурвал: 4 directions around the step input */}
        <div className="grid grid-cols-3 items-center justify-items-center gap-1
                        rounded-xl border border-cardline bg-black/20 p-2">
          <div />
          <Button
            variant="secondary"
            size="icon"
            onClick={() => jogEl(+1)}
            title={`+${stepN()}° EL`}
          >
            ▲
          </Button>
          <div />

          <Button
            variant="secondary"
            size="icon"
            onClick={() => jogAz(-1)}
            title={`−${stepN()}° AZ`}
          >
            ◀
          </Button>
          <Input
            className="h-9 text-center font-mono text-[12px]"
            style={{ width: 48, padding: '0 4px' }}
            value={step}
            onChange={(e) => setStep(e.target.value)}
            title="degrees per jog"
          />
          <Button
            variant="secondary"
            size="icon"
            onClick={() => jogAz(+1)}
            title={`+${stepN()}° AZ`}
          >
            ▶
          </Button>

          <div />
          <Button
            variant="secondary"
            size="icon"
            onClick={() => jogEl(-1)}
            title={`−${stepN()}° EL`}
          >
            ▼
          </Button>
          <div />
        </div>

        {/* precise absolute move */}
        <div className="flex flex-col gap-1 text-[12px]">
          <div className="flex items-center gap-2">
            <span className="w-5 text-inkmute">az</span>
            <Input
              className="h-8"
              value={az}
              onChange={(e) => setAz(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="w-5 text-inkmute">el</span>
            <Input
              className="h-8"
              value={el}
              onChange={(e) => setEl(e.target.value)}
            />
          </div>
          <Button
            size="sm"
            onClick={() => commands.move(parseFloat(az), parseFloat(el))}
          >
            Move
          </Button>
        </div>
      </div>
    </div>
  );
}
