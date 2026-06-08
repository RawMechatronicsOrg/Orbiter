/**
 * Machine Config — rig parameters reachable from the UI.
 *
 *  - **Calibration** — the rig has no accurate camera poses until a ChArUco
 *    hand-eye calibration has been run and applied. A sweep of poses
 *    captures a photo at each, detects the board, runs `cv2.calibrateHandEye`
 *    and applies the derived geometry. Pick an accuracy preset (fast / normal
 *    / full) — more poses are slower but more accurate. Until then a bright
 *    warning is shown and `model.calibrated` is false.
 *  - **Geometry** (arm_radius_mm, camera_offset_mm, camera_tilt_deg,
 *    camera_pan_deg) is derived from that calibration, not entered by hand.
 *    `base_height_mm` stays at the configured default (the hand-eye solver
 *    is invariant to it).
 *  - **Board** — the ChArUco params (squares, lengths, ArUco dict). These are
 *    now THE only user-entered geometry input, so they're editable here and
 *    pushed to the server with *Apply board*.
 *  - **Endpoints** — ESP IP (with mDNS auto-discovery) and camera URL.
 *  - **Encoder zero** — firmware-side encoder offsets (positioning is
 *    closed-loop, no steps-per-degree calibration).
 */

import { useEffect, useState, type ReactNode } from 'react';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { num } from './ui';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Card, CardContent, CardTitle } from '../components/ui/card';
import { GeometryStateModal } from './GeometryStateModal';

type CalPreset = 'fast' | 'normal' | 'full';

/** Calibration accuracy steps — denser sweep = slower but more accurate.
 *  `full` (the production full azimuth ring) is the MOST accurate. */
const CAL_PRESETS: ReadonlyArray<readonly [CalPreset, string, string]> = [
  ['fast',   'Fast',   '≈9 poses · quick, least accurate'],
  ['normal', 'Normal', '≈24 poses · balanced'],
  ['full',   'Fine',   '≈32 poses · slowest, MOST accurate'],
];

/** Common OpenCV predefined ArUco dictionaries, by their `cv2.aruco.DICT_*`
 *  integer id. The board defaults to DICT_4X4_50 (id 0). */
const ARUCO_DICTS: ReadonlyArray<readonly [number, string]> = [
  [0, 'DICT_4X4_50'],
  [1, 'DICT_4X4_100'],
  [2, 'DICT_4X4_250'],
  [4, 'DICT_5X5_50'],
  [5, 'DICT_5X5_100'],
  [8, 'DICT_6X6_50'],
  [10, 'DICT_6X6_250'],
];

/** Human-readable name for an ArUco dict id (falls back to the raw id). */
const dictName = (id: number): string =>
  ARUCO_DICTS.find(([i]) => i === id)?.[1] ?? `dict ${id}`;

const SubLabel = ({ children }: { children: string }) => (
  <div className="mb-1 mt-3 text-[11px] uppercase tracking-[0.14em] text-inkdim">
    {children}
  </div>
);

/** Progressive-disclosure section: a clickable header that reveals its body.
 *  Secondary panels (derived geometry, connections, advanced) start collapsed
 *  so a new operator sees the primary Step 1 → Step 2 flow first. */
function Collapsible({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mt-3 border-t border-cardline/50 pt-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-[11px] uppercase
                   tracking-[0.14em] text-inkdim hover:text-zinc-200"
      >
        <span>{title}</span>
        <span className="text-inkmute">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div>{children}</div>}
    </div>
  );
}

const ReadOnlyRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-baseline justify-between gap-3 py-0.5">
    <span className="text-[12px] text-inkdim">{label}</span>
    <span className="font-mono text-[13px] text-ink">{value}</span>
  </div>
);

/** Labelled numeric field used by the editable ChArUco board block.
 *  Kept as a plain text input (rather than type="number") so an empty /
 *  intermediate value doesn't spam onChange with coerced numbers — the
 *  parent validates with `Number.isFinite` on Apply. */
const BoardNumberRow = ({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  step?: string;
}) => (
  <label className="flex items-center gap-2">
    <span className="w-24 shrink-0 text-[13px] text-inkdim">{label}</span>
    <Input
      type="number"
      inputMode="decimal"
      step={step}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  </label>
);

// UI-side heuristic: when *Calibrate from board* is pressed we have no
// progress channel from server back yet, so we just keep the button
// disabled for this long and trust the WS `model` update to refresh
// the read-only geometry display when the solver applies its result.
// Server logs the real outcome (success or per-view detection failures)
// to /ws/log; check the log panel if nothing changed after this elapses.
const CALIBRATION_TIMEOUT_MS = 60_000;

export function MachineConfig({ commands }: { commands: Commands }) {
  // Reactive bindings to the model so the inputs and the resolved-IP line
  // update as soon as the server pushes a hydration patch — initial render
  // happens before the WS `model` message arrives, so a one-shot snapshot
  // (the previous pattern) left the fields empty even though the values
  // are persisted server-side.
  const modelEspIp = useViewerStore((s) =>
    typeof s.model.esp_ip === 'string' ? (s.model.esp_ip as string) : '',
  );
  const modelCameraUrl = useViewerStore((s) =>
    typeof s.model.camera_url === 'string' ? (s.model.camera_url as string) : '',
  );
  const modelAutoEsp = useViewerStore((s) =>
    typeof s.model.esp_autodiscover === 'boolean'
      ? (s.model.esp_autodiscover as boolean)
      : true,
  );

  const [espIp, setEspIp] = useState<string>(modelEspIp);
  const [cameraUrl, setCameraUrl] = useState<string>(modelCameraUrl);
  const [autoEsp, setAutoEsp] = useState<boolean>(modelAutoEsp);
  // Track whether the user has typed into the inputs themselves. While
  // they're "clean" we mirror server state into them (handles late
  // hydration + mDNS-discovered IP showing up). After the user starts
  // editing we stop overwriting their draft until Apply is pressed.
  const [espDirty, setEspDirty] = useState<boolean>(false);
  const [camDirty, setCamDirty] = useState<boolean>(false);
  useEffect(() => {
    if (!espDirty) setEspIp(modelEspIp);
  }, [modelEspIp, espDirty]);
  useEffect(() => {
    if (!camDirty) setCameraUrl(modelCameraUrl);
  }, [modelCameraUrl, camDirty]);
  useEffect(() => {
    setAutoEsp(modelAutoEsp);
  }, [modelAutoEsp]);

  const [calibrating, setCalibrating] = useState<boolean>(false);
  const [calStatus, setCalStatus] = useState<string | null>(null);
  // "Test accuracy" affordance — flips while we wait for the server to push
  // its result into model.calib_test_msg (and log to the panel).
  const [testing, setTesting] = useState<boolean>(false);

  const [geomOpen, setGeomOpen] = useState<boolean>(false);

  // Reactive — re-renders when these model fields change (e.g. when the
  // server applies a calibration result).
  const calibrated = useViewerStore((s) => s.model.calibrated === true);
  // Result of "Test accuracy" — a human string the server writes after it
  // captures a photo, detects the board, and compares optical vs encoder pose.
  const calTestMsg = useViewerStore((s) =>
    typeof s.model.calib_test_msg === 'string'
      ? (s.model.calib_test_msg as string)
      : null,
  );
  const armRadius  = useViewerStore((s) => num(s.model, 'arm_radius_mm', 0));
  const camOffset  = useViewerStore((s) => num(s.model, 'camera_offset_mm', 80));
  const baseHeight = useViewerStore((s) => num(s.model, 'base_height_mm', 45));
  const camTilt    = useViewerStore((s) => num(s.model, 'camera_tilt_deg', 0));
  const camPan     = useViewerStore((s) => num(s.model, 'camera_pan_deg', 0));

  const boardSx    = useViewerStore((s) => num(s.model, 'charuco_squares_x', 5));
  const boardSy    = useViewerStore((s) => num(s.model, 'charuco_squares_y', 7));
  const boardSq    = useViewerStore((s) => num(s.model, 'charuco_square_length_mm', 30));
  const boardMk    = useViewerStore((s) => num(s.model, 'charuco_marker_length_mm', 15));
  // aruco_dict_id may be absent on older persisted state — default to 0
  // (DICT_4X4_50), matching the server's default board.
  const boardDict  = useViewerStore((s) => num(s.model, 'aruco_dict_id', 0));

  // Editable board drafts. Seeded from the model and re-synced whenever the
  // server pushes a change AND the field is still "clean" (same dirty-flag
  // pattern as the endpoint inputs above). After Apply the field re-syncs.
  const [bSx, setBSx] = useState<string>(String(boardSx));
  const [bSy, setBSy] = useState<string>(String(boardSy));
  const [bSq, setBSq] = useState<string>(String(boardSq));
  const [bMk, setBMk] = useState<string>(String(boardMk));
  const [bDict, setBDict] = useState<number>(boardDict);
  const [boardDirty, setBoardDirty] = useState<boolean>(false);
  useEffect(() => {
    if (boardDirty) return;
    setBSx(String(boardSx));
    setBSy(String(boardSy));
    setBSq(String(boardSq));
    setBMk(String(boardMk));
    setBDict(boardDict);
  }, [boardSx, boardSy, boardSq, boardMk, boardDict, boardDirty]);

  const applyBoard = () => {
    // Only send finite numbers; an empty / mistyped field is dropped so it
    // keeps its current server value rather than clobbering it with NaN.
    const params: Record<string, number> = { aruco_dict_id: bDict };
    const add = (k: string, raw: string) => {
      const v = Number(raw);
      if (raw.trim() !== '' && Number.isFinite(v)) params[k] = v;
    };
    add('charuco_squares_x', bSx);
    add('charuco_squares_y', bSy);
    add('charuco_square_length_mm', bSq);
    add('charuco_marker_length_mm', bMk);
    commands.setBoardParams(params);
    setBoardDirty(false);
  };

  // Chosen accuracy step — defaults to the most accurate (full).
  const [calPreset, setCalPreset] = useState<CalPreset>('full');

  const runCalibration = (preset: CalPreset) => {
    setCalibrating(true);
    setCalStatus(
      `Sweeping poses (${preset}) and solving — this takes ~30–60 s. Keep the ` +
        'ChArUco board visible to the camera.',
    );
    commands.calibrateGeometry(preset);
    // Heuristic: switch the buttons back after the timeout. Real outcome
    // (success or solver error) shows in /ws/log, flips `model.calibrated`,
    // and refreshes the geometry rows above.
    setTimeout(() => {
      setCalibrating(false);
      setCalStatus('Done (check geometry above and the log panel).');
    }, CALIBRATION_TIMEOUT_MS);
  };

  // Capture at the current pose and compare optical pose vs encoder angles.
  // The server reports the delta via model.calib_test_msg + the log panel;
  // we just toggle the "testing…" affordance, which clears when a fresh
  // message arrives (effect below) or after a safety timeout.
  const runTestAccuracy = () => {
    setTesting(true);
    commands.raw('test_calibration_accuracy');
  };
  useEffect(() => {
    if (!testing) return;
    // A new result (calib_test_msg changed) means the test finished.
    setTesting(false);
  }, [calTestMsg]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Card>
      <CardTitle className="mb-2">Machine config</CardTitle>

      {/* ── Status — the first thing you see ── */}
      {calibrated ? (
        <div className="mb-3 rounded-md border border-emerald-500/50 bg-emerald-500/10 px-3 py-2 text-[12px] text-emerald-300">
          ✓ Calibrated — camera poses are solved. Re-run Step 2 if you change
          the board or rig.
        </div>
      ) : (
        <div className="mb-3 rounded-md border-2 border-amber-500 bg-amber-500/10 px-3 py-2.5">
          <div className="text-[12px] font-bold uppercase tracking-[0.08em] text-amber-300">
            ⚠ Not calibrated
          </div>
          <div className="mt-1 text-[12px] text-amber-100/90">
            The rig can't compute accurate camera poses yet. Set the board
            (Step 1), then pick an accuracy and Calibrate (Step 2):
          </div>
          <div className="mt-2">
            <Button
              size="sm"
              disabled={calibrating}
              onClick={() => runCalibration(calPreset)}
            >
              {calibrating ? 'Calibrating…' : `Calibrate now (${calPreset})`}
            </Button>
          </div>
          {calStatus && (
            <div className="mt-2 text-[11px] text-amber-100/80">{calStatus}</div>
          )}
        </div>
      )}

      {/* ── Step 1 · ChArUco board (the ONLY manual input) ── */}
      <Card className="my-3 border-cardline/80 bg-black/20">
        <CardTitle className="mb-2">Step 1 · ChArUco board</CardTitle>
        <CardContent>
          {/* Currently ACTIVE board — straight from the model (what the solver
              actually uses), distinct from the editable drafts below. */}
          <div className="mb-2 rounded-md border border-accent/40 bg-accent/5 px-2.5 py-1.5 text-[12px]">
            <span className="text-inkmute">active board: </span>
            <span className="font-mono text-zinc-100">
              {boardSx}×{boardSy} · {dictName(boardDict)} · {boardSq}mm sq / {boardMk}mm marker
            </span>
          </div>
          <BoardNumberRow
            label="squares X"
            value={bSx}
            onChange={(v) => { setBSx(v); setBoardDirty(true); }}
          />
          <BoardNumberRow
            label="squares Y"
            value={bSy}
            onChange={(v) => { setBSy(v); setBoardDirty(true); }}
          />
          <BoardNumberRow
            label="square (mm)"
            value={bSq}
            step="0.1"
            onChange={(v) => { setBSq(v); setBoardDirty(true); }}
          />
          <BoardNumberRow
            label="marker (mm)"
            value={bMk}
            step="0.1"
            onChange={(v) => { setBMk(v); setBoardDirty(true); }}
          />
          <label className="flex items-center gap-2">
            <span className="w-24 shrink-0 text-[13px] text-inkdim">ArUco dict</span>
            <select
              value={bDict}
              onChange={(e) => { setBDict(Number(e.target.value)); setBoardDirty(true); }}
              className="w-full rounded-md border border-fieldline bg-field px-2.5 py-1 font-mono text-[13px] text-zinc-100 focus:border-accent/40 focus:outline-1 focus:outline-accent"
            >
              {ARUCO_DICTS.map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <Button onClick={applyBoard} disabled={!boardDirty}>
            Apply board
          </Button>
          <div className="mt-1 text-[11px] text-inkmute">
            Edit a field and press Apply to change the active board. Changing
            the board requires a fresh calibration.
          </div>
        </CardContent>
      </Card>

      {/* ── Step 2 · calibration ── */}
      <SubLabel>Step 2 · Calibration</SubLabel>
      <CardContent>
        {/* Accuracy step — denser sweep = slower but more accurate. */}
        <div className="text-[11px] uppercase tracking-[0.1em] text-inkdim">
          accuracy
        </div>
        <div className="flex gap-1">
          {CAL_PRESETS.map(([p, label, hint]) => (
            <button
              key={p}
              type="button"
              disabled={calibrating}
              onClick={() => setCalPreset(p)}
              title={hint}
              className={`flex-1 rounded-md border px-2 py-1 text-[12px] transition disabled:opacity-50 ${
                calPreset === p
                  ? 'border-accent bg-accent/15 text-zinc-100'
                  : 'border-fieldline bg-field text-inkdim hover:text-zinc-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="text-[11px] text-inkmute">
          {CAL_PRESETS.find(([p]) => p === calPreset)?.[2]}
        </div>
        <Button disabled={calibrating} onClick={() => runCalibration(calPreset)}>
          {calibrating ? 'Calibrating…' : 'Calibrate from board'}
        </Button>
        {calStatus && (
          <div className="mt-2 text-[11px] text-inkmute">{calStatus}</div>
        )}
        {/* Test the active calibration: capture at the current pose, detect the
            board, and report the optical-vs-encoder delta. Result lands in
            model.calib_test_msg (below) and the log panel. */}
        <Button
          variant="secondary"
          disabled={calibrating || testing}
          onClick={runTestAccuracy}
        >
          {testing ? 'Testing…' : 'Test accuracy'}
        </Button>
        {calTestMsg && (
          <div className="mt-1 font-mono text-[12px] text-inkdim">{calTestMsg}</div>
        )}
      </CardContent>

      {/* ── derived geometry — collapsed (read-only result of Step 2) ── */}
      <Collapsible title="Geometry (derived)">
        <CardContent>
          <ReadOnlyRow label="arm radius"    value={`${armRadius.toFixed(1)} mm`} />
          <ReadOnlyRow label="camera offset" value={`${camOffset.toFixed(1)} mm`} />
          <ReadOnlyRow label="base height"   value={`${baseHeight.toFixed(1)} mm`} />
          <ReadOnlyRow label="camera tilt"   value={`${camTilt.toFixed(2)}°`} />
          <ReadOnlyRow label="camera pan"    value={`${camPan.toFixed(2)}°`} />
          <div className="mt-2 text-[11px] text-inkmute">
            Derived from the ChArUco hand-eye calibration; `base_height` keeps
            its configured default (the solver is invariant to it).
          </div>
        </CardContent>
      </Collapsible>

      {/* ── connections — collapsed ── */}
      <Collapsible title="Connections">
        <CardContent>
          <label className="flex items-center gap-2">
            <span className="w-24 shrink-0 text-[13px] text-inkdim">ESP IP</span>
            <Input
              value={espIp}
              placeholder={autoEsp ? 'auto (mDNS)' : '192.168.0.42'}
              onChange={(e) => {
                setEspIp(e.target.value);
                setEspDirty(true);
              }}
            />
          </label>
          <div className="pl-[6.5rem] text-[11px] text-inkmute">
            resolved:{' '}
            <span className="font-mono text-inkdim">
              {modelEspIp || '—'}
            </span>
          </div>
          <label className="flex cursor-pointer items-center gap-2 pl-[6.5rem] text-[12px] text-inkdim">
            <input
              type="checkbox"
              checked={autoEsp}
              onChange={(e) => {
                const v = e.target.checked;
                setAutoEsp(v);
                commands.raw('set_endpoints', { esp_autodiscover: v });
              }}
              className="h-3.5 w-3.5 cursor-pointer"
            />
            <span>Auto-discover via mDNS (orbiter.local)</span>
          </label>
          <label className="flex items-center gap-2">
            <span className="w-24 shrink-0 text-[13px] text-inkdim">camera URL</span>
            <Input
              value={cameraUrl}
              placeholder="http://phone-cam:8080"
              onChange={(e) => {
                setCameraUrl(e.target.value);
                setCamDirty(true);
              }}
            />
          </label>
          <Button
            onClick={() => {
              commands.raw('set_endpoints', {
                esp_ip: espIp.trim(),
                camera_url: cameraUrl.trim(),
                esp_autodiscover: autoEsp,
              });
              setEspDirty(false);
              setCamDirty(false);
            }}
          >
            Apply endpoints
          </Button>
        </CardContent>
      </Collapsible>

      {/* ── advanced — collapsed ── */}
      <Collapsible title="Advanced">
        <CardContent>
          <Button variant="secondary" onClick={() => setGeomOpen(true)}>
            Geometry state…
          </Button>
        </CardContent>
      </Collapsible>

      <GeometryStateModal
        open={geomOpen}
        onClose={() => setGeomOpen(false)}
        commands={commands}
      />
    </Card>
  );
}
