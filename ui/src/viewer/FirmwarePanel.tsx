/**
 * Firmware + encoder diagnostics — the ESP32 itself, through the server
 * (`/firmware/*` routes; the browser never talks to the device):
 *
 *  - **Encoders** — firmware `GET /test/encoder`: per-axis AS5600 reading,
 *    magnet status and the idle levels of each I2C line.
 *  - **Firmware** — `GET /health` (chip, IDF, uptime, heap), a `.bin` update
 *    over Wi-Fi (`POST /ota`, then watching the device through its 120 s
 *    rollback window), and continuous spin of one axis.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE, getJson } from './api';
import { usePoll } from './usePoll';
import { useViewerStore } from './modelStore';
import type { Commands } from './commands';
import { axisInfo, degPerS } from './ui';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardTitle } from '../components/ui/card';

// ── encoders ─────────────────────────────────────────────────────────────────

/** Idle level of an I2C line: 1 high, 0 LOW, -1 unknown. */
type Level = 1 | 0 | -1;

interface EncoderAxis {
  ok: boolean;
  raw?: number;
  angle_deg?: number;
  magnet_missing?: number;
  magnet_weak?: number;
  magnet_strong?: number;
  agc?: number;
  magnitude?: number;
  lines?: { sda: Level; scl: Level; external_pullups: Level };
}

interface EncoderReport {
  azimuth?: EncoderAxis;
  elevation?: EncoderAxis;
}

/** Pins of the Orbiter-Simple board wiring (firmware menuconfig choice). */
const ENCODER_BUSES = [
  { key: 'azimuth', label: 'AZ', pins: 'I2C0 · SDA 22 / SCL 23' },
  { key: 'elevation', label: 'EL', pins: 'I2C1 · SDA 4 / SCL 16' },
] as const;

function LevelChip({ name, level }: { name: string; level: Level | undefined }) {
  const [text, tone] =
    level === 1
      ? ['high', 'text-emerald-300 border-emerald-500/40']
      : level === 0
        ? ['LOW', 'text-red-300 border-red-500/60']
        : ['?', 'text-inkmute border-cardline'];
  return (
    <span className={`rounded border px-1.5 py-px font-mono text-[11px] ${tone}`}>
      {name} {text}
    </span>
  );
}

/** What the readings say is wrong, in the operator's terms. */
function diagnose(a: EncoderAxis): string[] {
  const out: string[] = [];
  const l = a.lines;
  if (l?.sda === 0 || l?.scl === 0) {
    out.push(
      `${l.sda === 0 ? 'SDA' : 'SCL'} is held LOW: a short, a wire on GND or ` +
        'a DIR pin, or an unpowered chip clamping the line.',
    );
  } else if (l?.external_pullups === 0) {
    out.push('No external pull-ups: the sensor board has no supply.');
  }
  if (a.magnet_missing) out.push('No magnet detected.');
  // AGC runs 0..128 at 3V3, ~64 at a good gap; pinned at the top = too far.
  else if (a.agc != null && a.agc >= 120) out.push('AGC at its ceiling — the magnet is too far.');
  if (a.magnet_weak) out.push('Magnet too weak — too far from the chip.');
  if (a.magnet_strong) out.push('Magnet too strong — too close to the chip.');
  if (!a.ok && out.length === 0) out.push('The chip does not answer on its bus.');
  return out;
}

export function EncoderDiagCard() {
  const [report, setReport] = useState<EncoderReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [auto, setAuto] = useState(false);
  const [busy, setBusy] = useState(false);

  const read = useCallback(async () => {
    setBusy(true);
    try {
      setReport(await getJson<EncoderReport>('/firmware/encoder'));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  usePoll(read, 1000, auto);

  return (
    <Card>
      <CardTitle className="mb-2">Encoders</CardTitle>
      <CardContent>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="secondary" disabled={busy} onClick={read}>
            Read encoders
          </Button>
          <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-inkdim">
            <input
              type="checkbox"
              checked={auto}
              onChange={(e) => setAuto(e.target.checked)}
              className="h-3.5 w-3.5 cursor-pointer"
            />
            every 1 s
          </label>
        </div>
        {error && <div className="text-[11px] text-red-300">{error}</div>}
        {report &&
          ENCODER_BUSES.map(({ key, label, pins }) => {
            const a = report[key];
            if (!a) return null;
            const problems = diagnose(a);
            return (
              <div
                key={key}
                className="rounded-md border border-cardline bg-black/20 px-2.5 py-1.5 text-[12px]"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-semibold text-ink">
                    {label} <span className="font-normal text-inkmute">{pins}</span>
                  </span>
                  <span className={a.ok ? 'text-emerald-300' : 'text-red-300'}>
                    {a.ok ? 'reads' : 'no reading'}
                  </span>
                </div>
                {a.ok && (
                  <div className="font-mono text-[12px] text-zinc-100">
                    raw {a.raw} · {a.angle_deg?.toFixed(2)}°
                    {a.agc != null && <span className="text-inkdim"> · agc {a.agc}</span>}
                    {a.magnitude != null && (
                      <span className="text-inkdim"> · mag {a.magnitude}</span>
                    )}
                  </div>
                )}
                <div className="mt-1 flex flex-wrap gap-1">
                  <LevelChip name="SDA" level={a.lines?.sda} />
                  <LevelChip name="SCL" level={a.lines?.scl} />
                  <LevelChip name="pull-ups" level={a.lines?.external_pullups} />
                </div>
                {problems.map((p) => (
                  <div key={p} className="mt-1 text-[11px] text-amber-200">
                    {p}
                  </div>
                ))}
              </div>
            );
          })}
        <div className="text-[11px] text-inkmute">
          Line levels are sampled at boot, before the I2C driver takes the
          pins: LOW = held down, “pull-ups ?” = not measured.
        </div>
      </CardContent>
    </Card>
  );
}

// ── firmware: health, update, spin ─────────────────────────────────────────

interface Health {
  device?: string;
  chip?: string;
  revision?: number;
  cores?: number;
  mac?: string;
  free_heap_bytes?: number;
  uptime_ms?: number;
  idf_version?: string;
  esp_ip?: string;
}

const fmtUptime = (ms: number): string => {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
};

/** The device rolls back to the previous image when a new one has not
 *  confirmed itself (its HTTP server up) within this long of booting. */
const ROLLBACK_WINDOW_S = 120;
/** How long to watch for the device after an update before giving up. */
const WATCH_S = ROLLBACK_WINDOW_S + 60;

/** The app descriptor of the uploaded image, as the server read it. */
interface ImageDesc {
  version: string;
  project: string;
  built: string;
  idf: string;
  elf_sha256: string;
}

type Update =
  | { phase: 'idle' }
  | { phase: 'uploading'; bytes: number }
  | { phase: 'watching'; since: number; seenDown: boolean }
  | { phase: 'done'; ok: boolean; text: string };

async function postImage(
  file: File,
): Promise<{ ok: boolean; text: string; image: ImageDesc | null }> {
  const r = await fetch(`${API_BASE}/firmware/ota`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  });
  let body: Record<string, unknown> = {};
  try {
    body = (await r.json()) as Record<string, unknown>;
  } catch {
    /* empty body */
  }
  // Firmware replies carry {status, message}; server-side refusals {detail}.
  const text = String(body.message ?? body.detail ?? `HTTP ${r.status}`);
  const image = (body.image as ImageDesc | undefined) ?? null;
  return { ok: r.ok, text: r.ok ? text : `HTTP ${r.status}: ${text}`, image };
}

export function FirmwareCard({ commands }: { commands: Commands }) {
  const espOnline = useViewerStore((s) => s.model.esp_online === true);
  const scanRunning = useViewerStore((s) => s.model.scan_running === true);
  const spinningAz = useViewerStore((s) => s.model.spinning_az === true);
  const spinningEl = useViewerStore((s) => s.model.spinning_el === true);
  const elZeroSet = useViewerStore((s) => s.model.encoder_zero_initialized === true);
  const motorCalActive = useViewerStore(
    (s) => (s.model.motor_cal as { active?: boolean } | null | undefined)?.active === true,
  );
  const geomCalRunning = useViewerStore((s) => s.model.calibration_running === true);
  const rigBusy = scanRunning || geomCalRunning || motorCalActive;
  const azAxis = useViewerStore((s) => axisInfo(s.model, 'az'));
  const elAxis = useViewerStore((s) => axisInfo(s.model, 'el'));

  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const readHealth = useCallback(async (): Promise<Health | null> => {
    try {
      const h = await getJson<Health>('/firmware/health');
      setHealth(h);
      setHealthError(null);
      return h;
    } catch (e) {
      setHealthError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, []);

  usePoll(readHealth, 10_000);

  // ── update over Wi-Fi ──
  const [file, setFile] = useState<File | null>(null);
  const [update, setUpdate] = useState<Update>({ phase: 'idle' });
  const [sentImage, setSentImage] = useState<ImageDesc | null>(null);
  const watchTimer = useRef<number | null>(null);
  useEffect(() => () => {
    if (watchTimer.current) window.clearTimeout(watchTimer.current);
  }, []);

  /** Poll /health until the device has rebooted. Booting long after the
   *  upload means the new image never confirmed and the bootloader went
   *  back to the previous one. */
  const watch = useCallback(
    (since: number, seenDown: boolean) => {
      watchTimer.current = window.setTimeout(async () => {
        const elapsed = (Date.now() - since) / 1000;
        const h = await readHealth();
        if (h && typeof h.uptime_ms === 'number' && h.uptime_ms / 1000 < elapsed) {
          const bootedAfter = elapsed - h.uptime_ms / 1000;
          // /health does not name the running image, so this is read from
          // timing: a watchdog rollback (HTTP never came up) takes the full
          // window; a crash during start-up rolls back within seconds and
          // cannot be told apart from a good boot here.
          setUpdate(
            bootedAfter < ROLLBACK_WINDOW_S / 2
              ? {
                  phase: 'done',
                  ok: true,
                  text:
                    `Back up ${bootedAfter.toFixed(0)} s after the upload. That is the new ` +
                    'image unless it crashed while starting and the bootloader went back ' +
                    "to the previous one — /health cannot tell; the firmware log's " +
                    '"OTA image confirmed" can.',
                }
              : {
                  phase: 'done',
                  ok: false,
                  text: `Came back only ${bootedAfter.toFixed(0)} s after the upload — the new image did not confirm itself and was rolled back to the previous one.`,
                },
          );
          return;
        }
        if (elapsed > WATCH_S) {
          setUpdate({
            phase: 'done',
            ok: false,
            text: `No answer for ${WATCH_S} s — check the device (serial log).`,
          });
          return;
        }
        const down = seenDown || h == null;
        setUpdate({ phase: 'watching', since, seenDown: down });
        watch(since, down);
      }, 2000);
    },
    [readHealth],
  );

  const upload = async () => {
    if (!file) return;
    if (
      !window.confirm(
        `Flash ${file.name} (${(file.size / 1024).toFixed(0)} KiB) to the ESP32? ` +
          'The motors are switched off while it writes, then it reboots.',
      )
    ) {
      return;
    }
    setUpdate({ phase: 'uploading', bytes: file.size });
    try {
      const res = await postImage(file);
      setSentImage(res.image);
      if (!res.ok) {
        setUpdate({ phase: 'done', ok: false, text: res.text });
        return;
      }
      const since = Date.now();
      setUpdate({ phase: 'watching', since, seenDown: false });
      watch(since, false);
    } catch (e) {
      setUpdate({ phase: 'done', ok: false, text: e instanceof Error ? e.message : String(e) });
    }
  };

  const updating = update.phase === 'uploading' || update.phase === 'watching';

  // ── spin ──
  const [spinAxis, setSpinAxis] = useState<'az' | 'el'>('az');
  const [spinDir, setSpinDir] = useState<'cw' | 'ccw'>('cw');
  const [spinHz, setSpinHz] = useState<Record<'az' | 'el', number>>({ az: 525, el: 525 });
  const band = spinAxis === 'az' ? azAxis : elAxis;
  const hz = band ? Math.min(band.hz_max, Math.max(band.hz_min, spinHz[spinAxis])) : spinHz[spinAxis];
  const spinning = spinAxis === 'az' ? spinningAz : spinningEl;
  const spinBlocked = !espOnline || rigBusy || updating || (spinAxis === 'el' && !elZeroSet);

  return (
    <Card>
      <CardTitle className="mb-2">Firmware</CardTitle>
      <CardContent>
        {/* health */}
        <div className="rounded-md border border-cardline bg-black/20 px-2.5 py-1.5 font-mono text-[12px]">
          {health ? (
            <>
              <div className="flex justify-between">
                <span className="text-inkmute">{health.device ?? 'device'}</span>
                <span className="text-zinc-100">{health.esp_ip}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-inkmute">chip</span>
                <span className="text-zinc-100">
                  {health.chip} rev {health.revision} · {health.cores} cores
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-inkmute">IDF</span>
                <span className="text-zinc-100">{health.idf_version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-inkmute">uptime</span>
                <span className="text-zinc-100">
                  {typeof health.uptime_ms === 'number' ? fmtUptime(health.uptime_ms) : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-inkmute">free heap</span>
                <span className="text-zinc-100">
                  {typeof health.free_heap_bytes === 'number'
                    ? `${(health.free_heap_bytes / 1024).toFixed(0)} KiB`
                    : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-inkmute">MAC</span>
                <span className="text-zinc-100">{health.mac}</span>
              </div>
            </>
          ) : (
            <span className="text-inkmute">no answer yet</span>
          )}
        </div>
        {healthError && <div className="text-[11px] text-red-300">{healthError}</div>}
        <Button size="sm" variant="secondary" onClick={() => void readHealth()}>
          Refresh
        </Button>

        {/* update over Wi-Fi */}
        <div className="mt-2 text-[11px] uppercase tracking-[0.1em] text-inkdim">
          Update over Wi-Fi
        </div>
        <input
          type="file"
          accept=".bin,application/octet-stream"
          disabled={updating}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-[12px] text-inkdim file:mr-2 file:rounded-md file:border file:border-zinc-800 file:bg-zinc-900 file:px-2 file:py-1 file:text-[12px] file:text-zinc-200"
        />
        <Button
          size="sm"
          disabled={!file || !espOnline || rigBusy || updating}
          onClick={() => void upload()}
          title="Send firmware/build/orbiter.bin to POST /ota"
        >
          {update.phase === 'uploading' ? 'Uploading…' : 'Upload firmware'}
        </Button>
        {update.phase === 'uploading' && (
          <div className="text-[11px] text-inkdim">
            Sending {(update.bytes / 1024).toFixed(0)} KiB — the firmware writes and
            verifies it before it answers.
          </div>
        )}
        {update.phase === 'watching' && (
          <div className="text-[11px] text-amber-200">
            Written; the device is rebooting. Watching it through the{' '}
            {ROLLBACK_WINDOW_S} s rollback window
            {update.seenDown ? ' (offline now)' : ''}…
          </div>
        )}
        {sentImage && update.phase !== 'idle' && (
          <div className="font-mono text-[11px] text-inkdim">
            sent {sentImage.project} {sentImage.version} · built {sentImage.built} ·
            IDF {sentImage.idf} · {sentImage.elf_sha256}
          </div>
        )}
        {update.phase === 'done' && (
          <div className={`text-[11px] ${update.ok ? 'text-emerald-300' : 'text-red-300'}`}>
            {update.text}
          </div>
        )}
        <div className="text-[11px] text-inkmute">
          The raw app image from <span className="font-mono">firmware/build/orbiter.bin</span>.
          A new image confirms itself once its HTTP server is up; if it does
          not within {ROLLBACK_WINDOW_S} s the bootloader goes back to the
          previous one. Calibration in NVS is kept.
        </div>

        {/* spin */}
        <div className="mt-2 text-[11px] uppercase tracking-[0.1em] text-inkdim">
          Spin (continuous)
        </div>
        <div className="flex gap-1">
          {(['az', 'el'] as const).map((a) => (
            <Button
              key={a}
              size="xs"
              variant={spinAxis === a ? 'default' : 'secondary'}
              onClick={() => setSpinAxis(a)}
            >
              {a.toUpperCase()}
            </Button>
          ))}
          <span className="w-2" />
          {(['cw', 'ccw'] as const).map((d) => (
            <Button
              key={d}
              size="xs"
              variant={spinDir === d ? 'default' : 'secondary'}
              onClick={() => setSpinDir(d)}
            >
              {d}
            </Button>
          ))}
        </div>
        {band && (
          <div>
            <div className="flex items-baseline justify-between text-[13px]">
              <span className="text-inkdim">step rate</span>
              <span className="font-mono text-ink">
                {hz} Hz <span className="text-inkdim">≈ {degPerS(hz, band)}</span>
              </span>
            </div>
            <input
              type="range"
              min={band.hz_min}
              max={band.hz_max}
              step={25}
              value={hz}
              onChange={(e) => setSpinHz({ ...spinHz, [spinAxis]: Number(e.target.value) })}
              className="w-full cursor-pointer accent-accent"
            />
          </div>
        )}
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="success"
            disabled={spinBlocked || spinning}
            title={spinAxis === 'el' && !elZeroSet ? 'Zero the EL encoder first (Step 2)' : undefined}
            onClick={() => commands.spin(spinAxis, spinDir, hz)}
          >
            ▶ Spin {spinAxis.toUpperCase()}
          </Button>
          <Button size="sm" variant="secondary" disabled={!spinning} onClick={() => commands.spinStop(spinAxis)}>
            ■ Stop {spinAxis.toUpperCase()}
          </Button>
          <Button
            size="sm"
            variant="destructive"
            disabled={!spinningAz && !spinningEl}
            onClick={() => commands.spinStop('both')}
          >
            Stop all
          </Button>
        </div>
        <div className="text-[11px] text-inkmute">
          Open loop at a driver step rate. spinning: AZ {spinningAz ? 'yes' : 'no'} · EL{' '}
          {spinningEl ? 'yes' : 'no'}. The firmware does not stop an EL spin at the
          travel limits; the server stops it a few degrees before them.
        </div>
      </CardContent>
    </Card>
  );
}
