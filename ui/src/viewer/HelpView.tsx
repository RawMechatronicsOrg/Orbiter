/**
 * Help tab — a scrollable ribbon of instructional cards. A static
 * getting-started guide, read top-to-bottom: what Orbiter is, the scan
 * workflow, why calibration matters, and where things live. No interactivity.
 */

import type { ReactNode } from 'react';
import {
  Card,
  CardContent,
  CardTitle,
} from '../components/ui/card';

/** Section card — a heading + free-form body, spaced for comfortable reading. */
function Section({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <Card className="px-5 py-4">
      <CardTitle className="mb-3 text-[12px] tracking-[0.16em] text-ink">
        {title}
      </CardTitle>
      <CardContent className="gap-3 text-[13px] leading-relaxed text-inkdim">
        {children}
      </CardContent>
    </Card>
  );
}

/** A leading inline keyword (UI label / control name) within prose. */
function Tag({ children }: { children: ReactNode }) {
  return <span className="font-semibold text-ink">{children}</span>;
}

/** Monospaced inline path / filename. */
function Path({ children }: { children: ReactNode }) {
  return (
    <code className="rounded bg-field px-1.5 py-0.5 font-mono text-[12px] text-inkdim">
      {children}
    </code>
  );
}

/** Unordered list with subtle accent bullets. */
function Bullets({ children }: { children: ReactNode }) {
  return (
    <ul className="ml-1 flex list-none flex-col gap-2">{children}</ul>
  );
}

function Bullet({ children }: { children: ReactNode }) {
  return (
    <li className="relative pl-4 before:absolute before:left-0 before:top-[0.55em] before:h-1 before:w-1 before:rounded-full before:bg-accent">
      {children}
    </li>
  );
}

/** Ordered list — numbered, matching the bullet rhythm. */
function Steps({ children }: { children: ReactNode }) {
  return (
    <ol className="ml-1 flex list-decimal flex-col gap-2.5 pl-5 marker:font-semibold marker:text-accent">
      {children}
    </ol>
  );
}

export function HelpView() {
  return (
    <div className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-stage">
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-5 py-8">
        <header className="mb-1 px-1">
          <h1 className="text-lg font-bold tracking-[0.16em] text-sky-200">
            ORBITER · HELP
          </h1>
          <p className="mt-1 text-[13px] text-inkmute">
            A getting-started guide — read top to bottom.
          </p>
        </header>

        <Section title="What this is">
          <p>
            Orbiter is a 2-axis (azimuth + elevation) camera turntable for
            photogrammetry. A phone camera on a motorized arm orbits an object
            that sits on a rotating platform, capturing many overlapping photos
            from known angles — the input for a 3D reconstruction (COLMAP).
          </p>
        </Section>

        <Section title="The workflow — do this in order">
          <Steps>
            <li>
              <Tag>Calibrate the motors</Tag> (right bar → Machine config → Step 1):
              the firmware measures each axis — steps per degree, which way it
              turns, its top speed and an angle-deviation profile — turning it
              full circle for 2–4 minutes. Take the arm off before calibrating
              EL. Until an axis has a result its moves are refused: the
              firmware would drive it the wrong way. <Tag>Motion speed</Tag>{' '}
              then shows °/s from the measured steps per degree.
            </li>
            <li>
              <Tag>Zero the encoders</Tag> (right bar → Machine config → Step 2):
              level the arm and press <Tag>Zero EL</Tag>; optionally{' '}
              <Tag>Zero AZ</Tag> to name the platform heading 0°. The geometry
              calibration stays blocked until the EL zero is set.
            </li>
            <li>
              <Tag>Calibrate geometry</Tag> (Steps 3–4): set the ChArUco board,
              pick an accuracy and <Tag>Calibrate from board</Tag>. The rig
              sweeps poses, detects the board, and solves where the camera sits
              at each angle.
            </li>
            <li>
              <Tag>Configure & run a scan</Tag> (left bar): set the Motion planner
              (elevation range / steps, azimuth step, actions) and press{' '}
              <Tag>Start scan</Tag> — the rig sweeps and shoots. Or take single
              shots with <Tag>Take shot</Tag>. Ring elevations must lie in the
              EL travel.
            </li>
            <li>
              <Tag>Review</Tag> in the <Tag>Library</Tag> tab: open a scan to see
              its cameras in 3D (click a frustum → full photo).
            </li>
            <li>
              <Tag>Export</Tag> for COLMAP: per scan,{' '}
              <Tag>Export SfM priors</Tag> (poses JSON) or{' '}
              <Tag>Download archive (+SfM)</Tag> (photos + manifest + priors).
            </li>
          </Steps>
        </Section>

        <Section title="Why calibration matters">
          <p>
            Without a geometry calibration the rig cannot compute accurate camera
            poses, so the exported priors are wrong and reconstruction suffers.
          </p>
          <p>
            Calibration is a one-time (per setup) ChArUco hand-eye solve. The only
            thing you enter by hand is the <Tag>ChArUco board spec</Tag> — squares,
            sizes, dictionary. Accuracy presets trade speed for a denser pose
            sweep (<Tag>Fine</Tag> = most accurate).
          </p>
          <p>
            A calibration describes the mechanics it was solved on — the rig
            model (GLB) the server loads. After a change of mechanics it shows
            amber, <Tag>from other mechanics</Tag>, and scans are refused until
            you calibrate again or <Tag>Reset calibration</Tag> (the rig is then
            drawn from the CAD model alone).
          </p>
        </Section>

        <Section title="Testing accuracy">
          <p>
            The <Tag>Test accuracy</Tag> button (Machine config) captures one photo
            at the current pose, detects the board, and compares the
            optically-measured pose to what the encoders predict via the
            calibration — reporting the delta (Δ rotation° / Δ position mm). A
            small delta means good agreement. (Needs a prior calibration.)
          </p>
        </Section>

        <Section title="Firmware, encoders and travel">
          <Bullets>
            <Bullet>
              <Tag>Encoders</Tag>: an AS5600 magnetic encoder on each axis, each
              on its own I2C bus — AZ on SDA 22 / SCL 23, EL on SDA 4 / SCL 16.
              The <Tag>Encoders</Tag> card (right bar) reads both: angle, magnet
              status and the idle level of each line. A LOW line is a short, a
              wire on GND or a DIR pin, or an unpowered chip; missing external
              pull-ups mean the sensor board has no supply.
            </Bullet>
            <Bullet>
              <Tag>EL travel</Tag> is −25…+90°: the arm meets the cross rail at
              about −28.8°. The firmware refuses a move outside it, the server
              validates moves, jogs and scan plans against it, and the navball
              shades the band below −25°. A <Tag>spin</Tag> is open loop: the
              server stops an EL spin a few degrees before the limits.
            </Bullet>
            <Bullet>
              <Tag>Speeds</Tag> are driver step rates (Hz, 1/16 microsteps); the
              °/s next to them comes from the steps per degree the server holds.
            </Bullet>
            <Bullet>
              <Tag>Updates over Wi-Fi</Tag>: the <Tag>Firmware</Tag> card uploads{' '}
              <Path>firmware/build/orbiter.bin</Path>. The device reboots into
              it; a new image confirms itself once its HTTP server is up, and
              if it does not within 120 s the bootloader goes back to the
              previous one. Encoder zeros in NVS are kept.
            </Bullet>
          </Bullets>
        </Section>

        <Section title="What the software is made of">
          <Bullets>
            <Bullet>
              <Tag>Server</Tag> (FastAPI / Python) — the single source of truth:
              holds the rig state, runs the calibration solver, stores scans +
              photos, and drives the camera.
            </Bullet>
            <Bullet>
              <Tag>3D scene over WebSocket</Tag> — the server computes the whole 3D
              scene and streams it; the browser only renders it (no 3D math in the
              browser).
            </Bullet>
            <Bullet>
              <Tag>Firmware (ESP32)</Tag> — a generic 2-axis IP actuator:
              closed-loop moves on the encoder readings, the EL travel limits,
              encoder zeros in NVS, updates over Wi-Fi.
            </Bullet>
            <Bullet>
              <Tag>Camera (phone)</Tag> — the CameraStreamer capture node app
              (or, legacy, IP Webcam): live preview, stills, and the phone IMU
              used as a redundant tilt indicator on the navball. The server
              detects which app answers on the camera URL.
            </Bullet>
          </Bullets>
        </Section>

        <Section title="Where things are stored">
          <Bullets>
            <Bullet>
              <Tag>Scans</Tag>: <Path>server/data/scans/&lt;scan_id&gt;/manifest.json</Path>{' '}
              — the scan document (<Path>sfm_priors.json</Path> appears there after
              export).
            </Bullet>
            <Bullet>
              <Tag>Photos</Tag>: a shared pool at{' '}
              <Path>server/data/captures/&lt;capture_id&gt;/</Path> (original.jpg +
              thumbnails + meta.json).
            </Bullet>
            <Bullet>
              <Tag>Config & calibration</Tag>:{' '}
              <Path>server/data/orbiter_state.json</Path> (board spec, derived
              geometry, intrinsics, etc.).
            </Bullet>
          </Bullets>
        </Section>

        <Section title="The phone camera">
          <p>
            Set the phone&apos;s address as the <Tag>camera URL</Tag> (Machine config
            → Connections, e.g. <Path>http://192.168.0.248:8080</Path>). The
            preview in the top-right corner shows the stream, its URL and the last
            error; click it for the full view.
          </p>
          <p>
            On the <Tag>capture node</Tag> every photo is shot on one locked
            manual profile (exposure, ISO, focus, white balance) and checked
            against it. Lock it once per setup in the full camera view:{' '}
            <Tag>Lock (meter)</Tag> meters the scene once and freezes it,{' '}
            <Tag>Re-lock saved</Tag> brings back the profile saved on the phone. A
            scan does not start on an unlocked phone. During a scan the rig moves
            on as soon as a frame is exposed; the photo is pulled off the phone in
            the background.
          </p>
          <p>
            A photo that cannot be taken fails its shot or scan point (shown in the
            log panel) — nothing is ever stored in its place.
          </p>
        </Section>

        <Section title="The navball">
          <p>
            To the left of the controls, the navball shows the live azimuth /
            elevation attitude. The purple marker is the phone-camera-based
            elevation estimate (a redundant sensor) and can be toggled off; it and
            its toggle only appear while the phone reports its IMU.
          </p>
        </Section>
      </div>
    </div>
  );
}
