/**
 * KSP-style navball — a single combined gimbal showing azimuth + elevation.
 *
 * A textured sphere (sky / ground hemispheres, dense heading + pitch grid)
 * rendered in its own tiny R3F canvas; it rotates with the rig's az/el while
 * a small fixed reticle marks the current attitude.
 */

import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';

const d2r = (deg: number) => (deg * Math.PI) / 180;

/** Shortest signed azimuth delta (degrees). */
function dAz(cur: number, tgt: number): number {
  return ((tgt - cur + 540) % 360) - 180;
}

/**
 * Project target attitude onto the 2D navball disc relative to current (centre).
 * Small-angle approximation — good for jog / move targets within ~90°.
 */
export function navballTargetMarker(
  azCur: number,
  elCur: number,
  azTgt: number,
  elTgt: number,
  radiusPx: number,
  centerPx: number,
): { x: number; y: number } | null {
  const da = dAz(azCur, azTgt);
  const de = elTgt - elCur;
  if (Math.abs(da) < 0.08 && Math.abs(de) < 0.08) return null;
  const scale = radiusPx * 0.88;
  return {
    x: centerPx + scale * Math.sin(d2r(da)) * Math.cos(d2r(de)),
    y: centerPx - scale * Math.sin(d2r(de)),
  };
}

/** Equirectangular navball texture drawn procedurally onto a 2D canvas. */
function makeNavballTexture(): THREE.Texture {
  const w = 2048;
  const h = 1024;
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext('2d')!;

  const yOfLat = (lat: number) => h / 2 - (lat / 90) * (h / 2);
  const xOfLon = (lon: number) => (lon / 360) * w;
  const hLine = (y: number) => {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  };
  const vLine = (x: number) => {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  };

  // Sky (top) and ground (bottom), each a gentle gradient toward the horizon.
  const sky = ctx.createLinearGradient(0, 0, 0, h / 2);
  sky.addColorStop(0, '#0c3866');
  sky.addColorStop(1, '#5a9fd4');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h / 2);

  const ground = ctx.createLinearGradient(0, h / 2, 0, h);
  ground.addColorStop(0, '#9a7038');
  ground.addColorStop(1, '#2e2014');
  ctx.fillStyle = ground;
  ctx.fillRect(0, h / 2, w, h / 2);

  // Minor grid — every 10°.
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  for (let lat = -80; lat <= 80; lat += 10) hLine(yOfLat(lat));
  for (let lon = 0; lon < 360; lon += 10) vLine(xOfLon(lon));

  // Major grid — every 30°.
  ctx.strokeStyle = 'rgba(255,255,255,0.42)';
  ctx.lineWidth = 2.5;
  for (let lat = -60; lat <= 60; lat += 30) hLine(yOfLat(lat));
  for (let lon = 0; lon < 360; lon += 30) vLine(xOfLon(lon));

  // Horizon.
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 5;
  hLine(h / 2);

  // Pre-distort labels by cos(lat) so the equirectangular → sphere mapping
  // (which stretches the texture horizontally by 1/cos(lat) near the poles)
  // ends up with text at roughly equator-width everywhere. Without this the
  // ±60° pitch numbers and grid would appear smeared / doubled-width.
  const drawAt = (text: string, x: number, y: number, latForScale: number) => {
    const sx = Math.max(0.15, Math.cos(d2r(latForScale)));
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(sx, 1);
    ctx.fillText(text, 0, 0);
    ctx.restore();
  };

  // Heading numbers along the horizon (lat=0 → cos=1, drawAt = passthrough).
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.font = '30px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let lon = 0; lon < 360; lon += 30) {
    drawAt(String(lon), xOfLon(lon), h / 2 - 22, 0);
  }

  // Pitch numbers at major latitudes — pre-shrunk so they don't smear at ±60°.
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.font = '24px ui-monospace, monospace';
  for (const lat of [-60, -30, 30, 60]) {
    const label = (lat > 0 ? '+' : '') + lat;
    for (const lon of [0, 90, 180, 270]) {
      drawAt(label, xOfLon(lon) + 48, yOfLat(lat), lat);
    }
  }

  // Cardinal letters — sit just below the horizon (effectively lat~0).
  ctx.fillStyle = '#fde68a';
  ctx.font = 'bold 56px sans-serif';
  for (const [lon, label] of [[0, 'N'], [90, 'E'], [180, 'S'], [270, 'W']] as const) {
    drawAt(label as string, xOfLon(lon as number), h / 2 + 58, 0);
  }

  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

function NavBallSphere({ az, el }: { az: number; el: number }) {
  const texture = useMemo(makeNavballTexture, []);
  // Heading scrolls the ball around its vertical axis; elevation pitches it.
  return (
    <mesh rotation={[d2r(el), d2r(-az), 0]}>
      <sphereGeometry args={[1, 64, 48]} />
      <meshBasicMaterial map={texture} toneMapped={false} />
    </mesh>
  );
}

export function NavBall({
  az,
  el,
  targetAz,
  targetEl,
  phoneEl,
  size = 132,
}: {
  az: number;
  el: number;
  targetAz?: number | null;
  targetEl?: number | null;
  /** Phone IMU: the phone's calibrated ESTIMATE of the rig elevation (deg) —
   *  built to track el so the marker sits on the reticle when the phone agrees
   *  with the rig and drifts off when they diverge. */
  phoneEl?: number | null;
  /** Phone IMU: bank/roll about the optical axis (deg). */
  phoneRoll?: number | null;
  size?: number;
}) {
  const center = size / 2;
  const targetPt =
    targetAz != null && targetEl != null
      ? navballTargetMarker(az, el, targetAz, targetEl, center, center)
      : null;
  // Phone tilt: we only know the lens's inclination, not its yaw — project
  // at the rig's current azimuth so the purple marker tracks above/below
  // the reticle by (phone_pitch − el). The closer to centre, the better the
  // rig's reported EL matches the phone's actual horizon-relative pitch.
  const phonePt =
    phoneEl != null
      ? navballTargetMarker(az, el, az, phoneEl, center, center)
      : null;
  const scale = size / 132;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <div className="absolute inset-0 overflow-hidden rounded-full border-2 border-[#324158] bg-[#0c1424]">
        <Canvas
          frameloop="demand"
          camera={{ position: [0, 0, 2.7], fov: 45 }}
          gl={{ alpha: true }}
        >
          <NavBallSphere az={az} el={el} />
        </Canvas>
      </div>
      <svg
        className="pointer-events-none absolute inset-0"
        viewBox={`0 0 ${size} ${size}`}
      >
        {/* Commanded move target (hollow ring). */}
        {targetPt && (
          <g
            stroke="#38bdf8"
            strokeWidth={2 * scale}
            fill="none"
            opacity={0.95}
          >
            <circle cx={targetPt.x} cy={targetPt.y} r={7 * scale} />
            <line
              x1={targetPt.x - 10 * scale}
              y1={targetPt.y}
              x2={targetPt.x - 4 * scale}
              y2={targetPt.y}
            />
            <line
              x1={targetPt.x + 4 * scale}
              y1={targetPt.y}
              x2={targetPt.x + 10 * scale}
              y2={targetPt.y}
            />
            <line
              x1={targetPt.x}
              y1={targetPt.y - 10 * scale}
              x2={targetPt.x}
              y2={targetPt.y - 4 * scale}
            />
            <line
              x1={targetPt.x}
              y1={targetPt.y + 4 * scale}
              x2={targetPt.x}
              y2={targetPt.y + 10 * scale}
            />
          </g>
        )}
        {/* Current attitude — fixed centre reticle. */}
        <g stroke="#fbbf24" strokeWidth={1.6 * scale} fill="none">
          <circle cx={center} cy={center} r={5 * scale} />
          <line
            x1={center - 9 * scale}
            y1={center}
            x2={center - 5 * scale}
            y2={center}
          />
          <line
            x1={center + 5 * scale}
            y1={center}
            x2={center + 9 * scale}
            y2={center}
          />
          <line
            x1={center}
            y1={center - 9 * scale}
            x2={center}
            y2={center - 5 * scale}
          />
        </g>
        {/* Phone IMU — purple horizon bar at the phone's elevation ESTIMATE (a
            redundant EL indicator). It sits on the reticle when the phone
            agrees with the rig's reported el, and drifts off if they diverge. */}
        {phonePt && (
          <g
            stroke="#c084fc"
            strokeWidth={2 * scale}
            fill="none"
            opacity={0.95}
          >
            <line
              x1={phonePt.x - 18 * scale}
              y1={phonePt.y}
              x2={phonePt.x - 4 * scale}
              y2={phonePt.y}
            />
            <line
              x1={phonePt.x + 4 * scale}
              y1={phonePt.y}
              x2={phonePt.x + 18 * scale}
              y2={phonePt.y}
            />
            <circle cx={phonePt.x} cy={phonePt.y} r={2.5 * scale} fill="#c084fc" />
          </g>
        )}
      </svg>
    </div>
  );
}
