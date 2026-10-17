/**
 * Typed command senders over the WebSocket. Mirrors `orbiter_server/commands.py`.
 * Commands are the only channel for mutating server state.
 *
 * Wrappers exist for the commands the UI calls today. Anything else can go
 * through `raw` until it earns a typed wrapper.
 */

import type { WsClient } from './WsClient';
import { useViewerStore } from './modelStore';
import { num } from './ui';

export type Axis = 'az' | 'el' | 'both';

/**
 * One eye of the binocular pair. Mirrors `orbiter_model._DEFAULT_EYE`.
 *
 * ORIENTATION ORDER: flips act in the sensor frame and are applied BEFORE
 * `quarter_turns_cw`. See `StereoView.eyeTransform` for the CSS that renders
 * it and `commands._cmd_set_stereo_rig` for the authoritative contract.
 */
export interface StereoEye {
  /** Upstream camserver id (`cam2`, `cam5`, ...). Empty until assigned. */
  camera_id: string;
  /** 0..3 — clockwise 90° steps applied after the flips. */
  quarter_turns_cw: number;
  flip_h: boolean;
  flip_v: boolean;
}

export interface Commands {
  move(az?: number, el?: number): void;
  /** Bump one axis by `deltaDeg` from its live reading. The server stops an
   *  EL jog at the travel limit and refuses one starting far outside it. */
  jog(axis: 'az' | 'el', deltaDeg: number): void;
  motors(enabled: boolean): void;
  /** Continuous rotation of one axis at a driver step rate (Hz) inside the
   *  band of `model.axes`. The server refuses an EL spin without the EL zero
   *  and stops one heading out of the travel. */
  spin(axis: 'az' | 'el', dir: 'cw' | 'ccw', stepHz: number): void;
  spinStop(axis: Axis): void;
  /** Firmware motor self-calibration of one axis (Machine config → Step 1).
   *  EL turns full circles, so the server refuses it without `armRemoved`. */
  motorCalibrate(axis: 'az' | 'el', armRemoved: boolean): void;
  motorCalibrateCancel(): void;
  /** Apply (or not) the axis' measured angle-deviation profile. */
  motorCompensation(axis: 'az' | 'el', enabled: boolean): void;
  /** Set or reset the firmware encoder zero (`calibrate_encoder`):
   *  `current` takes the present reading as 0, `reset` restores the
   *  firmware's default zero. Not the geometry calibration (Machine config →
   *  "Calibrate from board"). */
  zeroEncoder(axis: Axis, mode?: 'current' | 'reset'): void;
  rebootFirmware(): void;
  /** Rig geometry overrides and kinematic offsets (`set_machine_config`). */
  setMachineConfig(fields: Record<string, number>): void;
  /** Update the persisted ChArUco board params. Only the keys present in
   *  `params` are applied server-side. */
  setBoardParams(params: {
    charuco_squares_x?: number;
    charuco_squares_y?: number;
    charuco_square_length_mm?: number;
    charuco_marker_length_mm?: number;
    aruco_dict_id?: number;
  }): void;
  /** Run the ChArUco hand-eye geometry calibration and apply the result.
   *  `preset` controls the pose count / accuracy tradeoff
   *  (fast ≈ 9 poses, normal ≈ 24, full ≈ 32). */
  calibrateGeometry(preset: 'fast' | 'normal' | 'full'): void;
  /** Forget the geometry calibration — the rig goes back to the CAD-derived
   *  geometry (camera intrinsics are kept). */
  resetCalibration(): void;
  setMotionPlan(plan: Record<string, unknown>): void;
  /** Set the per-axis closed-loop top step rate (Hz) — the Machine-config
   *  speed knobs. Either or both axes; the server clamps to the band it
   *  publishes in `model.axes` and forwards the value to the firmware with the
   *  next /move. */
  setMoveSpeed(azHzMax?: number, elHzMax?: number): void;
  setRenderPref(prefs: Record<string, boolean>): void;
  /** Update the binocular pair config (Stereo tab). Only the keys present are
   *  applied server-side; each eye merges onto its current values. The server
   *  refuses a payload that points both eyes at the same camera. */
  setStereoRig(rig: {
    host?: string;
    token?: string;
    baseline_mm?: number;
    left?: Partial<StereoEye>;
    right?: Partial<StereoEye>;
  }): void;
  takeShot(): void;
  /** Delete one capture from the active scan session by its `capture_id`.
   *  Server removes it from `model.captures`, deletes the pool files and
   *  rewrites the active manifest; the scene frustum/photo-card and the UI
   *  list update via the resulting model/scene broadcast. */
  deleteCapture(captureId: string): void;
  saveScan(): void;
  newScan(): void;
  recreateScan(): void;
  /** Run the MotionPlanner sweep. Pass the current plan to apply + start in
   *  one click — the server applies it before launching (no set/start race). */
  startScan(plan?: Record<string, unknown>): void;
  /** Ask the running scan to abort at the next iteration. */
  stopScan(): void;
  /** Load a STORED scan into the 3D view (review frustums); `null` unloads.
   *  Stays on the device service — it reads manifests via orbiter_core. */
  setActiveSession(scanId: string | null): void;
  raw(name: string, args?: Record<string, unknown>): void;
}

export function makeCommands(client: WsClient): Commands {
  return {
    move: (az, el) => {
      const m = useViewerStore.getState().model;
      useViewerStore.getState().patchModel({
        move_target_az: az ?? num(m, 'az'),
        move_target_el: el ?? num(m, 'el'),
      });
      client.sendCommand('move', { az, el });
    },
    jog: (axis, deltaDeg) => client.sendCommand('jog', { axis, delta_deg: deltaDeg }),
    motors: (enabled) => client.sendCommand('motors', { enabled }),
    spin: (axis, dir, stepHz) =>
      client.sendCommand('spin', { axis, dir, step_hz: Math.round(stepHz) }),
    spinStop: (axis) => client.sendCommand('spin_stop', { axis }),
    motorCalibrate: (axis, armRemoved) =>
      client.sendCommand('motor_calibrate', { axis, arm_removed: armRemoved }),
    motorCalibrateCancel: () => client.sendCommand('motor_calibrate_cancel'),
    motorCompensation: (axis, enabled) =>
      client.sendCommand('motor_compensation', { axis, enabled }),
    zeroEncoder: (axis, mode = 'current') =>
      client.sendCommand('calibrate_encoder', { axis, mode }),
    rebootFirmware: () => client.sendCommand('reboot_firmware'),
    setMachineConfig: (fields) => client.sendCommand('set_machine_config', fields),
    setBoardParams: (params) => client.sendCommand('set_board_params', params),
    calibrateGeometry: (preset) =>
      client.sendCommand('calibrate_geometry', { apply: true, preset }),
    resetCalibration: () => client.sendCommand('reset_calibration'),
    setMotionPlan: (plan) =>
      client.sendCommand('set_motion_plan', { motion_plan: plan }),
    setMoveSpeed: (azHzMax, elHzMax) => {
      const args: Record<string, number> = {};
      if (azHzMax !== undefined) args.az_hz_max = Math.round(azHzMax);
      if (elHzMax !== undefined) args.el_hz_max = Math.round(elHzMax);
      client.sendCommand('set_move_speed', args);
    },
    setRenderPref: (prefs) => client.sendCommand('set_render_pref', prefs),
    setStereoRig: (rig) => client.sendCommand('set_stereo_rig', rig),
    takeShot: () => client.sendCommand('take_shot'),
    deleteCapture: (captureId) =>
      client.sendCommand('delete_capture', { capture_id: captureId }),
    saveScan: () => client.sendCommand('save_scan'),
    newScan: () => client.sendCommand('new_scan'),
    recreateScan: () => client.sendCommand('recreate_scan'),
    startScan: (plan) =>
      client.sendCommand('start_scan', plan ? { motion_plan: plan } : {}),
    stopScan: () => client.sendCommand('stop_scan'),
    setActiveSession: (scanId) =>
      client.sendCommand('set_active_session', { scan_id: scanId }),
    raw: (name, args = {}) => client.sendCommand(name, args),
  };
}
