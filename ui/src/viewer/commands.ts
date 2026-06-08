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

export interface Commands {
  move(az?: number, el?: number): void;
  motors(enabled: boolean): void;
  /** Zero / reset stepper encoders. Despite the name this is NOT the
   *  geometry/intrinsics calibration (that lives in Machine config →
   *  "Calibrate from board") — it's the firmware's encoder-position
   *  zeroing routine. Kept under the original protocol name to stay
   *  wire-compatible with the server. */
  calibrate(axis: Axis, mode?: string): void;
  rebootEsp(): void;
  setGeometry(fields: Record<string, number>): void;
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
  setMotionPlan(plan: Record<string, unknown>): void;
  setRenderPref(prefs: Record<string, boolean>): void;
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
    motors: (enabled) => client.sendCommand('motors', { enabled }),
    calibrate: (axis, mode = 'current') =>
      client.sendCommand('calibrate', { axis, mode }),
    rebootEsp: () => client.sendCommand('reboot_esp'),
    setGeometry: (fields) => client.sendCommand('set_geometry', fields),
    setBoardParams: (params) => client.sendCommand('set_board_params', params),
    calibrateGeometry: (preset) =>
      client.sendCommand('calibrate_geometry', { apply: true, preset }),
    setMotionPlan: (plan) =>
      client.sendCommand('set_motion_plan', { motion_plan: plan }),
    setRenderPref: (prefs) => client.sendCommand('set_render_pref', prefs),
    takeShot: () => client.sendCommand('take_shot'),
    deleteCapture: (captureId) =>
      client.sendCommand('delete_capture', { capture_id: captureId }),
    saveScan: () => client.sendCommand('save_scan'),
    newScan: () => client.sendCommand('new_scan'),
    recreateScan: () => client.sendCommand('recreate_scan'),
    startScan: (plan) =>
      client.sendCommand('start_scan', plan ? { motion_plan: plan } : {}),
    stopScan: () => client.sendCommand('stop_scan'),
    raw: (name, args = {}) => client.sendCommand(name, args),
  };
}
