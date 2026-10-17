"""Demo motion for the Orbiter rig: the table turns at a sinusoidally varying
speed while the elevation arm swings up and down.

Talks to the firmware directly over HTTP (keep-alive): AZ is driven open-loop
by velocity through /spin (which retunes a running spin in place), EL follows
a sinusoidal position target with velocity feed-forward plus a proportional
correction from the EL encoder. The script enforces its own EL band and stops
both axes on exit, on Ctrl-C and on any error.

    python tools/demo_motion.py                      # 2 minutes, defaults
    python tools/demo_motion.py --seconds 300 --el-center 20 --el-amp 15
"""
import argparse
import http.client
import json
import math
import time

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--host", default="192.168.0.149")
ap.add_argument("--seconds", type=float, default=120.0)
ap.add_argument("--az-mean", type=float, default=20.0, help="AZ mean speed, deg/s")
ap.add_argument("--az-swing", type=float, default=15.0, help="AZ speed swing, +/- deg/s")
ap.add_argument("--az-period", type=float, default=10.0, help="AZ speed period, s")
ap.add_argument("--el-center", type=float, default=20.0, help="EL swing centre, deg")
ap.add_argument("--el-amp", type=float, default=15.0, help="EL swing amplitude, deg")
ap.add_argument("--el-period", type=float, default=8.0, help="EL swing period, s")
ap.add_argument("--el-min", type=float, default=-15.0, help="EL hard floor, deg")
ap.add_argument("--el-max", type=float, default=80.0, help="EL hard ceiling, deg")
ap.add_argument("--rate", type=float, default=10.0, help="update rate, Hz")
args = ap.parse_args()

conn = http.client.HTTPConnection(args.host, 80, timeout=3)


def call(method, path, body=None):
    global conn
    for attempt in range(2):
        try:
            data = json.dumps(body) if body is not None else None
            conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return json.loads(r.read() or b"{}")
        except (http.client.HTTPException, OSError):
            conn.close()
            conn = http.client.HTTPConnection(args.host, 80, timeout=3)
            if attempt:
                raise


def spin(axis, dps, steps_per_deg, up_dir, down_dir):
    hz = int(abs(dps) * steps_per_deg)
    if hz < 20:
        call("POST", "/spin/stop", {"axis": axis})
        return
    call("POST", "/spin", {"axis": axis, "dir": up_dir if dps > 0 else down_dir, "step_hz": hz})


state = call("GET", "/state")
mc = state["motor_cal"]
if not (mc["az"]["valid"] and mc["el"]["valid"]):
    raise SystemExit("motor calibration missing on an axis — run it first")
az_spd, el_spd = mc["az"]["steps_per_deg"], mc["el"]["steps_per_deg"]
# DIR that raises the user angle: dir_sign says whether CCW raises RAW, the EL
# sign whether the user angle follows raw.
el_sign = state["calibration"].get("el_sign", 1)
el_up = "ccw" if mc["el"]["dir_sign"] * el_sign > 0 else "cw"
el_down = "cw" if el_up == "ccw" else "ccw"
az_fwd = "ccw" if mc["az"]["dir_sign"] > 0 else "cw"
az_back = "cw" if az_fwd == "ccw" else "ccw"
el_cap = 0.8 * mc["el"]["max_speed_dps"]
KP = 1.5   # deg/s of correction per deg of EL error

call("POST", "/motors", {"enabled": True})
print(f"AZ {args.az_mean - args.az_swing:.0f}..{args.az_mean + args.az_swing:.0f} deg/s, "
      f"EL {args.el_center - args.el_amp:.0f}..{args.el_center + args.el_amp:.0f} deg "
      f"every {args.el_period:g} s, for {args.seconds:g} s", flush=True)

t0 = time.time()
period = 1.0 / args.rate
next_print = 0.0
try:
    while (t := time.time() - t0) < args.seconds:
        el = call("GET", "/state")["elevation"]["angle_deg"]
        if not (args.el_min <= el <= args.el_max):
            raise SystemExit(f"EL {el:.1f} deg left the safe band {args.el_min}..{args.el_max} — stopped")

        w_az = 2 * math.pi / args.az_period
        az_dps = args.az_mean + args.az_swing * math.sin(w_az * t)

        w_el = 2 * math.pi / args.el_period
        target = args.el_center + args.el_amp * math.sin(w_el * t)
        v_ff = args.el_amp * w_el * math.cos(w_el * t)
        el_dps = max(-el_cap, min(el_cap, v_ff + KP * (target - el)))

        spin("az", az_dps, az_spd, az_fwd, az_back)
        spin("el", el_dps, el_spd, el_up, el_down)

        if t >= next_print:
            print(f"{t:6.1f}s  AZ {az_dps:5.1f} deg/s   EL {el:6.2f} -> {target:6.2f} deg ({el_dps:+5.1f} deg/s)",
                  flush=True)
            next_print += 2.0
        time.sleep(max(0.0, period - ((time.time() - t0) - t)))
finally:
    call("POST", "/spin/stop", {"axis": "both"})
    print("stopped", flush=True)
