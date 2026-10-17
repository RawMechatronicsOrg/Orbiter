"""Demo motion for the Orbiter rig: the table and the elevation arm play
motion patterns — a sinusoidal speed, Lissajous figures, a helix, a rocking
swing, nods between table steps — one pattern or all of them as a tour.

Talks to the firmware directly over HTTP (keep-alive) and drives both axes by
velocity through /spin, which retunes a running spin in place. Each pattern
gives, per axis, either a speed (open loop) or a position target with its
velocity feed-forward, corrected by a proportional term from that axis's
encoder. Directions and step rates come from the stored motor calibration and
EL sign. The script keeps its own EL band and stops both axes on exit, on
Ctrl-C and on any error.

    python tools/demo_motion.py                         # the tour, ~3.5 min
    python tools/demo_motion.py --pattern lissajous --seconds 90
    python tools/demo_motion.py --list
"""
import argparse
import http.client
import json
import math
import time

TAU = 2 * math.pi
EL_C = 20.0          # centre of every EL swing, deg


def sine(t, az0):
    """Table at a speed swinging 5..35 deg/s, arm 5..35 deg every 8 s."""
    return (("vel", 20 + 15 * math.sin(TAU * t / 10)),
            (EL_C + 15 * math.sin(TAU * t / 8), 15 * TAU / 8 * math.cos(TAU * t / 8)))


def lissajous(t, az0):
    """Table rocks +/-60 deg, arm swings 3:2 against it: a Lissajous figure."""
    a, wa = 60.0, TAU / 9
    b, wb = 15.0, TAU / 6
    return (("pos", az0 + a * math.sin(wa * t), a * wa * math.cos(wa * t)),
            (EL_C + b * math.sin(wb * t + math.pi / 2), b * wb * math.cos(wb * t + math.pi / 2)))


def figure8(t, az0):
    """Table rocks +/-45 deg, arm at twice the rate: a figure eight."""
    a, w = 45.0, TAU / 8
    b = 12.0
    return (("pos", az0 + a * math.sin(w * t), a * w * math.cos(w * t)),
            (EL_C + b * math.sin(2 * w * t), b * 2 * w * math.cos(2 * w * t)))


def helix(t, az0):
    """Table turns steadily, the arm climbs and descends slowly: a helix scan."""
    period, lo, hi = 30.0, 5.0, 40.0
    ph = (t % period) / period
    tri = 2 * ph if ph < 0.5 else 2 - 2 * ph                  # 0..1..0
    slope = (hi - lo) * 2 / period * (1 if ph < 0.5 else -1)
    return (("vel", 25.0), (lo + (hi - lo) * tri, slope))


def swing(t, az0):
    """Table rocks back and forth through zero speed, the arm rising with it."""
    w = TAU / 12
    return (("vel", 35 * math.sin(w * t)),
            (EL_C + 15 * math.sin(w * t - math.pi / 2), 15 * w * math.cos(w * t - math.pi / 2)))


def nod(t, az0):
    """Table steps 45 deg with a smooth start and stop; the arm nods at each stop."""
    step, cycle, move = 45.0, 4.0, 1.6
    k, ph = divmod(t, cycle)
    if ph < move:                                              # cosine-eased step
        u = ph / move
        pos = az0 + step * (k + (1 - math.cos(math.pi * u)) / 2)
        vel = step * math.pi / (2 * move) * math.sin(math.pi * u)
        el, el_v = EL_C, 0.0
    else:                                                      # dwell: one nod
        pos, vel = az0 + step * (k + 1), 0.0
        u = (ph - move) / (cycle - move)
        el = EL_C + 12 * math.sin(math.pi * u) ** 2
        el_v = 12 * math.pi / (cycle - move) * math.sin(TAU * u)
    return (("pos", pos, vel), (el, el_v))


PATTERNS = {"sine": sine, "lissajous": lissajous, "figure8": figure8,
            "helix": helix, "swing": swing, "nod": nod}
TOUR = [("sine", 30), ("lissajous", 36), ("figure8", 32), ("helix", 40), ("swing", 36), ("nod", 32)]

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--host", default="192.168.0.149")
ap.add_argument("--pattern", default="tour", choices=["tour", *PATTERNS])
ap.add_argument("--seconds", type=float, default=None, help="one pattern's duration (default 60)")
ap.add_argument("--el-min", type=float, default=-15.0, help="EL hard floor, deg")
ap.add_argument("--el-max", type=float, default=80.0, help="EL hard ceiling, deg")
ap.add_argument("--rate", type=float, default=10.0, help="update rate, Hz")
ap.add_argument("--list", action="store_true", help="list the patterns and exit")
args = ap.parse_args()

if args.list:
    for name, fn in PATTERNS.items():
        print(f"{name:10s} {fn.__doc__}")
    raise SystemExit

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


state = call("GET", "/state")
mc = state["motor_cal"]
if not (mc["az"]["valid"] and mc["el"]["valid"]):
    raise SystemExit("motor calibration missing on an axis — run it first")

# DIR that raises each user angle: dir_sign says whether CCW raises RAW; the
# EL user angle follows raw times el_sign, AZ follows raw.
el_sign = state["calibration"].get("el_sign", 1)
AXES = {
    "az": {"spd": mc["az"]["steps_per_deg"], "cap": 0.6 * mc["az"]["max_speed_dps"],
           "up": "ccw" if mc["az"]["dir_sign"] > 0 else "cw"},
    "el": {"spd": mc["el"]["steps_per_deg"], "cap": 0.6 * mc["el"]["max_speed_dps"],
           "up": "ccw" if mc["el"]["dir_sign"] * el_sign > 0 else "cw"},
}
KP = 1.5   # deg/s of correction per deg of position error


def drive(axis, dps):
    a = AXES[axis]
    dps = max(-a["cap"], min(a["cap"], dps))
    hz = int(abs(dps) * a["spd"])
    if hz < 20:
        call("POST", "/spin/stop", {"axis": axis})
        return dps
    down = "cw" if a["up"] == "ccw" else "ccw"
    call("POST", "/spin", {"axis": axis, "dir": a["up"] if dps > 0 else down, "step_hz": hz})
    return dps


plan = TOUR if args.pattern == "tour" else [(args.pattern, args.seconds or 60.0)]
call("POST", "/motors", {"enabled": True})
period = 1.0 / args.rate

# AZ is unwrapped from the firmware's [0, 360) so position patterns can turn
# through it; each pattern starts from wherever the table is.
az_prev = state["azimuth"]["angle_deg"]
az_unwrapped = az_prev
try:
    for name, seconds in plan:
        fn = PATTERNS[name]
        print(f"--- {name} ({seconds:g} s): {fn.__doc__}", flush=True)
        az0 = az_unwrapped
        el_start = call("GET", "/state")["elevation"]["angle_deg"]
        t0 = time.time()
        next_print = 0.0
        while (t := time.time() - t0) < seconds:
            s = call("GET", "/state")
            az, el = s["azimuth"]["angle_deg"], s["elevation"]["angle_deg"]
            az_unwrapped += (az - az_prev + 540) % 360 - 180
            az_prev = az
            if not (args.el_min <= el <= args.el_max):
                raise SystemExit(f"EL {el:.1f} deg left the safe band {args.el_min}..{args.el_max} — stopped")

            (az_mode, *az_cmd), (el_t, el_ff) = fn(t, az0)
            # Ease the first 2 s from where the arm is onto the pattern.
            blend = min(1.0, t / 2.0)
            el_t = el_start + (el_t - el_start) * blend
            el_ff *= blend

            if az_mode == "vel":
                az_dps = drive("az", az_cmd[0])
            else:
                az_t, az_ff = az_cmd
                az_dps = drive("az", az_ff + KP * (az_t - az_unwrapped))
            el_dps = drive("el", el_ff + KP * (el_t - el))

            if t >= next_print:
                print(f"{name:9s} {t:5.1f}s  AZ {az:6.1f} deg ({az_dps:+5.1f}/s)   "
                      f"EL {el:5.1f} -> {el_t:5.1f} deg ({el_dps:+5.1f}/s)", flush=True)
                next_print += 5.0
            time.sleep(max(0.0, period - ((time.time() - t0) - t)))
finally:
    call("POST", "/spin/stop", {"axis": "both"})
    print("stopped", flush=True)
