#include "motorcal.h"
#include "encoder.h"
#include "log_bus.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs.h"
#include <math.h>
#include <string.h>

static const char *TAG = "motorcal";

/* ── Tunables (ActuatorController's values; see its calib.c for the why) ─── */

/* COARSE: enough pulses to sweep a clearly measurable arc behind a reduction,
 * slow enough that nothing can slip. 500 microsteps = 8 deg on AZ (62/deg),
 * 5.6 deg on EL (89/deg). */
#define COARSE_STEPS          500u
#define COARSE_HZ             200u
#define COARSE_MIN_SWEEP_DEG  2.0f    /* below this: nothing actually moved   */
#define COARSE_TIMEOUT_MS     20000

/* SPEED: geometric ramp; each rung waits out the acceleration transient, then
 * measures over a fixed window. After the first slip the edge is bisected
 * between the last rung that tracked and the one that did not. */
#define SPEED_START_HZ        200u
#define SPEED_RAMP_FACTOR     1.18f
#define SPEED_SETTLE_MS       180
#define SPEED_WINDOW_MS       260     /* minimum; see SPEED_MIN_SWEEP_DEG    */
/*
 * Measure each rung over at least this much angle, not just over a fixed time.
 * The pendulum's axis was direct drive, where 200 Hz is 22 deg/s and 260 ms
 * sweeps ~6 deg. Behind Orbiter's 7:1 and 10:1 belts the same rung is 2-3
 * deg/s, 260 ms sweeps ~0.6 deg = 7 encoder counts, and a single count of
 * quantisation (0.088 deg) swings the ratio by 15% — the ramp then "slipped"
 * on its first rungs while the axis tracked cleanly at 4x the rate. 5 deg
 * keeps that error under 2%.
 */
#define SPEED_MIN_SWEEP_DEG   5.0f
#define SPEED_MAX_WINDOW_MS   4000
#define SPEED_TRACK_RATIO     0.90f   /* measured/commanded below this = slip */
#define SPEED_MARGIN          0.70f   /* derate applied to the tracked edge   */
#define SPEED_BISECT_PROBES   4u
#define SPEED_CEIL_HZ         STEPPER_MAX_STEP_HZ
#define SPEED_CEIL_DPS        900.0f
#define SPEED_TIMEOUT_MS      90000

/* REFINE: nominal revolution count, capped by a time budget so a heavily
 * geared-down axis gets fewer turns rather than a twenty-minute phase; ramps
 * into cruise from the rate COARSE proved the axis starts at. */
#define REFINE_REVS           25u
#define REFINE_MIN_REVS       5u
#define REFINE_BUDGET_S       180.0f
#define REFINE_CRUISE_FRAC    0.80f   /* of the calibrated max speed          */
#define REFINE_START_HZ       COARSE_HZ
#define REFINE_ACCEL_DPS2     300.0f
#define REFINE_TIMEOUT_MS     ((int)(REFINE_BUDGET_S * 1000) + 30000)

/* PROFILE: slow enough that every 5 deg bin collects samples at the tick rate,
 * floored so one revolution stays inside the timeout. */
#define PROFILE_MAX_DPS       45.0f
#define PROFILE_MIN_DPS       3.0f
#define PROFILE_FRAC          0.25f   /* of the calibrated max speed          */
#define PROFILE_MIN_FILL      0.75f   /* fraction of bins that must get samples */
#define PROFILE_TIMEOUT_MS    240000

#define SETTLE_MS             250     /* let the axis stop ringing before reading */
#define ENC_FAIL_ABORT        10      /* consecutive bad reads that kill a run */

/*
 * Tick period. One FreeRTOS tick at the 100 Hz default, so 10 ms: fast enough
 * that the axis turns far less than 180 deg between ticks (the unwrapping
 * below needs that) and that every profile bin collects ~10 samples at
 * PROFILE_MAX_DPS.
 */
#define TICK_MS               10

#define BIN_WIDTH_DEG         (360.0f / (float)MOTORCAL_PROFILE_BINS)

/* ── NVS ────────────────────────────────────────────────────────────────── */

#define NVS_NAMESPACE  "orbiter"
#define CAL_MAGIC      0x4D43414Cu   /* "MCAL" */
#define CAL_VERSION    1u

static const char *const NVS_KEYS[2] = { "mcal_az", "mcal_el" };
static const char *const AXIS_NAMES[2] = { "az", "el" };

/* Persisted image, separate from motorcal_result_t so the in-RAM struct can be
 * reshaped without misreading an old blob. */
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t bins;
    float    steps_per_deg_coarse;
    float    steps_per_deg;
    float    max_step_hz;
    float    max_speed_dps;
    int8_t   dir_sign;
    uint8_t  has_profile;
    uint8_t  compensation_enabled;
    uint8_t  _pad;
    uint16_t revolutions;
    uint16_t _pad2;
    int32_t  refine_steps;
    float    profile_peak_deg;
    float    profile_rms_deg;
    float    profile_deg[MOTORCAL_PROFILE_BINS];
} cal_blob_t;

/* ── State ──────────────────────────────────────────────────────────────── */

/*
 * s_res is what the rest of the firmware reads, one per axis; s_run is the run
 * in progress. A re-calibration that dies in its last phase must leave the
 * previously working numbers intact, so a run only lands in s_res once every
 * phase has passed. s_res is read from the pose and control tasks while the
 * runner may commit to it, hence the spinlock.
 */
static motorcal_result_t s_res[2];
static bool              s_comp[2] = { true, true };
static portMUX_TYPE      s_res_lk = portMUX_INITIALIZER_UNLOCKED;

static motorcal_result_t s_run;
static stepper_axis_t    s_axis;          /* axis of the current/last run   */
static volatile bool     s_active;
static volatile bool     s_cancel;

static motorcal_phase_t s_phase;
static int              s_step;           /* sub-step within the phase       */
static int64_t          s_phase_start_us;
static int64_t          s_wait_until_us;
static char             s_msg[96];
static portMUX_TYPE     s_msg_lk = portMUX_INITIALIZER_UNLOCKED;
static float            s_speed_dps;      /* what this run is commanding now */
static int              s_enc_fail;

/* Unwrapped angle accumulation: each tick's shortest-path delta folded into a
 * running total, so sweeps of many turns stay unambiguous. */
static float  s_prev_raw;
static bool   s_have_prev;
static double s_sweep_deg;                 /* since the last mark            */

/* SPEED phase */
static uint32_t s_ramp_hz;
static uint32_t s_last_good_hz;
static uint32_t s_bad_hz;                  /* lowest rate seen to slip; 0 while climbing */
static uint32_t s_probes_left;
static double   s_rung_start_deg;
static int64_t  s_rung_start_us;

/* REFINE phase */
static double   s_refine_target_deg;
static int64_t  s_refine_ramp_us;
static uint16_t s_refine_revs;

/* PROFILE phase: per-bin sums of the two raw quantities (measured sweep and
 * pulses), reduced once the sweep closes — see profile_reduce. */
static float    s_bin_x_sum[MOTORCAL_PROFILE_BINS];
static float    s_bin_step_sum[MOTORCAL_PROFILE_BINS];
static uint16_t s_bin_count[MOTORCAL_PROFILE_BINS];
static float    s_prof_x_end;
static float    s_prof_steps_end;

/* ── Small helpers ──────────────────────────────────────────────────────── */

static int64_t now_us(void) { return esp_timer_get_time(); }

static float wrap360(float d)
{
    d = fmodf(d, 360.0f);
    return (d < 0.0f) ? d + 360.0f : d;
}

/** Shortest signed delta on a circle, in (-180, +180]. */
static float angle_delta(float from, float to)
{
    return fmodf(to - from + 540.0f, 360.0f) - 180.0f;
}

static void set_msg(const char *m)
{
    portENTER_CRITICAL(&s_msg_lk);
    strncpy(s_msg, m, sizeof(s_msg) - 1);
    s_msg[sizeof(s_msg) - 1] = '\0';
    portEXIT_CRITICAL(&s_msg_lk);
}

/*
 * Restart the sweep accumulator and the pulse counter at the current angle.
 * s_have_prev stays true: this tick's reading is already the previous one, so
 * the next tick measures from exactly the mark instant.
 */
static void mark(void)
{
    s_sweep_deg = 0.0;
    stepper_reset_position(s_axis);
}

static void enter_phase(motorcal_phase_t p, const char *msg)
{
    s_phase          = p;
    s_step           = 0;
    s_phase_start_us = now_us();
    set_msg(msg);
    LOGX_EMIT_I(TAG, "%s: %s", AXIS_NAMES[s_axis], msg);
}

static void fail(const char *reason)
{
    stepper_stop(s_axis);
    s_speed_dps = 0.0f;
    s_phase     = MOTORCAL_FAILED;
    set_msg(reason);
    LOGX_EMIT_E(TAG, "%s calibration failed: %s", AXIS_NAMES[s_axis], reason);
}

static void wait_ms(int ms) { s_wait_until_us = now_us() + (int64_t)ms * 1000; }
static bool waited(void)    { return now_us() >= s_wait_until_us; }

static bool phase_timed_out(int limit_ms)
{
    return now_us() - s_phase_start_us > (int64_t)limit_ms * 1000;
}

/** Direction that makes the raw angle increase, per the COARSE probe. */
static stepper_dir_t dir_up(void)
{
    return (s_run.dir_sign >= 0) ? DIR_CCW : DIR_CW;
}

static uint32_t dps_to_hz_coarse(float dps)
{
    float hz = dps * s_run.steps_per_deg_coarse;
    if (hz < 1.0f) hz = 1.0f;
    return (uint32_t)(hz + 0.5f);
}

/*
 * Start continuous rotation from rest. The stepper refuses a spin while a
 * previous one is still running, so every (re)start stops first.
 */
static esp_err_t spin_from_rest(stepper_dir_t dir, uint32_t hz)
{
    stepper_stop(s_axis);
    return stepper_spin(s_axis, dir, hz);
}

/* ── Phase: COARSE ──────────────────────────────────────────────────────── */

static void tick_coarse(void)
{
    switch (s_step) {
    case 0:
        mark();
        stepper_stop(s_axis);
        if (stepper_move(s_axis, COARSE_STEPS, DIR_CCW, COARSE_HZ) != ESP_OK) {
            fail("could not start the coarse probe");
            return;
        }
        s_speed_dps = 0.0f;   /* unknown until we know steps-per-degree */
        s_step = 1;
        break;

    case 1:
        if (!stepper_is_busy(s_axis)) {
            wait_ms(SETTLE_MS);
            s_step = 2;
        } else if (phase_timed_out(COARSE_TIMEOUT_MS)) {
            fail("coarse probe did not finish");
        }
        break;

    case 2: {
        if (!waited()) return;
        float sweep = (float)fabs(s_sweep_deg);
        if (sweep < COARSE_MIN_SWEEP_DEG) {
            fail("no motion detected - check driver enable, wiring and magnet");
            return;
        }
        /* We commanded CCW, so the sign of the sweep says whether CCW counts
         * the raw angle up. */
        s_run.dir_sign             = (s_sweep_deg >= 0.0) ? 1 : -1;
        s_run.steps_per_deg_coarse = (float)COARSE_STEPS / sweep;

        LOGX_EMIT_I(TAG, "%s coarse: %.2f deg over %u steps -> %.3f steps/deg, dir_sign=%d",
                    AXIS_NAMES[s_axis], (double)sweep, (unsigned)COARSE_STEPS,
                    (double)s_run.steps_per_deg_coarse, s_run.dir_sign);

        enter_phase(MOTORCAL_SPEED, "ramping step rate to find the tracking limit");
        break;
    }
    default: break;
    }
}

/* ── Phase: SPEED ───────────────────────────────────────────────────────── */

static void speed_finish(void)
{
    stepper_stop(s_axis);
    s_speed_dps = 0.0f;

    if (s_last_good_hz == 0) {
        fail("rotor did not track even the lowest test rate");
        return;
    }
    s_run.max_step_hz   = (float)s_last_good_hz;
    s_run.max_speed_dps = (float)s_last_good_hz / s_run.steps_per_deg_coarse * SPEED_MARGIN;

    LOGX_EMIT_I(TAG, "%s speed: tracked to %u Hz%s -> max %.1f deg/s (x%.2f margin)",
                AXIS_NAMES[s_axis], (unsigned)s_last_good_hz,
                s_bad_hz ? " (edge bracketed)" : " (no slip found, ceiling)",
                (double)s_run.max_speed_dps, (double)SPEED_MARGIN);

    float cruise = s_run.max_speed_dps * REFINE_CRUISE_FRAC;
    uint32_t affordable = (uint32_t)(cruise * REFINE_BUDGET_S / 360.0f);
    if (affordable < REFINE_MIN_REVS) affordable = REFINE_MIN_REVS;
    s_refine_revs       = (uint16_t)((affordable > REFINE_REVS) ? REFINE_REVS : affordable);
    s_refine_target_deg = 360.0 * (double)s_refine_revs;

    enter_phase(MOTORCAL_REFINE, "measuring steps per degree over many turns");
}

static void tick_speed(void)
{
    if (phase_timed_out(SPEED_TIMEOUT_MS)) { speed_finish(); return; }

    switch (s_step) {
    case 0:
        s_ramp_hz      = SPEED_START_HZ;
        s_last_good_hz = 0;
        s_bad_hz       = 0;
        s_probes_left  = SPEED_BISECT_PROBES;
        mark();
        if (spin_from_rest(dir_up(), s_ramp_hz) != ESP_OK) {
            fail("could not start the speed ramp");
            return;
        }
        wait_ms(SPEED_SETTLE_MS);
        s_step = 1;
        break;

    case 1: {  /* let the rung's acceleration transient pass */
        if (!waited()) return;
        s_rung_start_deg = s_sweep_deg;
        s_rung_start_us  = now_us();
        float expected = (float)s_ramp_hz / s_run.steps_per_deg_coarse;
        int window = SPEED_WINDOW_MS;
        if (expected > 0.0f) {
            float need_ms = SPEED_MIN_SWEEP_DEG / expected * 1000.0f;
            if (need_ms > (float)window) window = (int)need_ms;
        }
        if (window > SPEED_MAX_WINDOW_MS) window = SPEED_MAX_WINDOW_MS;
        wait_ms(window);
        s_step = 2;
        break;
    }

    case 2: {  /* measure the rung */
        if (!waited()) return;
        double dt = (double)(now_us() - s_rung_start_us) / 1e6;
        if (dt <= 0.0) { s_step = 1; return; }

        float measured = (float)(fabs(s_sweep_deg - s_rung_start_deg) / dt);
        float expected = (float)s_ramp_hz / s_run.steps_per_deg_coarse;
        float ratio    = (expected > 0.0f) ? measured / expected : 0.0f;
        s_speed_dps    = measured;

        LOGX_EMIT_I(TAG, "%s speed rung %u Hz: expected %.1f, measured %.1f dps (%.0f%%)",
                    AXIS_NAMES[s_axis], (unsigned)s_ramp_hz, (double)expected,
                    (double)measured, (double)(ratio * 100.0f));

        uint32_t next;
        if (ratio < SPEED_TRACK_RATIO) {
            /* Slipped. The very first rung failing means the axis cannot turn. */
            if (s_last_good_hz == 0) { speed_finish(); return; }
            s_bad_hz = s_ramp_hz;
            if (s_probes_left == 0) { speed_finish(); return; }
        } else {
            s_last_good_hz = s_ramp_hz;
            if (expected >= SPEED_CEIL_DPS || s_ramp_hz >= SPEED_CEIL_HZ) {
                speed_finish();   /* tracking at the ceiling: no edge to find */
                return;
            }
            if (s_bad_hz == 0) {
                next = (uint32_t)((float)s_ramp_hz * SPEED_RAMP_FACTOR);
                if (next <= s_ramp_hz) next = s_ramp_hz + 1;
                if (next > SPEED_CEIL_HZ) next = SPEED_CEIL_HZ;
                s_ramp_hz = next;
                if (stepper_set_speed(s_axis, s_ramp_hz) != ESP_OK) { speed_finish(); return; }
                wait_ms(SPEED_SETTLE_MS);
                s_step = 1;
                break;
            }
            if (s_probes_left == 0) { speed_finish(); return; }
        }

        /* Bracketed: halve the gap until the two ends are within 1%. */
        if (s_bad_hz <= s_last_good_hz + s_last_good_hz / 100u) { speed_finish(); return; }
        s_probes_left--;
        next = s_last_good_hz + (s_bad_hz - s_last_good_hz) / 2u;
        if (next <= s_last_good_hz) { speed_finish(); return; }
        s_ramp_hz = next;

        LOGX_EMIT_I(TAG, "%s speed: bracketing %u..%u Hz, probing %u", AXIS_NAMES[s_axis],
                    (unsigned)s_last_good_hz, (unsigned)s_bad_hz, (unsigned)s_ramp_hz);

        /* Re-approach from a standstill: a slipped stepper keeps no phase, and
         * measuring out of that state measures the recovery, not the probe. */
        if (spin_from_rest(dir_up(), s_ramp_hz) != ESP_OK) { speed_finish(); return; }
        wait_ms(SPEED_SETTLE_MS);
        s_step = 1;
        break;
    }
    default: break;
    }
}

/* ── Phase: REFINE ──────────────────────────────────────────────────────── */

static void tick_refine(void)
{
    if (phase_timed_out(REFINE_TIMEOUT_MS)) {
        fail("multi-turn measurement did not finish in time");
        return;
    }

    switch (s_step) {
    case 0:
        mark();
        if (spin_from_rest(dir_up(), REFINE_START_HZ) != ESP_OK) {
            fail("could not start the multi-turn run");
            return;
        }
        s_refine_ramp_us = now_us();
        s_speed_dps      = (float)REFINE_START_HZ / s_run.steps_per_deg_coarse;
        s_step = 1;
        break;

    case 1: {
        /* Ramp on the clock rather than per tick. */
        float cruise = s_run.max_speed_dps * REFINE_CRUISE_FRAC;
        float start  = (float)REFINE_START_HZ / s_run.steps_per_deg_coarse;
        float want   = start + REFINE_ACCEL_DPS2 *
                       (float)((double)(now_us() - s_refine_ramp_us) / 1e6);
        if (want > cruise) want = cruise;
        if (want > s_speed_dps) {
            s_speed_dps = want;
            stepper_set_speed(s_axis, dps_to_hz_coarse(s_speed_dps));
        }
        if (fabs(s_sweep_deg) >= s_refine_target_deg) {
            stepper_stop(s_axis);
            s_speed_dps = 0.0f;
            wait_ms(SETTLE_MS);
            s_step = 2;
        }
        break;
    }

    case 2: {
        if (!waited()) return;
        /* Both totals include the post-stop settle: pulses emitted over the
         * angle the axis actually turned. */
        float sweep = (float)fabs(s_sweep_deg);
        int32_t pos = stepper_position(s_axis);
        int32_t steps = (pos < 0) ? -pos : pos;

        if (sweep < 180.0f || steps <= 0) {
            fail("multi-turn measurement collected no usable motion");
            return;
        }
        s_run.steps_per_deg = (float)steps / sweep;
        s_run.revolutions   = s_refine_revs;
        s_run.refine_steps  = steps;

        LOGX_EMIT_I(TAG, "%s refine: %ld steps over %.2f deg (%u turns) -> %.4f steps/deg",
                    AXIS_NAMES[s_axis], (long)steps, (double)sweep,
                    (unsigned)s_refine_revs, (double)s_run.steps_per_deg);

        enter_phase(MOTORCAL_PROFILE, "sweeping one slow turn to map angle deviation");
        break;
    }
    default: break;
    }
}

/* ── Phase: PROFILE ─────────────────────────────────────────────────────── */

static void profile_reset_bins(void)
{
    memset(s_bin_x_sum,    0, sizeof(s_bin_x_sum));
    memset(s_bin_step_sum, 0, sizeof(s_bin_step_sum));
    memset(s_bin_count,    0, sizeof(s_bin_count));
    s_prof_x_end     = 0.0f;
    s_prof_steps_end = 0.0f;
}

static void profile_sample(float raw_deg)
{
    /*
     * Both quantities signed and in the same sense — "degrees the angle has
     * gained". stepper_position() counts in the DIR sense (+ for CCW), so on an
     * axis where CCW counts the angle down it runs negative while the angle
     * climbs; multiplying by dir_sign puts it back in the angle's frame.
     */
    float steps = (float)stepper_position(s_axis) * (float)s_run.dir_sign;
    float x     = (float)s_sweep_deg;

    int bin = (int)(wrap360(raw_deg) / BIN_WIDTH_DEG);
    if (bin < 0) bin = 0;
    if (bin >= MOTORCAL_PROFILE_BINS) bin = MOTORCAL_PROFILE_BINS - 1;

    s_bin_x_sum[bin]    += x;
    s_bin_step_sum[bin] += steps;
    s_bin_count[bin]++;

    s_prof_x_end     = x;
    s_prof_steps_end = steps;
}

/**
 * Turn the per-bin sums into a gap-free deviation profile.
 *
 * Per sample, error = (measured sweep) - (sweep the pulse count implies). The
 * pulses-per-degree for the second term comes from this sweep's own closure:
 * over one full revolution the deviation returns to where it started, so total
 * pulses over total measured angle is the scale factor, and using it forces
 * the error to close at zero. (A least-squares line would also eat part of the
 * first harmonic — over one period a sinusoid correlates with a ramp.)
 */
static bool profile_reduce(void)
{
    if (!(fabsf(s_prof_x_end) > 180.0f) || !(fabsf(s_prof_steps_end) > 0.0f))
        return false;

    float k = s_prof_steps_end / s_prof_x_end;   /* both carry the sweep's sign */
    if (!(k > 0.0f)) return false;

    bool filled[MOTORCAL_PROFILE_BINS];
    int  n_filled = 0;
    for (int i = 0; i < MOTORCAL_PROFILE_BINS; i++) {
        if (s_bin_count[i] == 0) { filled[i] = false; s_run.profile_deg[i] = 0.0f; continue; }
        float n          = (float)s_bin_count[i];
        float mean_x     = s_bin_x_sum[i]    / n;
        float mean_steps = s_bin_step_sum[i] / n;
        s_run.profile_deg[i] = mean_x - mean_steps / k;
        filled[i] = true;
        n_filled++;
    }

    if ((float)n_filled < PROFILE_MIN_FILL * (float)MOTORCAL_PROFILE_BINS) {
        LOGX_EMIT_E(TAG, "%s profile: only %d/%d bins sampled",
                    AXIS_NAMES[s_axis], n_filled, MOTORCAL_PROFILE_BINS);
        return false;
    }

    /* Fill gaps by interpolating circularly between the nearest filled bins. */
    const int N = MOTORCAL_PROFILE_BINS;
    for (int i = 0; i < N; i++) {
        if (filled[i]) continue;
        int back = 0, fwd = 0;
        while (!filled[((i - back - 1) % N + N) % N] && back < N) back++;
        while (!filled[(i + fwd + 1) % N] && fwd < N) fwd++;
        int lo = ((i - back - 1) % N + N) % N;
        int hi = (i + fwd + 1) % N;
        float t = (float)(back + 1) / (float)(back + fwd + 2);
        s_run.profile_deg[i] = s_run.profile_deg[lo]
                             + t * (s_run.profile_deg[hi] - s_run.profile_deg[lo]);
    }

    /* Re-centre: the profile is a shape, not an offset. */
    float mean = 0.0f;
    for (int i = 0; i < N; i++) mean += s_run.profile_deg[i];
    mean /= (float)N;

    float peak = 0.0f, sumsq = 0.0f;
    for (int i = 0; i < N; i++) {
        s_run.profile_deg[i] -= mean;
        float a = fabsf(s_run.profile_deg[i]);
        if (a > peak) peak = a;
        sumsq += s_run.profile_deg[i] * s_run.profile_deg[i];
    }
    s_run.profile_peak_deg = peak;
    s_run.profile_rms_deg  = sqrtf(sumsq / (float)N);
    s_run.has_profile      = true;
    return true;
}

static void cal_save(stepper_axis_t axis);

static void tick_profile(float raw_deg)
{
    if (phase_timed_out(PROFILE_TIMEOUT_MS)) {
        fail("deviation sweep did not finish in time");
        return;
    }

    switch (s_step) {
    case 0: {
        float dps = s_run.max_speed_dps * PROFILE_FRAC;
        if (dps > PROFILE_MAX_DPS) dps = PROFILE_MAX_DPS;
        if (dps < PROFILE_MIN_DPS) dps = PROFILE_MIN_DPS;

        profile_reset_bins();
        mark();
        if (spin_from_rest(dir_up(), dps_to_hz_coarse(dps)) != ESP_OK) {
            fail("could not start the deviation sweep");
            return;
        }
        s_speed_dps = dps;
        s_step = 1;
        break;
    }

    case 1:
        profile_sample(raw_deg);
        if (fabs(s_sweep_deg) >= 360.0) {
            stepper_stop(s_axis);
            s_speed_dps = 0.0f;
            wait_ms(SETTLE_MS);
            s_step = 2;
        }
        break;

    case 2:
        if (!waited()) return;
        if (!profile_reduce()) {
            fail("deviation sweep collected too few samples per bin");
            return;
        }
        LOGX_EMIT_I(TAG, "%s profile: peak %.3f deg, rms %.3f deg", AXIS_NAMES[s_axis],
                    (double)s_run.profile_peak_deg, (double)s_run.profile_rms_deg);
        /* Every phase passed - the run becomes the axis's calibration. */
        s_run.valid = true;
        portENTER_CRITICAL(&s_res_lk);
        s_res[s_axis] = s_run;
        portEXIT_CRITICAL(&s_res_lk);
        cal_save(s_axis);
        enter_phase(MOTORCAL_DONE, "calibration complete");
        break;

    default: break;
    }
}

/* ── NVS ────────────────────────────────────────────────────────────────── */

static void cal_save(stepper_axis_t axis)
{
    motorcal_result_t r;
    motorcal_result(axis, &r);

    cal_blob_t b = {
        .magic                = CAL_MAGIC,
        .version              = CAL_VERSION,
        .bins                 = MOTORCAL_PROFILE_BINS,
        .steps_per_deg_coarse = r.steps_per_deg_coarse,
        .steps_per_deg        = r.steps_per_deg,
        .max_step_hz          = r.max_step_hz,
        .max_speed_dps        = r.max_speed_dps,
        .dir_sign             = r.dir_sign,
        .has_profile          = r.has_profile ? 1 : 0,
        .compensation_enabled = s_comp[axis] ? 1 : 0,
        .revolutions          = r.revolutions,
        .refine_steps         = r.refine_steps,
        .profile_peak_deg     = r.profile_peak_deg,
        .profile_rms_deg      = r.profile_rms_deg,
    };
    memcpy(b.profile_deg, r.profile_deg, sizeof(b.profile_deg));

    nvs_handle_t h;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_open failed: %s", esp_err_to_name(err));
        return;
    }
    err = nvs_set_blob(h, NVS_KEYS[axis], &b, sizeof(b));
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);

    if (err != ESP_OK) ESP_LOGE(TAG, "%s: nvs save failed: %s", AXIS_NAMES[axis], esp_err_to_name(err));
    else               ESP_LOGI(TAG, "%s: calibration saved to NVS", AXIS_NAMES[axis]);
}

static void cal_load(stepper_axis_t axis)
{
    s_res[axis].dir_sign = 1;   /* the pre-calibration assumption */

    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) return;

    cal_blob_t b;
    size_t len = sizeof(b);
    esp_err_t err = nvs_get_blob(h, NVS_KEYS[axis], &b, &len);
    nvs_close(h);

    if (err != ESP_OK || len != sizeof(b)) return;
    if (b.magic != CAL_MAGIC || b.version != CAL_VERSION || b.bins != MOTORCAL_PROFILE_BINS) {
        ESP_LOGW(TAG, "%s: stored calibration has an incompatible layout - ignoring",
                 AXIS_NAMES[axis]);
        return;
    }

    motorcal_result_t *r = &s_res[axis];
    r->valid                = true;
    r->steps_per_deg_coarse = b.steps_per_deg_coarse;
    r->steps_per_deg        = b.steps_per_deg;
    r->max_step_hz          = b.max_step_hz;
    r->max_speed_dps        = b.max_speed_dps;
    r->dir_sign             = (b.dir_sign < 0) ? -1 : 1;
    r->revolutions          = b.revolutions;
    r->refine_steps         = b.refine_steps;
    r->has_profile          = b.has_profile != 0;
    r->profile_peak_deg     = b.profile_peak_deg;
    r->profile_rms_deg      = b.profile_rms_deg;
    memcpy(r->profile_deg, b.profile_deg, sizeof(r->profile_deg));
    s_comp[axis] = b.compensation_enabled != 0;

    ESP_LOGI(TAG, "%s: loaded %.4f steps/deg, max %.1f dps, dir_sign=%d, profile=%s",
             AXIS_NAMES[axis], (double)r->steps_per_deg, (double)r->max_speed_dps,
             r->dir_sign, r->has_profile ? "yes" : "no");
}

void motorcal_init(void)
{
    cal_load(STEPPER_AZ);
    cal_load(STEPPER_EL);
    set_msg("idle");
}

/* ── Run ────────────────────────────────────────────────────────────────── */

static bool phase_running(void)
{
    return s_phase == MOTORCAL_COARSE || s_phase == MOTORCAL_SPEED ||
           s_phase == MOTORCAL_REFINE || s_phase == MOTORCAL_PROFILE;
}

static void tick(float raw_deg, bool enc_ok)
{
    if (!enc_ok) {
        if (++s_enc_fail >= ENC_FAIL_ABORT) fail("encoder became unreadable");
        return;
    }
    s_enc_fail = 0;

    if (s_have_prev) s_sweep_deg += (double)angle_delta(s_prev_raw, raw_deg);
    s_prev_raw  = raw_deg;
    s_have_prev = true;

    switch (s_phase) {
    case MOTORCAL_COARSE:  tick_coarse();         break;
    case MOTORCAL_SPEED:   tick_speed();          break;
    case MOTORCAL_REFINE:  tick_refine();         break;
    case MOTORCAL_PROFILE: tick_profile(raw_deg); break;
    default: break;
    }
}

esp_err_t motorcal_run(stepper_axis_t axis)
{
    if (axis != STEPPER_AZ && axis != STEPPER_EL) return ESP_ERR_INVALID_ARG;
    if (s_active) return ESP_ERR_INVALID_STATE;

    s_axis   = axis;
    s_cancel = false;
    s_active = true;

    memset(&s_run, 0, sizeof(s_run));
    s_run.dir_sign = 1;
    s_enc_fail     = 0;
    s_speed_dps    = 0.0f;
    s_prev_raw     = 0.0f;
    s_have_prev    = false;
    s_sweep_deg    = 0.0;

    enter_phase(MOTORCAL_COARSE, "probing steps per degree");

    const TickType_t period = pdMS_TO_TICKS(TICK_MS) ? pdMS_TO_TICKS(TICK_MS) : 1;
    TickType_t last = xTaskGetTickCount();
    while (phase_running()) {
        if (s_cancel) {
            fail("cancelled");
            break;
        }
        uint16_t raw = 0;
        bool ok = encoder_read_raw((enc_axis_t)axis, &raw) == ESP_OK;
        tick(ok ? (float)raw * (360.0f / 4096.0f) : 0.0f, ok);
        vTaskDelayUntil(&last, period);
    }

    stepper_stop(axis);
    s_speed_dps = 0.0f;
    s_active    = false;
    return (s_phase == MOTORCAL_DONE) ? ESP_OK : ESP_FAIL;
}

void motorcal_cancel(void)
{
    if (s_active) s_cancel = true;
}

bool motorcal_active(void) { return s_active; }

void motorcal_progress(stepper_axis_t *out_axis, motorcal_phase_t *out_phase,
                       float *out_progress, float *out_speed_dps,
                       char *out_msg, size_t msg_len)
{
    if (out_axis)      *out_axis = s_axis;
    if (out_phase)     *out_phase = s_phase;
    if (out_speed_dps) *out_speed_dps = s_speed_dps;

    if (out_progress) {
        /* Fixed weights per phase; within a phase, whatever it drives toward. */
        float p = 0.0f;
        switch (s_phase) {
        case MOTORCAL_COARSE:
            p = 0.05f;
            break;
        case MOTORCAL_SPEED: {
            float span = logf((float)SPEED_CEIL_HZ / (float)SPEED_START_HZ);
            float at   = (s_ramp_hz > SPEED_START_HZ)
                       ? logf((float)s_ramp_hz / (float)SPEED_START_HZ) : 0.0f;
            p = 0.10f + 0.15f * ((span > 0.0f) ? (at / span) : 0.0f);
            break;
        }
        case MOTORCAL_REFINE:
            p = 0.25f + 0.50f * (float)((s_refine_target_deg > 0.0)
                ? fabs(s_sweep_deg) / s_refine_target_deg : 0.0);
            break;
        case MOTORCAL_PROFILE:
            p = 0.75f + 0.25f * (float)(fabs(s_sweep_deg) / 360.0);
            break;
        case MOTORCAL_DONE:
            p = 1.0f;
            break;
        default:
            p = 0.0f;
            break;
        }
        if (p > 1.0f) p = 1.0f;
        *out_progress = p;
    }

    if (out_msg && msg_len) {
        portENTER_CRITICAL(&s_msg_lk);
        strncpy(out_msg, s_msg, msg_len - 1);
        out_msg[msg_len - 1] = '\0';
        portEXIT_CRITICAL(&s_msg_lk);
    }
}

/* ── Results ────────────────────────────────────────────────────────────── */

void motorcal_result(stepper_axis_t axis, motorcal_result_t *out)
{
    if (!out || axis > STEPPER_EL) return;
    portENTER_CRITICAL(&s_res_lk);
    *out = s_res[axis];
    portEXIT_CRITICAL(&s_res_lk);
}

bool motorcal_valid(stepper_axis_t axis)
{
    return axis <= STEPPER_EL && s_res[axis].valid;
}

int8_t motorcal_dir_sign(stepper_axis_t axis)
{
    if (axis > STEPPER_EL) return 1;
    return (s_res[axis].dir_sign < 0) ? -1 : 1;
}

float motorcal_max_step_hz(stepper_axis_t axis)
{
    if (axis > STEPPER_EL) return 0.0f;
    float hz;
    portENTER_CRITICAL(&s_res_lk);
    const motorcal_result_t *r = &s_res[axis];
    hz = (r->valid && r->max_speed_dps > 0.0f && r->steps_per_deg > 0.0f)
       ? r->max_speed_dps * r->steps_per_deg : 0.0f;
    portEXIT_CRITICAL(&s_res_lk);
    return hz;
}

float motorcal_correct(stepper_axis_t axis, float raw_deg)
{
    if (axis > STEPPER_EL || !s_comp[axis] || (s_active && s_axis == axis)) return raw_deg;

    float e0, e1, f;
    portENTER_CRITICAL(&s_res_lk);
    const motorcal_result_t *r = &s_res[axis];
    if (!r->has_profile) {
        portEXIT_CRITICAL(&s_res_lk);
        return raw_deg;
    }
    /* Bin i covers [i*W, (i+1)*W) and its value belongs at the bin CENTRE,
     * so interpolation runs between centres. */
    float t  = wrap360(raw_deg) / BIN_WIDTH_DEG - 0.5f;
    int   i0 = (int)floorf(t);
    f  = t - (float)i0;
    i0 = ((i0 % MOTORCAL_PROFILE_BINS) + MOTORCAL_PROFILE_BINS) % MOTORCAL_PROFILE_BINS;
    int i1 = (i0 + 1) % MOTORCAL_PROFILE_BINS;
    e0 = r->profile_deg[i0];
    e1 = r->profile_deg[i1];
    portEXIT_CRITICAL(&s_res_lk);

    return wrap360(raw_deg - (e0 + f * (e1 - e0)));
}

bool motorcal_compensation_enabled(stepper_axis_t axis)
{
    return axis <= STEPPER_EL && s_comp[axis];
}

void motorcal_set_compensation(stepper_axis_t axis, bool enabled)
{
    if (axis > STEPPER_EL || s_comp[axis] == enabled) return;
    s_comp[axis] = enabled;
    if (s_res[axis].valid) cal_save(axis);
    LOGX_EMIT_I(TAG, "%s profile compensation %s", AXIS_NAMES[axis],
                enabled ? "enabled" : "disabled");
}

const char *motorcal_phase_name(motorcal_phase_t p)
{
    switch (p) {
        case MOTORCAL_COARSE:  return "coarse";
        case MOTORCAL_SPEED:   return "speed";
        case MOTORCAL_REFINE:  return "refine";
        case MOTORCAL_PROFILE: return "profile";
        case MOTORCAL_DONE:    return "done";
        case MOTORCAL_FAILED:  return "failed";
        case MOTORCAL_IDLE:
        default:               return "idle";
    }
}
