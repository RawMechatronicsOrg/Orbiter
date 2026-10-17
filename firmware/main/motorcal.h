#pragma once

#include "esp_err.h"
#include "stepper.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Motor self-calibration, one axis at a time (ported from ActuatorController's
 * calib.c, which drives a single axis; here each axis keeps its own result).
 *
 * Four phases, run back to back:
 *
 *   COARSE   500 microsteps, measure the angle actually swept -> a first
 *            microsteps-per-degree figure, plus which DIR level makes the raw
 *            encoder angle count up (the handedness motion.c used to assume).
 *   SPEED    ramp the step rate while the axis spins free, comparing measured
 *            against commanded angular rate; bisect the slip edge. The last
 *            rate the rotor still tracked, times 0.7, becomes the speed ceiling.
 *   REFINE   many revolutions at cruise speed: total pulses over total angle.
 *   PROFILE  one slow revolution recording, per 5 deg bin of raw angle, how far
 *            the encoder drifts from where the pulse count says it should be —
 *            the off-centre-magnet / eccentric-pulley signature that
 *            motorcal_correct() subtracts back out.
 *
 * Every phase turns the axis through full revolutions, so the elevation arm
 * must be off. Angles in and out of this module are RAW encoder degrees, before
 * the encoder-zero offset: the deviations are fixed to the sensor and the
 * mechanics, not to wherever zero was last set.
 */

#define MOTORCAL_PROFILE_BINS 72   /* 5 deg resolution */

typedef enum {
    MOTORCAL_IDLE = 0,
    MOTORCAL_COARSE,
    MOTORCAL_SPEED,
    MOTORCAL_REFINE,
    MOTORCAL_PROFILE,
    MOTORCAL_DONE,
    MOTORCAL_FAILED,
} motorcal_phase_t;

typedef struct {
    bool     valid;                  /* a full run completed at least once      */
    float    steps_per_deg_coarse;   /* from the 500-step probe                 */
    float    steps_per_deg;          /* refined over many revolutions           */
    float    max_step_hz;            /* highest rate the rotor still tracked    */
    float    max_speed_dps;          /* = max_step_hz / steps_per_deg_coarse * 0.7 */
    int8_t   dir_sign;               /* +1 if DIR_CCW makes the raw angle grow  */
    uint16_t revolutions;            /* revolutions REFINE actually used        */
    int32_t  refine_steps;           /* pulses counted over those revolutions   */

    bool     has_profile;
    /** Per-bin angle error in degrees, indexed by raw angle, de-meaned. */
    float    profile_deg[MOTORCAL_PROFILE_BINS];
    float    profile_peak_deg;
    float    profile_rms_deg;
} motorcal_result_t;

/** Load both axes' stored calibration from NVS. Call once at boot. */
void motorcal_init(void);

/**
 * Run a full calibration of one axis. Blocks until it completes, fails or is
 * cancelled (minutes) — call it from the motion runner task, never from httpd.
 * The previous result stays in force unless every phase passes.
 *
 * @return ESP_OK when the run completed and was stored,
 *         ESP_ERR_INVALID_STATE if a run is already in flight,
 *         ESP_FAIL when a phase failed or the run was cancelled.
 */
esp_err_t motorcal_run(stepper_axis_t axis);

/** Ask the run in flight to stop at its next tick. No-op when idle. */
void motorcal_cancel(void);

/** True while a run is in flight. */
bool motorcal_active(void);

/**
 * State of the current or most recent run: its axis, phase, 0..1 progress,
 * the speed it is commanding (deg/s) and a short human-readable message.
 * Any pointer may be NULL.
 */
void motorcal_progress(stepper_axis_t *out_axis, motorcal_phase_t *out_phase,
                       float *out_progress, float *out_speed_dps,
                       char *out_msg, size_t msg_len);

/** Copy out an axis's stored result (check `.valid`). */
void motorcal_result(stepper_axis_t axis, motorcal_result_t *out);

/** +1 (default) or -1: whether DIR_CCW makes the axis's raw angle increase. */
int8_t motorcal_dir_sign(stepper_axis_t axis);

/** True once the axis has a stored calibration. */
bool motorcal_valid(stepper_axis_t axis);

/**
 * Calibrated speed ceiling as a step rate (max_speed_dps × steps_per_deg),
 * or 0 when the axis is uncalibrated.
 */
float motorcal_max_step_hz(stepper_axis_t axis);

/**
 * Linearise a raw encoder angle through the axis's deviation profile. Returns
 * `raw_deg` unchanged with no profile, with compensation off, and while a run
 * is in flight (calibration must measure the uncorrected axis).
 */
float motorcal_correct(stepper_axis_t axis, float raw_deg);

bool motorcal_compensation_enabled(stepper_axis_t axis);

/** Enable/disable profile compensation for an axis and persist the choice. */
void motorcal_set_compensation(stepper_axis_t axis, bool enabled);

const char *motorcal_phase_name(motorcal_phase_t p);

#ifdef __cplusplus
}
#endif
