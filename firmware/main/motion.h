#pragma once

#include "esp_err.h"
#include "stepper.h"
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    MOTION_IDLE,
    MOTION_MOVING,
    MOTION_SPINNING,
    MOTION_ERROR,
} motion_state_t;

typedef struct {
    float az_deg;
    float el_deg;
} motion_pos_t;

/**
 * Initialise motion controller.
 * Calls stepper_init() and encoder_init() internally.
 */
void motion_init(void);

/** Current machine state. */
motion_state_t motion_get_state(void);

/**
 * Current position.
 * Uses encoder readings when available, falls back to dead-reckoning.
 */
motion_pos_t motion_get_position(void);

/**
 * Move to target angles — closed-loop position control.
 *
 * Blocks until both axes settle within tolerance or `timeout_ms` expires. The
 * controller reads the output-shaft encoder each control tick and drives the
 * steppers at an error-proportional, slew-limited step rate; the encoder is
 * the sole position reference (no steps-per-revolution constant). Internally
 * the loop yields with vTaskDelay every tick, so this is normally invoked from
 * the motion_runner task while httpd keeps serving requests.
 *
 * @param az_deg      Target azimuth (ignored if has_az == false)
 * @param el_deg      Target elevation (ignored if has_el == false)
 * @param has_az      Move azimuth axis
 * @param has_el      Move elevation axis
 * @param timeout_ms  Max wait time (0 = use the internal hard ceiling)
 * @param out         Final position written here (may differ from target on timeout)
 *
 * @return ESP_OK         — target reached within tolerance
 *         ESP_ERR_TIMEOUT — time expired before arrival
 *         ESP_ERR_INVALID_STATE — another move/spin already in progress
 */
esp_err_t motion_move(float az_deg, float el_deg,
                      bool has_az, bool has_el,
                      uint32_t timeout_ms,
                      motion_pos_t *out);

/** Enable or disable both stepper drivers (coils). */
void motion_set_motors(bool enabled);

/** True if motors are currently enabled. */
bool motion_motors_enabled(void);

/**
 * Set software zero for one or both axes (backward-compat).
 * Captures current raw encoder reading as new zero; equivalent to
 * motion_set_calibration(do_az, do_el, CAL_MODE_CURRENT, 0, 0).
 */
void motion_zero(bool do_az, bool do_el);

/**
 * Calibration modes for motion_set_calibration().
 */
typedef enum {
    CAL_MODE_CURRENT  = 0,  /* capture current encoder raw_deg as new zero */
    CAL_MODE_EXPLICIT = 1,  /* use manual_*_raw values directly */
    CAL_MODE_RESET    = 2,  /* restore factory defaults (az=0, el=138.296) */
} motion_cal_mode_t;

/**
 * Update calibration offsets and persist to NVS.
 *
 * @param do_az          Apply to azimuth axis
 * @param do_el          Apply to elevation axis
 * @param mode           One of motion_cal_mode_t
 * @param manual_az_raw  Used only when mode == CAL_MODE_EXPLICIT
 * @param manual_el_raw  Used only when mode == CAL_MODE_EXPLICIT
 *
 * @return ESP_OK on success, error code if NVS write fails.
 */
esp_err_t motion_set_calibration(bool do_az, bool do_el,
                                 motion_cal_mode_t mode,
                                 float manual_az_raw, float manual_el_raw);

/** Get current azimuth zero offset (raw encoder degrees). */
float motion_get_az_zero_raw(void);

/** Get current elevation zero offset (raw encoder degrees). */
float motion_get_el_zero_raw(void);

/**
 * Start continuous rotation on one axis (non-blocking).
 * A second axis may spin while the first is active; /move remains blocked until all spins stop.
 * @return ESP_ERR_INVALID_STATE if this axis is already spinning, move mutex is busy, or stepper_spin fails.
 */
esp_err_t motion_spin(stepper_axis_t axis, stepper_dir_t dir, uint32_t step_hz);

/**
 * Stop continuous rotation for selected axes.
 * @param stop_az  Stop azimuth if spinning
 * @param stop_el  Stop elevation if spinning
 */
void motion_spin_stop(bool stop_az, bool stop_el);

/** True while azimuth axis is in continuous spin mode. */
bool motion_spinning_az(void);

/** True while elevation axis is in continuous spin mode. */
bool motion_spinning_el(void);
