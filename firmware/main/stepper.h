#pragma once

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    STEPPER_AZ = 0,
    STEPPER_EL = 1,
} stepper_axis_t;

typedef enum {
    DIR_CW  = 0,
    DIR_CCW = 1,
} stepper_dir_t;

/**
 * Initialise GPIO and timers for both axes.
 * Must be called before any other stepper function.
 */
void stepper_init(void);

/** Pull ENABLE low — both drivers active. */
void stepper_enable(void);

/** Pull ENABLE high — both drivers off (coils de-energised). */
void stepper_disable(void);

/**
 * Start a move on one axis.
 * Non-blocking — returns immediately, pulses are generated in hardware.
 *
 * @param axis     STEPPER_AZ or STEPPER_EL
 * @param steps    Number of full steps to move
 * @param dir      DIR_CW or DIR_CCW
 * @param step_hz  Step pulse frequency in Hz (pulses per second at motor)
 *
 * @return ESP_ERR_INVALID_STATE if the axis is already moving.
 */
esp_err_t stepper_move(stepper_axis_t axis, uint32_t steps,
                       stepper_dir_t dir, uint32_t step_hz);

/**
 * Change the step rate of an in-progress move without stopping the timer.
 *
 * Unlike `stepper_stop`+`stepper_move`, this just updates the periodic timer's
 * period in place via `esp_timer_restart`. The ISR pulse-edge state machine
 * is undisturbed, no microsteps are missed, no mechanical "stutter" at the
 * boundary. Use this to implement smooth ramp-up / ramp-down inside one
 * continuous `stepper_move`.
 *
 * @return ESP_OK on success, ESP_ERR_INVALID_STATE if the axis is idle,
 *         ESP_ERR_INVALID_ARG if step_hz == 0.
 */
esp_err_t stepper_set_speed(stepper_axis_t axis, uint32_t step_hz);

/**
 * Block until the axis finishes its move or timeout expires.
 *
 * @param timeout_ms  0 = wait forever
 * @return ESP_OK on completion, ESP_ERR_TIMEOUT if time ran out.
 */
esp_err_t stepper_wait(stepper_axis_t axis, uint32_t timeout_ms);

/** Immediately stop generating pulses on this axis. */
void stepper_stop(stepper_axis_t axis);

/** True if the axis is currently generating pulses. */
bool stepper_is_busy(stepper_axis_t axis);

/**
 * Start continuous rotation (non-blocking).
 * Runs until stepper_stop() is called.
 *
 * @return ESP_ERR_INVALID_STATE if the axis is already moving.
 */
esp_err_t stepper_spin(stepper_axis_t axis, stepper_dir_t dir, uint32_t step_hz);
