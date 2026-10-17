#pragma once

#include "esp_err.h"
#include <stdint.h>

typedef enum {
    ENC_AZ = 0,
    ENC_EL = 1,
} enc_axis_t;

/**
 * Initialise both buses and devices (Orbiter-Simple carrier, J3/J4):
 *   ENC_AZ → AS5600 on I2C0 (SDA=22, SCL=23, addr 0x36, 12-bit)
 *   ENC_EL → AS5600 on I2C1 (SDA=4,  SCL=16, addr 0x36, 12-bit)
 *
 * Must be called before encoder_read_*.
 */
esp_err_t encoder_init(void);

/**
 * Read the raw angle: 12-bit (0..4095) on both axes — AS5600 ANGLE register,
 * 360 / 4096 deg/count. Use encoder_read_angle for canonical degrees.
 * Returns an I2C error code when the sensor does not answer.
 */
esp_err_t encoder_read_raw(enc_axis_t axis, uint16_t *out_raw);

/**
 * Read angle in degrees (0.0–360.0) with median filtering, per-axis driver.
 */
esp_err_t encoder_read_angle(enc_axis_t axis, float *out_deg);

/**
 * Read magnet sanity flags (AS5600 STATUS reg 0x0B re-mapped), both axes:
 *     bit0 = magnet missing (!MD)
 *     bit1 = magnet too weak (ML)
 *     bit2 = magnet too strong (MH)
 *
 * Field/magnet diagnostics (AGC, magnitude) are in encoder_read_diag().
 */
esp_err_t encoder_read_errors(enc_axis_t axis, uint8_t *out_flags);

/**
 * Read diagnostic registers and log a one-line summary:
 * AS5600 STATUS/AGC/MAGNITUDE — MD/ML/MH + agc 0..255 + magnitude.
 *
 * Does not affect normal angle reading. Verbose per-frame trace via menuconfig
 *   "Orbiter Configuration" → ORBITER_AS5600_I2C_TRACE
 */
esp_err_t encoder_read_diag(enc_axis_t axis);
