#pragma once

#include "esp_err.h"
#include <stdint.h>

typedef enum {
    ENC_AZ = 0,
    ENC_EL = 1,
} enc_axis_t;

/**
 * Initialise both buses and devices:
 *   ENC_AZ → AS5600 over I2C (SDA=21, SCL=22, addr 0x36, 12-bit)
 *   ENC_EL → AS5048A over SPI (CLK=18, MISO=19, MOSI=23, CS=17, 14-bit)
 *
 * Must be called before encoder_read_*.
 */
esp_err_t encoder_init(void);

/**
 * Read native-resolution raw angle.
 *
 *   ENC_AZ → 12-bit (0..4095)  — AS5600 ANGLE register, 360 / 4096 deg/count
 *   ENC_EL → 14-bit (0..16383) — AS5048A angle, 360 / 16384 deg/count
 *
 * Callers MUST NOT assume a common bit width; use encoder_read_angle for
 * canonical degrees. Returns ESP_FAIL on bad parity/EF (EL) or I2C error (AZ).
 */
esp_err_t encoder_read_raw(enc_axis_t axis, uint16_t *out_raw);

/**
 * Read angle in degrees (0.0–360.0) with median filtering, per-axis driver.
 */
esp_err_t encoder_read_angle(enc_axis_t axis, float *out_deg);

/**
 * Read communication / sanity flags.
 *
 *   ENC_EL (AS5048A error reg 0x0001):
 *     bit0 = framing error  bit1 = invalid command  bit2 = parity error
 *     Reading the register CLEARS all three flags.
 *
 *   ENC_AZ (AS5600 STATUS reg 0x0B re-mapped):
 *     bit0 = magnet missing (!MD)
 *     bit1 = magnet too weak (ML)
 *     bit2 = magnet too strong (MH)
 *
 * Field/magnet diagnostics (AGC, magnitude) are in encoder_read_diag().
 */
esp_err_t encoder_read_errors(enc_axis_t axis, uint8_t *out_flags);

/**
 * Read diagnostic registers and log a one-line summary.
 *
 *   ENC_EL: AS5048A reg 0x3FFD — AGC, COMP_H/L, COF, OCF.
 *   ENC_AZ: AS5600 STATUS/AGC/MAGNITUDE — MD/ML/MH + agc 0..255 + magnitude.
 *
 * Does not affect normal angle reading. Verbose per-frame trace via menuconfig
 *   "Orbiter Configuration" → ORBITER_ENCODER_SPI_TRACE (AS5048A)
 *                          → ORBITER_AS5600_I2C_TRACE  (AS5600)
 */
esp_err_t encoder_read_diag(enc_axis_t axis);
