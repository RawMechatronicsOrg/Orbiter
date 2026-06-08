#pragma once

#include "esp_err.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * AS5600 magnetic angle sensor — I2C driver (replaces AS5048A on the AZ axis).
 *
 * Fixed device address 0x36 (only one AS5600 per bus). 12-bit resolution
 * → 0.088°/step (vs AS5048A 14-bit / 0.022°/step). See MACHINE.md.
 */

/** Initialise a private I2C master bus and attach the AS5600 device. */
esp_err_t as5600_init(int sda_gpio, int scl_gpio);

/** Read the processed ANGLE register (0x0E/0x0F), masked to 12 bits (0..4095). */
esp_err_t as5600_read_angle_raw12(uint16_t *out_raw);

/**
 * Read STATUS (0x0B), AGC (0x1A) and MAGNITUDE (0x1B/0x1C) for diagnostics.
 *
 *   status bits (AS5600 datasheet v1-06, p.21):
 *     bit5 MD — magnet detected
 *     bit4 ML — magnet too weak (AGC at max)
 *     bit3 MH — magnet too strong (AGC at min)
 *
 *   agc: 0..255, optimum ≈ 128.
 *   magnitude: 12-bit CORDIC field magnitude.
 *
 * Any pointer may be NULL to skip that field. Returns ESP_OK if all requested
 * reads succeed.
 */
esp_err_t as5600_read_status(uint8_t *out_status, uint8_t *out_agc, uint16_t *out_magnitude);

#ifdef __cplusplus
}
#endif
