#pragma once

#include "esp_err.h"
#include "driver/i2c_master.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * AS5600 magnetic angle sensor — I2C driver, used on both axes.
 *
 * Fixed device address 0x36 with no select pins, so each sensor gets its own
 * bus: one `as5600_t` per axis, each owning a private I2C master bus. 12-bit
 * resolution → 0.088°/step.
 */
typedef struct {
    i2c_master_bus_handle_t bus;
    i2c_master_dev_handle_t dev;
} as5600_t;

/** Initialise a private I2C master bus on these pins and attach the AS5600. */
esp_err_t as5600_init(as5600_t *s, int sda_gpio, int scl_gpio);

/** Read the processed ANGLE register (0x0E/0x0F), masked to 12 bits (0..4095). */
esp_err_t as5600_read_angle_raw12(as5600_t *s, uint16_t *out_raw);

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
esp_err_t as5600_read_status(as5600_t *s, uint8_t *out_status, uint8_t *out_agc,
                             uint16_t *out_magnitude);

#ifdef __cplusplus
}
#endif
