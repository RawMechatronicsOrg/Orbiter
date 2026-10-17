#include "enc_as5600.h"
#include "sdkconfig.h"

#include "driver/i2c_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "log_bus.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "rom/ets_sys.h"
#include <string.h>

static const char *TAG = "as5600";

#if CONFIG_ORBITER_AS5600_I2C_TRACE
#define ENC_I2C_TRACE(...) ESP_LOGI(TAG, __VA_ARGS__)
#else
#define ENC_I2C_TRACE(...) ((void)0)
#endif

/*
 * AS5600 register map (datasheet §6, table 1).
 * Only the registers we touch are listed.
 */
#define AS5600_REG_STATUS     0x0B  /* MD/ML/MH flags                          */
#define AS5600_REG_RAW_ANGLE  0x0C  /* 2 bytes, big-endian, mask 0x0FFF        */
#define AS5600_REG_ANGLE      0x0E  /* 2 bytes, processed (hysteresis + zero)  */
#define AS5600_REG_AGC        0x1A  /* 0..255                                  */
#define AS5600_REG_MAGNITUDE  0x1B  /* 2 bytes, mask 0x0FFF                    */

#define AS5600_I2C_ADDR       0x36  /* fixed 7-bit address                     */
#define AS5600_I2C_HZ         400000
#define AS5600_I2C_TIMEOUT_MS 30
#define AS5600_RETRIES        3
#define AS5600_RETRY_GAP_US   200

/*
 * Read both lines before the I2C driver owns them, so a dead bus says why:
 *   a line LOW with the internal pull-up on  -> shorted to GND, wired to a
 *                                              ground or DIR pin, or held down
 *                                              by an unpowered chip's clamp
 *                                              diode (a sensor without VCC)
 *   both high only while the internal pull-ups are on -> no external pull-ups
 *                                              (sensor board without 3V3/GND)
 * Ported from ActuatorController, where it ended an afternoon of guessing.
 */
static void log_idle_levels(as5600_t *s, int sda_gpio, int scl_gpio)
{
    s->idle_sda = s->idle_scl = s->idle_ext_pullups = -1;
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << sda_gpio) | (1ULL << scl_gpio),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    if (gpio_config(&io) != ESP_OK) return;

    ets_delay_us(200);   /* let the weak internal pull-ups win */
    int sda_pu = gpio_get_level(sda_gpio);
    int scl_pu = gpio_get_level(scl_gpio);
    s->idle_sda = (int8_t)sda_pu;
    s->idle_scl = (int8_t)scl_pu;

    if (!sda_pu || !scl_pu) {
        LOGX_EMIT_E(TAG, "SDA=%d(%s) SCL=%d(%s) — a line is held LOW. Shorted to "
                         "GND, wired to a ground or DIR pin, or a chip on it has no VCC.",
                    sda_gpio, sda_pu ? "high" : "LOW",
                    scl_gpio, scl_pu ? "high" : "LOW");
        return;
    }

    /* Both high — but by whose pull-up? */
    io.pull_up_en = GPIO_PULLUP_DISABLE;
    if (gpio_config(&io) != ESP_OK) return;
    ets_delay_us(500);   /* an external pull-up wins this quickly */
    int sda_hi_z = gpio_get_level(sda_gpio);
    int scl_hi_z = gpio_get_level(scl_gpio);
    s->idle_ext_pullups = (int8_t)(sda_hi_z && scl_hi_z);

    /* Leave the pins as we found them for the I2C driver. */
    io.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io);

    if (sda_hi_z && scl_hi_z) {
        LOGX_EMIT_I(TAG, "SDA=%d SCL=%d idle high, external pull-ups present",
                    sda_gpio, scl_gpio);
    } else {
        LOGX_EMIT_E(TAG, "SDA=%d SCL=%d held up ONLY by the internal ~45k pull-ups "
                         "(hi-Z reads sda=%d scl=%d). Check 3V3/GND to the sensor "
                         "board, or fit 2.2-4.7k pull-ups to 3V3.",
                    sda_gpio, scl_gpio, sda_hi_z, scl_hi_z);
    }
}

esp_err_t as5600_init(as5600_t *s, int sda_gpio, int scl_gpio)
{
    if (!s) return ESP_ERR_INVALID_ARG;
    if (s->dev) return ESP_OK;  /* idempotent */

    log_idle_levels(s, sda_gpio, scl_gpio);

    i2c_master_bus_config_t bus_cfg = {
        .clk_source                   = I2C_CLK_SRC_DEFAULT,
        .i2c_port                     = -1,             /* auto */
        .sda_io_num                   = sda_gpio,
        .scl_io_num                   = scl_gpio,
        .glitch_ignore_cnt            = 7,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t err = i2c_new_master_bus(&bus_cfg, &s->bus);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2c bus init failed: %s", esp_err_to_name(err));
        return err;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = AS5600_I2C_ADDR,
        .scl_speed_hz    = AS5600_I2C_HZ,
    };
    err = i2c_master_bus_add_device(s->bus, &dev_cfg, &s->dev);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2c add device 0x%02X failed: %s",
                 AS5600_I2C_ADDR, esp_err_to_name(err));
        i2c_del_master_bus(s->bus);
        s->bus = NULL;
        return err;
    }

    /* AS5600 power-on settling time per datasheet (~5 ms). */
    vTaskDelay(pdMS_TO_TICKS(10));

    ESP_LOGI(TAG, "init OK — SDA=%d SCL=%d addr=0x%02X @ %dHz%s",
             sda_gpio, scl_gpio, AS5600_I2C_ADDR, AS5600_I2C_HZ,
#if CONFIG_ORBITER_AS5600_I2C_TRACE
             " | I2C_TRACE=on"
#else
             ""
#endif
    );
    return ESP_OK;
}

/* Read N bytes from sequential registers starting at `reg`. */
static esp_err_t as5600_read_regs(as5600_t *s, uint8_t reg, uint8_t *buf, size_t n)
{
    if (!s || !s->dev || !buf || n == 0) return ESP_ERR_INVALID_STATE;

    esp_err_t err = ESP_FAIL;
    for (int attempt = 0; attempt < AS5600_RETRIES; attempt++) {
        if (attempt > 0) ets_delay_us(AS5600_RETRY_GAP_US);
        err = i2c_master_transmit_receive(
            s->dev, &reg, 1, buf, n, AS5600_I2C_TIMEOUT_MS);
        if (err == ESP_OK) {
            ENC_I2C_TRACE("read reg=0x%02X len=%u att=%d OK", reg, (unsigned)n, attempt);
            return ESP_OK;
        }
        ENC_I2C_TRACE("read reg=0x%02X len=%u att=%d FAIL: %s",
                      reg, (unsigned)n, attempt, esp_err_to_name(err));
    }
    return err;
}

esp_err_t as5600_read_angle_raw12(as5600_t *s, uint16_t *out_raw)
{
    if (!out_raw) return ESP_ERR_INVALID_ARG;
    uint8_t buf[2] = { 0 };
    esp_err_t err = as5600_read_regs(s, AS5600_REG_ANGLE, buf, sizeof(buf));
    if (err != ESP_OK) return err;

    /* 12-bit big-endian; bits[15:12] are reserved/zero. */
    *out_raw = (uint16_t)(((buf[0] << 8) | buf[1]) & 0x0FFF);
    return ESP_OK;
}

esp_err_t as5600_read_status(as5600_t *s, uint8_t *out_status, uint8_t *out_agc,
                             uint16_t *out_magnitude)
{
    esp_err_t err;

    if (out_status) {
        uint8_t st = 0;
        err = as5600_read_regs(s, AS5600_REG_STATUS, &st, 1);
        if (err != ESP_OK) return err;
        *out_status = st;
    }

    if (out_agc) {
        uint8_t agc = 0;
        err = as5600_read_regs(s, AS5600_REG_AGC, &agc, 1);
        if (err != ESP_OK) return err;
        *out_agc = agc;
    }

    if (out_magnitude) {
        uint8_t buf[2] = { 0 };
        err = as5600_read_regs(s, AS5600_REG_MAGNITUDE, buf, sizeof(buf));
        if (err != ESP_OK) return err;
        *out_magnitude = (uint16_t)(((buf[0] << 8) | buf[1]) & 0x0FFF);
    }

    return ESP_OK;
}
