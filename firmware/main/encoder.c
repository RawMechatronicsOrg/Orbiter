/*
 * Encoder dispatcher.
 *
 *   ENC_AZ → AS5600 over I2C (12-bit)   — see enc_as5600.c
 *   ENC_EL → AS5048A over SPI (14-bit)  — kept here
 *
 * Public API in encoder.h is unchanged. Callers must treat `encoder_read_raw`
 * as native bit-width per axis (12 for AZ, 14 for EL); for canonical angles
 * use `encoder_read_angle`.
 */

#include "encoder.h"
#include "enc_as5600.h"
#include "log_bus.h"
#include "sdkconfig.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "rom/ets_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <stdbool.h>

static const char *TAG = "encoder";

#if CONFIG_ORBITER_ENCODER_SPI_TRACE
#define ENC_SPI_TRACE(...) ESP_LOGI(TAG, __VA_ARGS__)
#else
#define ENC_SPI_TRACE(...) ((void)0)
#endif

/* ── Pin definitions ────────────────────────────────────────────────────── */
/* AS5048A (EL only) — SPI bus shared with future devices. */
#define PIN_CLK     18
#define PIN_MISO    19
#define PIN_MOSI    23
#define PIN_CS_EL   17   /* GPIO5 (former CS_AZ) is now FREE — AS5600 replaces it on I2C */

/* AS5600 (AZ only) — I2C bus. */
#define PIN_I2C_SDA 21
#define PIN_I2C_SCL 22

/*
 * CS is managed manually — ESP32 hardware CS fires simultaneously with CLK,
 * which violates AS5048A setup time. Manual CS gives us control over timing.
 */
#define SPI_HOST    SPI2_HOST
/*
 * SCK = 1 MHz. The AS5048A is rated to 10 MHz SPI; 1 MHz keeps a 10x margin
 * for the manual-CS timing and the harness wiring. A full angle read (6 ×
 * 16-bit transfers + CS settling) then takes ~1 ms, which lets pose_tick_task
 * stream telemetry at 100 Hz. The old 500 Hz value (~tens of ms per read)
 * capped pose telemetry at 10 Hz — a leftover from debugging the now-removed
 * second AS5048A; the surviving EL sensor reads cleanly at 1 MHz.
 */
#define SPI_FREQ_HZ 1000000

/* Manual CS: settle time after GPIO edges (long wires / dual AS5048A heritage). */
#define CS_GPIO_DELAY_US  45

/* After PARITY / noisy frame: read error reg (clears flags) + pause + full angle sequence again. */
#define ENC_ANGLE_READ_RETRIES   6
#define ENC_READ_RETRY_GAP_US    150

/* Circular median filter: number of raw samples per angle read (must be odd). */
#define ENC_FILTER_SAMPLES       3

/* AS5048A native resolution. */
#define AS5048A_BITS    14
#define AS5048A_COUNTS  (1 << AS5048A_BITS)   /* 16384 */
#define AS5048A_HALF    (AS5048A_COUNTS / 2)  /*  8192 */

/* AS5600 native resolution. */
#define AS5600_BITS     12
#define AS5600_COUNTS   (1 << AS5600_BITS)    /* 4096 */
#define AS5600_HALF     (AS5600_COUNTS / 2)   /* 2048 */

static spi_device_handle_t s_dev;   /* SPI bus; EL only. */

/*
 * Bus serialization mutex — guards every public read against concurrent access.
 *
 * Why here instead of letting callers lock: AS5048A on SPI is shared between
 * pose_tick_task (10 Hz from motion.c) and HTTP handlers that may call
 * encoder_read_raw / encoder_read_diag directly without going through motion.c.
 * ESP-IDF spi_master with queue_size=1 trips an assert (spi_master.c:1310
 * "ret_trans == trans_desc") if two tasks issue spi_device_transmit on the
 * same device concurrently. Same risk for I2C on AZ if two callers race.
 * Owning the lock at the encoder layer means every public entry point is
 * automatically safe and we don't have to audit every call site.
 */
static SemaphoreHandle_t s_io_mtx;

static inline void enc_io_lock(void)
{
    if (s_io_mtx) xSemaphoreTake(s_io_mtx, portMAX_DELAY);
}
static inline void enc_io_unlock(void)
{
    if (s_io_mtx) xSemaphoreGive(s_io_mtx);
}

/* ── Init ───────────────────────────────────────────────────────────────── */

esp_err_t encoder_init(void)
{
    /* Create bus lock BEFORE any read primitive runs — otherwise the
     * boot-time encoder_read_diag() at the bottom of this function would
     * race a concurrent reader (e.g. if pose_tick was already up). */
    if (!s_io_mtx) {
        s_io_mtx = xSemaphoreCreateMutex();
        if (!s_io_mtx) {
            ESP_LOGE(TAG, "encoder io mutex alloc failed");
            return ESP_ERR_NO_MEM;
        }
    }

    /* ── AS5600 (AZ) — I2C ────────────────────────────────────────────── */
    esp_err_t ret = as5600_init(PIN_I2C_SDA, PIN_I2C_SCL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "AS5600 init failed: %s", esp_err_to_name(ret));
        /* Continue — EL on SPI may still work. */
    }

    /* ── AS5048A (EL) — SPI with manual CS ────────────────────────────── */
    gpio_config_t cs_cfg = {
        .pin_bit_mask = (1ULL << PIN_CS_EL),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&cs_cfg);
    gpio_set_level(PIN_CS_EL, 1);

    spi_bus_config_t bus = {
        .miso_io_num   = PIN_MISO,
        .mosi_io_num   = PIN_MOSI,
        .sclk_io_num   = PIN_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4,
    };
    ret = spi_bus_initialize(SPI_HOST, &bus, SPI_DMA_DISABLED);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPI bus init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    spi_device_interface_config_t dev_cfg = {
        .mode           = 1,          /* CPOL=0, CPHA=1 per AS5048A datasheet */
        .clock_speed_hz = SPI_FREQ_HZ,
        .spics_io_num   = -1,         /* manual CS */
        .queue_size     = 1,
        .flags          = 0,
    };
    ret = spi_bus_add_device(SPI_HOST, &dev_cfg, &s_dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPI add device failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "init OK — AS5600(AZ) I2C SDA=%d SCL=%d | AS5048A(EL) SPI CLK=%d MOSI=%d MISO=%d CS=%d%s",
             PIN_I2C_SDA, PIN_I2C_SCL,
             PIN_CLK, PIN_MOSI, PIN_MISO, PIN_CS_EL,
#if CONFIG_ORBITER_ENCODER_SPI_TRACE
             " | SPI_TRACE=on"
#else
             ""
#endif
    );

    /* Boot-time sensor health snapshot — visible in UI via log_bus. */
    encoder_read_diag(ENC_AZ);
    encoder_read_diag(ENC_EL);

    return ESP_OK;
}

/* ── AS5048A (EL) — SPI primitives ──────────────────────────────────────── */

/* RX 16-bit word: D15 = even parity over D14..D0; D14 = EF; D13:D0 = data. */
static bool as5048a_rx_word_parity_ok(uint16_t w)
{
    return (__builtin_popcount((unsigned)w) & 1u) == 0u;
}

static uint16_t as5048a_spi_transfer16(int cs_pin, uint16_t tx_word)
{
    uint8_t tx[2] = { (tx_word >> 8) & 0xFF, tx_word & 0xFF };
    uint8_t rx[2] = { 0, 0 };

    spi_transaction_t t = {
        .length    = 16,
        .tx_buffer = tx,
        .rx_buffer = rx,
    };

    gpio_set_level(cs_pin, 0);
    ets_delay_us(CS_GPIO_DELAY_US);

    spi_device_transmit(s_dev, &t);

    ets_delay_us(CS_GPIO_DELAY_US);
    gpio_set_level(cs_pin, 1);
    ets_delay_us(CS_GPIO_DELAY_US);

    uint16_t rx_word = ((uint16_t)rx[0] << 8) | rx[1];

#if CONFIG_ORBITER_ENCODER_SPI_TRACE
    {
        bool p_ok = as5048a_rx_word_parity_ok(rx_word);
        ESP_LOGI(TAG,
                 "spi cs=%d tx=0x%04X rx=0x%04X [%02X %02X] P15=%u EF14=%u D13_0=0x%04X parity_%s",
                 cs_pin, tx_word, rx_word, rx[0], rx[1],
                 (unsigned)((rx_word >> 15) & 1u), (unsigned)((rx_word >> 14) & 1u),
                 (unsigned)(rx_word & 0x3FFFu), p_ok ? "OK" : "BAD");
    }
#endif

    return rx_word;
}

/* Read comm error reg 0x0001 — clears FRERR/INVCMD/PARITY on AS5048A (no log). */
static void as5048a_comm_error_clear_quiet(int cs_pin)
{
    as5048a_spi_transfer16(cs_pin, 0x4001);
    (void)as5048a_spi_transfer16(cs_pin, 0xC000);
}

static esp_err_t as5048a_read_raw(uint16_t *out_raw)
{
    int cs = PIN_CS_EL;

    /*
     * AS5048A SPI pipeline — response lags one frame behind command:
     *   Frame 1: send READ_ANGLE (0xFFFF) → receive previous response (discard)
     *   Frame 2: send NOP         (0xC000) → receive angle from Frame 1
     */
    for (int attempt = 0; attempt < ENC_ANGLE_READ_RETRIES; attempt++) {
        if (attempt > 0) {
            as5048a_comm_error_clear_quiet(cs);
            ets_delay_us(ENC_READ_RETRY_GAP_US);
        }

        uint16_t rsp1 = as5048a_spi_transfer16(cs, 0xFFFF);
        uint16_t word = as5048a_spi_transfer16(cs, 0xC000);
        (void)rsp1;  /* only used inside ENC_SPI_TRACE; silence warning when off */

        bool p_ok = as5048a_rx_word_parity_ok(word);
        bool ef   = (word & 0x4000u) != 0;

        ENC_SPI_TRACE("axis 1 read_raw att=%d rsp1=0x%04X rsp2=0x%04X EF=%d parity_%s angle14=0x%04X",
                      attempt, rsp1, word, ef ? 1 : 0,
                      p_ok ? "OK" : "BAD", (unsigned)(word & 0x3FFFu));

        if (!ef && p_ok) {
            *out_raw = word & 0x3FFF;
            return ESP_OK;
        }
    }
    return ESP_FAIL;
}

static esp_err_t as5048a_read_errors(uint8_t *out_flags)
{
    int cs = PIN_CS_EL;

    /* Read error register (addr=0x0001, R=1, parity=0): word = 0x4001 */
    as5048a_spi_transfer16(cs, 0x4001);
    uint16_t word = as5048a_spi_transfer16(cs, 0xC000);

    ENC_SPI_TRACE("axis 1 err-reg seq: rx_word_full=0x%04X parity_%s",
                  word, as5048a_rx_word_parity_ok(word) ? "OK" : "BAD");

    /*
     * Error register 0x0001 contains COMMUNICATION errors only.
     *   bit[0] — Framing error
     *   bit[1] — Invalid command
     *   bit[2] — Parity error
     * Reading this register CLEARS all three flags.
     */
    *out_flags = (uint8_t)(word & 0x07);
    LOGX_EMIT_W(TAG, "axis 1 comm-err reg: 0x%02X (FRERR=%d INVCMD=%d PARITY=%d)",
                *out_flags,
                (*out_flags >> 0) & 1,
                (*out_flags >> 1) & 1,
                (*out_flags >> 2) & 1);
    return ESP_OK;
}

static esp_err_t as5048a_read_diag(void)
{
    int cs = PIN_CS_EL;

    /*
     * Diagnostics register (addr=0x3FFD, R=1) — command word 0x7FFD.
     *   [7:0] AGC, [8] OCF, [9] COF, [10] COMP_L, [11] COMP_H
     */
    as5048a_spi_transfer16(cs, 0x7FFD);
    uint16_t word = as5048a_spi_transfer16(cs, 0xC000);

    ENC_SPI_TRACE("axis 1 diag seq: rx_word_full=0x%04X parity_%s",
                  word, as5048a_rx_word_parity_ok(word) ? "OK" : "BAD");

    uint16_t raw14  = word & 0x3FFF;
    uint8_t  agc    = (uint8_t)(raw14 & 0xFF);
    uint8_t  comp_h = (raw14 >> 11) & 1;
    uint8_t  comp_l = (raw14 >> 10) & 1;
    uint8_t  cof    = (raw14 >>  9) & 1;
    uint8_t  ocf    = (raw14 >>  8) & 1;

    LOGX_EMIT_I(TAG, "axis 1 AS5048A DIAG raw=0x%04X agc=0x%02X (%3u) COMP_H=%d COMP_L=%d COF=%d OCF=%d%s",
                raw14, agc, agc, comp_h, comp_l, cof, ocf,
                ocf ? "" : "  ← OCF=0! sensor not ready (check VCC)");

    return ESP_OK;
}

/* Filtered angle for AS5048A — median of N raws on a 14-bit circle. */
static esp_err_t as5048a_read_angle(float *out_deg)
{
    int32_t deltas[ENC_FILTER_SAMPLES];
    int     valid = 0;
    int32_t ref   = -1;

    for (int i = 0; i < ENC_FILTER_SAMPLES; i++) {
        uint16_t raw;
        if (as5048a_read_raw(&raw) != ESP_OK) continue;

        if (ref < 0) {
            ref = (int32_t)raw;
            deltas[valid++] = 0;
        } else {
            int32_t d = (int32_t)raw - ref;
            if (d >  AS5048A_HALF) d -= AS5048A_COUNTS;
            if (d < -AS5048A_HALF) d += AS5048A_COUNTS;
            deltas[valid++] = d;
        }
    }

    if (valid == 0) return ESP_FAIL;

    for (int i = 1; i < valid; i++) {
        int32_t key = deltas[i];
        int j = i - 1;
        while (j >= 0 && deltas[j] > key) { deltas[j + 1] = deltas[j]; j--; }
        deltas[j + 1] = key;
    }

    int32_t result = ((ref + deltas[valid / 2]) % AS5048A_COUNTS + AS5048A_COUNTS) % AS5048A_COUNTS;
    *out_deg = (float)result * (360.0f / (float)AS5048A_COUNTS);
    return ESP_OK;
}

/* Filtered angle for AS5600 — median of N raws on a 12-bit circle. */
static esp_err_t as5600_read_angle_filtered(float *out_deg)
{
    int32_t deltas[ENC_FILTER_SAMPLES];
    int     valid = 0;
    int32_t ref   = -1;

    for (int i = 0; i < ENC_FILTER_SAMPLES; i++) {
        uint16_t raw;
        if (as5600_read_angle_raw12(&raw) != ESP_OK) continue;

        if (ref < 0) {
            ref = (int32_t)raw;
            deltas[valid++] = 0;
        } else {
            int32_t d = (int32_t)raw - ref;
            if (d >  AS5600_HALF) d -= AS5600_COUNTS;
            if (d < -AS5600_HALF) d += AS5600_COUNTS;
            deltas[valid++] = d;
        }
    }

    if (valid == 0) return ESP_FAIL;

    for (int i = 1; i < valid; i++) {
        int32_t key = deltas[i];
        int j = i - 1;
        while (j >= 0 && deltas[j] > key) { deltas[j + 1] = deltas[j]; j--; }
        deltas[j + 1] = key;
    }

    int32_t result = ((ref + deltas[valid / 2]) % AS5600_COUNTS + AS5600_COUNTS) % AS5600_COUNTS;
    *out_deg = (float)result * (360.0f / (float)AS5600_COUNTS);
    return ESP_OK;
}

static esp_err_t as5600_log_diag(void)
{
    uint8_t  st = 0, agc = 0;
    uint16_t mag = 0;
    esp_err_t err = as5600_read_status(&st, &agc, &mag);
    if (err != ESP_OK) {
        LOGX_EMIT_W(TAG, "axis 0 AS5600 status read FAILED: %s", esp_err_to_name(err));
        return err;
    }
    /* Per AS5600 datasheet v1-06 p.21: MD=bit5, ML=bit4, MH=bit3. */
    uint8_t mh = (st >> 3) & 1;
    uint8_t ml = (st >> 4) & 1;
    uint8_t md = (st >> 5) & 1;
    LOGX_EMIT_I(TAG, "axis 0 AS5600 status=0x%02X MD=%d ML=%d MH=%d agc=%u mag=%u%s",
                st, md, ml, mh, agc, mag,
                md ? "" : "  ← magnet NOT detected!");
    return ESP_OK;
}

/* ── Public API dispatcher ──────────────────────────────────────────────── */
/*
 * Each entry point acquires s_io_mtx to serialise SPI/I2C traffic across
 * pose_tick_task, motion.c (move/settle loops) and HTTP handlers. The lock is
 * always taken at the public boundary — never inside the static helpers —
 * so it is not re-entrant and the order is unambiguous.
 */

esp_err_t encoder_read_raw(enc_axis_t axis, uint16_t *out_raw)
{
    if (!out_raw) return ESP_ERR_INVALID_ARG;
    esp_err_t ret = ESP_ERR_INVALID_ARG;
    enc_io_lock();
    if (axis == ENC_AZ) {
        /* AS5600: returns 12-bit (0..4095). Caller should NOT assume 14-bit. */
        ret = as5600_read_angle_raw12(out_raw);
    } else if (axis == ENC_EL) {
        ret = as5048a_read_raw(out_raw);
    }
    enc_io_unlock();
    return ret;
}

esp_err_t encoder_read_errors(enc_axis_t axis, uint8_t *out_flags)
{
    if (!out_flags) return ESP_ERR_INVALID_ARG;
    esp_err_t ret = ESP_ERR_INVALID_ARG;
    enc_io_lock();
    if (axis == ENC_AZ) {
        /* AS5600 has no per-frame comm error register — STATUS replaces it.
         * Per datasheet v1-06 p.21: MD=bit5, ML=bit4, MH=bit3.
         * Map into low bits so the legacy field stays meaningful:
         *   bit0 = !MD (magnet missing — analog of framing failure)
         *   bit1 = ML  (too weak)
         *   bit2 = MH  (too strong)
         */
        uint8_t st = 0;
        ret = as5600_read_status(&st, NULL, NULL);
        if (ret == ESP_OK) {
            *out_flags = (uint8_t)((!(st & 0x20) ? 0x01 : 0x00) |
                                   (((st >> 4) & 1) << 1) |
                                   (((st >> 3) & 1) << 2));
        }
    } else if (axis == ENC_EL) {
        ret = as5048a_read_errors(out_flags);
    }
    enc_io_unlock();
    return ret;
}

esp_err_t encoder_read_diag(enc_axis_t axis)
{
    esp_err_t ret = ESP_ERR_INVALID_ARG;
    enc_io_lock();
    if      (axis == ENC_AZ) ret = as5600_log_diag();
    else if (axis == ENC_EL) ret = as5048a_read_diag();
    enc_io_unlock();
    return ret;
}

esp_err_t encoder_read_angle(enc_axis_t axis, float *out_deg)
{
    if (!out_deg) return ESP_ERR_INVALID_ARG;
    esp_err_t ret = ESP_ERR_INVALID_ARG;
    enc_io_lock();
    if      (axis == ENC_AZ) ret = as5600_read_angle_filtered(out_deg);
    else if (axis == ENC_EL) ret = as5048a_read_angle(out_deg);
    enc_io_unlock();
    return ret;
}
