/*
 * Encoder dispatcher — one AS5600 per axis, each on its own I2C bus.
 *
 *   ENC_AZ → AS5600 on I2C0 (SDA=22, SCL=23)
 *   ENC_EL → AS5600 on I2C1 (SDA=4,  SCL=16)
 *
 * Pins are the Orbiter-Simple carrier's J3 (AZ) and J4 (EL); the wired bench
 * uses the same ones (menuconfig → Board wiring picks the steppers). The AS5600's
 * address is hardwired at 0x36, so the two sensors need two buses, not two
 * addresses. Each sensor's own DIR pin is not an ESP32 GPIO: strap it to GND
 * on the sensor board — left floating, the angle register jumps erratically.
 *
 * Both axes are 12-bit (0..4095); for canonical angles use encoder_read_angle.
 */

#include "encoder.h"
#include "enc_as5600.h"
#include "log_bus.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <stdbool.h>

static const char *TAG = "encoder";

/* ── Pin definitions (same on the wired bench and Orbiter-Simple) ───────── */
#define PIN_AZ_SDA  22
#define PIN_AZ_SCL  23
#define PIN_EL_SDA  4
#define PIN_EL_SCL  16

/* Circular median filter: number of raw samples per angle read (must be odd). */
#define ENC_FILTER_SAMPLES       3

/* AS5600 native resolution. */
#define AS5600_BITS     12
#define AS5600_COUNTS   (1 << AS5600_BITS)    /* 4096 */
#define AS5600_HALF     (AS5600_COUNTS / 2)   /* 2048 */

static as5600_t s_enc[2] = {   /* indexed by enc_axis_t */
    { .idle_sda = -1, .idle_scl = -1, .idle_ext_pullups = -1 },
    { .idle_sda = -1, .idle_scl = -1, .idle_ext_pullups = -1 },
};

/*
 * Bus serialization mutex — guards every public read against concurrent access.
 *
 * Why here instead of letting callers lock: the sensors are read by
 * pose_tick_task (motion.c) and by HTTP handlers that may call
 * encoder_read_raw / encoder_read_diag directly without going through
 * motion.c. Owning the lock at the encoder layer means every public entry
 * point is automatically safe and we don't have to audit every call site.
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

static as5600_t *enc_for(enc_axis_t axis)
{
    return (axis == ENC_AZ || axis == ENC_EL) ? &s_enc[axis] : NULL;
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

    /* A failed axis is logged and skipped — the other one may still work. */
    esp_err_t ret_az = as5600_init(&s_enc[ENC_AZ], PIN_AZ_SDA, PIN_AZ_SCL);
    if (ret_az != ESP_OK) {
        ESP_LOGE(TAG, "AZ AS5600 init failed: %s", esp_err_to_name(ret_az));
    }
    esp_err_t ret_el = as5600_init(&s_enc[ENC_EL], PIN_EL_SDA, PIN_EL_SCL);
    if (ret_el != ESP_OK) {
        ESP_LOGE(TAG, "EL AS5600 init failed: %s", esp_err_to_name(ret_el));
    }

    ESP_LOGI(TAG, "init — AS5600(AZ) SDA=%d SCL=%d | AS5600(EL) SDA=%d SCL=%d",
             PIN_AZ_SDA, PIN_AZ_SCL, PIN_EL_SDA, PIN_EL_SCL);

    /* Boot-time sensor health snapshot — visible in UI via log_bus. */
    encoder_read_diag(ENC_AZ);
    encoder_read_diag(ENC_EL);

    return (ret_az == ESP_OK || ret_el == ESP_OK) ? ESP_OK : ret_az;
}

/* ── AS5600 helpers ─────────────────────────────────────────────────────── */

/* Filtered angle — median of N raws on a 12-bit circle. */
static esp_err_t as5600_read_angle_filtered(as5600_t *s, float *out_deg)
{
    int32_t deltas[ENC_FILTER_SAMPLES];
    int     valid = 0;
    int32_t ref   = -1;

    for (int i = 0; i < ENC_FILTER_SAMPLES; i++) {
        uint16_t raw;
        if (as5600_read_angle_raw12(s, &raw) != ESP_OK) continue;

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

static esp_err_t as5600_log_diag(enc_axis_t axis)
{
    uint8_t  st = 0, agc = 0;
    uint16_t mag = 0;
    esp_err_t err = as5600_read_status(&s_enc[axis], &st, &agc, &mag);
    if (err != ESP_OK) {
        LOGX_EMIT_W(TAG, "axis %d AS5600 status read FAILED: %s", (int)axis, esp_err_to_name(err));
        return err;
    }
    /* Per AS5600 datasheet v1-06 p.21: MD=bit5, ML=bit4, MH=bit3. */
    uint8_t mh = (st >> 3) & 1;
    uint8_t ml = (st >> 4) & 1;
    uint8_t md = (st >> 5) & 1;
    LOGX_EMIT_I(TAG, "axis %d AS5600 status=0x%02X MD=%d ML=%d MH=%d agc=%u mag=%u%s",
                (int)axis, st, md, ml, mh, agc, mag,
                md ? "" : "  ← magnet NOT detected!");
    return ESP_OK;
}

/* ── Public API dispatcher ──────────────────────────────────────────────── */
/*
 * Each entry point acquires s_io_mtx to serialise I2C traffic across
 * pose_tick_task, motion.c (move/settle loops) and HTTP handlers. The lock is
 * always taken at the public boundary — never inside the static helpers —
 * so it is not re-entrant and the order is unambiguous.
 */

esp_err_t encoder_read_raw(enc_axis_t axis, uint16_t *out_raw)
{
    as5600_t *s = enc_for(axis);
    if (!out_raw || !s) return ESP_ERR_INVALID_ARG;
    enc_io_lock();
    esp_err_t ret = as5600_read_angle_raw12(s, out_raw);
    enc_io_unlock();
    return ret;
}

esp_err_t encoder_read_errors(enc_axis_t axis, uint8_t *out_flags)
{
    as5600_t *s = enc_for(axis);
    if (!out_flags || !s) return ESP_ERR_INVALID_ARG;
    /* The AS5600 has no comm error register — STATUS replaces it.
     * Per datasheet v1-06 p.21: MD=bit5, ML=bit4, MH=bit3. Mapped into
     * the low bits so the legacy field stays meaningful:
     *   bit0 = !MD (magnet missing)
     *   bit1 = ML  (too weak)
     *   bit2 = MH  (too strong)
     */
    uint8_t st = 0;
    enc_io_lock();
    esp_err_t ret = as5600_read_status(s, &st, NULL, NULL);
    enc_io_unlock();
    if (ret == ESP_OK) {
        *out_flags = (uint8_t)((!(st & 0x20) ? 0x01 : 0x00) |
                               (((st >> 4) & 1) << 1) |
                               (((st >> 3) & 1) << 2));
    }
    return ret;
}

esp_err_t encoder_read_diag(enc_axis_t axis)
{
    if (!enc_for(axis)) return ESP_ERR_INVALID_ARG;
    enc_io_lock();
    esp_err_t ret = as5600_log_diag(axis);
    enc_io_unlock();
    return ret;
}

void encoder_line_levels(enc_axis_t axis, int *sda, int *scl, int *ext_pullups)
{
    const as5600_t *s = enc_for(axis);
    if (sda)         *sda         = s ? s->idle_sda : -1;
    if (scl)         *scl         = s ? s->idle_scl : -1;
    if (ext_pullups) *ext_pullups = s ? s->idle_ext_pullups : -1;
}

esp_err_t encoder_read_magnet(enc_axis_t axis, enc_magnet_t *out)
{
    as5600_t *s = enc_for(axis);
    if (!out || !s) return ESP_ERR_INVALID_ARG;
    uint8_t  st = 0, agc = 0;
    uint16_t mag = 0;
    enc_io_lock();
    esp_err_t ret = as5600_read_status(s, &st, &agc, &mag);
    enc_io_unlock();
    if (ret != ESP_OK) return ret;
    /* Per AS5600 datasheet v1-06 p.21: MD=bit5, ML=bit4, MH=bit3. */
    out->detected   = (st >> 5) & 1;
    out->too_weak   = (st >> 4) & 1;
    out->too_strong = (st >> 3) & 1;
    out->agc        = agc;
    out->magnitude  = mag;
    return ESP_OK;
}

esp_err_t encoder_read_angle(enc_axis_t axis, float *out_deg)
{
    as5600_t *s = enc_for(axis);
    if (!out_deg || !s) return ESP_ERR_INVALID_ARG;
    enc_io_lock();
    esp_err_t ret = as5600_read_angle_filtered(s, out_deg);
    enc_io_unlock();
    return ret;
}
