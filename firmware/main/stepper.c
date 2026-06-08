#include "stepper.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "stepper";

/* ── Pin definitions ────────────────────────────────────────────────────── */
#define GPIO_STEP_AZ   25
#define GPIO_DIR_AZ    26
#define GPIO_STEP_EL   27
#define GPIO_DIR_EL    33   /* was GPIO14 (JTAG MTMS) — caused DIR floating at ~0.9V */
#define GPIO_ENABLE    32   /* active LOW, shared between both drivers */

/* ── Per-axis context ───────────────────────────────────────────────────── */
typedef struct {
    gpio_num_t        step_pin;
    gpio_num_t        dir_pin;
    esp_timer_handle_t timer;
    volatile int32_t  steps_remaining;
    volatile bool     step_high;
    SemaphoreHandle_t done_sem;
} stepper_ctx_t;

static stepper_ctx_t s_ctx[2];

/* ── Timer callback (runs in ISR context) ───────────────────────────────── */
static void IRAM_ATTR step_timer_cb(void *arg)
{
    stepper_ctx_t *s = (stepper_ctx_t *)arg;

    if (s->step_high) {
        /* Falling edge — count the step */
        gpio_set_level(s->step_pin, 0);
        s->step_high = false;
        s->steps_remaining--;

        if (s->steps_remaining <= 0) {
            esp_timer_stop(s->timer);
            BaseType_t woken = pdFALSE;
            xSemaphoreGiveFromISR(s->done_sem, &woken);
            portYIELD_FROM_ISR(woken);
        }
    } else {
        /* Rising edge */
        gpio_set_level(s->step_pin, 1);
        s->step_high = true;
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void stepper_init(void)
{
    /* Configure all output pins at once */
    uint64_t pin_mask =
        (1ULL << GPIO_STEP_AZ) | (1ULL << GPIO_DIR_AZ) |
        (1ULL << GPIO_STEP_EL) | (1ULL << GPIO_DIR_EL) |
        (1ULL << GPIO_ENABLE);

    gpio_config_t io = {
        .pin_bit_mask = pin_mask,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&io);

    /* Start with drivers disabled */
    gpio_set_level(GPIO_ENABLE, 1);
    gpio_set_level(GPIO_STEP_AZ, 0);
    gpio_set_level(GPIO_STEP_EL, 0);

    const gpio_num_t step_pins[2] = { GPIO_STEP_AZ, GPIO_STEP_EL };
    const gpio_num_t dir_pins[2]  = { GPIO_DIR_AZ,  GPIO_DIR_EL  };
    const char      *names[2]     = { "step_az",    "step_el"    };

    for (int i = 0; i < 2; i++) {
        stepper_ctx_t *s = &s_ctx[i];
        s->step_pin        = step_pins[i];
        s->dir_pin         = dir_pins[i];
        s->steps_remaining = 0;
        s->step_high       = false;
        s->done_sem        = xSemaphoreCreateBinary();

        esp_timer_create_args_t ta = {
            .callback              = step_timer_cb,
            .arg                   = s,
            .name                  = names[i],
            .dispatch_method       = ESP_TIMER_ISR,  /* ISR — точный тайминг для микростепа */
            .skip_unhandled_events = true,
        };
        esp_timer_create(&ta, &s->timer);
    }

    ESP_LOGI(TAG, "init OK — AZ: STEP=%d DIR=%d | EL: STEP=%d DIR=%d | EN=%d",
             GPIO_STEP_AZ, GPIO_DIR_AZ,
             GPIO_STEP_EL, GPIO_DIR_EL,
             GPIO_ENABLE);
}

void stepper_enable(void)  { gpio_set_level(GPIO_ENABLE, 0); }
void stepper_disable(void) { gpio_set_level(GPIO_ENABLE, 1); }

esp_err_t stepper_move(stepper_axis_t axis, uint32_t steps,
                       stepper_dir_t dir, uint32_t step_hz)
{
    if (axis >= 2 || steps == 0 || step_hz == 0) return ESP_ERR_INVALID_ARG;

    stepper_ctx_t *s = &s_ctx[axis];
    if (s->steps_remaining > 0) return ESP_ERR_INVALID_STATE;

    /* Drain any stale semaphore token from a previous move */
    xSemaphoreTake(s->done_sem, 0);

    /*
     * Direction goes straight to the GPIO pin — no software inversion.
     * Mapping from DIR_CW/DIR_CCW to physical shaft rotation is determined
     * entirely by motor wiring + magnet orientation. Current wiring is such
     * that DIR_CCW GPIO → encoder reading increases. The sign convention is
     * resolved in motion.c (delta ≥ 0 → DIR_CCW). See COORDINATES.md §6.
     */
    gpio_set_level(s->dir_pin, (int)dir);
    s->steps_remaining = (int32_t)steps;
    s->step_high       = false;

    /* Timer fires at 2× step_hz — one callback per edge */
    uint64_t period_us = 1000000ULL / ((uint64_t)step_hz * 2);
    if (period_us < 10) period_us = 10;

    return esp_timer_start_periodic(s->timer, period_us);
}

esp_err_t stepper_set_speed(stepper_axis_t axis, uint32_t step_hz)
{
    if (axis >= 2) return ESP_ERR_INVALID_ARG;
    if (step_hz == 0) return ESP_ERR_INVALID_ARG;

    stepper_ctx_t *s = &s_ctx[axis];
    if (s->steps_remaining <= 0) return ESP_ERR_INVALID_STATE;

    /* Same conversion as stepper_move — 2× rate, one ISR per edge. */
    uint64_t period_us = 1000000ULL / ((uint64_t)step_hz * 2);
    if (period_us < 10) period_us = 10;

    /*
     * esp_timer_restart() changes the period of a RUNNING periodic timer
     * without firing a spurious callback or restarting the edge state
     * machine. The next callback will arrive after `period_us` from now, and
     * subsequent ones at the new cadence. This is the key API call that lets
     * us do trapezoidal acceleration without inter-stage micro-coasts.
     */
    return esp_timer_restart(s->timer, period_us);
}

esp_err_t stepper_wait(stepper_axis_t axis, uint32_t timeout_ms)
{
    if (axis >= 2) return ESP_ERR_INVALID_ARG;
    TickType_t ticks = (timeout_ms == 0)
                     ? portMAX_DELAY
                     : pdMS_TO_TICKS(timeout_ms);
    return (xSemaphoreTake(s_ctx[axis].done_sem, ticks) == pdTRUE)
           ? ESP_OK
           : ESP_ERR_TIMEOUT;
}

void stepper_stop(stepper_axis_t axis)
{
    if (axis >= 2) return;
    stepper_ctx_t *s = &s_ctx[axis];
    esp_timer_stop(s->timer);
    s->steps_remaining = 0;
    gpio_set_level(s->step_pin, 0);
    s->step_high = false;
    xSemaphoreTake(s->done_sem, 0);   /* drain any stale token */
}

bool stepper_is_busy(stepper_axis_t axis)
{
    if (axis >= 2) return false;
    return s_ctx[axis].steps_remaining > 0;
}

esp_err_t stepper_spin(stepper_axis_t axis, stepper_dir_t dir, uint32_t step_hz)
{
    /* INT32_MAX steps ≈ 67 000 output revolutions — effectively infinite */
    return stepper_move(axis, (uint32_t)0x7FFFFFFF, dir, step_hz);
}
