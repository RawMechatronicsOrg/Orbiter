#include "motion_runner.h"
#include "stepper.h"
#include "log_bus.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include <stdio.h>
#include <string.h>

static const char *TAG = "motion_runner";

/* ── Internal types ─────────────────────────────────────────────────────── */

typedef struct {
    uint32_t id;
    mr_cmd_t cmd;
} mr_queued_t;

/* ── State ──────────────────────────────────────────────────────────────── */

static QueueHandle_t      s_q;            /* depth 1 */
static SemaphoreHandle_t  s_snap_mtx;     /* protects s_snap */
static mr_snapshot_t      s_snap;         /* current/last */
static volatile bool      s_busy;         /* set on submit, cleared on done */
static volatile uint32_t  s_next_id = 1;  /* monotonic, starts at 1 (0 reserved) */

/* ── Helpers ────────────────────────────────────────────────────────────── */

const char *motion_runner_status_str(mr_status_t s)
{
    switch (s) {
        case MR_STATUS_IDLE:         return "idle";
        case MR_STATUS_RUNNING:      return "running";
        case MR_STATUS_DONE_OK:      return "done";
        case MR_STATUS_DONE_ERR:     return "error";
        case MR_STATUS_DONE_TIMEOUT: return "timeout";
        default:                     return "unknown";
    }
}

const char *motion_runner_kind_str(mr_kind_t k)
{
    switch (k) {
        case MR_KIND_MOVE: return "move";
        case MR_KIND_JOG:  return "jog";
        default:           return "unknown";
    }
}

static void snap_set(const mr_snapshot_t *src)
{
    xSemaphoreTake(s_snap_mtx, portMAX_DELAY);
    s_snap = *src;
    xSemaphoreGive(s_snap_mtx);
}

static void snap_update_status(mr_status_t new_status, uint64_t finished_ms)
{
    xSemaphoreTake(s_snap_mtx, portMAX_DELAY);
    s_snap.status      = new_status;
    s_snap.finished_ms = finished_ms;
    xSemaphoreGive(s_snap_mtx);
}

static void snap_set_move_result(motion_pos_t final, uint32_t duration_ms)
{
    xSemaphoreTake(s_snap_mtx, portMAX_DELAY);
    s_snap.result.move.final       = final;
    s_snap.result.move.duration_ms = duration_ms;
    xSemaphoreGive(s_snap_mtx);
}

static void snap_set_jog_result(uint32_t duration_ms)
{
    xSemaphoreTake(s_snap_mtx, portMAX_DELAY);
    s_snap.result.jog.duration_ms = duration_ms;
    xSemaphoreGive(s_snap_mtx);
}

/* ── Command execution ─────────────────────────────────────────────────── */

static void run_move(uint32_t id, const mr_cmd_t *cmd)
{
    uint64_t t0 = esp_timer_get_time() / 1000ULL;
    motion_pos_t final = { 0 };

    esp_err_t ret = motion_move(cmd->u.move.az_deg, cmd->u.move.el_deg,
                                cmd->u.move.has_az, cmd->u.move.has_el,
                                cmd->u.move.timeout_ms, &final);
    uint32_t dur = (uint32_t)(esp_timer_get_time() / 1000ULL - t0);
    uint64_t now_ms = esp_timer_get_time() / 1000ULL;

    snap_set_move_result(final, dur);

    const char *status_str;
    mr_status_t status;
    if (ret == ESP_OK) {
        status     = MR_STATUS_DONE_OK;
        status_str = "done";
    } else if (ret == ESP_ERR_TIMEOUT) {
        status     = MR_STATUS_DONE_TIMEOUT;
        status_str = "timeout";
    } else {
        status     = MR_STATUS_DONE_ERR;
        status_str = "error";
    }
    snap_update_status(status, now_ms);

    char inner[160];
    snprintf(inner, sizeof(inner),
             "\"azimuth_deg\":%.3f,\"elevation_deg\":%.3f,\"duration_ms\":%lu",
             (double)final.az_deg, (double)final.el_deg,
             (unsigned long)dur);
    log_bus_emit_task(id, status_str, inner);
}

static void run_jog(uint32_t id, const mr_cmd_t *cmd)
{
    uint64_t t0 = esp_timer_get_time() / 1000ULL;

    stepper_enable();
    esp_err_t ret = stepper_move(cmd->u.jog.axis,
                                 cmd->u.jog.steps,
                                 cmd->u.jog.dir,
                                 cmd->u.jog.step_hz);
    if (ret == ESP_OK) {
        /* Block until the stepper finishes; this runs on the runner task
         * (priority 5) — NOT on httpd's worker — so httpd stays responsive
         * to /state polls and other requests during the jog. */
        stepper_wait(cmd->u.jog.axis, 0);
    }

    uint32_t dur = (uint32_t)(esp_timer_get_time() / 1000ULL - t0);
    uint64_t now_ms = esp_timer_get_time() / 1000ULL;

    snap_set_jog_result(dur);

    const char *status_str;
    mr_status_t status;
    if (ret == ESP_OK) {
        status     = MR_STATUS_DONE_OK;
        status_str = "done";
    } else if (ret == ESP_ERR_INVALID_STATE) {
        status     = MR_STATUS_DONE_ERR;
        status_str = "error";
    } else {
        status     = MR_STATUS_DONE_ERR;
        status_str = "error";
    }
    snap_update_status(status, now_ms);

    char inner[160];
    snprintf(inner, sizeof(inner),
             "\"axis\":\"%s\",\"steps\":%lu,\"step_hz\":%lu,\"dir\":\"%s\","
             "\"duration_ms\":%lu",
             cmd->u.jog.axis == STEPPER_AZ ? "az" : "el",
             (unsigned long)cmd->u.jog.steps,
             (unsigned long)cmd->u.jog.step_hz,
             cmd->u.jog.dir == DIR_CW ? "cw" : "ccw",
             (unsigned long)dur);
    log_bus_emit_task(id, status_str, inner);
}

/* ── Runner task ────────────────────────────────────────────────────────── */

/*
 * Priority 5 — same as httpd. We need to be at least httpd's level so that
 * pose-tick (priority 4) can still preempt-via-inheritance both of us when
 * holding the encoder mutex. Stack 8 KB is generous: motion_move recurses
 * through axis_move_with_ramp + encoder_read_locked + LOGX_EMIT_I (snprintf)
 * and we'd rather not chase stack overflows during a long scan.
 */
static void motion_runner_task(void *arg)
{
    (void)arg;
    for (;;) {
        mr_queued_t qd;
        if (xQueueReceive(s_q, &qd, portMAX_DELAY) != pdTRUE) continue;

        uint64_t now_ms = esp_timer_get_time() / 1000ULL;

        mr_snapshot_t fresh = { 0 };
        fresh.id          = qd.id;
        fresh.kind        = qd.cmd.kind;
        fresh.status      = MR_STATUS_RUNNING;
        fresh.started_ms  = now_ms;
        fresh.finished_ms = 0;
        snap_set(&fresh);

        log_bus_emit_task(qd.id, "running", NULL);

        switch (qd.cmd.kind) {
            case MR_KIND_MOVE: run_move(qd.id, &qd.cmd); break;
            case MR_KIND_JOG:  run_jog(qd.id, &qd.cmd);  break;
            default:
                snap_update_status(MR_STATUS_DONE_ERR,
                                   esp_timer_get_time() / 1000ULL);
                log_bus_emit_task(qd.id, "error", "\"reason\":\"unknown_kind\"");
                break;
        }

        s_busy = false;
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void motion_runner_init(void)
{
    if (s_q) return;   /* idempotent */
    s_q        = xQueueCreate(1, sizeof(mr_queued_t));
    s_snap_mtx = xSemaphoreCreateMutex();
    memset(&s_snap, 0, sizeof(s_snap));
    s_busy     = false;
    xTaskCreate(motion_runner_task, "mot_runner", 8192, NULL, 5, NULL);
    ESP_LOGI(TAG, "init OK");
}

uint32_t motion_runner_submit(const mr_cmd_t *cmd)
{
    if (!cmd || !s_q) return 0;
    if (s_busy)       return 0;

    /* Reserve the slot BEFORE assigning id so two concurrent submits can't
     * both think they got accepted. xQueueSend with timeout 0 + s_busy gate
     * gives us atomicity since the underlying queue is depth-1. */
    s_busy = true;

    mr_queued_t qd;
    qd.id  = s_next_id++;
    if (s_next_id == 0) s_next_id = 1;  /* wrap, never hand out 0 */
    qd.cmd = *cmd;

    if (xQueueSend(s_q, &qd, 0) != pdTRUE) {
        s_busy = false;
        return 0;
    }

    /* Emit accepted frame BEFORE task picks it up — gives UI a fence to
     * start awaiting completion without missing the running→done transition
     * (the runner task may not yet have scheduled). */
    log_bus_emit_task(qd.id, "accepted", NULL);

    return qd.id;
}

bool motion_runner_snapshot(mr_snapshot_t *out)
{
    if (!out) return false;
    if (!s_snap_mtx) {
        memset(out, 0, sizeof(*out));
        return true;
    }
    xSemaphoreTake(s_snap_mtx, portMAX_DELAY);
    *out = s_snap;
    xSemaphoreGive(s_snap_mtx);
    return true;
}
