#pragma once

#include "esp_err.h"
#include "motion.h"
#include "stepper.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * motion_runner — single-slot async executor for long-running motion commands.
 *
 * Rationale: ESP-IDF httpd serves all requests on ONE worker task. A blocking
 * call inside an HTTP handler (e.g. motion_move taking 10-30 s) stalls every
 * other request — /state, /test/encoder, AND the queued WS pose-frame sends —
 * for the entire duration. The user sees the dials freeze.
 *
 * The runner moves that blocking work into its own task. Handlers submit a
 * command and return 202 + task_id immediately; the runner executes and
 * broadcasts a `kind:"task"` WS frame on completion (via log_bus_emit_task).
 *
 * Capacity is 1: only one move/jog may be in flight at a time — matches the
 * underlying motion_move contract which rejects concurrent calls with
 * ESP_ERR_INVALID_STATE.
 */

typedef enum {
    MR_KIND_MOVE = 1,
    MR_KIND_JOG  = 2,
} mr_kind_t;

typedef enum {
    MR_STATUS_IDLE         = 0,
    MR_STATUS_RUNNING      = 1,
    MR_STATUS_DONE_OK      = 2,
    MR_STATUS_DONE_ERR     = 3,
    MR_STATUS_DONE_TIMEOUT = 4,
} mr_status_t;

typedef struct {
    mr_kind_t kind;
    union {
        struct {
            float    az_deg;
            float    el_deg;
            bool     has_az;
            bool     has_el;
            uint32_t timeout_ms;
        } move;
        struct {
            stepper_axis_t axis;
            uint32_t       steps;
            uint32_t       step_hz;
            stepper_dir_t  dir;
        } jog;
    } u;
} mr_cmd_t;

typedef struct {
    uint32_t    id;             /* 0 == nothing ever ran */
    mr_kind_t   kind;
    mr_status_t status;
    uint64_t    started_ms;
    uint64_t    finished_ms;
    union {
        struct { motion_pos_t final; uint32_t duration_ms; } move;
        struct { uint32_t duration_ms; } jog;
    } result;
} mr_snapshot_t;

/* Initialise queue, snapshot mutex, and start the runner task. */
void motion_runner_init(void);

/*
 * Submit a command. Returns the assigned task_id (monotonically increasing
 * starting from 1) on success, or 0 if the runner is already busy with
 * another task (caller should respond 409 busy).
 *
 * On success, log_bus_emit_task(task_id, "accepted", NULL) is broadcast
 * synchronously before this returns, so the UI knows the runner picked up
 * the request even before motion starts.
 */
uint32_t motion_runner_submit(const mr_cmd_t *cmd);

/* Snapshot the current/last task state. Always returns true and writes
 * out (id=0 when nothing has run yet). */
bool motion_runner_snapshot(mr_snapshot_t *out);

/* Map status enum → short string used in WS frames + /state. */
const char *motion_runner_status_str(mr_status_t s);
const char *motion_runner_kind_str(mr_kind_t k);

#ifdef __cplusplus
}
#endif
