#pragma once

#include "esp_err.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Log bus — explicit event channel for the UI.
 *
 *   LOGX_EMIT(level, tag, "fmt", ...) — write to UART (ESP_LOGx) AND queue a
 *   compact JSON frame for any clients connected to GET /ws/log.
 *
 * We do NOT hijack esp_log_set_vprintf — that would catch internal IDF logs
 * and risk recursion through Wi-Fi/HTTP layers. Only explicit operational
 * events are pushed (motion, http_handlers, encoder failures).
 */

typedef enum {
    LOG_BUS_INFO = 0,
    LOG_BUS_WARN = 1,
    LOG_BUS_ERR  = 2,
} log_bus_level_t;

void log_bus_init(void);

/* Register WebSocket route on the given httpd server. */
esp_err_t log_bus_register_ws(httpd_handle_t server, const char *uri);

/* Append a formatted event to the bus and broadcast to active WS clients. */
void log_bus_emit(log_bus_level_t lvl, const char *tag, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

/*
 * Broadcast a compact pose frame on the same /ws/log channel. Sent at the
 * pose-tick cadence (~10 Hz) — the UI uses these to update az/el/state with
 * lower latency than polling /state. Skipped silently if no clients.
 *
 *   { "kind":"pose","ts_ms":N,"az":X.X,"el":Y.Y,"st":"idle",
 *     "motors":true,"sp_az":false,"sp_el":false }
 */
void log_bus_emit_pose(float az_deg, float el_deg, const char *state_str,
                       bool motors_on, bool spin_az, bool spin_el);

/*
 * Broadcast a task-completion (or transition) frame for a long-running motion
 * command submitted via motion_runner_submit(). UI awaits these to know when
 * a /move or /calibrate/spr async command finished.
 *
 *   { "kind":"task","task_id":N,"status":"accepted|done|error|timeout|busy",
 *     "result":{ ... } }
 *
 * `result_json_inner` is splat into the "result":{...} block VERBATIM (no
 * surrounding braces). Pass NULL or "" for a frame without result. Caller is
 * responsible for valid JSON content (no top-level keys outside the object).
 */
void log_bus_emit_task(uint32_t task_id, const char *status,
                       const char *result_json_inner);

/*
 * Convenience: emit to log bus + ESP_LOGx in a single call site.
 * Keep messages short (UI displays them as one row).
 */
#define LOGX_EMIT_I(tag, ...) do { \
    ESP_LOGI(tag, __VA_ARGS__); \
    log_bus_emit(LOG_BUS_INFO, tag, __VA_ARGS__); \
} while (0)

#define LOGX_EMIT_W(tag, ...) do { \
    ESP_LOGW(tag, __VA_ARGS__); \
    log_bus_emit(LOG_BUS_WARN, tag, __VA_ARGS__); \
} while (0)

#define LOGX_EMIT_E(tag, ...) do { \
    ESP_LOGE(tag, __VA_ARGS__); \
    log_bus_emit(LOG_BUS_ERR, tag, __VA_ARGS__); \
} while (0)

#ifdef __cplusplus
}
#endif
