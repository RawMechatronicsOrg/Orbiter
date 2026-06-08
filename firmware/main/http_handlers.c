#include "http_handlers.h"
#include "motion.h"
#include "motion_runner.h"
#include "encoder.h"
#include "stepper.h"
#include "log_bus.h"
#include "esp_chip_info.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "esp_idf_version.h"
#include "esp_system.h"
#include "esp_log.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static const char *TAG = "http";

/* ── Minimal JSON helpers (no external deps) ────────────────────────────── */

/** Read full request body into buf. Returns bytes read, -1 on error. */
static int read_body(httpd_req_t *req, char *buf, size_t buf_len)
{
    int total = 0, remaining = req->content_len;
    if (remaining == 0 || remaining >= (int)buf_len) return -1;
    while (remaining > 0) {
        int n = httpd_req_recv(req, buf + total,
                               remaining < (int)(buf_len - total)
                               ? remaining : (int)(buf_len - total));
        if (n <= 0) return -1;
        total += n; remaining -= n;
    }
    buf[total] = '\0';
    return total;
}

/* ── CORS OPTIONS handler ───────────────────────────────────────────────── */

esp_err_t handler_options(httpd_req_t *req)
{
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin",  "*");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "Content-Type");
    httpd_resp_send(req, NULL, 0);
    return ESP_OK;
}

/** Send JSON string with given HTTP status code. */
static esp_err_t send_json(httpd_req_t *req, const char *json, int status)
{
    if (status != 200) {
        httpd_resp_set_status(req,
            status == 202 ? "202 Accepted"        :
            status == 408 ? "408 Request Timeout" :
            status == 409 ? "409 Conflict"        :
            status == 400 ? "400 Bad Request"     :
                            "500 Internal Server Error");
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, json);
}

/** Extract float value for "key" from flat JSON string. Returns false if not found. */
static bool json_float(const char *json, const char *key, float *out)
{
    char needle[48];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ') p++;
    return sscanf(p, "%f", out) == 1;
}

/** Extract unsigned int value for "key". */
static bool json_uint(const char *json, const char *key, uint32_t *out)
{
    char needle[48];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ') p++;
    unsigned int v;
    if (sscanf(p, "%u", &v) != 1) return false;
    *out = (uint32_t)v;
    return true;
}

/** Extract quoted string value for "key" into out (max len bytes). */
static bool json_str(const char *json, const char *key, char *out, size_t len)
{
    char needle[48];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ') p++;
    if (*p != '"') return false;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < len - 1) out[i++] = *p++;
    out[i] = '\0';
    return true;
}

static const char *state_str(motion_state_t s)
{
    switch (s) {
        case MOTION_IDLE:     return "idle";
        case MOTION_MOVING:   return "moving";
        case MOTION_SPINNING: return "spinning";
        case MOTION_ERROR:    return "error";
        default:              return "unknown";
    }
}

/* ── GET /health ────────────────────────────────────────────────────────── */

esp_err_t handler_health(httpd_req_t *req)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    esp_chip_info_t chip;
    esp_chip_info(&chip);

    char buf[320];
    snprintf(buf, sizeof(buf),
        "{"
        "\"status\":\"ok\","
        "\"device\":\"Orbiter\","
        "\"chip\":\"ESP32-D0WD-V3\","
        "\"revision\":%d,"
        "\"cores\":%d,"
        "\"mac\":\"%02x:%02x:%02x:%02x:%02x:%02x\","
        "\"free_heap_bytes\":%lu,"
        "\"uptime_ms\":%llu,"
        "\"idf_version\":\"%s\""
        "}",
        chip.revision, chip.cores,
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
        (unsigned long)esp_get_free_heap_size(),
        (unsigned long long)(esp_timer_get_time() / 1000ULL),
        esp_get_idf_version());

    ESP_LOGI(TAG, "GET /health");
    return send_json(req, buf, 200);
}

/* ── GET /state ─────────────────────────────────────────────────────────── */

esp_err_t handler_state(httpd_req_t *req)
{
    motion_state_t state = motion_get_state();
    motion_pos_t   pos   = motion_get_position();

    mr_snapshot_t snap = { 0 };
    motion_runner_snapshot(&snap);

    /*
     * Format runner.result inline so UI's polling loop can settle a task
     * without needing the /ws/log channel. Empty object {} when nothing has
     * ever run (snap.id == 0); per-kind fields otherwise.
     */
    char runner_result[224] = "{}";
    if (snap.id != 0) {
        if (snap.kind == MR_KIND_MOVE) {
            snprintf(runner_result, sizeof(runner_result),
                "{\"azimuth_deg\":%.3f,\"elevation_deg\":%.3f,\"duration_ms\":%lu}",
                (double)snap.result.move.final.az_deg,
                (double)snap.result.move.final.el_deg,
                (unsigned long)snap.result.move.duration_ms);
        } else if (snap.kind == MR_KIND_JOG) {
            snprintf(runner_result, sizeof(runner_result),
                "{\"duration_ms\":%lu}",
                (unsigned long)snap.result.jog.duration_ms);
        }
    }

    char buf[896];
    snprintf(buf, sizeof(buf),
        "{"
        "\"state\":\"%s\","
        "\"motors_enabled\":%s,"
        "\"spinning_az\":%s,"
        "\"spinning_el\":%s,"
        "\"azimuth\":{\"angle_deg\":%.3f},"
        "\"elevation\":{\"angle_deg\":%.3f},"
        "\"calibration\":{"
            "\"az_zero_raw_deg\":%.3f,"
            "\"el_zero_raw_deg\":%.3f"
        "},"
        "\"runner\":{"
            "\"id\":%lu,"
            "\"kind\":\"%s\","
            "\"status\":\"%s\","
            "\"result\":%s"
        "}"
        "}",
        state_str(state),
        motion_motors_enabled() ? "true" : "false",
        motion_spinning_az() ? "true" : "false",
        motion_spinning_el() ? "true" : "false",
        (double)pos.az_deg, (double)pos.el_deg,
        (double)motion_get_az_zero_raw(),
        (double)motion_get_el_zero_raw(),
        (unsigned long)snap.id,
        motion_runner_kind_str(snap.kind),
        motion_runner_status_str(snap.status),
        runner_result);

    ESP_LOGI(TAG, "GET /state → AZ=%.2f EL=%.2f %s",
             pos.az_deg, pos.el_deg, state_str(state));
    return send_json(req, buf, 200);
}

/* ── POST /move ─────────────────────────────────────────────────────────── */

/*
 * Async since Phase 2: handler submits to motion_runner and returns 202
 * immediately with task_id. The actual move runs on the runner task while
 * httpd stays free to serve /state and pump WS frames. UI awaits the
 * `kind:"task"` WS frame matching task_id for completion semantics
 * (status: "done" | "timeout" | "error" + result block).
 */
esp_err_t handler_move(httpd_req_t *req)
{
    char body[256];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    float az_deg = 0, el_deg = 0;
    bool  has_az = json_float(body, "azimuth_deg",   &az_deg);
    bool  has_el = json_float(body, "elevation_deg", &el_deg);
    uint32_t tmo = 15000;
    json_uint(body, "timeout_ms", &tmo);

    if (!has_az && !has_el)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"provide azimuth_deg or elevation_deg\"}",
            400);

    /*
     * Azimuth is continuous — accept any float. motion.c normalises to [0, 360)
     * via angle_delta which always picks the shortest path. See COORDINATES.md §3.1.
     */
    if (has_el && (el_deg < -36.0f || el_deg > 90.0f))
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"elevation_deg out of range -36..+90\"}",
            400);

    mr_cmd_t cmd = {
        .kind = MR_KIND_MOVE,
        .u.move = {
            .az_deg     = az_deg,
            .el_deg     = el_deg,
            .has_az     = has_az,
            .has_el     = has_el,
            .timeout_ms = tmo,
        },
    };
    uint32_t id = motion_runner_submit(&cmd);
    if (id == 0)
        return send_json(req, "{\"status\":\"busy\"}", 409);

    char resp[120];
    snprintf(resp, sizeof(resp),
        "{\"status\":\"accepted\",\"task_id\":%lu,\"kind\":\"move\"}",
        (unsigned long)id);

    LOGX_EMIT_I(TAG, "POST /move → 202 task_id=%lu (az=%s%.2f el=%s%.2f tmo=%lu)",
                (unsigned long)id,
                has_az ? "" : "skip:", (double)az_deg,
                has_el ? "" : "skip:", (double)el_deg,
                (unsigned long)tmo);
    return send_json(req, resp, 202);
}

/* ── POST /motors ───────────────────────────────────────────────────────── */

esp_err_t handler_motors(httpd_req_t *req)
{
    char body[64];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    /* Accept {"enabled":true} or {"enabled":false} */
    const char *p = strstr(body, "\"enabled\"");
    if (!p) return send_json(req, "{\"status\":\"error\",\"message\":\"enabled required\"}", 400);
    p += strlen("\"enabled\"");
    while (*p == ' ' || *p == ':' || *p == ' ') p++;
    bool en = (strncmp(p, "true", 4) == 0);
    motion_set_motors(en);

    char resp[64];
    snprintf(resp, sizeof(resp), "{\"status\":\"ok\",\"enabled\":%s}", en ? "true" : "false");
    LOGX_EMIT_I(TAG, "POST /motors → %s", en ? "enabled" : "disabled");
    return send_json(req, resp, 200);
}

/* ── POST /zero ─────────────────────────────────────────────────────────── */

esp_err_t handler_zero(httpd_req_t *req)
{
    char body[64];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    char axis_str[8] = {0};
    if (!json_str(body, "axis", axis_str, sizeof(axis_str)))
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis required\"}", 400);

    bool do_az = (strcmp(axis_str, "az")   == 0 || strcmp(axis_str, "both") == 0);
    bool do_el = (strcmp(axis_str, "el")   == 0 || strcmp(axis_str, "both") == 0);
    if (!do_az && !do_el)
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis must be az, el or both\"}", 400);

    motion_zero(do_az, do_el);
    motion_pos_t pos = motion_get_position();

    char resp[120];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"azimuth_deg\":%.3f,\"elevation_deg\":%.3f}",
             (double)pos.az_deg, (double)pos.el_deg);
    LOGX_EMIT_I(TAG, "POST /zero axis=%s", axis_str);
    return send_json(req, resp, 200);
}

/* ── POST /calibrate ────────────────────────────────────────────────────── */

esp_err_t handler_calibrate(httpd_req_t *req)
{
    char body[128];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    char axis_str[8] = {0};
    char mode_str[12] = "current";
    if (!json_str(body, "axis", axis_str, sizeof(axis_str)))
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis required\"}", 400);
    json_str(body, "mode", mode_str, sizeof(mode_str));

    bool do_az = (strcmp(axis_str, "az") == 0 || strcmp(axis_str, "both") == 0);
    bool do_el = (strcmp(axis_str, "el") == 0 || strcmp(axis_str, "both") == 0);
    if (!do_az && !do_el)
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis must be az, el or both\"}", 400);

    motion_cal_mode_t mode;
    float manual_az = 0.0f, manual_el = 0.0f;
    if      (strcmp(mode_str, "current")  == 0) mode = CAL_MODE_CURRENT;
    else if (strcmp(mode_str, "explicit") == 0) {
        mode = CAL_MODE_EXPLICIT;
        if (do_az && !json_float(body, "az_raw_deg", &manual_az))
            return send_json(req, "{\"status\":\"error\",\"message\":\"az_raw_deg required for explicit mode\"}", 400);
        if (do_el && !json_float(body, "el_raw_deg", &manual_el))
            return send_json(req, "{\"status\":\"error\",\"message\":\"el_raw_deg required for explicit mode\"}", 400);
    }
    else if (strcmp(mode_str, "reset")    == 0) mode = CAL_MODE_RESET;
    else return send_json(req, "{\"status\":\"error\",\"message\":\"mode must be current, explicit or reset\"}", 400);

    esp_err_t ret = motion_set_calibration(do_az, do_el, mode, manual_az, manual_el);
    if (ret != ESP_OK)
        return send_json(req, "{\"status\":\"error\",\"message\":\"calibration failed\"}", 500);

    char resp[180];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"axis\":\"%s\",\"mode\":\"%s\","
             "\"az_zero_raw_deg\":%.3f,\"el_zero_raw_deg\":%.3f}",
             axis_str, mode_str,
             (double)motion_get_az_zero_raw(),
             (double)motion_get_el_zero_raw());
    LOGX_EMIT_I(TAG, "POST /calibrate axis=%s mode=%s → az=%.3f el=%.3f",
                axis_str, mode_str,
                (double)motion_get_az_zero_raw(),
                (double)motion_get_el_zero_raw());
    return send_json(req, resp, 200);
}

/* ── POST /spin ─────────────────────────────────────────────────────────── */

esp_err_t handler_spin(httpd_req_t *req)
{
    char body[128];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    char axis_str[4] = {0};
    char dir_str[8]  = "cw";
    uint32_t step_hz = 525;  /* calibrated ~9°/sec (see motion.c CAL_DEFAULT_SPR) */

    if (!json_str(body, "axis", axis_str, sizeof(axis_str)))
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis required\"}", 400);
    json_str(body,  "dir",     dir_str, sizeof(dir_str));
    json_uint(body, "step_hz", &step_hz);

    stepper_axis_t axis;
    if      (strcmp(axis_str, "az") == 0) axis = STEPPER_AZ;
    else if (strcmp(axis_str, "el") == 0) axis = STEPPER_EL;
    else return send_json(req, "{\"status\":\"error\",\"message\":\"axis must be az or el\"}", 400);

    if (step_hz == 0 || step_hz > 10000) step_hz = 525;
    stepper_dir_t dir = (strcmp(dir_str, "ccw") == 0) ? DIR_CCW : DIR_CW;

    esp_err_t ret = motion_spin(axis, dir, step_hz);
    if (ret == ESP_ERR_INVALID_STATE)
        return send_json(req, "{\"status\":\"busy\"}", 409);
    if (ret != ESP_OK)
        return send_json(req, "{\"status\":\"error\",\"message\":\"spin failed\"}", 500);

    char resp[80];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"axis\":\"%s\",\"dir\":\"%s\",\"step_hz\":%lu}",
             axis_str, dir_str, (unsigned long)step_hz);
    LOGX_EMIT_I(TAG, "POST /spin axis=%s dir=%s hz=%lu", axis_str, dir_str, (unsigned long)step_hz);
    return send_json(req, resp, 200);
}

/* ── POST /spin/stop ────────────────────────────────────────────────────── */

esp_err_t handler_spin_stop(httpd_req_t *req)
{
    char body[64];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    char axis_str[8] = {0};
    if (!json_str(body, "axis", axis_str, sizeof(axis_str)))
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis required\"}", 400);

    bool stop_az = (strcmp(axis_str, "az")   == 0 || strcmp(axis_str, "both") == 0);
    bool stop_el = (strcmp(axis_str, "el")   == 0 || strcmp(axis_str, "both") == 0);
    if (!stop_az && !stop_el)
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis must be az, el or both\"}", 400);

    motion_spin_stop(stop_az, stop_el);
    LOGX_EMIT_I(TAG, "POST /spin/stop axis=%s", axis_str);
    return send_json(req, "{\"status\":\"ok\"}", 200);
}

/* ── POST /reboot ───────────────────────────────────────────────────────── */

/*
 * esp_restart() never returns, so it cannot be called inline: the HTTP
 * response would never flush and the caller would just see a dropped
 * connection. Instead the handler acks 200, then a one-shot esp_timer fires
 * the restart ~500 ms later — long enough for the TCP response to leave the
 * device and for the storage-api proxy to read it.
 */
static void reboot_timer_cb(void *arg)
{
    (void)arg;
    esp_restart();
}

esp_err_t handler_reboot(httpd_req_t *req)
{
    LOGX_EMIT_W(TAG, "POST /reboot → firmware restart in 500 ms");

    static esp_timer_handle_t reboot_timer = NULL;
    if (reboot_timer == NULL) {
        const esp_timer_create_args_t args = {
            .callback = reboot_timer_cb,
            .name     = "reboot",
        };
        if (esp_timer_create(&args, &reboot_timer) != ESP_OK)
            return send_json(req,
                "{\"status\":\"error\",\"message\":\"reboot timer create failed\"}", 500);
    }
    /* ESP_ERR_INVALID_STATE here just means a reboot is already pending. */
    esp_timer_start_once(reboot_timer, 500000 /* µs = 500 ms */);
    return send_json(req, "{\"status\":\"ok\",\"message\":\"rebooting\"}", 200);
}

/* ── GET /test/encoder ──────────────────────────────────────────────────── */

esp_err_t handler_test_encoder(httpd_req_t *req)
{
    /* Sizes account for the extended AZ error JSON (magnet_* synonyms). */
    char buf[400];
    char az_part[180], el_part[120];

    uint16_t raw; float deg;

    /* ── Azimuth encoder (AS5600 over I2C, 12-bit) ── */
    if (encoder_read_raw(ENC_AZ, &raw) == ESP_OK) {
        deg = (float)raw * (360.0f / 4096.0f);
        snprintf(az_part, sizeof(az_part),
                 "\"azimuth\":{\"ok\":true,\"raw\":%u,\"angle_deg\":%.3f}",
                 raw, (double)deg);
        ESP_LOGI(TAG, "encoder AZ: raw=%u angle=%.3f", raw, (double)deg);
    } else {
        uint8_t flags = 0;
        encoder_read_errors(ENC_AZ, &flags);
        encoder_read_diag(ENC_AZ);
        /* For AS5600 the legacy fields carry magnet status (see encoder.h):
         *   bit0 = magnet missing, bit1 = too weak, bit2 = too strong.
         * Expose synonyms alongside legacy frerr/invcmd/parity. */
        snprintf(az_part, sizeof(az_part),
                 "\"azimuth\":{\"ok\":false,\"frerr\":%d,\"invcmd\":%d,\"parity\":%d,"
                 "\"magnet_missing\":%d,\"magnet_weak\":%d,\"magnet_strong\":%d}",
                 (flags >> 0) & 1, (flags >> 1) & 1, (flags >> 2) & 1,
                 (flags >> 0) & 1, (flags >> 1) & 1, (flags >> 2) & 1);
        ESP_LOGW(TAG, "encoder AZ: read failed");
    }

    /* ── Elevation encoder (AS5048A over SPI, CS=GPIO17, 14-bit) ── */
    if (encoder_read_raw(ENC_EL, &raw) == ESP_OK) {
        deg = (float)raw * (360.0f / 16384.0f);
        snprintf(el_part, sizeof(el_part),
                 "\"elevation\":{\"ok\":true,\"raw\":%u,\"angle_deg\":%.3f}",
                 raw, (double)deg);
        ESP_LOGI(TAG, "encoder EL: raw=%u angle=%.3f", raw, (double)deg);
    } else {
        uint8_t flags = 0;
        encoder_read_errors(ENC_EL, &flags);
        encoder_read_diag(ENC_EL);
        snprintf(el_part, sizeof(el_part),
                 "\"elevation\":{\"ok\":false,\"frerr\":%d,\"invcmd\":%d,\"parity\":%d}",
                 (flags >> 0) & 1, (flags >> 1) & 1, (flags >> 2) & 1);
        ESP_LOGW(TAG, "encoder EL: read failed");
    }

    snprintf(buf, sizeof(buf), "{%s,%s}", az_part, el_part);
    return send_json(req, buf, 200);
}

/* ── POST /test/jog ─────────────────────────────────────────────────────── */

esp_err_t handler_test_jog(httpd_req_t *req)
{
    char body[128];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    char axis_str[4] = {0};
    char dir_str[8]  = "cw";
    uint32_t steps = 0, step_hz = 200;

    if (!json_str(body, "axis", axis_str, sizeof(axis_str)) || !json_uint(body, "steps", &steps))
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"axis and steps required\"}", 400);

    json_str(body,  "dir",     dir_str, sizeof(dir_str));
    json_uint(body, "step_hz", &step_hz);

    stepper_axis_t axis;
    if      (strcmp(axis_str, "az") == 0) axis = STEPPER_AZ;
    else if (strcmp(axis_str, "el") == 0) axis = STEPPER_EL;
    else return send_json(req,
            "{\"status\":\"error\",\"message\":\"axis must be az or el\"}", 400);

    if (steps == 0 || steps > 100000)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"steps must be 1-100000\"}", 400);

    if (step_hz == 0 || step_hz > 5000) step_hz = 200;

    stepper_dir_t dir = (strcmp(dir_str, "ccw") == 0) ? DIR_CCW : DIR_CW;

    LOGX_EMIT_I(TAG, "jog: axis=%s steps=%lu dir=%s hz=%lu (submit)",
                axis_str, (unsigned long)steps, dir_str, (unsigned long)step_hz);

    /*
     * Async-submit via motion_runner. The runner task executes the blocking
     * stepper_wait() so httpd stays free for /state polls + other requests.
     * Returning 202 + task_id immediately makes the long-running test/jog
     * behave like /move and /calibrate/spr auto (UI awaits via polling).
     */
    mr_cmd_t cmd = {
        .kind = MR_KIND_JOG,
        .u.jog = {
            .axis    = axis,
            .steps   = steps,
            .step_hz = step_hz,
            .dir     = dir,
        },
    };
    uint32_t task_id = motion_runner_submit(&cmd);
    if (task_id == 0)
        return send_json(req, "{\"status\":\"busy\"}", 409);

    char resp[96];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"accepted\",\"task_id\":%lu}",
             (unsigned long)task_id);
    return send_json(req, resp, 202);
}
