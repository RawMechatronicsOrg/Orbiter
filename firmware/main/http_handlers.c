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
#include "esp_ota_ops.h"
#include "motorcal.h"
#include "esp_app_format.h"
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
            status == 413 ? "413 Content Too Large" :
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
        } else if (snap.kind == MR_KIND_MOTOR_CAL) {
            snprintf(runner_result, sizeof(runner_result),
                "{\"axis\":\"%s\",\"duration_ms\":%lu}",
                snap.result.motor_cal.axis == STEPPER_AZ ? "az" : "el",
                (unsigned long)snap.result.motor_cal.duration_ms);
        }
    }

    /* Motor calibration per axis — what the server needs to turn °/s into
     * step rates and to show whether the axes are calibrated. */
    char motor_cal[320];
    {
        motorcal_result_t az, el;
        motorcal_result(STEPPER_AZ, &az);
        motorcal_result(STEPPER_EL, &el);
        snprintf(motor_cal, sizeof(motor_cal),
            "{\"active\":%s,"
            "\"az\":{\"valid\":%s,\"steps_per_deg\":%.4f,\"max_speed_dps\":%.2f,\"dir_sign\":%d},"
            "\"el\":{\"valid\":%s,\"steps_per_deg\":%.4f,\"max_speed_dps\":%.2f,\"dir_sign\":%d}}",
            motorcal_active() ? "true" : "false",
            az.valid ? "true" : "false", (double)az.steps_per_deg, (double)az.max_speed_dps, az.dir_sign,
            el.valid ? "true" : "false", (double)el.steps_per_deg, (double)el.max_speed_dps, el.dir_sign);
    }

    char buf[1280];
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
            "\"el_zero_raw_deg\":%.3f,"
            "\"el_sign\":%d"
        "},"
        "\"runner\":{"
            "\"id\":%lu,"
            "\"kind\":\"%s\","
            "\"status\":\"%s\","
            "\"result\":%s"
        "},"
        "\"motor_cal\":%s"
        "}",
        state_str(state),
        motion_motors_enabled() ? "true" : "false",
        motion_spinning_az() ? "true" : "false",
        motion_spinning_el() ? "true" : "false",
        (double)pos.az_deg, (double)pos.el_deg,
        (double)motion_get_az_zero_raw(),
        (double)motion_get_el_zero_raw(),
        motion_get_el_sign(),
        (unsigned long)snap.id,
        motion_runner_kind_str(snap.kind),
        motion_runner_status_str(snap.status),
        runner_result,
        motor_cal);

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

    /* Optional per-axis top step rate (UI speed knobs, via the server).
     * Absent / 0 → the firmware uses its compiled default cap. motion_move
     * clamps to a safe band, so no range check is needed here. */
    uint32_t az_hz_max = 0, el_hz_max = 0;
    json_uint(body, "az_hz_max", &az_hz_max);
    json_uint(body, "el_hz_max", &el_hz_max);

    if (!has_az && !has_el)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"provide azimuth_deg or elevation_deg\"}",
            400);

    /*
     * Azimuth is continuous — accept any float. motion.c normalises to [0, 360)
     * via angle_delta which always picks the shortest path. See COORDINATES.md §3.1.
     */
    /* Written as "not inside" so a NaN from the parser is rejected too. */
    if (has_el && !(el_deg >= MOTION_EL_MIN_DEG && el_deg <= MOTION_EL_MAX_DEG)) {
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "{\"status\":\"error\",\"message\":\"elevation_deg out of range %d..%+d\"}",
                 CONFIG_ORBITER_EL_MIN_DEG, CONFIG_ORBITER_EL_MAX_DEG);
        return send_json(req, msg, 400);
    }

    mr_cmd_t cmd = {
        .kind = MR_KIND_MOVE,
        .u.move = {
            .az_deg     = az_deg,
            .el_deg     = el_deg,
            .has_az     = has_az,
            .has_el     = has_el,
            .az_hz_max  = az_hz_max,
            .el_hz_max  = el_hz_max,
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
    uint32_t step_hz = 525;  /* v2: ~8.4 °/s on AZ, ~5.9 °/s on EL (see motion.c) */

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

    if (motorcal_active())
        return send_json(req, "{\"status\":\"busy\",\"message\":\"motor calibration running\"}", 409);

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

/* Restart ~500 ms from now. ESP_ERR_INVALID_STATE means one is already pending. */
static esp_err_t schedule_reboot(void)
{
    static esp_timer_handle_t reboot_timer = NULL;
    if (reboot_timer == NULL) {
        const esp_timer_create_args_t args = {
            .callback = reboot_timer_cb,
            .name     = "reboot",
        };
        esp_err_t err = esp_timer_create(&args, &reboot_timer);
        if (err != ESP_OK) return err;
    }
    esp_err_t err = esp_timer_start_once(reboot_timer, 500000 /* µs = 500 ms */);
    return err == ESP_ERR_INVALID_STATE ? ESP_OK : err;
}

esp_err_t handler_reboot(httpd_req_t *req)
{
    LOGX_EMIT_W(TAG, "POST /reboot → firmware restart in 500 ms");
    if (schedule_reboot() != ESP_OK)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"reboot timer create failed\"}", 500);
    return send_json(req, "{\"status\":\"ok\",\"message\":\"rebooting\"}", 200);
}

/* ── POST /ota ──────────────────────────────────────────────────────────── */

/*
 * Firmware update over Wi-Fi. The request body is the raw app image
 * (build/orbiter.bin), streamed into the inactive OTA slot:
 *
 *   curl --data-binary @build/orbiter.bin http://<esp-ip>/ota
 *
 * On success that slot becomes the boot partition and the device restarts.
 * The new image starts in PENDING_VERIFY; app_main confirms it once the HTTP
 * server is up, and if that does not happen in time the bootloader goes back
 * to the previous image (see ota_confirm_* in orbiter_main.c). So a build
 * that cannot join Wi-Fi still leaves the device reachable.
 */
#define OTA_CHUNK          4096
#define OTA_MAX_TIMEOUTS   5      /* consecutive recv timeouts before giving up */

static esp_err_t ota_fail(httpd_req_t *req, const char *step, esp_err_t err, int status)
{
    char msg[160];
    LOGX_EMIT_E(TAG, "POST /ota failed at %s: %s", step, esp_err_to_name(err));
    snprintf(msg, sizeof(msg),
             "{\"status\":\"error\",\"message\":\"%s failed: %s\"}",
             step, esp_err_to_name(err));
    return send_json(req, msg, status);
}

/* The image must be this project: refuse e.g. the pendulum firmware by mistake. */
static bool ota_image_is_ours(const char *head, size_t len)
{
    const size_t off = sizeof(esp_image_header_t) + sizeof(esp_image_segment_header_t);
    if (len < off + sizeof(esp_app_desc_t)) return false;
    const esp_app_desc_t *incoming = (const esp_app_desc_t *)(head + off);
    const esp_app_desc_t *running  = esp_app_get_description();
    return incoming->magic_word == ESP_APP_DESC_MAGIC_WORD &&
           strncmp(incoming->project_name, running->project_name,
                   sizeof(incoming->project_name)) == 0;
}

esp_err_t handler_ota(httpd_req_t *req)
{
    motion_state_t st = motion_get_state();
    if (st == MOTION_MOVING || st == MOTION_SPINNING || motorcal_active())
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"axes are moving, stop them first\"}", 409);
    if (req->content_len == 0)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"empty body, send build/orbiter.bin\"}", 400);

    const esp_partition_t *slot = esp_ota_get_next_update_partition(NULL);
    if (!slot)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"no OTA slot, flash the OTA partition table over USB once\"}", 500);
    if (req->content_len > slot->size)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"image is larger than the OTA slot\"}", 413);

    /* Flash writes stall non-IRAM interrupts, the step timer among them:
     * no driver may be holding or stepping while the image is written. */
    motion_set_motors(false);
    LOGX_EMIT_W(TAG, "POST /ota → %u bytes into %s",
                (unsigned)req->content_len, slot->label);

    char *buf = malloc(OTA_CHUNK);
    if (!buf) return ota_fail(req, "malloc", ESP_ERR_NO_MEM, 500);

    esp_ota_handle_t ota = 0;
    bool        begun    = false;
    esp_err_t   err      = ESP_OK;
    const char *step     = "receive";
    size_t      left     = req->content_len;
    int         timeouts = 0;

    while (left > 0) {
        int n = httpd_req_recv(req, buf, left < OTA_CHUNK ? left : OTA_CHUNK);
        if (n == HTTPD_SOCK_ERR_TIMEOUT && ++timeouts < OTA_MAX_TIMEOUTS) continue;
        if (n <= 0) { err = ESP_ERR_TIMEOUT; break; }
        timeouts = 0;

        if (!begun) {
            /* The first chunk carries the app descriptor. */
            if (!ota_image_is_ours(buf, (size_t)n)) {
                free(buf);
                return send_json(req,
                    "{\"status\":\"error\",\"message\":\"not an Orbiter firmware image\"}", 400);
            }
            step = "esp_ota_begin";
            err = esp_ota_begin(slot, OTA_WITH_SEQUENTIAL_WRITES, &ota);
            if (err != ESP_OK) break;
            begun = true;
        }
        step = "esp_ota_write";
        err = esp_ota_write(ota, buf, n);
        if (err != ESP_OK) break;
        left -= (size_t)n;
    }
    free(buf);

    if (err != ESP_OK) {
        if (begun) esp_ota_abort(ota);
        return ota_fail(req, step, err, 500);
    }
    err = esp_ota_end(ota);   /* verifies the image checksum and hash */
    if (err != ESP_OK) return ota_fail(req, "esp_ota_end", err, 400);
    err = esp_ota_set_boot_partition(slot);
    if (err != ESP_OK) return ota_fail(req, "esp_ota_set_boot_partition", err, 500);

    LOGX_EMIT_W(TAG, "POST /ota → image OK in %s, restarting in 500 ms", slot->label);
    if (schedule_reboot() != ESP_OK)
        return send_json(req,
            "{\"status\":\"ok\",\"message\":\"updated, reboot to run it\"}", 200);
    return send_json(req, "{\"status\":\"ok\",\"message\":\"updated, rebooting\"}", 200);
}

/* ── POST /calibrate/el_sign ────────────────────────────────────────────── */

/*
 * {"sign": 1|-1} — which way the user elevation runs against the raw EL
 * reading ("+ = up"). Persisted; the encoder zero is kept, so the horizontal
 * stays 0 whichever sign is set.
 */
esp_err_t handler_el_sign(httpd_req_t *req)
{
    char body[48];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);
    float f;
    if (!json_float(body, "sign", &f) || (f != 1.0f && f != -1.0f))
        return send_json(req, "{\"status\":\"error\",\"message\":\"sign must be 1 or -1\"}", 400);
    motion_state_t st = motion_get_state();
    if (st == MOTION_MOVING || st == MOTION_SPINNING)
        return send_json(req, "{\"status\":\"busy\"}", 409);
    if (motion_set_el_sign((int)f) != ESP_OK)
        return send_json(req, "{\"status\":\"error\",\"message\":\"nvs write failed\"}", 500);
    char resp[64];
    snprintf(resp, sizeof(resp), "{\"status\":\"ok\",\"el_sign\":%d}", motion_get_el_sign());
    return send_json(req, resp, 200);
}

/* ── Motor calibration ──────────────────────────────────────────────────── */

static bool json_bool(const char *json, const char *key, bool *out)
{
    char pat[48];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return false;
    p += strlen(pat);
    while (*p == ' ' || *p == ':') p++;
    if (strncmp(p, "true", 4) == 0)  { *out = true;  return true; }
    if (strncmp(p, "false", 5) == 0) { *out = false; return true; }
    return false;
}

static bool axis_from(const char *s, stepper_axis_t *out)
{
    if (strcmp(s, "az") == 0) { *out = STEPPER_AZ; return true; }
    if (strcmp(s, "el") == 0) { *out = STEPPER_EL; return true; }
    return false;
}

/*
 * POST /motor/calibrate {"axis":"az"|"el", "arm_removed":true}
 *
 * Runs the four-phase motor calibration (motorcal.h) on the motion runner:
 * 202 + task_id, progress on GET /motor/calibration, the end as a `task`
 * frame on /ws/log like a move. Every phase spins the axis through full
 * turns, so elevation refuses to start unless the caller confirms the arm is
 * off — a full EL turn with the arm on drives it into the frame.
 */
esp_err_t handler_motor_calibrate(httpd_req_t *req)
{
    char body[96];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    char axis_str[4] = {0};
    stepper_axis_t axis;
    if (!json_str(body, "axis", axis_str, sizeof(axis_str)) || !axis_from(axis_str, &axis))
        return send_json(req, "{\"status\":\"error\",\"message\":\"axis must be az or el\"}", 400);

    bool arm_removed = false;
    json_bool(body, "arm_removed", &arm_removed);
    if (axis == STEPPER_EL && !arm_removed)
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"elevation calibration turns the axis full circle: "
            "remove the arm and send arm_removed:true\"}", 400);

    motion_state_t st = motion_get_state();
    if (st == MOTION_MOVING || st == MOTION_SPINNING || motorcal_active())
        return send_json(req, "{\"status\":\"busy\"}", 409);

    mr_cmd_t cmd = { .kind = MR_KIND_MOTOR_CAL, .u.motor_cal = { .axis = axis } };
    uint32_t task_id = motion_runner_submit(&cmd);
    if (task_id == 0)
        return send_json(req, "{\"status\":\"busy\"}", 409);

    LOGX_EMIT_I(TAG, "POST /motor/calibrate axis=%s → task %lu", axis_str, (unsigned long)task_id);
    char resp[80];
    snprintf(resp, sizeof(resp), "{\"status\":\"accepted\",\"task_id\":%lu}", (unsigned long)task_id);
    return send_json(req, resp, 202);
}

/* POST /motor/calibrate/cancel — stop a run in flight; the old result stays. */
esp_err_t handler_motor_calibrate_cancel(httpd_req_t *req)
{
    char body[16];
    read_body(req, body, sizeof(body));   /* body ignored */
    bool was = motorcal_active();
    motorcal_cancel();
    return send_json(req, was ? "{\"status\":\"ok\",\"message\":\"cancelling\"}"
                              : "{\"status\":\"ok\",\"message\":\"nothing running\"}", 200);
}

/* One axis's stored result as JSON, profile included (72 values). */
static int motor_axis_json(stepper_axis_t axis, char *out, size_t len)
{
    motorcal_result_t r;
    motorcal_result(axis, &r);
    int n = snprintf(out, len,
        "{\"valid\":%s,\"steps_per_deg\":%.4f,\"steps_per_deg_coarse\":%.4f,"
        "\"max_step_hz\":%.0f,\"max_speed_dps\":%.2f,\"dir_sign\":%d,"
        "\"revolutions\":%u,\"refine_steps\":%ld,\"compensation\":%s,"
        "\"has_profile\":%s,\"profile_peak_deg\":%.3f,\"profile_rms_deg\":%.3f,"
        "\"profile_deg\":[",
        r.valid ? "true" : "false", (double)r.steps_per_deg, (double)r.steps_per_deg_coarse,
        (double)r.max_step_hz, (double)r.max_speed_dps, r.dir_sign,
        (unsigned)r.revolutions, (long)r.refine_steps,
        motorcal_compensation_enabled(axis) ? "true" : "false",
        r.has_profile ? "true" : "false", (double)r.profile_peak_deg, (double)r.profile_rms_deg);
    for (int i = 0; i < MOTORCAL_PROFILE_BINS && n > 0 && (size_t)n < len; i++) {
        n += snprintf(out + n, len - (size_t)n, "%s%.3f", i ? "," : "",
                      r.has_profile ? (double)r.profile_deg[i] : 0.0);
    }
    if (n > 0 && (size_t)n < len) n += snprintf(out + n, len - (size_t)n, "]}");
    return n;
}

/*
 * GET /motor/calibration — the run in flight (or the last one) and both
 * axes' stored results: steps/deg, speed ceiling, handedness, the deviation
 * profile and whether it is applied.
 */
esp_err_t handler_motor_calibration(httpd_req_t *req)
{
    const size_t len = 3072;
    char *buf = malloc(len);
    if (!buf) return send_json(req, "{\"status\":\"error\",\"message\":\"no memory\"}", 500);

    stepper_axis_t axis;
    motorcal_phase_t phase;
    float progress, speed;
    char msg[96];
    motorcal_progress(&axis, &phase, &progress, &speed, msg, sizeof(msg));

    int n = snprintf(buf, len,
        "{\"run\":{\"active\":%s,\"axis\":\"%s\",\"phase\":\"%s\","
        "\"progress\":%.3f,\"speed_dps\":%.2f,\"message\":\"%s\"},\"az\":",
        motorcal_active() ? "true" : "false", axis == STEPPER_AZ ? "az" : "el",
        motorcal_phase_name(phase), (double)progress, (double)speed, msg);
    n += motor_axis_json(STEPPER_AZ, buf + n, len - (size_t)n);
    n += snprintf(buf + n, len - (size_t)n, ",\"el\":");
    n += motor_axis_json(STEPPER_EL, buf + n, len - (size_t)n);
    snprintf(buf + n, len - (size_t)n, "}");

    esp_err_t ret = send_json(req, buf, 200);
    free(buf);
    return ret;
}

/* POST /motor/compensation {"axis":"az"|"el","enabled":true|false} */
esp_err_t handler_motor_compensation(httpd_req_t *req)
{
    char body[64];
    if (read_body(req, body, sizeof(body)) < 0)
        return send_json(req, "{\"status\":\"error\",\"message\":\"bad body\"}", 400);

    char axis_str[4] = {0};
    stepper_axis_t axis;
    bool en;
    if (!json_str(body, "axis", axis_str, sizeof(axis_str)) || !axis_from(axis_str, &axis) ||
        !json_bool(body, "enabled", &en))
        return send_json(req,
            "{\"status\":\"error\",\"message\":\"axis (az|el) and enabled (bool) required\"}", 400);

    motorcal_set_compensation(axis, en);
    char resp[80];
    snprintf(resp, sizeof(resp), "{\"status\":\"ok\",\"axis\":\"%s\",\"enabled\":%s}",
             axis_str, en ? "true" : "false");
    return send_json(req, resp, 200);
}

/* ── GET /test/encoder ──────────────────────────────────────────────────── */

/*
 * One axis as JSON: whether the sensor answers, its raw 12-bit angle, the
 * magnet field (detected / too weak / too strong, AGC, magnitude) and the bus
 * line levels found at boot. Everything a wiring or magnet problem needs,
 * without a serial cable.
 */
static void encoder_axis_json(enc_axis_t axis, const char *name, char *out, size_t len)
{
    int sda, scl, pu;
    encoder_line_levels(axis, &sda, &scl, &pu);

    char angle[64];
    uint16_t raw;
    bool ok = encoder_read_raw(axis, &raw) == ESP_OK;
    if (ok) {
        snprintf(angle, sizeof(angle), "\"raw\":%u,\"angle_deg\":%.3f,",
                 raw, (double)raw * (360.0 / 4096.0));
    } else {
        angle[0] = '\0';
    }

    char magnet[160];
    enc_magnet_t m;
    if (encoder_read_magnet(axis, &m) == ESP_OK) {
        snprintf(magnet, sizeof(magnet),
                 "\"magnet_missing\":%d,\"magnet_weak\":%d,\"magnet_strong\":%d,"
                 "\"agc\":%u,\"magnitude\":%u,",
                 !m.detected, m.too_weak, m.too_strong, m.agc, m.magnitude);
    } else {
        magnet[0] = '\0';
    }

    snprintf(out, len,
             "\"%s\":{\"ok\":%s,%s%s\"lines\":{\"sda\":%d,\"scl\":%d,\"external_pullups\":%d}}",
             name, ok ? "true" : "false", angle, magnet, sda, scl, pu);
    if (!ok) ESP_LOGW(TAG, "encoder %s: read failed", name);
}

esp_err_t handler_test_encoder(httpd_req_t *req)
{
    char az[300], el[300], buf[640];
    encoder_axis_json(ENC_AZ, "azimuth", az, sizeof(az));
    encoder_axis_json(ENC_EL, "elevation", el, sizeof(el));
    snprintf(buf, sizeof(buf), "{%s,%s}", az, el);
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
