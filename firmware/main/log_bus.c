#include "log_bus.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/ringbuf.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"   /* setsockopt + TCP_NODELAY */
#include "lwip/tcp.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * log_bus — async fan-out of structured WS frames over /ws/log.
 *
 * Phase-3 architecture:
 *
 *   producers (pose_tick @10Hz, LOGX_EMIT_*, motion_runner)
 *       │ format JSON → stack buffer (NO malloc in hot path)
 *       ▼
 *   ┌─────────────┐         ┌────────────┐
 *   │ s_pose_q    │         │ s_log_rb   │   (FreeRTOS queue / ringbuf)
 *   │ depth 8,    │         │ 4 KB ring  │
 *   │ drop-OLDEST │         │ drop-newest│
 *   └─────────────┘         └────────────┘
 *           │   pump_task @prio 3   │
 *           └───────────┬───────────┘
 *                       ▼
 *   foreach connected client:
 *       malloc ws_send_ctx_t (one alloc per dispatch)
 *       httpd_queue_work(ws_send_worker, ctx)
 *
 *   httpd worker task (single-threaded):
 *       httpd_ws_send_frame_async(fd, frame)   ← SAFE here (httpd context)
 *       record success/fail per client
 *
 * Lifecycle:
 *   • clients_record_send increments consecutive_errs on failure, resets
 *     on success. Pump purges clients with err≥MAX_ERRS or >MAX_SILENCE_MS.
 *   • Pump pings every PING_INTERVAL_MS so a stalled UI side gets detected
 *     within ~10 sec instead of TCP keepalive's minutes.
 *   • Stats line every STATS_INTERVAL_MS to UART for observability.
 *
 * Why ringbuf for log but queue for pose:
 *   • pose frames are bounded (≤160 B) and time-critical — fixed-size
 *     queue + drop-oldest gives O(1) drop semantics
 *   • log frames are variable-length up to ~540 B; ringbuf packs efficiently
 *     and drop-newest is fine (better to keep older context than spammy
 *     newest entry on overflow)
 */

#define MAX_CLIENTS         4

#define POSE_ITEM_LEN       176       /* JSON frame for kind:"pose" */
#define POSE_QUEUE_DEPTH    16        /* 16 × 40ms = 640ms buffer @25Hz pose */

#define LOG_RB_BYTES        4096      /* covers ~7 max-length log frames */
#define LOG_FRAME_MAX       600       /* hard cap on any log/task frame */
#define LOG_TEXT_MAX        220       /* matches old MSG_BUF_LEN */

#define MAX_CONSEC_ERRS     5
#define MAX_SILENCE_MS      60000U    /* purge if no successful send in 60s */
#define PING_INTERVAL_MS    5000U
#define STATS_INTERVAL_MS   30000U
/*
 * Pump idle-sleep. MUST be >= one FreeRTOS tick (10 ms at CONFIG_FREERTOS_HZ
 * = 100): pdMS_TO_TICKS() of a sub-tick value rounds to 0, and vTaskDelay(0)
 * does NOT yield — the pump would spin, starve the idle task and trip the
 * task watchdog. 10 ms keeps dispatch latency at one pose period.
 */
#define PUMP_IDLE_TICK_MS   10
_Static_assert(pdMS_TO_TICKS(PUMP_IDLE_TICK_MS) > 0,
               "PUMP_IDLE_TICK_MS must be >= one FreeRTOS tick or the pump "
               "spins (vTaskDelay(0) never yields)");

static const char *TAG = "log_bus";

/* ── Client table ──────────────────────────────────────────────────────── */

typedef struct {
    int      fd;
    uint16_t consecutive_errs;
    uint64_t last_send_ms;
    uint64_t last_ping_ms;
} ws_client_t;

static SemaphoreHandle_t s_mtx;
static httpd_handle_t    s_server;
static ws_client_t       s_clients[MAX_CLIENTS];
static int               s_nfds = 0;

/* ── Send queues ───────────────────────────────────────────────────────── */

static QueueHandle_t     s_pose_q;
static RingbufHandle_t   s_log_rb;

/* ── Stats ─────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t sent;
    uint32_t dropped_pose;
    uint32_t dropped_log;
    uint32_t send_errs;
    uint32_t purged;
    uint32_t pings_sent;
} stats_t;
static stats_t s_stats;
static atomic_uint s_seq;

/* ── Helpers ───────────────────────────────────────────────────────────── */

static inline uint64_t now_ms(void)
{
    return esp_timer_get_time() / 1000ULL;
}

static const char *level_str(log_bus_level_t lvl)
{
    switch (lvl) {
        case LOG_BUS_WARN: return "W";
        case LOG_BUS_ERR:  return "E";
        default:           return "I";
    }
}

/* ── Client management ─────────────────────────────────────────────────── */

static void clients_add(int fd)
{
    xSemaphoreTake(s_mtx, portMAX_DELAY);
    for (int i = 0; i < s_nfds; i++) {
        if (s_clients[i].fd == fd) {
            xSemaphoreGive(s_mtx);
            return;
        }
    }
    /* Stale-FD purge. When MAX_CLIENTS is full, the browser reloading the
       tab can leave the previous fd in the table — its CLOSE frame may
       never arrive (page navigated away, lwIP didn't observe a FIN) and
       we silently drop the new client. Before refusing, probe each entry
       with a non-blocking `getsockopt(SO_ERROR)`: an invalid / closed
       socket returns EBADF or has a pending error, free that slot.
       Without this, the device hit the 4-client cap after ~4 page-loads
       and stopped sending pose to fresh UI sessions. */
    if (s_nfds >= MAX_CLIENTS) {
        int err = 0;
        socklen_t errlen = sizeof(err);
        for (int i = s_nfds - 1; i >= 0; i--) {
            int probe_fd = s_clients[i].fd;
            int rc = getsockopt(probe_fd, SOL_SOCKET, SO_ERROR, &err, &errlen);
            if (rc != 0 || err != 0) {
                ESP_LOGI(TAG, "purging stale ws fd=%d (rc=%d err=%d)",
                         probe_fd, rc, err);
                s_clients[i] = s_clients[s_nfds - 1];
                s_nfds--;
            }
        }
    }
    if (s_nfds < MAX_CLIENTS) {
        s_clients[s_nfds].fd               = fd;
        s_clients[s_nfds].consecutive_errs = 0;
        s_clients[s_nfds].last_send_ms     = now_ms();
        s_clients[s_nfds].last_ping_ms     = now_ms();
        s_nfds++;
    } else {
        ESP_LOGW(TAG, "ws client table full (%d), dropping fd=%d",
                 MAX_CLIENTS, fd);
    }
    xSemaphoreGive(s_mtx);

    /* Disable Nagle: pose frames are tiny (~130 B) at 25 Hz; with Nagle the
     * UI dials lag ~200 ms per frame during motion. */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    /*
     * Bound send-side blocking. Without this, a stalled WS client (background
     * tab, NAT/AP hiccup) lets lwIP's send buffer fill, then lwip_send blocks
     * the single httpd worker thread for up to TCP_MSL (60 s here) — every
     * other client misses frames in that window and the pump's dispatch
     * backlog accretes malloc'd payloads in the heap. 1 s is long enough that
     * a momentarily-congested AP doesn't false-positive, short enough that a
     * truly dead client gets purged within a couple of MAX_CONSEC_ERRS cycles.
     */
    struct timeval snd_tmo = { .tv_sec = 1, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &snd_tmo, sizeof(snd_tmo));
}

static void clients_remove_locked(int fd)
{
    for (int i = 0; i < s_nfds; i++) {
        if (s_clients[i].fd == fd) {
            s_clients[i] = s_clients[s_nfds - 1];
            s_nfds--;
            return;
        }
    }
}

static void clients_remove(int fd)
{
    xSemaphoreTake(s_mtx, portMAX_DELAY);
    clients_remove_locked(fd);
    xSemaphoreGive(s_mtx);
}

/*
 * Called from ws_send_worker (httpd context). Light-touch: just bump
 * counters on the matching client. Purging is done in the pump.
 */
static void clients_record_send(int fd, bool ok)
{
    xSemaphoreTake(s_mtx, portMAX_DELAY);
    for (int i = 0; i < s_nfds; i++) {
        if (s_clients[i].fd != fd) continue;
        if (ok) {
            s_clients[i].consecutive_errs = 0;
            s_clients[i].last_send_ms     = now_ms();
            s_stats.sent++;
        } else {
            s_clients[i].consecutive_errs++;
            s_stats.send_errs++;
        }
        break;
    }
    xSemaphoreGive(s_mtx);
}

static void clients_purge_dead(void)
{
    xSemaphoreTake(s_mtx, portMAX_DELAY);
    uint64_t t = now_ms();
    int j = 0;
    for (int i = 0; i < s_nfds; i++) {
        bool alive = (s_clients[i].consecutive_errs < MAX_CONSEC_ERRS) &&
                     (t - s_clients[i].last_send_ms < MAX_SILENCE_MS);
        if (alive) {
            if (i != j) s_clients[j] = s_clients[i];
            j++;
        } else {
            ESP_LOGW(TAG, "purge client fd=%d errs=%u silence=%llums",
                     s_clients[i].fd,
                     (unsigned)s_clients[i].consecutive_errs,
                     (unsigned long long)(t - s_clients[i].last_send_ms));
            s_stats.purged++;
        }
    }
    s_nfds = j;
    xSemaphoreGive(s_mtx);
}

static int clients_snapshot(int *fds_out, int max)
{
    xSemaphoreTake(s_mtx, portMAX_DELAY);
    int n = (s_nfds < max) ? s_nfds : max;
    for (int i = 0; i < n; i++) fds_out[i] = s_clients[i].fd;
    xSemaphoreGive(s_mtx);
    return n;
}

/* ── WS send worker (runs on httpd task) ───────────────────────────────── */

typedef struct {
    int                  fd;
    size_t               len;
    httpd_ws_type_t      type;
    char                *payload;   /* NULL allowed for PING */
} ws_send_ctx_t;

static void ws_send_worker(void *arg)
{
    ws_send_ctx_t *c = (ws_send_ctx_t *)arg;
    httpd_ws_frame_t frame = {
        .final   = true,
        .type    = c->type,
        .payload = c->payload ? (uint8_t *)c->payload : NULL,
        .len     = c->len,
    };
    esp_err_t err = httpd_ws_send_frame_async(s_server, c->fd, &frame);
    clients_record_send(c->fd, err == ESP_OK);
    if (c->payload) free(c->payload);
    free(c);
}

/*
 * Allocate ctx + payload copy and queue to httpd worker. On enqueue failure
 * we record it as a send_err on this client so the lifecycle code purges
 * persistently-failing clients (e.g. httpd queue stuck full).
 */
static void dispatch_one(int fd, httpd_ws_type_t type,
                         const char *payload, size_t len)
{
    ws_send_ctx_t *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        clients_record_send(fd, false);
        return;
    }
    ctx->fd   = fd;
    ctx->type = type;
    ctx->len  = len;
    if (payload && len > 0) {
        ctx->payload = malloc(len + 1);
        if (!ctx->payload) {
            free(ctx);
            clients_record_send(fd, false);
            return;
        }
        memcpy(ctx->payload, payload, len);
        ctx->payload[len] = '\0';
    } else {
        ctx->payload = NULL;
    }

    if (httpd_queue_work(s_server, ws_send_worker, ctx) != ESP_OK) {
        ESP_LOGW(TAG, "queue_work full fd=%d heap=%u",
                 fd, (unsigned)esp_get_free_heap_size());
        if (ctx->payload) free(ctx->payload);
        free(ctx);
        clients_record_send(fd, false);
    }
}

static void broadcast_text(const char *payload, size_t len)
{
    int fds[MAX_CLIENTS];
    int n = clients_snapshot(fds, MAX_CLIENTS);
    for (int i = 0; i < n; i++) {
        dispatch_one(fds[i], HTTPD_WS_TYPE_TEXT, payload, len);
    }
}

static void maybe_ping_clients(void)
{
    uint64_t t = now_ms();
    int fds[MAX_CLIENTS];
    int n;
    xSemaphoreTake(s_mtx, portMAX_DELAY);
    n = s_nfds;
    int to_ping_count = 0;
    int to_ping[MAX_CLIENTS];
    for (int i = 0; i < n; i++) {
        if (t - s_clients[i].last_ping_ms >= PING_INTERVAL_MS) {
            to_ping[to_ping_count++] = s_clients[i].fd;
            s_clients[i].last_ping_ms = t;
        }
    }
    memcpy(fds, to_ping, sizeof(int) * to_ping_count);
    xSemaphoreGive(s_mtx);

    for (int i = 0; i < to_ping_count; i++) {
        dispatch_one(fds[i], HTTPD_WS_TYPE_PING, NULL, 0);
        s_stats.pings_sent++;
    }
}

/* ── Pump task ─────────────────────────────────────────────────────────── */

static void log_bus_pump_task(void *arg)
{
    (void)arg;
    static char pose_item[POSE_ITEM_LEN];

    uint64_t last_stats_ms = now_ms();
    uint64_t last_purge_ms = now_ms();

    for (;;) {
        bool did_work = false;

        /* Pose has priority — it's time-critical for UI freshness. */
        if (xQueueReceive(s_pose_q, pose_item, 0) == pdTRUE) {
            broadcast_text(pose_item, strlen(pose_item));
            did_work = true;
        }

        /* Log ringbuf — variable-length items. */
        size_t item_len = 0;
        void *log_item = xRingbufferReceive(s_log_rb, &item_len, 0);
        if (log_item) {
            broadcast_text((const char *)log_item, item_len);
            vRingbufferReturnItem(s_log_rb, log_item);
            did_work = true;
        }

        uint64_t t = now_ms();

        /* Housekeeping runs every loop regardless of did_work — otherwise a
         * pump that's constantly draining a busy pose queue would never
         * purge stale clients or emit stats. */
        if (t - last_purge_ms >= 1000) {
            clients_purge_dead();
            maybe_ping_clients();
            last_purge_ms = t;
        }
        if (t - last_stats_ms >= STATS_INTERVAL_MS) {
            xSemaphoreTake(s_mtx, portMAX_DELAY);
            int nclients = s_nfds;
            xSemaphoreGive(s_mtx);
            ESP_LOGI(TAG,
                "stats: clients=%d sent=%lu dropped_pose=%lu "
                "dropped_log=%lu send_errs=%lu purged=%lu pings=%lu heap=%u",
                nclients,
                (unsigned long)s_stats.sent,
                (unsigned long)s_stats.dropped_pose,
                (unsigned long)s_stats.dropped_log,
                (unsigned long)s_stats.send_errs,
                (unsigned long)s_stats.purged,
                (unsigned long)s_stats.pings_sent,
                (unsigned)esp_get_free_heap_size());
            last_stats_ms = t;
        }

        if (!did_work) {
            /* Idle sleep — yields the core (pose/log arrival is paced by the
             * producers). One tick (10 ms) is the floor; see PUMP_IDLE_TICK_MS.
             * A burst of queued frames still drains immediately via the
             * did_work fast path above, so 100 Hz telemetry isn't batched. */
            vTaskDelay(pdMS_TO_TICKS(PUMP_IDLE_TICK_MS));
        }
    }
}

/* ── Public API: lifecycle ─────────────────────────────────────────────── */

void log_bus_init(void)
{
    if (s_mtx) return;   /* idempotent */
    s_mtx     = xSemaphoreCreateMutex();
    s_pose_q  = xQueueCreate(POSE_QUEUE_DEPTH, POSE_ITEM_LEN);
    s_log_rb  = xRingbufferCreate(LOG_RB_BYTES, RINGBUF_TYPE_NOSPLIT);
    atomic_store(&s_seq, 0u);
    memset(&s_stats, 0, sizeof(s_stats));
    /* Pump priority 3 — below httpd (5), pose_tick (4), motion_runner (5).
     * Below pose_tick is intentional: if encoder bus is contested, pose-tick
     * produces first; pump drains right after. Stack 4 KB covers snprintf +
     * httpd_queue_work + 2 mallocs. */
    xTaskCreate(log_bus_pump_task, "log_pump", 4096, NULL, 3, NULL);
}

/* ── Public API: producers ─────────────────────────────────────────────── */

void log_bus_emit(log_bus_level_t lvl, const char *tag, const char *fmt, ...)
{
    /* Log messages now go to UART via the standard ESP_LOG facility —
       simpler than the WS multiplexor and frees up TCP socket budget on
       the device. The WS channel keeps streaming `pose` and `task`
       frames (UI needs them live for state sync); log lines come via
       the USB serial monitor instead. The ring buffer / pump still
       exists for pose/task but we no longer push log payloads into it.
       Skipping the early `s_server` guard is fine — ESP_LOG works even
       when the HTTP server hasn't started yet (it's whatever
       esp_log_set_default_level() decided). */
    char msg[LOG_TEXT_MAX];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (n < 0) return;

    const char *t = tag ? tag : TAG;
    switch (lvl) {
    case LOG_BUS_ERR:  ESP_LOGE(t, "%s", msg); break;
    case LOG_BUS_WARN: ESP_LOGW(t, "%s", msg); break;
    default:           ESP_LOGI(t, "%s", msg); break;
    }
}

void log_bus_emit_pose(float az_deg, float el_deg, const char *state_str,
                       bool motors_on, bool spin_az, bool spin_el)
{
    if (!s_mtx || !s_server) return;
    if (s_nfds == 0) return;      /* fast path — nobody listening */

    char item[POSE_ITEM_LEN];
    int plen = snprintf(item, sizeof(item),
        "{\"kind\":\"pose\",\"ts_ms\":%llu,\"az\":%.3f,\"el\":%.3f,"
        "\"st\":\"%s\",\"motors\":%s,\"sp_az\":%s,\"sp_el\":%s}",
        (unsigned long long)now_ms(),
        (double)az_deg, (double)el_deg,
        state_str ? state_str : "unknown",
        motors_on ? "true" : "false",
        spin_az   ? "true" : "false",
        spin_el   ? "true" : "false");
    if (plen <= 0) return;
    if (plen >= (int)sizeof(item)) plen = (int)sizeof(item) - 1;

    /* Drop-OLDEST on overflow: freshness matters more than completeness
     * for pose. Single producer (pose_tick) so this dequeue+enqueue dance
     * doesn't race with itself; the retry is guaranteed to succeed. */
    if (xQueueSend(s_pose_q, item, 0) != pdTRUE) {
        s_stats.dropped_pose++;
        char dummy[POSE_ITEM_LEN];
        xQueueReceive(s_pose_q, dummy, 0);
        (void)xQueueSend(s_pose_q, item, 0);
    }
}

void log_bus_emit_task(uint32_t task_id, const char *status,
                       const char *result_json_inner)
{
    if (!s_mtx || !s_server) return;
    if (s_nfds == 0) return;

    char payload[LOG_FRAME_MAX];
    int plen;
    if (result_json_inner && result_json_inner[0]) {
        plen = snprintf(payload, sizeof(payload),
            "{\"kind\":\"task\",\"task_id\":%lu,\"ts_ms\":%llu,"
            "\"status\":\"%s\",\"result\":{%s}}",
            (unsigned long)task_id, (unsigned long long)now_ms(),
            status ? status : "unknown",
            result_json_inner);
    } else {
        plen = snprintf(payload, sizeof(payload),
            "{\"kind\":\"task\",\"task_id\":%lu,\"ts_ms\":%llu,"
            "\"status\":\"%s\"}",
            (unsigned long)task_id, (unsigned long long)now_ms(),
            status ? status : "unknown");
    }
    if (plen <= 0) return;
    if (plen >= (int)sizeof(payload)) plen = (int)sizeof(payload) - 1;

    /* Task frames share the log ringbuf — they're rare (one per motion
     * command) and have similar latency requirements as log lines. */
    if (xRingbufferSend(s_log_rb, payload, (size_t)plen, 0) != pdTRUE) {
        s_stats.dropped_log++;
    }
}

/* ── WS endpoint handler ───────────────────────────────────────────────── */

/*
 * Register a freshly-connected /ws/log client.
 *
 * ESP-IDF v6 no longer dispatches the WebSocket handshake GET to the URI
 * handler — httpd_uri.c explicitly skips it ("do not call the uri->handler").
 * The client is therefore attached from the post-handshake callback below;
 * the ws_handler HTTP_GET path is kept only as a fallback for older IDFs.
 * clients_add() dedups by fd, so both paths firing for one socket is safe.
 */
static void ws_client_attach(httpd_req_t *req)
{
    int fd = httpd_req_to_sockfd(req);
    if (fd < 0) {
        ESP_LOGW(TAG, "ws attach: invalid sockfd");
        return;
    }
    clients_add(fd);
    ESP_LOGI(TAG, "client connected fd=%d (clients=%d)", fd, s_nfds);
}

#if CONFIG_HTTPD_WS_POST_HANDSHAKE_CB_SUPPORT
/* Invoked by httpd immediately after the WS handshake response is sent. */
static esp_err_t ws_post_handshake(httpd_req_t *req)
{
    ws_client_attach(req);
    return ESP_OK;
}
#endif

static esp_err_t ws_handler(httpd_req_t *req)
{
    if (req->method == HTTP_GET) {
        /* Fallback only — reached on pre-v6 IDFs that still dispatch the
         * handshake GET here. On v6+ ws_post_handshake() does the attach. */
        ws_client_attach(req);
        return ESP_OK;
    }

    httpd_ws_frame_t frame = { 0 };
    frame.type = HTTPD_WS_TYPE_TEXT;
    esp_err_t ret = httpd_ws_recv_frame(req, &frame, 0);
    if (ret != ESP_OK) return ret;

    if (frame.type == HTTPD_WS_TYPE_CLOSE ||
        frame.type == HTTPD_WS_TYPE_PONG) {
        if (frame.type == HTTPD_WS_TYPE_CLOSE) {
            int fd = httpd_req_to_sockfd(req);
            clients_remove(fd);
            ESP_LOGI(TAG, "client closed fd=%d", fd);
        } else {
            /* Pong → mark this client as fresh so it's not purged for
             * silence even if pose/log is briefly idle. */
            int fd = httpd_req_to_sockfd(req);
            xSemaphoreTake(s_mtx, portMAX_DELAY);
            for (int i = 0; i < s_nfds; i++) {
                if (s_clients[i].fd == fd) {
                    s_clients[i].last_send_ms = now_ms();
                    break;
                }
            }
            xSemaphoreGive(s_mtx);
        }
    }
    return ESP_OK;
}

esp_err_t log_bus_register_ws(httpd_handle_t server, const char *uri)
{
    if (!server) return ESP_ERR_INVALID_ARG;
    log_bus_init();
    s_server = server;

    httpd_uri_t route = {
        .uri          = uri,
        .method       = HTTP_GET,
        .handler      = ws_handler,
        .user_ctx     = NULL,
        .is_websocket = true,
#if CONFIG_HTTPD_WS_POST_HANDSHAKE_CB_SUPPORT
        /* ESP-IDF v6: the handshake GET is not dispatched to .handler, so the
         * client is registered here instead. Requires
         * CONFIG_HTTPD_WS_POST_HANDSHAKE_CB_SUPPORT (set in sdkconfig). */
        .ws_post_handshake_cb = ws_post_handshake,
#endif
    };
    return httpd_register_uri_handler(server, &route);
}
