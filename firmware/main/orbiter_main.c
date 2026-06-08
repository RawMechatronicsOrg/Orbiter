/*
 * Orbiter — main entry point
 *
 * Boot sequence:
 *   1. NVS init
 *   2. WiFi STA connect
 *   3. Motion controller init (steppers + encoders)
 *   4. HTTP server start
 */

#include "motion.h"
#include "motion_runner.h"
#include "http_handlers.h"
#include "log_bus.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "mdns.h"

static const char *TAG = "orbiter";

/* ── WiFi ───────────────────────────────────────────────────────────────── */

/* Set credentials in `idf.py menuconfig` -> Orbiter Configuration. */
#define WIFI_SSID      CONFIG_ORBITER_WIFI_SSID
#define WIFI_PASS      CONFIG_ORBITER_WIFI_PASSWORD
#define WIFI_MAX_RETRY CONFIG_ORBITER_WIFI_MAX_RETRY
/* After WIFI_MAX_RETRY back-to-back failures we pause this long and try again.
 * Never giving up was the original bug — a transient AP blip locked the device
 * out until power-cycle. 5 s is short enough to recover quickly, long enough
 * not to spam the AP. */
#define WIFI_RETRY_BACKOFF_US (5 * 1000 * 1000ULL)

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static int                s_retry_num = 0;
static esp_timer_handle_t s_wifi_retry_timer;

static void wifi_retry_cb(void *arg)
{
    (void)arg;
    ESP_LOGW(TAG, "WiFi backoff elapsed — retrying connect");
    s_retry_num = 0;
    esp_wifi_connect();
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        /* Association succeeded — reset retry budget even before DHCP returns
         * an IP, so a renew-only flap can't burn through it again. */
        s_retry_num = 0;
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < WIFI_MAX_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGW(TAG, "WiFi retry %d/%d", s_retry_num, WIFI_MAX_RETRY);
        } else {
            /* Don't give up — back off and try forever. */
            ESP_LOGE(TAG, "WiFi: %d retries exhausted, backing off %llus",
                     WIFI_MAX_RETRY, WIFI_RETRY_BACKOFF_US / 1000000ULL);
            if (s_wifi_retry_timer) {
                esp_timer_stop(s_wifi_retry_timer);
                esp_timer_start_once(s_wifi_retry_timer, WIFI_RETRY_BACKOFF_US);
            }
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "IP: " IPSTR, IP2STR(&e->ip_info.ip));
        s_retry_num = 0;
        xEventGroupClearBits(s_wifi_event_group, WIFI_FAIL_BIT);
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    const esp_timer_create_args_t retry_args = {
        .callback = wifi_retry_cb,
        .name     = "wifi_retry",
    };
    ESP_ERROR_CHECK(esp_timer_create(&retry_args, &s_wifi_retry_timer));

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t h1, h2;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &h1));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &h2));

    wifi_config_t wifi_cfg = {
        .sta = {
            .ssid     = WIFI_SSID,
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* Disable WiFi power-save. Default is WIFI_PS_MIN_MODEM which lets the
       radio sleep ~tens of ms between RX windows — fine for IoT polling
       but disastrous for our RT control loop (4 polls/s /state + WS @ 10 Hz
       pose + sweep loop /move). Symptom we saw on the host side
       was rising HTTP latency, `httpx.ReadError("")` mid-sweep, and the
       device eventually going silent without sending a TCP FIN. WIFI_PS_NONE
       keeps the radio hot — costs ~80 mA more but the device is USB-fed,
       so power is not the constraint here, reliability is. */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_LOGI(TAG, "Connecting to %s ...", WIFI_SSID);
    xEventGroupWaitBits(s_wifi_event_group,
                        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                        pdFALSE, pdFALSE, portMAX_DELAY);
}

/* ── mDNS ────────────────────────────────────────────────────────────────
 *
 * Advertise the device so the server can find it without a hardcoded IP.
 * Hostname becomes `orbiter.local` (resolvable via mDNS on macOS / Linux
 * with avahi / Windows with Bonjour, and via Python's `zeroconf` without
 * any host-side service).
 *
 * Two service records are added on the same HTTP port:
 *   _orbiter._tcp  — our own service type; the server browses for this.
 *   _http._tcp     — standard HTTP advertisement so generic LAN scanners
 *                    (dns-sd, avahi-browse, mobile network tools) show
 *                    the device with a friendly name.
 *
 * TXT records carry minimal identity (chip + firmware tag) so a multi-rig
 * setup can disambiguate by board if it ever needs to.
 */
static void start_mdns(void)
{
    esp_err_t err = mdns_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mDNS init failed: %s", esp_err_to_name(err));
        return;
    }
    ESP_ERROR_CHECK(mdns_hostname_set("orbiter"));
    ESP_ERROR_CHECK(mdns_instance_name_set("Orbiter Turntable"));

    mdns_txt_item_t txt[] = {
        {"version", "0.2"},
        {"chip",    "esp32"},
    };
    const size_t txt_n = sizeof(txt) / sizeof(txt[0]);
    ESP_ERROR_CHECK(mdns_service_add(NULL, "_orbiter", "_tcp",
                                     CONFIG_ORBITER_HTTP_PORT, txt, txt_n));
    ESP_ERROR_CHECK(mdns_service_add(NULL, "_http", "_tcp",
                                     CONFIG_ORBITER_HTTP_PORT, NULL, 0));

    LOGX_EMIT_I(TAG, "mDNS up — orbiter.local on port %d (+ _orbiter._tcp)",
                CONFIG_ORBITER_HTTP_PORT);
}

/* ── HTTP server ────────────────────────────────────────────────────────── */

static void start_http_server(void)
{
    httpd_config_t config  = HTTPD_DEFAULT_CONFIG();
    config.server_port     = CONFIG_ORBITER_HTTP_PORT;
    config.lru_purge_enable = true;
    config.max_uri_handlers = 26;   /* default is 8, we now have 21 routes + /ws/log */

    /*
     * UI polls /state every 250 ms and browsers (Firefox/Chrome) tend to
     * open 4–6 parallel keep-alive sockets to the same origin. The IDF
     * default of 7 leaves no headroom — once it's exhausted httpd starts
     * LRU-evicting live sockets and the recv loop logs ECONNRESET (104)
     * spam. 12 fits a typical browser pool with a buffer for /move,
     * /test/jog, manual /test/encoder pokes, etc.
     */
    config.max_open_sockets = 12;

    /*
     * Suppress the "httpd_sock_err: error in recv : 104" WARN flood. ECONNRESET
     * during keep-alive recycling is normal and expected at our poll rate; it
     * doesn't indicate a real problem. Real errors stay visible at ERROR.
     */
    esp_log_level_set("httpd_txrx", ESP_LOG_ERROR);

    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP server");
        return;
    }

    httpd_uri_t routes[] = {
        { .uri = "/health",       .method = HTTP_GET,     .handler = handler_health       },
        { .uri = "/state",        .method = HTTP_GET,     .handler = handler_state        },
        { .uri = "/move",         .method = HTTP_POST,    .handler = handler_move         },
        { .uri = "/move",         .method = HTTP_OPTIONS, .handler = handler_options      },
        { .uri = "/motors",       .method = HTTP_POST,    .handler = handler_motors       },
        { .uri = "/motors",       .method = HTTP_OPTIONS, .handler = handler_options      },
        { .uri = "/zero",         .method = HTTP_POST,    .handler = handler_zero         },
        { .uri = "/zero",         .method = HTTP_OPTIONS, .handler = handler_options      },
        { .uri = "/calibrate",    .method = HTTP_POST,    .handler = handler_calibrate    },
        { .uri = "/calibrate",    .method = HTTP_OPTIONS, .handler = handler_options      },
        { .uri = "/spin",         .method = HTTP_POST,    .handler = handler_spin         },
        { .uri = "/spin",         .method = HTTP_OPTIONS, .handler = handler_options      },
        { .uri = "/spin/stop",    .method = HTTP_POST,    .handler = handler_spin_stop    },
        { .uri = "/spin/stop",    .method = HTTP_OPTIONS, .handler = handler_options      },
        { .uri = "/reboot",       .method = HTTP_POST,    .handler = handler_reboot       },
        { .uri = "/reboot",       .method = HTTP_OPTIONS, .handler = handler_options      },
        { .uri = "/test/encoder", .method = HTTP_GET,     .handler = handler_test_encoder },
        { .uri = "/test/jog",     .method = HTTP_POST,    .handler = handler_test_jog     },
        { .uri = "/test/jog",     .method = HTTP_OPTIONS, .handler = handler_options      },
    };
    for (int i = 0; i < (int)(sizeof(routes)/sizeof(routes[0])); i++) {
        httpd_register_uri_handler(server, &routes[i]);
    }

    if (log_bus_register_ws(server, "/ws/log") != ESP_OK) {
        ESP_LOGE(TAG, "Failed to register /ws/log");
    }

    LOGX_EMIT_I(TAG, "HTTP server ready on port %d (+ /ws/log)", config.server_port);
}

/* ── Entry point ────────────────────────────────────────────────────────── */

void app_main(void)
{
    log_bus_init();
    ESP_LOGI(TAG, "=== Orbiter v0.2 starting ===");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_sta();
    start_mdns();
    motion_init();
    motion_runner_init();
    start_http_server();
    LOGX_EMIT_I(TAG, "Orbiter ready");
}
