#pragma once

#include "esp_http_server.h"

esp_err_t handler_health(httpd_req_t *req);
esp_err_t handler_state(httpd_req_t *req);
esp_err_t handler_move(httpd_req_t *req);
esp_err_t handler_motors(httpd_req_t *req);     /* POST /motors      */
esp_err_t handler_zero(httpd_req_t *req);       /* POST /zero        */
esp_err_t handler_calibrate(httpd_req_t *req);  /* POST /calibrate   */
esp_err_t handler_spin(httpd_req_t *req);       /* POST /spin        */
esp_err_t handler_spin_stop(httpd_req_t *req);  /* POST /spin/stop   */
esp_err_t handler_reboot(httpd_req_t *req);     /* POST /reboot      */
esp_err_t handler_ota(httpd_req_t *req);        /* POST /ota — firmware image body */

/* ── Motor calibration (motorcal.h) ── */
esp_err_t handler_motor_calibrate(httpd_req_t *req);        /* POST /motor/calibrate        */
esp_err_t handler_motor_calibrate_cancel(httpd_req_t *req); /* POST /motor/calibrate/cancel */
esp_err_t handler_motor_calibration(httpd_req_t *req);      /* GET  /motor/calibration      */
esp_err_t handler_motor_compensation(httpd_req_t *req);     /* POST /motor/compensation     */
esp_err_t handler_options(httpd_req_t *req);    /* OPTIONS * — CORS preflight */

/* ── Test endpoints ── */
esp_err_t handler_test_encoder(httpd_req_t *req);  /* GET  /test/encoder */
esp_err_t handler_test_jog(httpd_req_t *req);      /* POST /test/jog     */
