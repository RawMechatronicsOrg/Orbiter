import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Dev-time reverse proxy to the storage-api so the browser can hit
// same-origin URLs (no CORS dance). Routes mirror what the v0.1 UI
// actually calls: /ws/scene (WebSocket), /scans, /captures, /config, /stereo,
// /firmware (ESP32 diagnostics + updates), and the live MJPEG camera stream
// under /camera (long-lived multipart response — http-proxy pipes it without
// buffering). ORBITER_API points a second dev UI (e.g. on :5174) at another
// server.
const API = process.env.ORBITER_API ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/ws': { target: API, ws: true, changeOrigin: true },
      '/captures': { target: API, changeOrigin: true },
      '/scans': { target: API, changeOrigin: true },
      '/config': { target: API, changeOrigin: true },
      '/camera': { target: API, changeOrigin: true },
      '/firmware': { target: API, changeOrigin: true },
      // Stereo tab: JSON passthrough to camserver. Only the JSON is proxied —
      // the MJPEG previews point straight at camserver, bypassing us entirely.
      '/stereo': { target: API, changeOrigin: true },
    },
  },
});
