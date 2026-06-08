import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Dev-time reverse proxy to the storage-api so the browser can hit
// same-origin URLs (no CORS dance). Routes mirror what the v0.1 UI
// actually calls: /ws/scene (WebSocket), /scans, /captures, /config, and the
// live MJPEG camera stream under /camera (long-lived multipart response —
// http-proxy pipes it without buffering).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/ws': { target: 'http://127.0.0.1:8000', ws: true, changeOrigin: true },
      '/captures': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/scans': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/config': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/camera': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
});
