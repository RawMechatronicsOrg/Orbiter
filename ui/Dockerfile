# syntax=docker/dockerfile:1.6

# ── Build stage ─────────────────────────────────────────────────────────────
# Compiles the Vite/React app to static assets in /app/dist.
FROM node:20-alpine AS build
WORKDIR /app

# Install deps with the locked manifest for reproducible builds. The lock
# file is committed so `npm ci` (NOT `npm install`) gives byte-identical
# node_modules across hosts and CI.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# Copy the rest of the sources and build. tsc + vite build produces a
# static bundle in ./dist with hashed asset names ready for CDN caching.
COPY . .
RUN npm run build

# ── Runtime stage ───────────────────────────────────────────────────────────
# nginx:alpine serves the built static files. The default upstream config
# already handles SPA fallback well enough for our single-page app — no
# custom nginx.conf required for v0.1.
FROM nginx:alpine AS runtime

# Drop the default placeholder page and copy our build artifacts.
RUN rm -rf /usr/share/nginx/html/*
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
