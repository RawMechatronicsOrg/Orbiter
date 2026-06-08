"""mDNS service discovery for the ESP32 firmware.

The firmware advertises itself as `orbiter.local` plus an `_orbiter._tcp`
service on its HTTP port (see `firmware/main/orbiter_main.c::start_mdns`).
This module browses for that service and pushes the resolved IPv4 address
into `model.esp_ip` whenever the device shows up — letting the operator
skip the "find the ESP's IP and paste it" step entirely.

The browser is gated by `model.esp_autodiscover` (persisted, default ON).
When the user toggles it off from the UI, call `sync_to_model()` to stop
the browser; toggle back on to restart.

Threading model: the `zeroconf` library invokes listener callbacks on its
own thread. We bridge to the asyncio loop via `call_soon_threadsafe` so
`model.update()` and its subscribers run in the right context.
"""

from __future__ import annotations

import asyncio
import logging

from orbiter_model import model

log = logging.getLogger("orbiter.discovery")

#: Service type the firmware advertises. Must match the strings passed to
#: `mdns_service_add(NULL, "_orbiter", "_tcp", …)` in the firmware.
_SERVICE_TYPE = "_orbiter._tcp.local."

# Lazy imports of zeroconf classes — keeps app importable even if the lib
# is missing (we just log and disable discovery).
_zc = None  # type: ignore[assignment]
_browser = None  # type: ignore[assignment]
_lock = asyncio.Lock()


class _Listener:
    """Bridges zeroconf threaded callbacks to the asyncio loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _resolve(self, type_: str, name: str) -> str | None:
        if _zc is None:
            return None
        from zeroconf import IPVersion

        info = _zc.get_service_info(type_, name, timeout=2000)
        if info is None:
            return None
        addrs = info.parsed_scoped_addresses(IPVersion.V4Only)
        if not addrs:
            return None
        # Strip any %scope suffix from link-local addresses — esp_proxy
        # builds `http://<ip>` URLs that don't accept the suffix.
        return addrs[0].split("%", 1)[0]

    def _dispatch(self, type_: str, name: str, action: str) -> None:
        ip = self._resolve(type_, name)
        if ip:
            log.info("mDNS %s: %s @ %s", action, name, ip)
            self._loop.call_soon_threadsafe(_apply_discovery, ip)
        else:
            log.debug("mDNS %s: %s (no IPv4 address)", action, name)

    # The zeroconf API names these methods; signature is fixed.
    def add_service(self, _zc, type_, name):  # noqa: D401, ARG002
        self._dispatch(type_, name, "found")

    def update_service(self, _zc, type_, name):  # noqa: D401, ARG002
        self._dispatch(type_, name, "updated")

    def remove_service(self, _zc, type_, name):  # noqa: D401, ARG002
        # Don't clear esp_ip on removal: the device is probably just
        # transiently unreachable. esp_proxy's poll loop owns `esp_online`
        # and will flip it false on the next failed /state read.
        log.info("mDNS removed: %s", name)


def _apply_discovery(ip: str) -> None:
    """Update `model.esp_ip`. Called on the asyncio loop."""
    # Re-check the toggle in case the user disabled discovery between
    # zeroconf's resolution and our loop tick.
    if not model.esp_autodiscover:
        return
    if model.esp_ip == ip:
        return
    log.info("autodiscover: setting esp_ip = %s (was %r)", ip, model.esp_ip)
    model.update(esp_ip=ip)


async def _start_browser() -> None:
    global _zc, _browser
    if _zc is not None:
        return
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        log.warning(
            "zeroconf not installed — mDNS discovery disabled "
            "(install with: pip install 'zeroconf>=0.130')"
        )
        return
    try:
        loop = asyncio.get_running_loop()
        _zc = Zeroconf()
        _browser = ServiceBrowser(_zc, _SERVICE_TYPE, listener=_Listener(loop))
        log.info("mDNS browser started for %s", _SERVICE_TYPE)
    except OSError as exc:
        # Common cause: firewall blocking multicast, no usable network
        # interface (no Wi-Fi), or another Zeroconf instance bound the
        # mDNS socket. Discovery is opt-in convenience — don't crash boot.
        log.warning("could not start mDNS browser: %s", exc)
        _zc = None
        _browser = None


async def _stop_browser() -> None:
    global _zc, _browser
    if _zc is None:
        return
    _browser = None
    try:
        _zc.close()
    except Exception:  # noqa: BLE001 — zeroconf close on a half-initialised state is noisy
        log.exception("zeroconf close raised — proceeding")
    _zc = None
    log.info("mDNS browser stopped")


async def sync_to_model() -> None:
    """Reconcile browser running-state with `model.esp_autodiscover`.

    Call after toggling the field. Safe to call repeatedly — idempotent.
    """
    async with _lock:
        if model.esp_autodiscover:
            await _start_browser()
        else:
            await _stop_browser()


async def start() -> None:
    """Initial start during app lifespan startup."""
    await sync_to_model()


async def stop() -> None:
    """Lifespan shutdown — always stop, regardless of the toggle."""
    async with _lock:
        await _stop_browser()
