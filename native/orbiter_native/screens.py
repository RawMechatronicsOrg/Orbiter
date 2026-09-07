"""Which GPU draws the window, against which GPU drives the monitor it is on.

Windows gives a process one OpenGL implementation: the primary display
adapter's. A window shown on a monitor that a different adapter drives is
still drawn by that one, and the desktop (csrss, DWM) then copies every
presented frame between the two GPUs. Measured on the lab PC - AMD iGPU on the
primary monitor, this window on the GTX 1650 SUPER's - that copy alone held
40% of the 1650's 3D engine at 30 frames/s; maximised, with the cloud being
turned, it saturates the GPU that composes that monitor and the desktop
stalls.

This module knows two things: the adapter behind the monitor a window is on,
and whether an OpenGL renderer string names the same GPU. `app.MainWindow`
turns a mismatch into a warning. Windows only: elsewhere the adapter is not
knowable from here and nothing is reported.
"""

from __future__ import annotations

import ctypes
import re
import sys

_MONITOR_DEFAULTTONULL = 0


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint32), ("rcMonitor", _RECT), ("rcWork", _RECT),
                ("dwFlags", ctypes.c_uint32), ("szDevice", ctypes.c_wchar * 32)]


class _DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("DeviceName", ctypes.c_wchar * 32),
                ("DeviceString", ctypes.c_wchar * 128), ("StateFlags", ctypes.c_uint32),
                ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128)]


def adapter_of_window(hwnd: int) -> str | None:
    """The display adapter driving the monitor `hwnd` is on - "NVIDIA GeForce
    GTX 1650 SUPER", "AMD Radeon(TM) Graphics" - or None: not Windows, no such
    window, or a window on no monitor at all."""
    if sys.platform != "win32" or not hwnd:
        return None
    try:
        user32 = ctypes.windll.user32
    except (AttributeError, OSError):
        return None
    user32.MonitorFromWindow.restype = ctypes.c_void_p
    user32.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    user32.EnumDisplayDevicesW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32,
                                           ctypes.c_void_p, ctypes.c_uint32]
    monitor = user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONULL)
    if not monitor:
        return None
    info = _MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(info)
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return None
    want = info.szDevice.lower()
    i = 0
    while True:
        dev = _DISPLAY_DEVICEW()
        dev.cb = ctypes.sizeof(dev)
        if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dev), 0):
            return None
        if dev.DeviceName.lower() == want:
            return dev.DeviceString or None
        i += 1


def _gpu_words(name: str) -> str:
    """A GPU name reduced to what both the driver and GL agree on: no
    trademark marks, no GL suffix ("/PCIe/SSE2"), one space between words."""
    s = name.split("/")[0].lower()
    s = re.sub(r"\((tm|r|c)\)|\u2122|\u00ae", " ", s)
    return " ".join(s.split())


def same_gpu(renderer: str, adapter: str) -> bool:
    """Does the OpenGL renderer string name the display adapter?

    The two come from different places - GL's GL_RENDERER and the display
    driver's DeviceString - and differ in dressing, not in substance:
    "NVIDIA GeForce GTX 1650 SUPER/PCIe/SSE2" against "NVIDIA GeForce GTX 1650
    SUPER". Reduced to their words, one names the other.
    """
    a, b = _gpu_words(renderer), _gpu_words(adapter)
    return bool(a) and bool(b) and (a == b or a.startswith(b) or b.startswith(a))
