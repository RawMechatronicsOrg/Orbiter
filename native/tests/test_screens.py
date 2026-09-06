"""screens.py: the pure half - does a renderer string name the adapter."""

from __future__ import annotations

from orbiter_native.screens import adapter_of_window, same_gpu


def test_same_gpu_ignores_trademarks_and_gl_suffixes() -> None:
    assert same_gpu("AMD Radeon(TM) Graphics", "AMD Radeon(TM) Graphics")
    assert same_gpu("NVIDIA GeForce GTX 1650 SUPER/PCIe/SSE2", "NVIDIA GeForce GTX 1650 SUPER")
    assert same_gpu("Intel(R) UHD Graphics 630", "Intel(R) UHD Graphics 630")
    assert same_gpu("AMD Radeon\u2122 Graphics", "AMD Radeon(TM) Graphics")


def test_same_gpu_tells_two_gpus_apart() -> None:
    assert not same_gpu("AMD Radeon(TM) Graphics", "NVIDIA GeForce GTX 1650 SUPER")
    assert not same_gpu("NVIDIA GeForce RTX 5060 Ti/PCIe/SSE2", "NVIDIA GeForce GTX 1650 SUPER")
    assert not same_gpu("", "NVIDIA GeForce GTX 1650 SUPER")


def test_adapter_of_window_is_none_without_a_window() -> None:
    # 0 is never a window: on Windows no monitor is found for it, elsewhere the
    # question is never asked. Either way the caller gets "unknown", not a name.
    assert adapter_of_window(0) is None
