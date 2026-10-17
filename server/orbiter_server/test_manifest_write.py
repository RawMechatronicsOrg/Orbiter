"""Manifest writes survive Windows refusing a rename for a moment (another
handle on manifest.json) and never interleave."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

import storage


def test_a_refused_rename_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_replace = os.replace
    refusals = iter([True, True, False])

    def flaky_replace(src: str, dst: Path) -> None:
        if next(refusals):
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", flaky_replace)
    monkeypatch.setattr(storage, "_REPLACE_BACKOFF_S", 0.0)
    target = tmp_path / "manifest.json"
    storage._atomic_write_json(target, {"n": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"n": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


def test_a_rename_that_stays_refused_raises_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_src: str, _dst: Path) -> None:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(storage.os, "replace", refuse)
    monkeypatch.setattr(storage, "_REPLACE_BACKOFF_S", 0.0)
    with pytest.raises(PermissionError):
        storage._atomic_write_json(tmp_path / "manifest.json", {"n": 1})
    assert list(tmp_path.iterdir()) == []


def test_concurrent_writers_do_not_interleave(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    errors: list[BaseException] = []

    def write(n: int) -> None:
        try:
            for i in range(20):
                storage._atomic_write_json(target, {"writer": n, "i": i})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert json.loads(target.read_text(encoding="utf-8"))["i"] == 19
