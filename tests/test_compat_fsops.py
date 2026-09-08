"""Tests for Windows-aware filesystem operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye._compat import fsops


def test_replace_matches_os_replace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")

    fsops.replace(source, destination)

    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "new"


def test_unlink_missing_ok_semantics(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    fsops.unlink(missing, missing_ok=True)

    with pytest.raises(FileNotFoundError):
        fsops.unlink(missing)


def test_posix_never_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def denied(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        raise PermissionError("denied")

    monkeypatch.setattr(fsops, "IS_WINDOWS", False)
    monkeypatch.setattr(fsops.os, "replace", denied)

    with pytest.raises(PermissionError, match="denied"):
        fsops.replace(tmp_path / "source", tmp_path / "destination")
    assert calls == 1


def test_windows_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def flaky(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("locked")

    monkeypatch.setattr(fsops, "IS_WINDOWS", True)
    monkeypatch.setattr(fsops.os, "replace", flaky)
    monkeypatch.setattr(fsops.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(fsops.time, "sleep", lambda delay: None)

    fsops.replace(tmp_path / "source", tmp_path / "destination")

    assert calls == 3


def test_windows_reraises_after_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0
    now = 0.0

    def denied(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        raise PermissionError("still locked")

    def monotonic() -> float:
        return now

    def sleep(delay: float) -> None:
        nonlocal now
        now += delay

    monkeypatch.setattr(fsops, "IS_WINDOWS", True)
    monkeypatch.setattr(fsops.os, "replace", denied)
    monkeypatch.setattr(fsops.time, "monotonic", monotonic)
    monkeypatch.setattr(fsops.time, "sleep", sleep)

    with pytest.raises(PermissionError, match="still locked"):
        fsops.replace(tmp_path / "source", tmp_path / "destination")
    assert calls > 10
    assert now == pytest.approx(fsops._RETRY_BUDGET_S)
