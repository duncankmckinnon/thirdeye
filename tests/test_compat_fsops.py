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


# -- read_text -----------------------------------------------------------------
#
# The mirror of the rename/unlink cases above. Windows refuses to open a file
# another process is atomically replacing (ERROR_SHARING_VIOLATION, surfaced as
# PermissionError), so a reader racing a writer's os.replace fails even though
# the file is present and intact a millisecond later.


def test_read_text_returns_content(tmp_path: Path) -> None:
    path = tmp_path / "meta.yaml"
    path.write_text("hello: world\n", encoding="utf-8")

    assert fsops.read_text(path) == "hello: world\n"


def test_read_text_propagates_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fsops.read_text(tmp_path / "missing")


def test_read_text_posix_never_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "meta.yaml"
    path.write_text("x", encoding="utf-8")
    calls = 0

    def denied(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise PermissionError("denied")

    monkeypatch.setattr(fsops, "IS_WINDOWS", False)
    monkeypatch.setattr(fsops.Path, "read_text", denied)

    with pytest.raises(PermissionError, match="denied"):
        fsops.read_text(path)
    assert calls == 1


def test_read_text_windows_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "meta.yaml"
    path.write_text("x", encoding="utf-8")
    calls = 0

    def flaky(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("locked by a concurrent replace")
        return "recovered"

    monkeypatch.setattr(fsops, "IS_WINDOWS", True)
    monkeypatch.setattr(fsops.Path, "read_text", flaky)
    monkeypatch.setattr(fsops.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(fsops.time, "sleep", lambda delay: None)

    assert fsops.read_text(path) == "recovered"
    assert calls == 3


def test_read_text_windows_reraises_after_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "meta.yaml"
    path.write_text("x", encoding="utf-8")
    now = 0.0

    def denied(*args: object, **kwargs: object) -> str:
        raise PermissionError("still locked")

    def sleep(delay: float) -> None:
        nonlocal now
        now += delay

    monkeypatch.setattr(fsops, "IS_WINDOWS", True)
    monkeypatch.setattr(fsops.Path, "read_text", denied)
    monkeypatch.setattr(fsops.time, "monotonic", lambda: now)
    monkeypatch.setattr(fsops.time, "sleep", sleep)

    with pytest.raises(PermissionError, match="still locked"):
        fsops.read_text(path)
    assert now == pytest.approx(fsops._RETRY_BUDGET_S)
