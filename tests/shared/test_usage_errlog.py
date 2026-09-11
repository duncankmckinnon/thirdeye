from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.paths import usage_log_path
from thirdeye.usage.errlog import log_capture_error, safe_capture


def test_log_creates_logs_dir_lazily(tmp_path: Path) -> None:
    """The logs/ directory should be created on first write."""
    assert not (tmp_path / "logs").exists()
    log_capture_error(
        thirdeye_home=tmp_path,
        phase="open_source",
        error=FileNotFoundError("nope"),
        platform="claude",
        session_id="abc",
    )
    assert (tmp_path / "logs").is_dir()
    assert usage_log_path(tmp_path).exists()


def test_log_writes_structured_jsonl(tmp_path: Path) -> None:
    log_capture_error(
        thirdeye_home=tmp_path,
        phase="parse_transcript",
        error=FileNotFoundError("nope"),
        platform="claude",
        session_id="abc",
        source_path="/x/y/z",
    )
    entry = json.loads(usage_log_path(tmp_path).read_text().strip())
    assert entry["phase"] == "parse_transcript"
    assert entry["platform"] == "claude"
    assert entry["session_id"] == "abc"
    assert entry["source_path"] == "/x/y/z"
    assert entry["error_class"] == "FileNotFoundError"
    assert entry["level"] == "warn"
    # ts is ISO8601 ending in Z
    assert entry["ts"].endswith("Z")


def test_log_appends_multiple_entries(tmp_path: Path) -> None:
    log_capture_error(thirdeye_home=tmp_path, phase="one", message="first")
    log_capture_error(thirdeye_home=tmp_path, phase="two", message="second")
    lines = usage_log_path(tmp_path).read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["phase"] == "one"
    assert json.loads(lines[1])["phase"] == "two"


def test_log_without_error_object(tmp_path: Path) -> None:
    """Calling without an error should still write a valid entry."""
    log_capture_error(
        thirdeye_home=tmp_path,
        phase="index_sync",
        message="sidecar shrank",
        session_id="abc",
    )
    entry = json.loads(usage_log_path(tmp_path).read_text().strip())
    assert entry["error_class"] == ""
    assert entry["message"] == "sidecar shrank"


def test_log_falls_back_to_stderr_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """If the log write itself fails, fall back to stderr without raising."""
    # Make the logs dir un-creatable by making tmp_path a file
    bad_home = tmp_path / "not_a_dir"
    bad_home.write_text("blocking")
    # Now logs/ can't be created under bad_home
    log_capture_error(
        thirdeye_home=bad_home,
        phase="test",
        message="should fall back",
    )
    captured = capsys.readouterr()
    assert "error log write failed" in captured.err


def test_safe_capture_swallows_exceptions(tmp_path: Path) -> None:
    @safe_capture(phase="extract", platform="claude")
    def boom(*, thirdeye_home: Path, session_id: str) -> int:
        raise RuntimeError("kapow")

    result = boom(thirdeye_home=tmp_path, session_id="abc")
    assert result is None
    log = usage_log_path(tmp_path)
    assert log.exists()
    entry = json.loads(log.read_text().strip())
    assert entry["phase"] == "extract"
    assert entry["platform"] == "claude"
    assert entry["error_class"] == "RuntimeError"
    assert entry["session_id"] == "abc"


def test_safe_capture_returns_inner_value_on_success(tmp_path: Path) -> None:
    @safe_capture(phase="x", platform="claude")
    def ok(*, thirdeye_home: Path) -> str:
        return "ok"

    assert ok(thirdeye_home=tmp_path) == "ok"
    # No log file should have been created
    assert not usage_log_path(tmp_path).exists()


def test_safe_capture_without_thirdeye_home_writes_stderr(
    capsys: pytest.CaptureFixture,
) -> None:
    @safe_capture(phase="x", platform="codex")
    def boom() -> int:
        raise ValueError("oops")

    result = boom()
    assert result is None
    captured = capsys.readouterr()
    assert "codex/x" in captured.err
    assert "ValueError" in captured.err or "oops" in captured.err


def test_safe_capture_preserves_function_metadata() -> None:
    @safe_capture(phase="x", platform="claude")
    def my_func(*, thirdeye_home: Path) -> int:
        """My docstring."""
        return 1

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."
