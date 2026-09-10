"""CLI tests for thirdeye copilot sync/watch/status and package registration."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.config import Config
from thirdeye.platforms.copilot.capture import iter_captured_records
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.types import SourcePaths, SyncResult

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
CLI_FIXTURE = FIXTURES / "cli-1.0.83"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
HOOK_BIN = "thirdeye-copilot-hook"


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "thirdeye"
    monkeypatch.setenv("THIRDEYE_HOME", str(home))
    return home


def _empty_result(**overrides: int) -> SyncResult:
    result: SyncResult = {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 0,
    }
    result.update(overrides)  # type: ignore[typeddict-item]
    return result


def _write_transcript(home: Path, native_id: str) -> None:
    session_dir = home / "session-state" / native_id
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CLI_FIXTURE / "events.jsonl", session_dir / "events.jsonl")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")


def _write_database(home: Path, *, session_id: str = NATIVE_SESSION_ID) -> None:
    home.mkdir(parents=True, exist_ok=True)
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, created_at TEXT);
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER,
                content TEXT,
                updated_at TEXT
            );
            CREATE TABLE assistant_usage_events (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER,
                agent_id TEXT,
                parent_tool_call_id TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                reasoning_tokens INTEGER,
                total_nano_aiu INTEGER,
                request_multiplier REAL,
                duration_ms INTEGER,
                time_to_first_token_ms REAL,
                output_ttft_ms REAL,
                inter_token_latency_ms REAL,
                initiator TEXT,
                api_endpoint TEXT,
                reasoning_effort TEXT,
                finish_reason TEXT,
                content_filter_triggered INTEGER,
                token_details_json TEXT,
                created_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions (id, cwd, created_at) VALUES (?, ?, ?)",
            (session_id, "/tmp/probe", "2026-09-10T17:08:00.000Z"),
        )
        connection.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, session_id, 0, "turn-0", "2026-09-10T17:08:10.000Z"),
        )
        connection.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (2, session_id, 1, "turn-1", "2026-09-10T17:08:11.000Z"),
        )
        usage_rows = CLI_FIXTURE / "assistant-usage-events.json"
        if usage_rows.is_file():
            for row in json.loads(usage_rows.read_text(encoding="utf-8")):
                columns = ", ".join(row)
                placeholders = ", ".join("?" for _ in row)
                connection.execute(
                    f"INSERT INTO assistant_usage_events ({columns}) VALUES ({placeholders})",
                    tuple(row.values()),
                )
        connection.commit()
    finally:
        connection.close()


def _minimal_status(*, configured: bool = False, errors: list[dict[str, Any]] | None = None) -> dict:
    return {
        "paths": {
            "home": "/tmp/copilot",
            "source_key": "abc123",
            "session_root": "/tmp/copilot/session-state",
            "database": "/tmp/copilot/session-store.db",
        },
        "installation": {"configured": configured, "hooks_file": "/tmp/copilot/hooks/thirdeye.json"},
        "capabilities": {
            "transcripts": {"exists": True, "readable": True},
            "database": {"exists": True, "readable": True},
            "database_wal": {"exists": False, "readable": False},
        },
        "last_observed_hook": None,
        "last_successful_import": None,
        "sessions": [],
        "pending": {
            "spool_records": 0,
            "spool_sessions": [],
            "followup": 0,
            "leases": 0,
            "journals": 0,
        },
        "errors": errors or [],
    }


# -- command registration ------------------------------------------------------


def test_copilot_group_appears_in_main_help() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "copilot" in result.output


def test_copilot_help_lists_subcommands() -> None:
    result = CliRunner().invoke(main, ["copilot", "--help"])
    assert result.exit_code == 0, result.output
    for subcommand in ("sync", "watch", "status"):
        assert subcommand in result.output


def test_sync_help_documents_flags() -> None:
    result = CliRunner().invoke(main, ["copilot", "sync", "--help"])
    assert result.exit_code == 0, result.output
    assert "--session-id" in result.output
    assert "--source-home" in result.output


def test_watch_help_documents_interval_and_source_home() -> None:
    result = CliRunner().invoke(main, ["copilot", "watch", "--help"])
    assert result.exit_code == 0, result.output
    assert "--interval" in result.output
    assert "--source-home" in result.output


def test_status_help_documents_source_home() -> None:
    result = CliRunner().invoke(main, ["copilot", "status", "--help"])
    assert result.exit_code == 0, result.output
    assert "--source-home" in result.output


def test_copilot_commands_have_no_export_flag() -> None:
    runner = CliRunner()
    for args in (
        ["copilot", "--help"],
        ["copilot", "sync", "--help"],
        ["copilot", "watch", "--help"],
        ["copilot", "status", "--help"],
    ):
        result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output
        assert "--export" not in result.output


def test_pyproject_registers_copilot_hook_entrypoint() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert f"{HOOK_BIN} =" in text
    assert "thirdeye.platforms.copilot.hooks:main" in text


def test_copilot_hook_entrypoint_is_importable() -> None:
    from importlib.metadata import entry_points

    scripts = entry_points(group="console_scripts")
    hook = next((ep for ep in scripts if ep.name == HOOK_BIN), None)
    assert hook is not None
    module_path, _, attr = hook.value.partition(":")
    module = __import__(module_path, fromlist=[attr])
    assert callable(getattr(module, attr))


# -- sync ----------------------------------------------------------------------


def test_sync_invokes_capture_sync(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_sync(config: Config, paths: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append((config.root, paths["home"], session_id))
        return _empty_result(sessions=2, records_written=5)

    monkeypatch.setattr("thirdeye.commands.copilot.capture_sync", fake_sync)
    result = CliRunner().invoke(main, ["copilot", "sync"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][2] is None
    assert "5" in result.output or "records" in result.output.lower()


def test_sync_passes_session_id(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str | None] = {"session_id": "unset"}

    def fake_sync(_config: Config, _paths: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        captured["session_id"] = session_id
        return _empty_result()

    monkeypatch.setattr("thirdeye.commands.copilot.capture_sync", fake_sync)
    result = CliRunner().invoke(main, ["copilot", "sync", "--session-id", NATIVE_SESSION_ID])
    assert result.exit_code == 0, result.output
    assert captured["session_id"] == NATIVE_SESSION_ID


def test_sync_resolves_explicit_source_home(
    isolated_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / "custom copilot home"
    source_home.mkdir()
    resolved: dict[str, str] = {}

    def fake_resolve(source_home_arg: Path | None = None) -> SourcePaths:
        paths = resolve_sources(source_home_arg)
        resolved["home"] = paths["home"]
        return paths

    def fake_sync(_config: Config, paths: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        resolved["sync_home"] = paths["home"]
        return _empty_result()

    monkeypatch.setattr("thirdeye.commands.copilot.resolve_sources", fake_resolve)
    monkeypatch.setattr("thirdeye.commands.copilot.capture_sync", fake_sync)
    result = CliRunner().invoke(main, ["copilot", "sync", "--source-home", str(source_home)])
    assert result.exit_code == 0, result.output
    assert Path(resolved["home"]) == source_home.resolve()
    assert Path(resolved["sync_home"]) == source_home.resolve()


def test_sync_empty_discovery_exits_zero(isolated_home: Path, tmp_path: Path) -> None:
    empty_home = tmp_path / "empty-copilot"
    empty_home.mkdir()
    result = CliRunner().invoke(main, ["copilot", "sync", "--source-home", str(empty_home)])
    assert result.exit_code == 0, result.output


def test_sync_missing_session_id_exits_nonzero(
    isolated_home: Path,
    tmp_path: Path,
) -> None:
    empty_home = tmp_path / "empty-copilot"
    empty_home.mkdir()
    result = CliRunner().invoke(
        main,
        ["copilot", "sync", "--source-home", str(empty_home), "--session-id", "missing-session-id"],
    )
    assert result.exit_code != 0, result.output
    assert "No such command" not in result.output
    assert "missing-session-id" in result.output
    assert "errors=1" in result.output


def test_sync_session_with_capture_errors_exits_nonzero(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sync(
        _config: Config,
        _paths: SourcePaths,
        *,
        session_id: str | None = None,
    ) -> SyncResult:
        return _empty_result(errors=1)

    monkeypatch.setattr("thirdeye.commands.copilot.capture_sync", fake_sync)
    result = CliRunner().invoke(
        main,
        ["copilot", "sync", "--session-id", NATIVE_SESSION_ID],
    )
    assert result.exit_code != 0, result.output
    assert NATIVE_SESSION_ID in result.output
    assert "errors=1" in result.output


def test_sync_capture_value_error_becomes_click_exception(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sync(
        _config: Config,
        _paths: SourcePaths,
        *,
        session_id: str | None = None,
    ) -> SyncResult:
        raise ValueError("invalid native session routing")

    monkeypatch.setattr("thirdeye.commands.copilot.capture_sync", fake_sync)
    result = CliRunner().invoke(main, ["copilot", "sync"])
    assert result.exit_code != 0, result.output
    assert "invalid native session routing" in result.output


def test_sync_prints_counts(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sync(_config: Config, _paths: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        return _empty_result(sessions=1, records_written=12, duplicate_records=3, pending=2, errors=0)

    monkeypatch.setattr("thirdeye.commands.copilot.capture_sync", fake_sync)
    result = CliRunner().invoke(main, ["copilot", "sync"])
    assert result.exit_code == 0, result.output
    for token in ("1", "12", "3", "2"):
        assert token in result.output


def test_sync_fixture_session_end_to_end(isolated_home: Path, tmp_path: Path) -> None:
    source_home = tmp_path / "copilot-home"
    source_home.mkdir()
    _write_transcript(source_home, NATIVE_SESSION_ID)
    _write_database(source_home, session_id=NATIVE_SESSION_ID)

    result = CliRunner().invoke(main, ["copilot", "sync", "--source-home", str(source_home)])
    assert result.exit_code == 0, result.output

    paths = resolve_sources(source_home)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(Config.load(), stored))
    kinds = {record["source_kind"] for record in captured}
    assert "transcript" in kinds
    assert "database" in kinds
    assert sum(1 for record in captured if record["source_kind"] == "transcript") == 76


def test_sync_rejects_native_id_with_path_separators(isolated_home: Path) -> None:
    result = CliRunner().invoke(main, ["copilot", "sync", "--session-id", "../escape"])
    assert result.exit_code != 0, result.output
    assert "No such command" not in result.output


# -- watch ---------------------------------------------------------------------


def test_watch_rejects_interval_below_minimum(isolated_home: Path) -> None:
    result = CliRunner().invoke(main, ["copilot", "watch", "--interval", "0.05"])
    assert result.exit_code != 0, result.output
    assert "No such command" not in result.output
    assert "0.1" in result.output or "interval" in result.output.lower()


def test_watch_rejects_non_finite_interval(isolated_home: Path) -> None:
    result = CliRunner().invoke(main, ["copilot", "watch", "--interval", "inf"])
    assert result.exit_code != 0, result.output
    assert "No such command" not in result.output


def test_watch_invokes_watch_module(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_watch(_config: Config, _paths: SourcePaths, *, interval: float = 1.0) -> None:
        calls.append(interval)

    monkeypatch.setattr("thirdeye.commands.copilot.watch_loop", fake_watch)
    result = CliRunner().invoke(main, ["copilot", "watch", "--interval", "2.5"])
    assert result.exit_code == 0, result.output
    assert calls == [2.5]


def test_watch_passes_source_home(
    isolated_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / "watch home"
    source_home.mkdir()
    seen: dict[str, str] = {}

    def fake_watch(_config: Config, paths: SourcePaths, *, interval: float = 1.0) -> None:
        seen["home"] = paths["home"]

    monkeypatch.setattr("thirdeye.commands.copilot.watch_loop", fake_watch)
    result = CliRunner().invoke(
        main,
        ["copilot", "watch", "--source-home", str(source_home), "--interval", "1"],
    )
    assert result.exit_code == 0, result.output
    assert Path(seen["home"]) == source_home.resolve()


# -- status --------------------------------------------------------------------


def test_status_invokes_capture_status(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}

    def fake_status(_config: Config, _paths: SourcePaths) -> dict:
        called["count"] += 1
        return _minimal_status(configured=False)

    monkeypatch.setattr("thirdeye.commands.copilot.capture_status", fake_status)
    result = CliRunner().invoke(main, ["copilot", "status"])
    assert result.exit_code == 0, result.output
    assert called["count"] == 1


def test_status_exits_zero_when_hooks_missing_but_sources_ok(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "thirdeye.commands.copilot.capture_status",
        lambda _config, _paths: _minimal_status(configured=False, errors=[]),
    )
    result = CliRunner().invoke(main, ["copilot", "status"])
    assert result.exit_code == 0, result.output
    assert "not configured" in result.output.lower() or "configured" in result.output.lower()


def test_status_exits_nonzero_on_source_errors(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "thirdeye.commands.copilot.capture_status",
        lambda _config, _paths: _minimal_status(
            configured=True,
            errors=[{"kind": "source_unreadable", "message": "database unreadable"}],
        ),
    )
    result = CliRunner().invoke(main, ["copilot", "status"])
    assert result.exit_code != 0, result.output
    assert "source_unreadable" in result.output
    assert "database unreadable" in result.output


def test_status_prints_string_errors(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "thirdeye.commands.copilot.capture_status",
        lambda _config, _paths: _minimal_status(errors=["legacy string error"]),  # type: ignore[list-item]
    )
    result = CliRunner().invoke(main, ["copilot", "status"])
    assert result.exit_code != 0, result.output
    assert "legacy string error" in result.output


def test_status_prints_paths_and_guidance(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "thirdeye.commands.copilot.capture_status",
        lambda _config, _paths: _minimal_status(configured=True),
    )
    result = CliRunner().invoke(main, ["copilot", "status"])
    assert result.exit_code == 0, result.output
    assert "/tmp/copilot" in result.output
    assert "hooks" in result.output.lower() or "configured" in result.output.lower()


def test_status_resolves_source_home_with_spaces(
    isolated_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / "copilot home with spaces"
    source_home.mkdir()
    seen: dict[str, str] = {}

    def fake_status(_config: Config, paths: SourcePaths) -> dict:
        seen["home"] = paths["home"]
        return _minimal_status()

    monkeypatch.setattr("thirdeye.commands.copilot.capture_status", fake_status)
    result = CliRunner().invoke(main, ["copilot", "status", "--source-home", str(source_home)])
    assert result.exit_code == 0, result.output
    assert Path(seen["home"]) == source_home.resolve()


# -- add/remove wiring ---------------------------------------------------------


def test_add_help_mentions_copilot() -> None:
    result = CliRunner().invoke(main, ["add", "--help"])
    assert result.exit_code == 0, result.output
    assert "--copilot" in result.output


def test_remove_help_mentions_copilot() -> None:
    result = CliRunner().invoke(main, ["remove", "--help"])
    assert result.exit_code == 0, result.output
    assert "--copilot" in result.output


def test_add_copilot_dispatches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    from thirdeye.commands.add import PLATFORMS

    mock_platform = MagicMock()
    mock_platform.display_name = "GitHub Copilot CLI"
    mock_cls = MagicMock(return_value=mock_platform)
    monkeypatch.setitem(PLATFORMS, "copilot", mock_cls)

    result = CliRunner().invoke(main, ["add", "--copilot"])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_platform.install.assert_called_once()


def test_remove_copilot_dispatches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    from thirdeye.commands.add import PLATFORMS

    mock_platform = MagicMock()
    mock_platform.display_name = "GitHub Copilot CLI"
    mock_cls = MagicMock(return_value=mock_platform)
    monkeypatch.setitem(PLATFORMS, "copilot", mock_cls)

    result = CliRunner().invoke(main, ["remove", "--copilot"])
    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_platform.uninstall.assert_called_once()
