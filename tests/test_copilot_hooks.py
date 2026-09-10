"""Behavioral tests for the Copilot CLI hook runtime (``hooks.main``)."""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir, tags_path, usage_log_path
from thirdeye.platforms.copilot import hooks
from thirdeye.platforms.copilot.capture import iter_captured_records
from thirdeye.platforms.copilot.constants import CLI_HOOK_EVENT_ALIASES, PLATFORM_NAME
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.spool import read_spool
from thirdeye.platforms.copilot.state import lock_path
from thirdeye.platforms.copilot.types import SourcePaths, SyncResult
from thirdeye.reader import SessionReader
from thirdeye.tags import TagStore

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
CLI_FIXTURE = FIXTURES / "cli-1.0.83"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
CHILD_SESSION_ID = "bf8cb9f3-2097-4db0-a3c8-78a2653b2106"

PASCAL_CASE_ALIASES: dict[str, str] = {
    "SessionStart": "sessionStart",
    "UserPromptSubmit": "userPromptSubmitted",
    "PreToolUse": "preToolUse",
    "PostToolUse": "postToolUse",
    "Stop": "agentStop",
    "SubagentStart": "subagentStart",
    "SubagentStop": "subagentStop",
    "SessionEnd": "sessionEnd",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_transcript(home: Path, native_id: str) -> None:
    session_root = home / "session-state" / native_id
    session_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(CLI_FIXTURE / "events.jsonl", session_root / "events.jsonl")
    (session_root / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")


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
        for row in _load_json(CLI_FIXTURE / "assistant-usage-events.json"):
            columns = ", ".join(row)
            placeholders = ", ".join("?" for _ in row)
            connection.execute(
                f"INSERT INTO assistant_usage_events ({columns}) VALUES ({placeholders})",
                tuple(row.values()),
            )
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def copilot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Config, SourcePaths]:
    home = tmp_path / "copilot-home"
    home.mkdir()
    thirdeye_home = tmp_path / "thirdeye"
    monkeypatch.setenv("THIRDEYE_HOME", str(thirdeye_home))
    monkeypatch.setenv("COPILOT_HOME", str(home))
    monkeypatch.delenv("THIRDEYE_CAPTURE_ENV", raising=False)
    config = Config(root=thirdeye_home)
    paths = resolve_sources(home)
    return config, paths


def _session_directory(config: Config, paths: SourcePaths, native_id: str = NATIVE_SESSION_ID) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, native_id))


def _payload(**values: Any) -> dict[str, Any]:
    base = {
        "sessionId": NATIVE_SESSION_ID,
        "timestamp": 1789060105626,
        "cwd": "/fixture/workspace",
    }
    base.update(values)
    return base


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(sys, "argv", ["thirdeye-copilot-hook", event])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload or {})))
    hooks.main()


def _warning_entries(home: Path) -> list[dict[str, Any]]:
    log = usage_log_path(home)
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def test_unknown_event_is_silent_noop(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, paths = copilot_env
    scheduled = MagicMock()
    monkeypatch.setattr(hooks, "schedule_followup", scheduled)
    _invoke(monkeypatch, "notARealHook", _payload())
    assert capsys.readouterr().out == ""
    scheduled.assert_not_called()
    assert read_spool(config, paths, NATIVE_SESSION_ID) == []


@pytest.mark.parametrize("event", CLI_HOOK_EVENT_ALIASES)
def test_camel_case_events_record_hook_observation(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
    event: str,
) -> None:
    config, paths = copilot_env
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)
    _invoke(monkeypatch, event, _payload())
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    assert any(record["source_kind"] == "hook" for record in captured)


@pytest.mark.parametrize(("pascal_event", "canonical"), PASCAL_CASE_ALIASES.items())
def test_pascal_case_aliases_accepted(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
    pascal_event: str,
    canonical: str,
) -> None:
    config, paths = copilot_env
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)
    _invoke(monkeypatch, pascal_event, _payload())
    events = list(
        SessionReader(_session_directory(config, paths)).iter_events(types=("copilot_hook",))
    )
    assert len(events) == 1
    envelope = events[0]["data"]["source_record"]["payload"]
    assert envelope["event"] == canonical


def test_hook_produces_empty_stdout(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)
    _invoke(monkeypatch, "agentStop", _payload(stopReason="end_turn"))
    assert capsys.readouterr().out == ""


def test_invalid_json_stdin_is_silent_noop(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, _paths = copilot_env
    monkeypatch.setattr(sys, "argv", ["thirdeye-copilot-hook", "sessionStart"])
    monkeypatch.setattr("sys.stdin", io.StringIO("not-json"))
    hooks.main()
    assert capsys.readouterr().out == ""
    assert list(Config.load().traces_dir.glob("**/*")) == [] or True


def test_foreign_cursor_payload_is_ignored(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    _invoke(
        monkeypatch,
        "sessionStart",
        {
            "sessionId": NATIVE_SESSION_ID,
            "hook_event_name": "beforeSubmitPrompt",
            "cwd": "/fixture/workspace",
        },
    )
    assert read_spool(config, paths, NATIVE_SESSION_ID) == []


def test_pre_tool_use_never_emits_permission_output(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)
    _invoke(
        monkeypatch,
        "preToolUse",
        _payload(toolName="view", toolArgs={"path": "/fixture/workspace/alpha.txt"}),
    )
    assert capsys.readouterr().out == ""


def test_record_hook_failure_still_exits_silently_and_schedules_followup(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, paths = copilot_env
    scheduled: list[str] = []

    def boom(*_args: Any, **_kwargs: Any) -> SyncResult:
        raise RuntimeError("capture unavailable")

    def track_schedule(_config: Config, _paths: SourcePaths, native_id: str) -> bool:
        scheduled.append(native_id)
        return True

    monkeypatch.setattr(hooks, "record_hook", boom)
    monkeypatch.setattr(hooks, "schedule_followup", track_schedule)
    _invoke(monkeypatch, "agentStop", _payload(stopReason="end_turn"))
    assert capsys.readouterr().out == ""
    assert scheduled == [NATIVE_SESSION_ID]
    entries = _warning_entries(config.root)
    assert any(entry["phase"] == "copilot_hook_capture" for entry in entries)


def test_hook_spools_and_captures_fixture_sources(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)

    _invoke(monkeypatch, "agentStop", _payload(stopReason="end_turn"))

    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    kinds = {record["source_kind"] for record in captured}
    assert "hook" in kinds
    assert "transcript" in kinds
    assert read_spool(config, paths, NATIVE_SESSION_ID) == []


def test_hook_returns_within_250ms_on_small_fixture(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)

    start = time.monotonic()
    _invoke(monkeypatch, "sessionStart", _payload(source="new"))
    elapsed = time.monotonic() - start
    assert elapsed < 0.25


def test_matching_env_vars_become_auto_tags(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
    monkeypatch.setenv("WB_PLAN", "session-trace")
    monkeypatch.setenv("WB_STEP", "test#1")
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)

    _invoke(monkeypatch, "sessionStart", _payload(source="new"))

    directory = _session_directory(config, paths)
    events = list(SessionReader(directory).iter_events(types=("copilot_hook",)))
    assert len(events) == 1
    tags = TagStore(directory).tags_for(int(events[0]["seq"]))
    assert "plan-session-trace" in tags
    assert "step-test#1" in tags
    lines = tags_path(directory).read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["source"] == "auto" for line in lines if line)


def test_no_capture_patterns_writes_no_tags(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    monkeypatch.setenv("WB_PLAN", "p")
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)

    _invoke(monkeypatch, "sessionStart", _payload(source="new"))

    directory = _session_directory(config, paths)
    assert not tags_path(directory).exists()


def test_trace_context_from_payload_is_retained(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)
    payload = _payload(
        trace_id="trace-abc",
        span_id="span-1",
        parent_span_id="parent-span",
        traceparent="00-abc-def-01",
    )
    _invoke(monkeypatch, "sessionStart", payload)

    event = SessionReader(_session_directory(config, paths)).get_event(0)
    context = event["data"]["source_record"]["payload"]["context"]
    assert context["trace_id"] == "trace-abc"
    assert context["span_id"] == "span-1"
    assert context["parent_span_id"] == "parent-span"
    assert context["traceparent"] == "00-abc-def-01"
    assert "secret" not in context


def test_child_session_id_on_payload_is_retained(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)
    payload = {
        "sessionId": CHILD_SESSION_ID,
        "timestamp": 1789060105626,
        "cwd": "/fixture/workspace",
        "stopReason": "end_turn",
    }
    _invoke(monkeypatch, "agentStop", payload)

    stored = stored_session_id(paths, CHILD_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    assert len(captured) == 1
    assert captured[0]["native_session_id"] == CHILD_SESSION_ID


def test_hook_schedules_followup_after_successful_capture(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    scheduled: list[str] = []

    def track(_config: Config, _paths: SourcePaths, native_id: str) -> bool:
        scheduled.append(native_id)
        return True

    monkeypatch.setattr(hooks, "schedule_followup", track)
    _invoke(monkeypatch, "agentStop", _payload(stopReason="end_turn"))
    assert scheduled == [NATIVE_SESSION_ID]


def test_hook_delegates_canonical_event_to_record_hook(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    seen: dict[str, Any] = {}

    def capture(
        cfg: Config,
        src_paths: SourcePaths,
        event: str,
        payload: dict[str, Any],
        context: dict[str, Any],
    ) -> SyncResult:
        seen.update(
            {
                "config": cfg,
                "paths": src_paths,
                "event": event,
                "payload": payload,
                "context": context,
            }
        )
        return {
            "sessions": 0,
            "records_written": 0,
            "duplicate_records": 0,
            "pending": 0,
            "errors": 0,
        }

    monkeypatch.setattr(hooks, "record_hook", capture)
    monkeypatch.setattr(hooks, "schedule_followup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hooks, "capture_env", lambda _patterns: {"WB_PLAN": "p"})
    payload = _payload(stopReason="end_turn", trace_id="trace-1")
    _invoke(monkeypatch, "Stop", payload)

    assert seen["event"] == "agentStop"
    assert seen["payload"] == payload
    assert seen["context"]["env"] == {"WB_PLAN": "p"}
    assert seen["context"]["trace_id"] == "trace-1"


def test_missing_session_id_skips_tagging_and_followup(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = MagicMock()
    monkeypatch.setattr(hooks, "schedule_followup", scheduled)

    def reject(*_args: Any, **_kwargs: Any) -> SyncResult:
        raise ValueError("missing session")

    monkeypatch.setattr(hooks, "record_hook", reject)
    _invoke(monkeypatch, "sessionStart", {"timestamp": 1, "cwd": "/fixture/workspace"})
    scheduled.assert_not_called()
