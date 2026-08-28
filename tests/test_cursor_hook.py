from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.platforms.cursor import hook
from thirdeye.store import Store


def _invoke(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert hook.main() == 0


def _events() -> list[dict]:
    return list(Store(Config.load()).reader("session-1").iter_events())


def test_hook_captures_cursor_turn_and_dispatches_logfire_export(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    exported = []
    monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *args: exported.append(args))
    common = {
        "conversation_id": "session-1",
        "generation_id": "generation-1",
        "cwd": "/repo",
    }
    _invoke(monkeypatch, {**common, "hook_event_name": "beforeSubmitPrompt", "prompt": "hello"})
    _invoke(
        monkeypatch,
        {**common, "hook_event_name": "afterAgentResponse", "text": "hi", "model": "gpt-5"},
    )
    _invoke(
        monkeypatch,
        {
            **common,
            "hook_event_name": "stop",
            "model": "gpt-5",
            "input_tokens": 2,
            "output_tokens": 1,
        },
    )
    events = list(Store(Config.load()).reader("session-1").iter_events())
    assert [event["t"] for event in events] == [
        "user_message",
        "assistant_message",
        "turn_stop",
    ]
    assert len(exported) == 1
    turn = exported[0][-1]
    assert turn["input_message"] == "hello"
    assert turn["output_message"] == "hi"


def test_hook_always_prints_permissive_response(monkeypatch, capfd):
    monkeypatch.setitem(hook._HANDLERS, "beforeSubmitPrompt", lambda payload: 1 / 0)
    _invoke(monkeypatch, {"hook_event_name": "beforeSubmitPrompt"})
    assert '"permission": "allow"' in capfd.readouterr().out


def test_unknown_event_is_permissive_and_noop(tmp_path: Path, monkeypatch, capfd):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "hook_event_name": "futureCursorEvent",
            "payload": "must not be stored",
        },
    )

    assert capfd.readouterr().out == '{"continue": true}'
    assert list(Store(Config.load()).list_sessions()) == []


def test_hook_fires_live_export_only_when_shell_tool_completes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.emit_live_tools",
        lambda *args: emitted.append(args),
    )
    common = {
        "conversation_id": "session-1",
        "generation_id": "generation-1",
        "cwd": "/repo",
    }
    _invoke(
        monkeypatch,
        {**common, "hook_event_name": "beforeShellExecution", "command": "pytest"},
    )
    assert emitted == []

    _invoke(
        monkeypatch,
        {**common, "hook_event_name": "afterShellExecution", "output": "passed"},
    )
    assert len(emitted) == 1


def test_before_read_records_noninstant_normalized_call(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": "beforeReadFile",
            "tool_name": "Read",
            "path": "src/app.py",
        },
    )

    events = _events()
    assert len(events) == 1
    assert events[0]["t"] == "tool_call"
    assert events[0]["data"]["tool_name"] == "read_file"
    assert events[0]["data"]["cursor_tool_family"] == "read_file"
    assert "cursor_instant" not in events[0]["data"]


@pytest.mark.parametrize("tool_name", ["Read", "read", "read_file", "view", "view_file"])
def test_post_read_alias_records_noninstant_result(tmp_path: Path, monkeypatch, tool_name: str):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": "postToolUse",
            "tool_name": tool_name,
            "result": "file contents",
        },
    )

    events = _events()
    assert len(events) == 1
    assert events[0]["t"] == "tool_result"
    assert events[0]["data"]["tool_name"] == "read_file"
    assert events[0]["data"]["cursor_tool_family"] == "read_file"
    assert "cursor_instant" not in events[0]["data"]


def test_read_result_triggers_live_export_once(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.emit_live_tools",
        lambda *args: emitted.append(args),
    )
    common = {
        "conversation_id": "session-1",
        "generation_id": "generation-1",
        "cwd": "/repo",
    }

    _invoke(
        monkeypatch,
        {**common, "hook_event_name": "beforeReadFile", "path": "src/app.py"},
    )
    assert emitted == []

    _invoke(
        monkeypatch,
        {
            **common,
            "hook_event_name": "postToolUse",
            "tool_name": "Read",
            "result": "file contents",
        },
    )
    assert len(emitted) == 1


@pytest.mark.parametrize(
    "tool_name",
    [
        "shell",
        "terminal",
        "bash",
        "run_command",
        "run_shell",
        "mcp",
        "mcp_execution",
        "edit_file",
        "edit",
        "write_file",
        "write",
        "create_file",
        "delete_file",
        "tab_file_read",
        "tab_file_edit",
        # Cursor is not guaranteed to send the alias lowercased.
        "Shell",
        "Edit",
        "MCP",
    ],
)
def test_post_tool_skips_dedicated_after_aliases(tmp_path: Path, monkeypatch, tool_name: str):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.emit_live_tools",
        lambda *args: emitted.append(args),
    )

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": "postToolUse",
            "tool_name": tool_name,
            "result": "ignored duplicate",
        },
    )

    assert list(Store(Config.load()).list_sessions()) == []
    assert emitted == []


def test_generic_post_tool_remains_instant(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.emit_live_tools",
        lambda *args: emitted.append(args),
    )

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": "postToolUse",
            "tool_name": "search_web",
            "result": {"matches": 3},
        },
    )

    events = _events()
    assert len(events) == 1
    assert events[0]["t"] == "tool_result"
    assert events[0]["data"]["tool_name"] == "search_web"
    assert events[0]["data"]["cursor_instant"] is True
    assert "cursor_tool_family" not in events[0]["data"]
    assert len(emitted) == 1


def test_read_without_session_is_noop(tmp_path: Path, monkeypatch, capfd):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.emit_live_tools",
        lambda *args: emitted.append(args),
    )

    _invoke(
        monkeypatch,
        {"hook_event_name": "beforeReadFile", "generation_id": "generation-1"},
    )
    _invoke(
        monkeypatch,
        {
            "hook_event_name": "postToolUse",
            "generation_id": "generation-1",
            "tool_name": "Read",
        },
    )

    assert capfd.readouterr().out == '{"permission": "allow"}{"continue": true}'
    assert list(Store(Config.load()).list_sessions()) == []
    assert emitted == []


def test_read_without_generation_records_but_does_not_export_live(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.emit_live_tools",
        lambda *args: emitted.append(args),
    )
    common = {"conversation_id": "session-1", "cwd": "/repo"}

    _invoke(
        monkeypatch,
        {**common, "hook_event_name": "beforeReadFile", "path": "src/app.py"},
    )
    _invoke(
        monkeypatch,
        {
            **common,
            "hook_event_name": "postToolUse",
            "tool_name": "view_file",
            "result": "file contents",
        },
    )

    events = _events()
    assert [event["t"] for event in events] == ["tool_call", "tool_result"]
    assert all(event["data"]["tool_name"] == "read_file" for event in events)
    assert all(event["data"]["cursor_tool_family"] == "read_file" for event in events)
    assert all("cursor_instant" not in event["data"] for event in events)
    assert emitted == []


def test_hook_records_turn_stop_even_without_generation_id(tmp_path: Path, monkeypatch):
    """A `stop` with no generation_id still belongs in the event log.

    It is the only payload carrying the model and token counts. Without a
    generation_id there is nothing to correlate a turn span to, but dropping
    the event loses that data from local history too.
    """
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    exported = []
    monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *args: exported.append(args))
    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "cwd": "/repo",
            "hook_event_name": "stop",
            "model": "gpt-5",
            "input_tokens": 2,
            "output_tokens": 1,
        },
    )
    events = list(Store(Config.load()).reader("session-1").iter_events())
    assert [event["t"] for event in events] == ["turn_stop"]
    assert events[0]["data"]["model"] == "gpt-5"
    assert exported == []


def test_subagent_stop_dispatches_to_subagent_message(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    exported = []
    monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *args: exported.append(args))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": "subagentStop",
            "subagent_id": "agent-1",
            "subagent_type": "explore",
            "status": "completed",
        },
    )

    events = list(Store(Config.load()).reader("session-1").iter_events())
    assert len(events) == 1
    assert events[0]["t"] == "subagent_message"
    assert exported == []


def test_subagent_stop_preserves_generation_and_counts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    payload = {
        "conversation_id": "session-1",
        "generation_id": "generation-1",
        "cwd": "/repo",
        "hook_event_name": "subagentStop",
        "subagent_id": "agent-1",
        "subagent_type": "generalPurpose",
        "status": "completed",
        "task": "Inspect the authentication flow",
        "description": "Exploring authentication",
        "summary": "Located the relevant middleware",
        "duration_ms": 45_000,
        "message_count": 12,
        "tool_call_count": 8,
        "loop_count": 2,
        "modified_files": ["src/auth.py"],
        "agent_transcript_path": "/tmp/subagent-transcript.txt",
    }

    _invoke(monkeypatch, payload)

    events = list(Store(Config.load()).reader("session-1").iter_events())
    assert len(events) == 1
    assert events[0]["data"] == {
        "generation_id": "generation-1",
        "subagent_id": "agent-1",
        "subagent_type": "generalPurpose",
        "status": "completed",
        "task": "Inspect the authentication flow",
        "description": "Exploring authentication",
        "summary": "Located the relevant middleware",
        "duration_ms": 45_000,
        "message_count": 12,
        "tool_call_count": 8,
        "loop_count": 2,
        "modified_files": ["src/auth.py"],
        "agent_transcript_path": "/tmp/subagent-transcript.txt",
    }


def test_subagent_stop_without_conversation_id_is_noop(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": "subagentStop",
            "subagent_id": "agent-1",
            "subagent_type": "explore",
            "message_count": 3,
            "tool_call_count": 1,
            "loop_count": 0,
        },
    )

    assert list(Store(Config.load()).list_sessions()) == []
