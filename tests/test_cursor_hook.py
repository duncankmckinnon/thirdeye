from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import usage_log_path
from thirdeye.platforms.cursor import hook, tracing
from thirdeye.platforms.cursor.subagents import cursor_subagent_generation_id
from thirdeye.span_ids import tool_span_id, turn_span_id
from thirdeye.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


def _invoke(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert hook.main() == 0


def _events() -> list[dict]:
    return list(Store(Config.load()).reader("session-1").iter_events())


def _warning_entries(home: Path) -> list[dict]:
    log = usage_log_path(home)
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _capture_detached_jobs(tmp_path: Path, monkeypatch) -> list[Path]:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    Config(root=tmp_path).write_logfire_settings(LogfireSettings(enabled=True, token="test-token"))
    jobs: list[Path] = []
    monkeypatch.setattr("thirdeye.otel_export._spawn", jobs.append)
    return jobs


def _cursor_payload(event: str, generation_id: str = "parent-gen", **values) -> dict:
    return {
        "conversation_id": "session-1",
        "generation_id": generation_id,
        "cwd": "/repo",
        "hook_event_name": event,
        **values,
    }


def _job(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    ("event_name", "handler"),
    [
        ("SessionStart", hook._session_start),
        ("UserPromptSubmit", hook._before_submit),
        ("PostToolUse", hook._post_tool_use),
        ("SubagentStop", hook._subagent_stop),
        ("Stop", hook._stop),
    ],
)
def test_pascalcase_foreign_event_writes_nothing(
    tmp_path: Path,
    monkeypatch,
    event_name: str,
    handler,
):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    monkeypatch.setitem(hook._HANDLERS, event_name, handler)
    monkeypatch.setattr(hook, "capture_env", lambda patterns: {})
    exported = []
    monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *args: exported.append(args))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": event_name,
            "prompt": "#must-not-tag",
            "tool_name": "search_web",
            "result": {"matches": 3},
        },
    )

    assert list(Store(Config.load()).list_sessions()) == []
    assert exported == []


def test_pascalcase_foreign_event_logs_one_warning(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "hook_event_name": "SessionStart",
        },
    )

    entries = _warning_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["level"] == "warn"
    assert entries[0]["phase"] == "foreign_payload"
    assert entries[0]["platform"] == "cursor"
    assert entries[0]["session_id"] == "session-1"
    assert "SessionStart" in entries[0]["message"]


def test_pascalcase_foreign_event_still_prints_permissive_response(
    tmp_path: Path,
    monkeypatch,
    capfd,
):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "hook_event_name": "SessionStart",
        },
    )

    assert capfd.readouterr().out == '{"continue": true}'


def test_genuine_cursor_camelcase_event_records_unchanged(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "cwd": "/repo",
            "hook_event_name": "beforeSubmitPrompt",
            "prompt": "hello",
        },
    )

    events = _events()
    assert len(events) == 1
    assert events[0]["t"] == "user_message"
    assert events[0]["data"] == {
        "generation_id": "generation-1",
        "prompt": "hello",
    }
    assert _warning_entries(tmp_path) == []


def test_missing_event_name_remains_noop_without_warning(tmp_path: Path, monkeypatch, capfd):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "prompt": "must not be stored",
        },
    )

    assert capfd.readouterr().out == '{"continue": true}'
    assert list(Store(Config.load()).list_sessions()) == []
    assert _warning_entries(tmp_path) == []


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


def test_pre_tool_task_records_noninstant_call(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    payload = json.loads((FIXTURES / "cursor-pre-tool-use.json").read_text())

    _invoke(monkeypatch, payload)

    events = list(Store(Config.load()).reader("cursor-session-1").iter_events())
    assert len(events) == 1
    assert events[0]["t"] == "tool_call"
    assert events[0]["data"]["tool_name"] == "Task"
    assert events[0]["data"]["tool_use_id"] == "call-123"
    assert events[0]["data"]["tool_input"] == payload["tool_input"]
    assert events[0]["data"]["generation_id"] == "parent-generation-1"
    assert "cursor_instant" not in events[0]["data"]


@pytest.mark.parametrize(
    "tool_name", ["shell", "Shell", "read_file", "view", "edit_file", "mcp", "mcp_execution"]
)
def test_pre_tool_skips_dedicated_callbacks(tmp_path: Path, monkeypatch, tool_name: str):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "hook_event_name": "preToolUse",
            "tool_name": tool_name,
            "tool_use_id": "tool-1",
        },
    )

    assert list(Store(Config.load()).list_sessions()) == []


def test_generic_pre_post_tool_builds_pair(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    live_event_counts = []
    monkeypatch.setattr(hook, "_emit_live", lambda *_args: live_event_counts.append(len(_events())))
    common = {
        "conversation_id": "session-1",
        "generation_id": "generation-1",
        "cwd": "/repo",
        "tool_name": "search_web",
        "tool_use_id": "search-1",
    }

    _invoke(
        monkeypatch,
        {
            **common,
            "hook_event_name": "preToolUse",
            "tool_input": {"query": "sample"},
        },
    )
    assert live_event_counts == []

    _invoke(
        monkeypatch,
        {
            **common,
            "hook_event_name": "postToolUse",
            "result": {"matches": 3},
        },
    )

    events = _events()
    assert [event["t"] for event in events] == ["tool_call", "tool_result"]
    assert all(event["data"]["generation_id"] == "generation-1" for event in events)
    assert all(event["data"]["tool_use_id"] == "search-1" for event in events)
    assert all("cursor_instant" not in event["data"] for event in events)
    assert live_event_counts == [2]


def test_generic_post_without_pre_is_noninstant_result(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(hook, "_emit_live", lambda *args: emitted.append(args))

    _invoke(
        monkeypatch,
        {
            "conversation_id": "session-1",
            "generation_id": "generation-1",
            "hook_event_name": "postToolUse",
            "tool_name": "search_web",
            "tool_use_id": "search-1",
            "result": {"matches": 3},
        },
    )

    events = _events()
    assert len(events) == 1
    assert events[0]["t"] == "tool_result"
    assert events[0]["data"]["tool_use_id"] == "search-1"
    assert "cursor_instant" not in events[0]["data"]
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


def test_subagent_start_dispatches_fixture(tmp_path: Path, monkeypatch, capfd):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    payload = json.loads((FIXTURES / "cursor-subagent-start.json").read_text())

    _invoke(monkeypatch, payload)

    assert capfd.readouterr().out == '{"continue": true}'
    events = list(Store(Config.load()).reader("cursor-session-1").iter_events())
    assert len(events) == 1
    assert events[0]["t"] == "subagent_start"
    assert events[0]["data"] == {
        key: value
        for key, value in payload.items()
        if key not in {"conversation_id", "hook_event_name"}
    }


def test_subagent_start_accepts_camel_case(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    payload = {
        "hookEventName": "subagentStart",
        "conversationId": "session-1",
        "generationId": "parent-generation-1",
        "subagentId": "subagent-1",
        "subagentType": "explore",
        "task": "Inspect sample",
        "toolCallId": "call-123",
        "subagentModel": "claude-4-sonnet",
        "isParallelWorker": True,
        "gitBranch": "fixture-branch",
    }

    _invoke(monkeypatch, payload)

    event = _events()[0]
    assert event["t"] == "subagent_start"
    assert event["data"] == {
        "generation_id": "parent-generation-1",
        "subagentId": "subagent-1",
        "subagentType": "explore",
        "task": "Inspect sample",
        "toolCallId": "call-123",
        "subagentModel": "claude-4-sonnet",
        "isParallelWorker": True,
        "gitBranch": "fixture-branch",
    }


def test_subagent_start_without_conversation_id_is_noop(tmp_path: Path, monkeypatch, capfd):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "hook_event_name": "subagentStart",
            "generation_id": "parent-generation-1",
            "subagent_id": "subagent-1",
        },
    )

    assert capfd.readouterr().out == '{"continue": true}'
    assert list(Store(Config.load()).list_sessions()) == []


def test_subagent_start_accepts_partial_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _invoke(
        monkeypatch,
        {
            "hook_event_name": "subagentStart",
            "conversation_id": "session-1",
            "subagent_id": "subagent-1",
        },
    )

    event = _events()[0]
    assert event["t"] == "subagent_start"
    assert event["data"] == {"subagent_id": "subagent-1"}


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


def test_subagent_stop_strips_bogus_session_generation_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    payload = {
        "conversation_id": "session-1",
        "generation_id": "session-1",
        "cwd": "/repo",
        "hook_event_name": "subagentStop",
        "subagent_id": "agent-1",
        "subagent_type": "explore",
        "task": "work",
    }

    _invoke(monkeypatch, payload)

    events = list(Store(Config.load()).reader("session-1").iter_events())
    assert len(events) == 1
    assert "generation_id" not in events[0]["data"]


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


def test_subagent_stop_exports_under_dispatching_task(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    child_generation = cursor_subagent_generation_id("call-123")

    _invoke(monkeypatch, _cursor_payload("beforeSubmitPrompt", prompt="turn A"))
    _invoke(
        monkeypatch,
        _cursor_payload(
            "preToolUse", tool_name="Task", tool_use_id="call-123", tool_input={"task": "go"}
        ),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="child-1", tool_call_id="call-123", task="go"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload(
            "preToolUse",
            child_generation,
            tool_name="search_web",
            tool_use_id="child-tool",
        ),
    )
    start_seq = next(event["seq"] for event in _events() if event["t"] == "subagent_start")
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="child-1"))

    loaded = [_job(path) for path in jobs]
    task_job = next(job for job in loaded if job["kind"] == "spans")
    [task_span] = task_job["spans"]
    assert task_span["name"] == "tool: Task"
    assert int(task_span["span_id"]) == tool_span_id("cursor", "session-1", "call-123")
    job = next(job for job in loaded if job["kind"] == "subagent_turn")
    assert int(job["parent_span_id"]) == tool_span_id("cursor", "session-1", "call-123")
    assert job["turn"]["turn_id"] == str(start_seq)


def test_background_subagent_ignores_new_open_turn(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)

    _invoke(monkeypatch, _cursor_payload("beforeSubmitPrompt", prompt="turn A"))
    turn_a_seq = _events()[-1]["seq"]
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", tool_name="Task", tool_use_id="call-A"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="child-A", tool_call_id="call-A"),
    )
    _invoke(monkeypatch, _cursor_payload("stop", model="gpt-5"))
    _invoke(
        monkeypatch,
        _cursor_payload("beforeSubmitPrompt", "turn-B-gen", prompt="turn B"),
    )
    turn_b_seq = _events()[-1]["seq"]
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="child-A"))

    job = next(_job(path) for path in jobs if _job(path)["kind"] == "subagent_turn")
    parent_id = int(job["parent_span_id"])
    assert parent_id == tool_span_id("cursor", "session-1", "call-A")
    assert parent_id != turn_span_id("cursor", "session-1", turn_a_seq)
    assert parent_id != turn_span_id("cursor", "session-1", turn_b_seq)


def test_subagent_stop_can_precede_parent_task_post(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)

    _invoke(monkeypatch, _cursor_payload("beforeSubmitPrompt", prompt="turn A"))
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", tool_name="Task", tool_use_id="call-A"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="child-A", tool_call_id="call-A"),
    )
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="child-A"))

    child_job = next(_job(path) for path in jobs if _job(path)["kind"] == "subagent_turn")
    recorded_parent = int(child_job["parent_span_id"])
    _invoke(
        monkeypatch,
        _cursor_payload(
            "postToolUse", tool_name="Task", tool_use_id="call-A", result={"status": "done"}
        ),
    )
    task_job = next(_job(path) for path in jobs if _job(path)["kind"] == "spans")
    assert sum(_job(path)["kind"] == "spans" for path in jobs) == 1
    [task_span] = task_job["spans"]
    assert int(task_span["span_id"]) == recorded_parent
    assert recorded_parent == tool_span_id("cursor", "session-1", "call-A")


def test_nested_subagent_uses_nested_task_parent(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    outer_generation = cursor_subagent_generation_id("call-A")

    _invoke(monkeypatch, _cursor_payload("beforeSubmitPrompt", prompt="turn A"))
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", tool_name="Task", tool_use_id="call-A"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="outer", tool_call_id="call-A"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", outer_generation, tool_name="Task", tool_use_id="call-N"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload(
            "subagentStart",
            outer_generation,
            subagent_id="nested",
            tool_call_id="call-N",
        ),
    )
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="nested"))

    loaded = [_job(path) for path in jobs]
    child_job = next(job for job in loaded if job["kind"] == "subagent_turn")
    parent_id = int(child_job["parent_span_id"])
    assert parent_id == tool_span_id("cursor", "session-1", "call-N")
    assert parent_id != tool_span_id("cursor", "session-1", "call-A")
    task_span_ids = {
        int(span["span_id"]) for job in loaded if job["kind"] == "spans" for span in job["spans"]
    }
    assert parent_id in task_span_ids


def test_start_without_task_id_uses_exact_parent_generation_turn(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    exported = []
    monkeypatch.setattr(
        "thirdeye.otel_export.export_subagent_turn",
        lambda *args, **kwargs: exported.append((args, kwargs)),
    )

    _invoke(monkeypatch, _cursor_payload("beforeSubmitPrompt", prompt="turn A"))
    parent_turn_seq = _events()[-1]["seq"]
    _invoke(monkeypatch, _cursor_payload("subagentStart", subagent_id="child-A"))
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="child-A"))

    assert len(exported) == 1
    _, kwargs = exported[0]
    assert kwargs == {"parent_span_id": str(turn_span_id("cursor", "session-1", parent_turn_seq))}


def test_unresolved_modern_parent_logs_sanitized_diagnostic(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    sensitive_task = "secret task contents"
    sensitive_path = "/private/secret/transcript.jsonl"

    _invoke(
        monkeypatch,
        _cursor_payload(
            "subagentStart",
            "unknown-generation",
            subagent_id="child-unparented",
            task=sensitive_task,
        ),
    )
    _invoke(
        monkeypatch,
        _cursor_payload(
            "subagentStop",
            subagent_id="child-unparented",
            agent_transcript_path=sensitive_path,
        ),
    )

    entries = _warning_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["phase"] == "cursor_subagent_parent_resolution"
    assert entries[0]["platform"] == "cursor"
    assert entries[0]["session_id"] == "session-1"
    assert "child-unparented" in entries[0]["message"]
    assert sensitive_task not in entries[0]["message"]
    assert sensitive_path not in entries[0]["message"]


def test_duplicate_stop_writes_raw_event_but_spawns_one_job(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="child-A", tool_call_id="call-A"),
    )
    stop = _cursor_payload("subagentStop", subagent_id="child-A")

    _invoke(monkeypatch, stop)
    _invoke(monkeypatch, stop)

    assert [event["t"] for event in _events()] == [
        "subagent_start",
        "subagent_message",
        "subagent_message",
    ]
    assert len(jobs) == 1


def test_resume_task_exports_reused_subagent_as_new_run(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", tool_name="Task", tool_use_id="call-1"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="call-1", tool_call_id="call-1"),
    )
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="call-1"))
    _invoke(
        monkeypatch,
        _cursor_payload(
            "preToolUse",
            tool_name="Task",
            tool_use_id="call-2",
            tool_input={"resume": "agent-uuid", "prompt": "Continue the review"},
        ),
    )
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="call-2"))

    subagent_jobs = [_job(path) for path in jobs if _job(path)["kind"] == "subagent_turn"]
    assert len(subagent_jobs) == 2
    resumed = subagent_jobs[1]
    assert int(resumed["parent_span_id"]) == tool_span_id("cursor", "session-1", "call-2")
    assert resumed["turn"]["input_message"] == "Continue the review"
    task_jobs = [_job(path) for path in jobs if _job(path)["kind"] == "spans"]
    resume_task = next(
        span
        for job in task_jobs
        for span in job["spans"]
        if span["name"] == "tool: Task" and span["tool_call_id"] == "call-2"
    )
    assert int(resume_task["span_id"]) == int(resumed["parent_span_id"])


def test_resume_with_subagent_start_keeps_prompt(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", tool_name="Task", tool_use_id="call-1"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="call-1", tool_call_id="call-1"),
    )
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="call-1"))
    _invoke(
        monkeypatch,
        _cursor_payload(
            "preToolUse",
            tool_name="Task",
            tool_use_id="call-2",
            tool_input={"resume": "agent-uuid", "prompt": "Continue the review"},
        ),
    )
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="call-2", tool_call_id="call-2"),
    )
    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="call-2"))

    resumed = [_job(path) for path in jobs if _job(path)["kind"] == "subagent_turn"][1]
    assert resumed["turn"]["input_message"] == "Continue the review"
    assert int(resumed["parent_span_id"]) == tool_span_id("cursor", "session-1", "call-2")


def test_resolver_exception_is_fail_open(tmp_path: Path, monkeypatch, capfd):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="child-A", tool_call_id="call-A"),
    )
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.tracing.resolve_subagent_export",
        lambda *_args: (_ for _ in ()).throw(OSError("unreadable event log")),
    )

    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="child-A"))

    assert capfd.readouterr().out == '{"continue": true}{"continue": true}'
    assert [event["t"] for event in _events()] == ["subagent_start", "subagent_message"]


def test_transient_resolver_exception_retries_persisted_stop(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="child-A", tool_call_id="call-A"),
    )
    real_resolve = tracing.resolve_subagent_export
    attempts = 0

    def flaky_resolve(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient event snapshot failure")
        return real_resolve(*args)

    monkeypatch.setattr(
        "thirdeye.platforms.cursor.tracing.resolve_subagent_export",
        flaky_resolve,
    )

    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="child-A"))

    assert attempts == 2
    assert len(jobs) == 1


def test_subagent_exporter_exception_is_fail_open(tmp_path: Path, monkeypatch, capfd):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    _invoke(
        monkeypatch,
        _cursor_payload("subagentStart", subagent_id="child-A", tool_call_id="call-A"),
    )
    monkeypatch.setattr(
        "thirdeye.otel_export.export_subagent_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("export failed")),
    )

    _invoke(monkeypatch, _cursor_payload("subagentStop", subagent_id="child-A"))

    assert capfd.readouterr().out == '{"continue": true}{"continue": true}'
    assert _events()[-1]["t"] == "subagent_message"
