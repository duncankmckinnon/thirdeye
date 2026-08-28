from __future__ import annotations

import io
import json
from pathlib import Path

from thirdeye.config import Config
from thirdeye.platforms.cursor import hook
from thirdeye.store import Store


def _invoke(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert hook.main() == 0


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
