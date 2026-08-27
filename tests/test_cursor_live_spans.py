from __future__ import annotations

from pathlib import Path

from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.live_spans import emit_live_tools
from thirdeye.platforms.cursor.tracing import build_turn
from thirdeye.span_ids import chat_span_id, turn_span_id
from thirdeye.store import Store


def _config(root: Path) -> Config:
    return Config(
        root=root,
        logfire=LogfireSettings(enabled=True, token="test-write-token"),
    )


def _append(store: Store, sid: str, event_type: str, data: dict) -> int:
    return store.append_event(
        session_id=sid, platform="cursor", cwd="/repo", t=event_type, data=data
    )


def test_completed_tool_is_emitted_live_once_and_omitted_at_stop(
    tmp_path: Path, monkeypatch
):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    turn_seq = _append(
        store, sid, "user_message", {"generation_id": generation, "prompt": "test"}
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "tool_call_id": "call-1",
            "command": "pytest",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "tool_call_id": "call-1",
            "output": "passed",
        },
    )
    exported: list[list[dict]] = []

    def capture(*args):
        exported.append(args[-1])
        return True

    monkeypatch.setattr("thirdeye.platforms.cursor.live_spans.export_spans", capture)
    sd = session_dir(tmp_path, "cursor", sid)
    emit_live_tools(config, sd, sid, "/repo", generation, result_seq)
    emit_live_tools(config, sd, sid, "/repo", generation, result_seq)

    assert len(exported) == 1
    span = exported[0][0]
    assert span["parent_span_id"] == chat_span_id(sid, generation)
    assert span["turn_span_id"] == str(turn_span_id(sid, turn_seq))
    assert span["tool_name"] == "shell"
    assert span["tool_call_id"] == "call-1"
    assert span["attributes"] == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "shell",
        "gen_ai.tool.call.id": "call-1",
        "gen_ai.tool.call.arguments": "pytest",
        "gen_ai.tool.call.result": "passed",
    }

    stop_seq = _append(
        store,
        sid,
        "turn_stop",
        {"generation_id": generation, "model": "gpt-5", "input_tokens": 2},
    )
    turn = build_turn(
        session_dir_=sd,
        session_id=sid,
        generation_id=generation,
        stop_seq=stop_seq,
    )
    assert turn is not None
    assert turn["llm_calls"][0]["tool_calls"] == []


def test_failed_live_dispatch_is_retried(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "test"})
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "Grep",
            "tool_use_id": "call-1",
            "cursor_instant": True,
            "tool_input": {"pattern": "TODO"},
            "tool_output": '{"matches": 3}',
            "duration": 25,
        },
    )
    outcomes = iter((False, True))
    dispatches = []

    def capture(*args):
        dispatches.append(args[-1])
        return next(outcomes)

    monkeypatch.setattr("thirdeye.platforms.cursor.live_spans.export_spans", capture)
    sd = session_dir(tmp_path, "cursor", sid)
    emit_live_tools(config, sd, sid, "/repo", generation, result_seq)
    emit_live_tools(config, sd, sid, "/repo", generation, result_seq)

    assert len(dispatches) == 2
    span = dispatches[-1][0]
    assert span["tool_call_id"] == "call-1"
    assert span["attributes"]["gen_ai.tool.call.arguments"] == {"pattern": "TODO"}
    assert span["attributes"]["gen_ai.tool.call.result"] == {"matches": 3}
    assert span["start_ts"] < span["end_ts"]
