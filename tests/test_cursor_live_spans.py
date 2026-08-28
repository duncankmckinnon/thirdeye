from __future__ import annotations

from pathlib import Path

from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.live_spans import committed_tool_call_ids, emit_live_tools
from thirdeye.platforms.cursor.tracing import build_turn
from thirdeye.span_ids import chat_span_id, tool_span_id, trace_id_for_session, turn_span_id
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


def test_completed_tool_is_emitted_live_once_and_omitted_at_stop(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "test"})
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
    assert span["parent_span_id"] == chat_span_id("cursor", sid, generation)
    assert span["turn_span_id"] == str(turn_span_id("cursor", sid, turn_seq))
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


def test_live_tool_parent_matches_cursor_chat_and_turn_ids(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation, call_id = "shared-session", "generation-1", "call-1"
    turn_seq = _append(
        store,
        sid,
        "user_message",
        {"generation_id": generation, "prompt": "read the file"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "tool_call_id": call_id,
            "file_path": "README.md",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "tool_call_id": call_id,
            "output": "contents",
        },
    )
    exports: list[tuple] = []

    def capture(*args):
        exports.append(args)
        return True

    monkeypatch.setattr("thirdeye.platforms.cursor.live_spans.export_spans", capture)

    emit_live_tools(
        config,
        session_dir(tmp_path, "cursor", sid),
        sid,
        "/repo",
        generation,
        result_seq,
    )

    assert len(exports) == 1
    export = exports[0]
    span = export[-1][0]
    assert export[-2] == trace_id_for_session("cursor", sid)
    assert span["span_id"] == tool_span_id("cursor", sid, call_id)
    assert span["parent_span_id"] == chat_span_id("cursor", sid, generation)
    assert span["turn_span_id"] == str(turn_span_id("cursor", sid, turn_seq))


def test_live_read_commit_prevents_stop_duplicate(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-read"
    _append(
        store,
        sid,
        "user_message",
        {"generation_id": generation, "prompt": "read both files"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "file_path": "src/thirdeye/store.py",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "file_path": "src/thirdeye/store.py",
            "output": "class Store: ...",
        },
    )
    exported: list[list[dict]] = []

    def capture(*args):
        exported.append(args[-1])
        return True

    monkeypatch.setattr("thirdeye.platforms.cursor.live_spans.export_spans", capture)
    sd = session_dir(tmp_path, "cursor", sid)

    emit_live_tools(config, sd, sid, "/repo", generation, result_seq)

    assert len(exported) == 1
    assert len(exported[0]) == 1
    live_tool = exported[0][0]
    assert live_tool["tool_name"] == "read_file"
    assert live_tool["attributes"]["gen_ai.tool.call.arguments"] == "src/thirdeye/store.py"
    assert live_tool["attributes"]["gen_ai.tool.call.result"] == "class Store: ..."
    assert committed_tool_call_ids(sd, generation) == {live_tool["tool_call_id"]}

    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
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
