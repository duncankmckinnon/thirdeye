from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from thirdeye import otel_export
from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import otel_state_path, session_dir
from thirdeye.platforms.cursor import hook
from thirdeye.platforms.cursor.live_spans import (
    committed_interaction_ids,
    committed_tool_call_ids,
    emit_live_interactions,
    emit_live_tools,
    _interaction_state_entry,
)
from thirdeye.platforms.cursor.subagents import cursor_subagent_generation_id
from thirdeye.platforms.cursor.tracing import build_turn, resolve_subagent_export
from thirdeye.reader import SessionReader
from thirdeye.span_ids import (
    chat_span_id,
    interaction_span_id,
    tool_span_id,
    trace_id_for_session,
    turn_span_id,
)
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


def _canonical_interaction_id(
    generation: str, kind: str, source_seq: int, correlation_id: str = ""
) -> str:
    return f"{generation}:{kind}:{correlation_id or '-'}:{source_seq}"


def _live_state_path(sd: Path) -> Path:
    return sd / "cursor-live-state.json"


def _expected_interaction_attributes(
    *,
    kind: str,
    payload: dict,
    generation_id: str,
    source_type: str,
    source_seq: int,
    correlation_id: str = "",
    duplicate_seqs: tuple[int, ...] = (),
) -> dict:
    attributes = {
        "thirdeye.interaction.kind": kind,
        "thirdeye.interaction.payload": payload,
        "thirdeye.interaction.correlation_id": correlation_id,
        "thirdeye.interaction.source_type": source_type,
        "thirdeye.interaction.source_seq": source_seq,
        "thirdeye.interaction.generation_id": generation_id,
    }
    if duplicate_seqs:
        attributes["thirdeye.interaction.duplicate_seqs"] = list(duplicate_seqs)
    return attributes


def _spans_by_name(spans: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for span in spans:
        grouped.setdefault(span["name"], []).append(span)
    return grouped


def test_completed_tool_is_emitted_live_once_and_omitted_at_stop(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "test"})
    call_seq = _append(
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
        "thirdeye.event.call_seq": call_seq,
        "thirdeye.event.result_seq": result_seq,
        "thirdeye.tool.call.payload": {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "tool_call_id": "call-1",
            "command": "pytest",
        },
        "thirdeye.tool.result.payload": {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "tool_call_id": "call-1",
            "output": "passed",
        },
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


def test_live_tool_uses_persisted_trace_id(tmp_path: Path, monkeypatch):
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
            "tool_output": "match",
        },
    )
    sd = session_dir(tmp_path, "cursor", sid)
    persisted_trace_id = "0123456789abcdef0123456789abcdef"
    otel_state_path(sd).write_text(json.dumps({"trace_id": persisted_trace_id}))
    exports: list[tuple] = []

    def capture(*args):
        exports.append(args)
        return True

    monkeypatch.setattr("thirdeye.platforms.cursor.live_spans.export_spans", capture)

    emit_live_tools(config, sd, sid, "/repo", generation, result_seq)

    assert len(exports) == 1
    assert exports[0][-2] == int(persisted_trace_id, 16)


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


def test_live_export_skips_when_prompt_generation_mismatches_tools(tmp_path: Path, monkeypatch):
    """Tools must not create an agent-turn anchor when no user_message shares their generation."""
    config = _config(tmp_path)
    store = Store(config)
    sid = "cursor-session"
    prompt_gen, tool_gen = "gen-prompt", "gen-tools"
    _append(store, sid, "user_message", {"generation_id": prompt_gen, "prompt": "go"})
    tool_seq = _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": tool_gen,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "command": "ls",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": tool_gen,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "output": "ok",
        },
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    emit_live_tools(config, sd, sid, "/repo", tool_gen, result_seq)

    assert exported == []
    assert committed_tool_call_ids(sd, tool_gen) == set()
    assert tool_seq < result_seq


# --------------------------------------------------------------------------- #
# Live interaction sweep: reasoning, assistant messages, dedupe, retry, state,
# and exact parent IDs. `emit_live_interactions` generalizes the tool-only
# live path without wiring hooks yet.
# --------------------------------------------------------------------------- #


def test_live_reasoning_parents_to_turn_span(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    thought_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "consider options", "model": "gpt-5"},
    )
    exported: list[list[dict]] = []

    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    emit_live_interactions(config, sd, sid, "/repo", generation, thought_seq)

    assert len(exported) == 1
    span = exported[0][0]
    interaction_id = _canonical_interaction_id(generation, "reasoning", thought_seq)
    expected_turn_id = turn_span_id("cursor", sid, turn_seq)
    assert span["name"] == "reasoning"
    assert span["span_id"] == interaction_span_id("cursor", sid, interaction_id)
    assert span["parent_span_id"] == expected_turn_id
    assert span["turn_seq"] == turn_seq
    assert span["turn_span_id"] == str(expected_turn_id)
    assert span["attributes"] == _expected_interaction_attributes(
        kind="reasoning",
        payload={
            "generation_id": generation,
            "text": "consider options",
            "model": "gpt-5",
        },
        generation_id=generation,
        source_type="assistant_thought",
        source_seq=thought_seq,
    )
    assert committed_interaction_ids(sd, generation) == {interaction_id}


def test_duplicate_reasoning_emits_one_span_with_duplicate_seqs(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    timestamps = iter(
        [
            "1970-01-01T00:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
        ]
    )
    monkeypatch.setattr("thirdeye.writer._utc_iso_ms", lambda: next(timestamps))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    first_seq = _append(
        store,
        sid,
        "assistant_thought",
        {
            "generation_id": generation,
            "text": "plan",
            "model": "claude-4",
            "speed": "fast",
        },
    )
    duplicate_seq = _append(
        store,
        sid,
        "assistant_thought",
        {
            "generation_id": generation,
            "text": "plan",
            "model": "gpt-5",
            "model_id": "gpt-5.6",
            "speed": "slow",
        },
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)

    emit_live_interactions(config, sd, sid, "/repo", generation, duplicate_seq)

    assert len(exported) == 1
    assert len(exported[0]) == 1
    span = exported[0][0]
    interaction_id = _canonical_interaction_id(generation, "reasoning", first_seq)
    assert span["name"] == "reasoning"
    assert span["span_id"] == interaction_span_id("cursor", sid, interaction_id)
    assert span["attributes"] == _expected_interaction_attributes(
        kind="reasoning",
        payload={
            "generation_id": generation,
            "text": "plan",
            "model": "claude-4",
            "speed": "fast",
        },
        generation_id=generation,
        source_type="assistant_thought",
        source_seq=first_seq,
        duplicate_seqs=(duplicate_seq,),
    )
    assert committed_interaction_ids(sd, generation) == {interaction_id}


def test_same_reasoning_text_at_later_timestamp_emits_second_span(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    # First append_event also stamps SessionMeta.started_at from the same clock.
    timestamps = iter(
        [
            "1970-01-01T00:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
            "2026-09-02T12:00:00.100Z",
            "2026-09-02T12:00:01.000Z",
        ]
    )
    monkeypatch.setattr("thirdeye.writer._utc_iso_ms", lambda: next(timestamps))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    first_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan"},
    )
    second_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan"},
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)

    emit_live_interactions(config, sd, sid, "/repo", generation, first_seq)
    emit_live_interactions(config, sd, sid, "/repo", generation, second_seq)

    assert len(exported) == 2
    first_id = _canonical_interaction_id(generation, "reasoning", first_seq)
    second_id = _canonical_interaction_id(generation, "reasoning", second_seq)
    reasoning_spans = exported[0] + exported[1]
    assert {span["span_id"] for span in reasoning_spans} == {
        interaction_span_id("cursor", sid, first_id),
        interaction_span_id("cursor", sid, second_id),
    }
    assert committed_interaction_ids(sd, generation) == {first_id, second_id}


def test_same_reasoning_text_within_same_second_emits_second_span(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    timestamps = iter(
        [
            "1970-01-01T00:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
            "2026-09-02T12:00:00.100Z",
            "2026-09-02T12:00:00.900Z",
        ]
    )
    monkeypatch.setattr("thirdeye.writer._utc_iso_ms", lambda: next(timestamps))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    first_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan"},
    )
    second_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan"},
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)

    emit_live_interactions(config, sd, sid, "/repo", generation, first_seq)
    emit_live_interactions(config, sd, sid, "/repo", generation, second_seq)

    assert len(exported) == 2
    first_id = _canonical_interaction_id(generation, "reasoning", first_seq)
    second_id = _canonical_interaction_id(generation, "reasoning", second_seq)
    assert committed_interaction_ids(sd, generation) == {first_id, second_id}


def test_live_reasoning_reexports_when_duplicate_arrives_after_first_sweep(
    tmp_path: Path, monkeypatch
):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    timestamps = iter(
        [
            "1970-01-01T00:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
            "2026-09-02T12:00:00.000Z",
        ]
    )
    monkeypatch.setattr("thirdeye.writer._utc_iso_ms", lambda: next(timestamps))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    first_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan", "model": "claude-4"},
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    interaction_id = _canonical_interaction_id(generation, "reasoning", first_seq)

    emit_live_interactions(config, sd, sid, "/repo", generation, first_seq)
    assert len(exported) == 1
    assert "thirdeye.interaction.duplicate_seqs" not in exported[0][0]["attributes"]

    duplicate_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan", "model": "gpt-5"},
    )
    emit_live_interactions(config, sd, sid, "/repo", generation, duplicate_seq)

    assert len(exported) == 2
    assert exported[1][0]["span_id"] == interaction_span_id("cursor", sid, interaction_id)
    assert exported[1][0]["attributes"]["thirdeye.interaction.duplicate_seqs"] == [duplicate_seq]
    assert committed_interaction_ids(sd, generation) == {interaction_id}


def test_live_state_classifies_colon_containing_tool_and_interaction_ids(
    tmp_path: Path, monkeypatch
):
    config = _config(tmp_path)
    store = Store(config)
    sid = "cursor-session"
    generation = "gen:with:colons"
    tool_call_id = "x:reasoning:y:1"
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    thought_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "stateful"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "tool_call_id": tool_call_id,
            "command": "ls",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "tool_call_id": tool_call_id,
            "output": "ok",
        },
    )
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    emit_live_interactions(config, sd, sid, "/repo", generation, result_seq)

    interaction_id = _canonical_interaction_id(generation, "reasoning", thought_seq)
    assert committed_tool_call_ids(sd, generation) == {tool_call_id}
    assert committed_interaction_ids(sd, generation) == {interaction_id}
    persisted = json.loads(_live_state_path(sd).read_text())
    assert f"i:{interaction_id}" not in persisted[generation]
    assert _interaction_state_entry(interaction_id) in persisted[generation]
    assert tool_call_id in persisted[generation]


def test_live_state_treats_i_colon_prefixed_tool_id_as_tool_not_interaction(
    tmp_path: Path, monkeypatch
):
    config = _config(tmp_path)
    store = Store(config)
    sid = "cursor-session"
    generation = "generation-1"
    tool_call_id = "i:legacy-tool-like-id"
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "tool_call_id": tool_call_id,
            "command": "ls",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "tool_call_id": tool_call_id,
            "output": "ok",
        },
    )
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    state_path = _live_state_path(sd)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({generation: [tool_call_id]}, separators=(",", ":")))

    emit_live_interactions(config, sd, sid, "/repo", generation, result_seq)

    assert committed_tool_call_ids(sd, generation) == {tool_call_id}
    assert committed_interaction_ids(sd, generation) == set()
    persisted = json.loads(state_path.read_text())
    assert persisted[generation] == [tool_call_id]


def test_live_assistant_message_parents_to_turn_span(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    response_seq = _append(
        store,
        sid,
        "assistant_message",
        {"generation_id": generation, "text": "done", "parts": [{"type": "text", "content": "done"}]},
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    emit_live_interactions(config, sd, sid, "/repo", generation, response_seq)

    assert len(exported) == 1
    span = exported[0][0]
    interaction_id = _canonical_interaction_id(generation, "assistant_message", response_seq)
    expected_turn_id = turn_span_id("cursor", sid, turn_seq)
    assert span["name"] == "interaction: assistant_message"
    assert span["span_id"] == interaction_span_id("cursor", sid, interaction_id)
    assert span["parent_span_id"] == expected_turn_id
    assert span["attributes"] == _expected_interaction_attributes(
        kind="assistant_message",
        payload={
            "generation_id": generation,
            "text": "done",
            "parts": [{"type": "text", "content": "done"}],
        },
        generation_id=generation,
        source_type="assistant_message",
        source_seq=response_seq,
    )


def test_user_message_and_lifecycle_events_are_not_exported_live(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    user_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    lifecycle_seq = _append(
        store,
        sid,
        "subagent_start",
        {
            "generation_id": generation,
            "subagent_id": "child-1",
            "tool_call_id": "call-task",
        },
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation, "model": "gpt-5"})
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)

    emit_live_interactions(config, sd, sid, "/repo", generation, user_seq)
    emit_live_interactions(config, sd, sid, "/repo", generation, lifecycle_seq)
    emit_live_interactions(config, sd, sid, "/repo", generation, stop_seq)

    assert exported == []
    assert committed_interaction_ids(sd, generation) == set()


def test_live_interaction_sweep_parents_tools_under_chat_and_reasoning_under_turn(
    tmp_path: Path, monkeypatch
):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    thought_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "search first"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "Grep",
            "tool_use_id": "call-1",
            "tool_input": {"pattern": "TODO"},
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "Grep",
            "tool_use_id": "call-1",
            "tool_output": "match",
        },
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    emit_live_interactions(config, sd, sid, "/repo", generation, result_seq)

    assert len(exported) == 1
    by_name = _spans_by_name(exported[0])
    assert len(by_name["reasoning"]) == 1
    assert len(by_name["tool: Grep"]) == 1
    reasoning = by_name["reasoning"][0]
    tool = by_name["tool: Grep"][0]
    expected_turn_id = turn_span_id("cursor", sid, turn_seq)
    expected_chat_id = chat_span_id("cursor", sid, generation)
    reasoning_id = _canonical_interaction_id(generation, "reasoning", thought_seq)
    assert reasoning["parent_span_id"] == expected_turn_id
    assert reasoning["span_id"] == interaction_span_id("cursor", sid, reasoning_id)
    assert tool["parent_span_id"] == expected_chat_id
    assert tool["span_id"] == tool_span_id("cursor", sid, "call-1")
    assert tool["turn_span_id"] == str(expected_turn_id)
    assert committed_interaction_ids(sd, generation) == {reasoning_id}
    assert committed_tool_call_ids(sd, generation) == {"call-1"}


def test_live_interaction_sweep_is_idempotent(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    thought_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan"},
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)

    emit_live_interactions(config, sd, sid, "/repo", generation, thought_seq)
    emit_live_interactions(config, sd, sid, "/repo", generation, thought_seq)

    assert len(exported) == 1
    interaction_id = _canonical_interaction_id(generation, "reasoning", thought_seq)
    assert committed_interaction_ids(sd, generation) == {interaction_id}


def test_failed_live_interaction_dispatch_is_retried(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    thought_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "retry me"},
    )
    outcomes = iter((False, True))
    dispatches: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: dispatches.append(args[-1]) or next(outcomes),
    )
    sd = session_dir(tmp_path, "cursor", sid)
    interaction_id = _canonical_interaction_id(generation, "reasoning", thought_seq)

    emit_live_interactions(config, sd, sid, "/repo", generation, thought_seq)
    assert committed_interaction_ids(sd, generation) == set()
    emit_live_interactions(config, sd, sid, "/repo", generation, thought_seq)

    assert len(dispatches) == 2
    span = dispatches[-1][0]
    assert span["name"] == "reasoning"
    assert span["span_id"] == interaction_span_id("cursor", sid, interaction_id)
    assert committed_interaction_ids(sd, generation) == {interaction_id}


def test_live_interaction_state_keeps_generation_list_json_compatible(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    thought_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "stateful"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "tool_call_id": "call-1",
            "command": "ls",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "tool_call_id": "call-1",
            "output": "ok",
        },
    )
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    state_path = _live_state_path(sd)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({generation: ["legacy-tool"]}, separators=(",", ":")))

    emit_live_interactions(config, sd, sid, "/repo", generation, result_seq)

    persisted = json.loads(state_path.read_text())
    assert isinstance(persisted, dict)
    assert isinstance(persisted[generation], list)
    interaction_id = _canonical_interaction_id(generation, "reasoning", thought_seq)
    assert set(persisted[generation]) == {
        "legacy-tool",
        "call-1",
        _interaction_state_entry(interaction_id),
    }
    assert committed_tool_call_ids(sd, generation) == {"legacy-tool", "call-1"}
    assert committed_interaction_ids(sd, generation) == {interaction_id}


def test_live_interaction_uses_persisted_trace_id(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = Store(config)
    sid, generation = "cursor-session", "generation-1"
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    thought_seq = _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "trace"},
    )
    sd = session_dir(tmp_path, "cursor", sid)
    persisted_trace_id = "0123456789abcdef0123456789abcdef"
    otel_state_path(sd).write_text(json.dumps({"trace_id": persisted_trace_id}))
    exports: list[tuple] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exports.append(args) or True,
    )

    emit_live_interactions(config, sd, sid, "/repo", generation, thought_seq)

    assert len(exports) == 1
    assert exports[0][-2] == int(persisted_trace_id, 16)


# --------------------------------------------------------------------------- #
# Concurrency: parent / child / nested Cursor subagent exports race, but every
# deterministic span, claim, and committed-cursor entry lands exactly once.
# These follow `test_failed_live_dispatch_is_retried`: the local export/job
# boundary is monkeypatched and persisted state is inspected. `threading.Barrier`
# only aligns the calls whose ordering is under test; no assertion depends on
# which aligned thread wins.
# --------------------------------------------------------------------------- #


def _hook_env(tmp_path: Path, monkeypatch) -> list[Path]:
    """Wire a logfire-enabled home and capture detached OTel jobs as files."""
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    Config(root=tmp_path).write_logfire_settings(LogfireSettings(enabled=True, token="test-token"))
    jobs: list[Path] = []
    monkeypatch.setattr("thirdeye.otel_export._spawn", jobs.append)
    return jobs


def _cursor_payload(session_id: str, generation_id: str | None = "parent-gen", **values):
    payload = {"conversation_id": session_id, "cwd": "/repo", **values}
    if generation_id is not None:
        payload["generation_id"] = generation_id
    return payload


def _event_at(sd: Path, seq: int) -> dict:
    return next(iter(SessionReader(sd).iter_events(seq_range=(seq, seq + 1))))


def _run_aligned(fns, timeout: float = 5.0) -> None:
    """Start `fns` on threads aligned by a barrier; join each with a deadline."""
    barrier = threading.Barrier(len(fns))
    errors: list[BaseException] = []

    def wrap(fn):
        def inner() -> None:
            try:
                barrier.wait(timeout=timeout)
                fn()
            except BaseException as exc:  # noqa: BLE001 -- surfaced via assert below
                errors.append(exc)

        return inner

    threads = [threading.Thread(target=wrap(fn)) for fn in fns]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "aligned thread did not terminate"
    assert errors == [], f"aligned thread raised: {errors!r}"


def _worker_exporter(monkeypatch):
    """Configure a local OTel exporter for exercising queued child jobs."""
    logfire = pytest.importorskip("logfire")
    from logfire.testing import TestExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    otel_export._state["id_generator"] = None
    exporter = TestExporter()
    instance = logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
        advanced=logfire.AdvancedOptions(id_generator=otel_export._id_generator()),
    )
    monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)
    return exporter


def _run_subagent_job(job: dict) -> None:
    otel_export._export_subagent_turn_inner(
        config=Config.load(),
        session_dir_=Path(job["session_dir"]),
        session_id=job["session_id"],
        platform=job["platform"],
        cwd=job["cwd"],
        trace_id=job["trace_id"],
        parent_span_id=job["parent_span_id"],
        turn=job["turn"],
    )


def _run_spans_job(job: dict) -> None:
    otel_export._export_spans_batch(
        config=Config.load(),
        session_dir_=Path(job["session_dir"]),
        session_id=job["session_id"],
        platform=job["platform"],
        cwd=job["cwd"],
        trace_id=job["trace_id"],
        spans=job["spans"],
    )


def _span_with_id(spans: list[dict], span_id: int) -> dict:
    matches = [span for span in spans if span["context"]["span_id"] == span_id]
    assert len(matches) == 1
    return matches[0]


def test_child_generation_is_not_live_committed_without_user_turn(tmp_path: Path, monkeypatch):
    """Script 1: a child generic pre/post pair has no `user_message` of its own,
    so `emit_live_tools` must leave it uncommitted for `resolve_subagent_export`;
    the tool is still reconstructed exactly once when the child is built at stop.
    """
    config = _config(tmp_path)
    store = Store(config)
    sid = "cursor-session"
    child_generation = cursor_subagent_generation_id("call-123")

    _append(store, sid, "user_message", {"generation_id": "parent-gen", "prompt": "dispatch"})
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": "parent-gen",
            "tool_name": "Task",
            "tool_call_id": "call-123",
            "tool_input": {"task": "go"},
        },
    )
    start_seq = _append(
        store,
        sid,
        "subagent_start",
        {
            "generation_id": "parent-gen",
            "subagent_id": "child-1",
            "tool_call_id": "call-123",
            "task": "go",
        },
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": child_generation,
            "tool_name": "search_web",
            "cursor_tool_family": "search",
            "tool_use_id": "child-tool",
            "tool_input": {"q": "cursor"},
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": child_generation,
            "tool_name": "search_web",
            "cursor_tool_family": "search",
            "tool_use_id": "child-tool",
            "tool_output": "hit",
        },
    )
    stop_seq = _append(
        store, sid, "subagent_message", {"subagent_id": "child-1", "status": "completed"}
    )
    sd = session_dir(tmp_path, "cursor", sid)

    exported: list = []
    state_writes: list = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans._write_state",
        lambda *args: state_writes.append(args),
    )
    emit_live_tools(config, sd, sid, "/repo", child_generation, result_seq)

    assert exported == []
    assert state_writes == []
    assert committed_tool_call_ids(sd, child_generation) == set()

    resolved = resolve_subagent_export(sd, sid, _event_at(sd, stop_seq))
    assert resolved is not None
    assert resolved.tool_call_id == "call-123"
    assert resolved.turn["turn_id"] == str(start_seq)
    assert resolved.turn["turn_span_id"] == str(turn_span_id("cursor", sid, start_seq))
    llm_call = resolved.turn["llm_calls"][0]
    assert llm_call["call_id"] == child_generation
    assert [t["tool_call_id"] for t in llm_call["tool_calls"]] == ["child-tool"]

    exporter = _worker_exporter(monkeypatch)
    parent_id = tool_span_id("cursor", sid, "call-123")
    otel_export._export_subagent_turn_inner(
        config=config,
        session_dir_=sd,
        session_id=sid,
        platform="cursor",
        cwd="/repo",
        trace_id=trace_id_for_session("cursor", sid),
        parent_span_id=parent_id,
        turn=resolved.turn,
    )
    spans = exporter.exported_spans_as_dict()
    turn_span = _span_with_id(spans, turn_span_id("cursor", sid, start_seq))
    chat_span = _span_with_id(spans, chat_span_id("cursor", sid, child_generation))
    tool_span = _span_with_id(spans, tool_span_id("cursor", sid, "child-tool"))
    assert turn_span["parent"]["span_id"] == parent_id
    assert chat_span["parent"]["span_id"] == turn_span["context"]["span_id"]
    assert tool_span["parent"]["span_id"] == chat_span["context"]["span_id"]


def test_parent_task_post_racing_child_stop_keeps_deterministic_parent(tmp_path: Path, monkeypatch):
    """Script 2: the dispatching Task's live export and the child stop run on
    aligned threads. One Task tool span id is produced and the child-turn job's
    parent id equals it, regardless of job creation order.
    """
    jobs = _hook_env(tmp_path, monkeypatch)
    sid = "cursor-session"
    child_gen = cursor_subagent_generation_id("call-A")
    store = Store(Config.load())

    _append(store, sid, "user_message", {"generation_id": "parent-gen", "prompt": "turn A"})
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": "parent-gen", "tool_name": "Task", "tool_call_id": "call-A"},
    )
    start_seq = _append(
        store,
        sid,
        "subagent_start",
        {"generation_id": "parent-gen", "subagent_id": "child-A", "tool_call_id": "call-A"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": child_gen,
            "tool_name": "search_web",
            "cursor_tool_family": "search",
            "tool_use_id": "child-tool",
        },
    )
    _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": child_gen,
            "tool_name": "search_web",
            "cursor_tool_family": "search",
            "tool_use_id": "child-tool",
            "tool_output": "hit",
        },
    )
    sd = session_dir(tmp_path, "cursor", sid)

    _run_aligned(
        [
            lambda: hook._post_tool_use(
                _cursor_payload(
                    sid, tool_name="Task", tool_use_id="call-A", result={"status": "done"}
                )
            ),
            lambda: hook._subagent_stop(
                _cursor_payload(sid, generation_id=None, subagent_id="child-A")
            ),
        ]
    )

    loaded = [json.loads(path.read_text()) for path in jobs]
    spans_jobs = [job for job in loaded if job["kind"] == "spans"]
    sub_jobs = [job for job in loaded if job["kind"] == "subagent_turn"]

    task_tool_id = tool_span_id("cursor", sid, "call-A")
    assert len(spans_jobs) == 1
    task_spans = [s for s in spans_jobs[0]["spans"] if s["name"] == "tool: Task"]
    assert len(task_spans) == 1
    assert int(task_spans[0]["span_id"]) == task_tool_id
    assert int(task_spans[0]["parent_span_id"]) == chat_span_id("cursor", sid, "parent-gen")

    assert len(sub_jobs) == 1
    child_turn = sub_jobs[0]["turn"]
    assert int(sub_jobs[0]["parent_span_id"]) == task_tool_id
    assert child_turn["turn_id"] == str(start_seq)
    assert child_turn["turn_span_id"] == str(turn_span_id("cursor", sid, start_seq))
    assert child_turn["llm_calls"][0]["call_id"] == child_gen
    assert [t["tool_call_id"] for t in child_turn["llm_calls"][0]["tool_calls"]] == ["child-tool"]

    exporter = _worker_exporter(monkeypatch)
    _run_subagent_job(sub_jobs[0])
    emitted = exporter.exported_spans_as_dict()
    child_turn_span = _span_with_id(emitted, turn_span_id("cursor", sid, start_seq))
    child_chat_span = _span_with_id(emitted, chat_span_id("cursor", sid, child_gen))
    child_tool_span = _span_with_id(emitted, tool_span_id("cursor", sid, "child-tool"))
    assert child_turn_span["parent"]["span_id"] == task_tool_id
    assert child_chat_span["parent"]["span_id"] == child_turn_span["context"]["span_id"]
    assert child_tool_span["parent"]["span_id"] == child_chat_span["context"]["span_id"]

    assert committed_tool_call_ids(sd, "parent-gen") == {"call-A"}
    assert committed_tool_call_ids(sd, child_gen) == set()


def test_parallel_child_stops_do_not_cross_attribute_tools(tmp_path: Path, monkeypatch):
    """Script 3: two starts with distinct Task ids and derived generations stop
    on aligned threads. Two distinct child turns result, each parented to its own
    Task span, with no tool id owned by the other child.
    """
    jobs = _hook_env(tmp_path, monkeypatch)
    sid = "cursor-session"
    gen_a = cursor_subagent_generation_id("call-A")
    gen_b = cursor_subagent_generation_id("call-B")
    store = Store(Config.load())

    _append(store, sid, "user_message", {"generation_id": "parent-gen", "prompt": "fan out"})
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": "parent-gen", "tool_name": "Task", "tool_call_id": "call-A"},
    )
    start_a = _append(
        store,
        sid,
        "subagent_start",
        {"generation_id": "parent-gen", "subagent_id": "child-A", "tool_call_id": "call-A"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": "parent-gen", "tool_name": "Task", "tool_call_id": "call-B"},
    )
    start_b = _append(
        store,
        sid,
        "subagent_start",
        {"generation_id": "parent-gen", "subagent_id": "child-B", "tool_call_id": "call-B"},
    )
    for gen, tool_id in ((gen_a, "tool-A"), (gen_b, "tool-B")):
        _append(
            store,
            sid,
            "tool_call",
            {
                "generation_id": gen,
                "tool_name": "read_file",
                "cursor_tool_family": "read",
                "tool_use_id": tool_id,
            },
        )
        _append(
            store,
            sid,
            "tool_result",
            {
                "generation_id": gen,
                "tool_name": "read_file",
                "cursor_tool_family": "read",
                "tool_use_id": tool_id,
                "tool_output": "x",
            },
        )
    sd = session_dir(tmp_path, "cursor", sid)

    _run_aligned(
        [
            lambda: hook._subagent_stop(
                _cursor_payload(sid, generation_id=None, subagent_id="child-A")
            ),
            lambda: hook._subagent_stop(
                _cursor_payload(sid, generation_id=None, subagent_id="child-B")
            ),
        ]
    )

    sub_jobs = [
        job for path in jobs if (job := json.loads(path.read_text()))["kind"] == "subagent_turn"
    ]
    assert len(sub_jobs) == 2
    by_parent = {int(job["parent_span_id"]): job for job in sub_jobs}
    assert set(by_parent) == {
        tool_span_id("cursor", sid, "call-A"),
        tool_span_id("cursor", sid, "call-B"),
    }
    job_a = by_parent[tool_span_id("cursor", sid, "call-A")]
    job_b = by_parent[tool_span_id("cursor", sid, "call-B")]

    assert job_a["turn"]["turn_id"] == str(start_a)
    assert job_a["turn"]["turn_span_id"] == str(turn_span_id("cursor", sid, start_a))
    assert job_a["turn"]["llm_calls"][0]["call_id"] == gen_a
    assert [t["tool_call_id"] for t in job_a["turn"]["llm_calls"][0]["tool_calls"]] == ["tool-A"]

    assert job_b["turn"]["turn_id"] == str(start_b)
    assert job_b["turn"]["turn_span_id"] == str(turn_span_id("cursor", sid, start_b))
    assert job_b["turn"]["llm_calls"][0]["call_id"] == gen_b
    assert [t["tool_call_id"] for t in job_b["turn"]["llm_calls"][0]["tool_calls"]] == ["tool-B"]

    assert start_a != start_b
    assert turn_span_id("cursor", sid, start_a) != turn_span_id("cursor", sid, start_b)

    exporter = _worker_exporter(monkeypatch)
    for job in sub_jobs:
        _run_subagent_job(job)
    emitted = exporter.exported_spans_as_dict()
    for start_seq, generation, task_id, child_tool_id in (
        (start_a, gen_a, "call-A", "tool-A"),
        (start_b, gen_b, "call-B", "tool-B"),
    ):
        turn_span = _span_with_id(emitted, turn_span_id("cursor", sid, start_seq))
        chat_span = _span_with_id(emitted, chat_span_id("cursor", sid, generation))
        tool_span = _span_with_id(emitted, tool_span_id("cursor", sid, child_tool_id))
        assert turn_span["parent"]["span_id"] == tool_span_id("cursor", sid, task_id)
        assert chat_span["parent"]["span_id"] == turn_span["context"]["span_id"]
        assert tool_span["parent"]["span_id"] == chat_span["context"]["span_id"]


def test_nested_and_outer_stop_race_have_independent_claims(tmp_path: Path, monkeypatch):
    """Script 4: the full nested lifecycle is stored, then the nested and outer
    stops run on aligned threads. Each child turn keys a distinct
    `subagent:<turn_id>` claim and the nested turn parents to its own Task span.
    """
    jobs = _hook_env(tmp_path, monkeypatch)
    sid = "cursor-session"
    gen_a = cursor_subagent_generation_id("call-A")
    gen_n = cursor_subagent_generation_id("call-N")
    store = Store(Config.load())

    _append(store, sid, "user_message", {"generation_id": "parent-gen", "prompt": "root"})
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": "parent-gen", "tool_name": "Task", "tool_call_id": "call-A"},
    )
    outer_start = _append(
        store,
        sid,
        "subagent_start",
        {"generation_id": "parent-gen", "subagent_id": "outer", "tool_call_id": "call-A"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": gen_a,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "tool_use_id": "read-A",
        },
    )
    _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": gen_a,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "tool_use_id": "read-A",
            "tool_output": "y",
        },
    )
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": gen_a, "tool_name": "Task", "tool_call_id": "call-N"},
    )
    nested_start = _append(
        store,
        sid,
        "subagent_start",
        {"generation_id": gen_a, "subagent_id": "nested", "tool_call_id": "call-N"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": gen_n,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "tool_use_id": "nested-read",
        },
    )
    _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": gen_n,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            "tool_use_id": "nested-read",
            "tool_output": "z",
        },
    )
    _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": gen_a,
            "tool_name": "Task",
            "tool_call_id": "call-N",
            "tool_output": "nested done",
        },
    )
    sd = session_dir(tmp_path, "cursor", sid)

    _run_aligned(
        [
            lambda: hook._subagent_stop(
                _cursor_payload(sid, generation_id=None, subagent_id="nested")
            ),
            lambda: hook._subagent_stop(
                _cursor_payload(sid, generation_id=None, subagent_id="outer")
            ),
        ]
    )

    loaded = [json.loads(path.read_text()) for path in jobs]
    sub_jobs = [job for job in loaded if job["kind"] == "subagent_turn"]
    assert len(sub_jobs) == 2
    by_parent = {int(job["parent_span_id"]): job for job in sub_jobs}
    outer_tool_id = tool_span_id("cursor", sid, "call-A")
    nested_tool_id = tool_span_id("cursor", sid, "call-N")
    assert nested_tool_id != outer_tool_id
    assert set(by_parent) == {outer_tool_id, nested_tool_id}
    outer_job = by_parent[outer_tool_id]
    nested_job = by_parent[nested_tool_id]

    assert outer_job["turn"]["turn_id"] == str(outer_start)
    assert outer_job["turn"]["turn_span_id"] == str(turn_span_id("cursor", sid, outer_start))
    assert outer_job["turn"]["llm_calls"][0]["call_id"] == gen_a
    assert [t["tool_call_id"] for t in outer_job["turn"]["llm_calls"][0]["tool_calls"]] == [
        "read-A"
    ]
    task_span_ids = {
        int(span["span_id"]) for job in loaded if job["kind"] == "spans" for span in job["spans"]
    }
    assert task_span_ids == {outer_tool_id, nested_tool_id}

    assert nested_job["turn"]["turn_id"] == str(nested_start)
    assert nested_job["turn"]["turn_span_id"] == str(turn_span_id("cursor", sid, nested_start))
    assert nested_job["turn"]["llm_calls"][0]["call_id"] == gen_n
    assert [t["tool_call_id"] for t in nested_job["turn"]["llm_calls"][0]["tool_calls"]] == [
        "nested-read"
    ]

    outer_claim = f"subagent:{outer_start}"
    nested_claim = f"subagent:{nested_start}"
    exporter = _worker_exporter(monkeypatch)
    for job in loaded:
        if job["kind"] == "spans":
            _run_spans_job(job)
    for job in sub_jobs:
        _run_subagent_job(job)

    assert otel_export._turn_claim_path(sd, outer_claim).read_text() == "sent"
    assert otel_export._turn_claim_path(sd, nested_claim).read_text() == "sent"
    emitted = exporter.exported_spans_as_dict()
    outer_turn_span = _span_with_id(emitted, turn_span_id("cursor", sid, outer_start))
    outer_chat_span = _span_with_id(emitted, chat_span_id("cursor", sid, gen_a))
    read_a_span = _span_with_id(emitted, tool_span_id("cursor", sid, "read-A"))
    nested_task_span = _span_with_id(emitted, nested_tool_id)
    nested_turn_span = _span_with_id(emitted, turn_span_id("cursor", sid, nested_start))
    nested_chat_span = _span_with_id(emitted, chat_span_id("cursor", sid, gen_n))
    nested_read_span = _span_with_id(emitted, tool_span_id("cursor", sid, "nested-read"))

    assert outer_turn_span["parent"]["span_id"] == outer_tool_id
    assert outer_chat_span["parent"]["span_id"] == outer_turn_span["context"]["span_id"]
    assert read_a_span["parent"]["span_id"] == outer_chat_span["context"]["span_id"]
    assert nested_task_span["parent"]["span_id"] == outer_chat_span["context"]["span_id"]
    assert nested_turn_span["parent"]["span_id"] == nested_tool_id
    assert nested_chat_span["parent"]["span_id"] == nested_turn_span["context"]["span_id"]
    assert nested_read_span["parent"]["span_id"] == nested_chat_span["context"]["span_id"]


def test_failed_child_job_dispatch_is_retryable(tmp_path: Path, monkeypatch):
    """Script 5: the first local job/export boundary raises before any claim or
    committed-cursor entry becomes permanent; repeating the same logical event
    writes exactly one usable job and commits the parent Task exactly once.
    """
    jobs = _hook_env(tmp_path, monkeypatch)
    sid = "cursor-session"
    child_gen = cursor_subagent_generation_id("call-A")
    store = Store(Config.load())

    _append(store, sid, "user_message", {"generation_id": "parent-gen", "prompt": "turn A"})
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": "parent-gen", "tool_name": "Task", "tool_call_id": "call-A"},
    )
    start_seq = _append(
        store,
        sid,
        "subagent_start",
        {"generation_id": "parent-gen", "subagent_id": "child-A", "tool_call_id": "call-A"},
    )
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": child_gen,
            "tool_name": "search_web",
            "cursor_tool_family": "search",
            "tool_use_id": "child-tool",
        },
    )
    _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": child_gen,
            "tool_name": "search_web",
            "cursor_tool_family": "search",
            "tool_use_id": "child-tool",
            "tool_output": "hit",
        },
    )
    stop_seq = _append(
        store, sid, "subagent_message", {"subagent_id": "child-A", "status": "completed"}
    )
    sd = session_dir(tmp_path, "cursor", sid)
    config = Config.load()

    resolved = resolve_subagent_export(sd, sid, _event_at(sd, stop_seq))
    assert resolved is not None
    claim_id = f"subagent:{resolved.turn['turn_id']}"

    real_write_job = otel_export._write_job
    attempts = {"n": 0}

    def flaky_write_job(home, payload):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("disk full")
        return real_write_job(home, payload)

    monkeypatch.setattr(otel_export, "_write_job", flaky_write_job)

    otel_export.export_subagent_turn(
        config, sd, sid, "cursor", "/repo", resolved.turn, tool_use_id=resolved.tool_call_id
    )
    assert jobs == []
    assert not otel_export._turn_claim_path(sd, claim_id).exists()

    otel_export.export_subagent_turn(
        config, sd, sid, "cursor", "/repo", resolved.turn, tool_use_id=resolved.tool_call_id
    )
    assert len(jobs) == 1
    job = json.loads(jobs[0].read_text())
    assert job["kind"] == "subagent_turn"
    assert int(job["parent_span_id"]) == tool_span_id("cursor", sid, "call-A")
    assert job["turn"]["turn_id"] == str(start_seq)
    assert job["turn"]["turn_span_id"] == str(turn_span_id("cursor", sid, start_seq))
    assert job["turn"]["llm_calls"][0]["call_id"] == child_gen
    assert [t["tool_call_id"] for t in job["turn"]["llm_calls"][0]["tool_calls"]] == ["child-tool"]
    assert not otel_export.turn_export_sent(sd, claim_id)
    assert not otel_export._turn_claim_path(sd, claim_id).exists()

    exporter = _worker_exporter(monkeypatch)
    _run_subagent_job(job)
    assert otel_export.turn_export_sent(sd, claim_id)
    emitted = exporter.exported_spans_as_dict()
    turn_span = _span_with_id(emitted, turn_span_id("cursor", sid, start_seq))
    chat_span = _span_with_id(emitted, chat_span_id("cursor", sid, child_gen))
    tool_span = _span_with_id(emitted, tool_span_id("cursor", sid, "child-tool"))
    assert turn_span["parent"]["span_id"] == tool_span_id("cursor", sid, "call-A")
    assert chat_span["parent"]["span_id"] == turn_span["context"]["span_id"]
    assert tool_span["parent"]["span_id"] == chat_span["context"]["span_id"]

    # The dispatching Task's live tool commit is likewise retried, never marked
    # committed by the failed attempt.
    outcomes = iter((False, True))
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: next(outcomes),
    )
    task_result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": "parent-gen",
            "tool_name": "Task",
            "tool_call_id": "call-A",
            "tool_output": "done",
        },
    )
    emit_live_tools(config, sd, sid, "/repo", "parent-gen", task_result_seq)
    assert committed_tool_call_ids(sd, "parent-gen") == set()
    emit_live_tools(config, sd, sid, "/repo", "parent-gen", task_result_seq)
    assert committed_tool_call_ids(sd, "parent-gen") == {"call-A"}
    assert committed_tool_call_ids(sd, child_gen) == set()


def test_task_parent_emits_when_task_event_has_no_generation_id(tmp_path: Path, monkeypatch):
    from thirdeye.platforms.cursor.live_spans import emit_task_parent_span

    config = _config(tmp_path)
    store = Store(config)
    sid = "cursor-session"
    user_seq = _append(store, sid, "user_message", {"generation_id": "parent-gen", "prompt": "go"})
    task_seq = _append(
        store,
        sid,
        "tool_call",
        {"tool_name": "Task", "tool_use_id": "call-orphan", "tool_input": {"prompt": "child"}},
    )
    exported: list[list[dict]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.live_spans.export_spans",
        lambda *args: exported.append(args[-1]) or True,
    )
    sd = session_dir(tmp_path, "cursor", sid)
    emit_task_parent_span(config, sd, sid, "/repo", "call-orphan", task_seq)

    assert len(exported) == 1
    span = exported[0][0]
    assert span["name"] == "tool: Task"
    assert int(span["span_id"]) == tool_span_id("cursor", sid, "call-orphan")
    assert span["turn_seq"] in {user_seq, task_seq}
