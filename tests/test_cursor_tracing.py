from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.subagents import cursor_subagent_generation_id
from thirdeye.platforms.cursor.tracing import (
    build_turn,
    resolve_subagent_export,
    usage_from_payload,
)
from thirdeye.span_ids import turn_span_id
from thirdeye.store import Store


def _append(store: Store, sid: str, event_type: str, data: dict) -> int:
    return store.append_event(
        session_id=sid, platform="cursor", cwd="/repo", t=event_type, data=data
    )


def test_usage_from_payload_treats_cursor_input_tokens_as_cache_inclusive():
    """Cursor ``stop`` reports ``input_tokens`` as the turn total, cache included."""
    assert usage_from_payload(
        {
            "input_tokens": 1_180_993,
            "output_tokens": 8146,
            "cache_read_tokens": 1_007_022,
            "cache_write_tokens": 173_957,
        }
    ) == {
        "input_tokens": 1_180_993,
        "output_tokens": 8146,
        "cache_read_input_tokens": 1_007_022,
        "cache_creation_input_tokens": 173_957,
    }


def test_usage_from_payload_without_cache_fields_passes_input_through():
    assert usage_from_payload({"input_tokens": 42, "output_tokens": 7}) == {
        "input_tokens": 42,
        "output_tokens": 7,
    }


def test_build_turn_uses_otel_gen_ai_tool_conventions(tmp_path: Path):
    sid, generation = "cursor-session", "gen-1"
    config = Config(root=tmp_path)
    store = Store(config)
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "run tests"})
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "command": "pytest",
        },
    )
    _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "output": "passed",
            "exit_code": 0,
        },
    )
    _append(
        store,
        sid,
        "assistant_message",
        {"generation_id": generation, "text": "All tests passed", "model": "claude-4"},
    )
    stop_seq = _append(
        store,
        sid,
        "turn_stop",
        {
            "generation_id": generation,
            "model": "claude-4",
            "input_tokens": 33,
            "output_tokens": 5,
            "cache_read_tokens": 20,
            "cache_write_tokens": 3,
        },
    )
    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id=generation,
        stop_seq=stop_seq,
    )
    assert turn is not None
    assert turn["input_message"] == "run tests"
    assert turn["output_message"] == "All tests passed"
    call = turn["llm_calls"][0]
    assert call["provider"] == "anthropic"
    assert call["usage"]["input_tokens"] == 33
    tool = call["tool_calls"][0]
    assert tool["attributes"]["gen_ai.operation.name"] == "execute_tool"
    assert tool["attributes"]["gen_ai.tool.name"] == "shell"
    assert tool["attributes"]["gen_ai.tool.call.arguments"] == "pytest"
    assert tool["attributes"]["gen_ai.tool.call.result"] == "passed"
    assert not any(key.startswith("openinference") for key in tool["attributes"])


def test_build_turn_uses_cursor_scoped_turn_id(tmp_path: Path):
    sid, generation = "shared-session", "gen-1"
    store = Store(Config(root=tmp_path))
    turn_seq = _append(
        store,
        sid,
        "user_message",
        {"generation_id": generation, "prompt": "scope this turn"},
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id=generation,
        stop_seq=stop_seq,
    )

    assert turn is not None
    assert turn["turn_span_id"] == str(turn_span_id("cursor", sid, turn_seq))


def test_build_turn_ignores_other_generations(tmp_path: Path):
    sid = "cursor-session"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": "old", "prompt": "old prompt"})
    _append(store, sid, "user_message", {"generation_id": "new", "prompt": "new prompt"})
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": "new", "model": "gpt-5"})
    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id="new",
        stop_seq=stop_seq,
    )
    assert turn is not None
    assert turn["input_message"] == "new prompt"
    assert turn["llm_calls"][0]["provider"] == "openai"


def test_build_turn_uses_open_prompt_when_stop_generation_is_successor(tmp_path: Path):
    """Cursor Stop often labels the next loop; live tools already used the prompt gen."""
    from thirdeye.span_ids import chat_span_id

    sid = "cursor-session"
    store = Store(Config(root=tmp_path))
    prompt_seq = _append(
        store, sid, "user_message", {"generation_id": "prompt-gen", "prompt": "go"}
    )
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": "prompt-gen", "tool_name": "Grep", "tool_use_id": "call-g"},
    )
    _append(
        store,
        sid,
        "assistant_message",
        {"generation_id": "prompt-gen", "text": "done"},
    )
    stop_seq = _append(
        store,
        sid,
        "turn_stop",
        {
            "generation_id": "successor-gen",
            "model": "cursor-grok-4.6-medium",
            "input_tokens": 10,
            "output_tokens": 2,
        },
    )
    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id="successor-gen",
        stop_seq=stop_seq,
    )
    assert turn is not None
    assert turn["turn_id"] == str(prompt_seq)
    assert turn["input_message"] == "go"
    assert turn["llm_calls"][0]["call_id"] == "prompt-gen"
    assert turn["llm_calls"][0]["usage"]["input_tokens"] == 10
    assert chat_span_id("cursor", sid, "prompt-gen") != chat_span_id("cursor", sid, "successor-gen")


def test_build_turn_keeps_tools_when_generation_has_no_llm_signal(tmp_path: Path):
    """A generation carrying only tool activity still exports its tool calls.

    Cursor can close a generation with a bare `turn_stop` (no model, no usage)
    and no assistant message. The tools still happened and must not vanish.
    """
    sid, generation = "cursor-session", "gen-tools-only"
    store = Store(Config(root=tmp_path))
    _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "command": "ls",
        },
    )
    _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "output": "README.md",
        },
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id=generation,
        stop_seq=stop_seq,
    )
    assert turn is not None
    assert len(turn["llm_calls"]) == 1
    tools = turn["llm_calls"][0]["tool_calls"]
    assert [tool["name"] for tool in tools] == ["shell"]
    assert tools[0]["attributes"]["gen_ai.tool.call.arguments"] == "ls"


def _shell_call(store: Store, sid: str, generation: str, **data) -> int:
    return _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            **data,
        },
    )


def _shell_result(store: Store, sid: str, generation: str, **data) -> int:
    return _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            **data,
        },
    )


def _read_call(store: Store, sid: str, generation: str, **data) -> int:
    return _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            **data,
        },
    )


def _read_result(store: Store, sid: str, generation: str, **data) -> int:
    return _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "read_file",
            "cursor_tool_family": "read",
            **data,
        },
    )


def _pairs(turn) -> list[tuple]:
    return [
        (
            tool["attributes"].get("gen_ai.tool.call.arguments"),
            tool["attributes"].get("gen_ai.tool.call.result"),
        )
        for tool in turn["llm_calls"][0]["tool_calls"]
    ]


def _build(tmp_path: Path, sid: str, generation: str, stop_seq: int):
    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id=generation,
        stop_seq=stop_seq,
    )
    assert turn is not None
    return turn


def test_same_family_tools_pair_on_payload_signature(tmp_path: Path):
    """When the result echoes the command, pairing is exact regardless of order."""
    sid, generation = "cursor-session", "gen-sig"
    store = Store(Config(root=tmp_path))
    _shell_call(store, sid, generation, command="make build")
    _shell_call(store, sid, generation, command="make test")
    _shell_result(store, sid, generation, command="make build", output="built")
    _shell_result(store, sid, generation, command="make test", output="tested")
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    assert sorted(_pairs(_build(tmp_path, sid, generation, stop_seq))) == [
        ("make build", "built"),
        ("make test", "tested"),
    ]


def test_unlabelled_same_family_tools_pair_in_dispatch_order(tmp_path: Path):
    """With no echoed signature, fall back to dispatch order, not reverse order.

    Cursor's shell callbacks carry no tool call id, so sequential completion
    (the common case) must pair first-in with first-out.
    """
    sid, generation = "cursor-session", "gen-fifo"
    store = Store(Config(root=tmp_path))
    _shell_call(store, sid, generation, command="first")
    _shell_call(store, sid, generation, command="second")
    _shell_result(store, sid, generation, output="out-first")
    _shell_result(store, sid, generation, output="out-second")
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    assert _pairs(_build(tmp_path, sid, generation, stop_seq)) == [
        ("first", "out-first"),
        ("second", "out-second"),
    ]


def test_read_pair_builds_exactly_one_tool_span(tmp_path: Path):
    sid, generation = "cursor-session", "gen-read"
    store = Store(Config(root=tmp_path))
    _read_call(store, sid, generation, file_path="src/thirdeye/store.py")
    _read_result(
        store,
        sid,
        generation,
        file_path="src/thirdeye/store.py",
        output="file contents",
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    tools = _build(tmp_path, sid, generation, stop_seq)["llm_calls"][0]["tool_calls"]

    assert len(tools) == 1
    assert tools[0]["name"] == "read_file"


def test_read_pair_contains_arguments_and_result(tmp_path: Path):
    sid, generation = "cursor-session", "gen-read-values"
    store = Store(Config(root=tmp_path))
    _read_call(store, sid, generation, filePath="docs/architecture.md")
    _read_result(
        store,
        sid,
        generation,
        filePath="docs/architecture.md",
        result={"contents": "# Architecture\n", "truncated": False},
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    tool = _build(tmp_path, sid, generation, stop_seq)["llm_calls"][0]["tool_calls"][0]

    assert tool["attributes"]["gen_ai.tool.call.arguments"] == "docs/architecture.md"
    assert tool["attributes"]["gen_ai.tool.call.result"] == {
        "contents": "# Architecture\n",
        "truncated": False,
    }


def test_concurrent_reads_pair_by_file_path(tmp_path: Path):
    """Every supported path spelling disambiguates reversed read completions."""
    sid, generation = "cursor-session", "gen-concurrent-reads"
    store = Store(Config(root=tmp_path))
    reads = [
        ("file_path", "src/first.py", "first contents"),
        ("filePath", "src/second.py", "second contents"),
        ("path", "src/third.py", "third contents"),
    ]
    for key, path, _result in reads:
        _read_call(store, sid, generation, **{key: path})
    for key, path, result in reversed(reads):
        _read_result(store, sid, generation, output=result, **{key: path})
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    assert _pairs(_build(tmp_path, sid, generation, stop_seq)) == [
        ("src/third.py", "third contents"),
        ("src/second.py", "second contents"),
        ("src/first.py", "first contents"),
    ]


def test_read_paths_pair_across_supported_key_spellings(tmp_path: Path):
    sid, generation = "cursor-session", "gen-read-path-spellings"
    store = Store(Config(root=tmp_path))
    _read_call(store, sid, generation, file_path="src/first.py")
    _read_call(store, sid, generation, filePath="src/second.py")
    _read_result(store, sid, generation, path="src/second.py", output="second contents")
    _read_result(store, sid, generation, path="src/first.py", output="first contents")
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    assert _pairs(_build(tmp_path, sid, generation, stop_seq)) == [
        ("src/second.py", "second contents"),
        ("src/first.py", "first contents"),
    ]


def test_read_explicit_call_id_takes_precedence_over_path(tmp_path: Path):
    sid, generation = "cursor-session", "gen-read-call-id-precedence"
    store = Store(Config(root=tmp_path))
    _read_call(store, sid, generation, tool_call_id="call-1", path="src/first.py")
    _read_call(store, sid, generation, tool_call_id="call-2", path="src/second.py")
    _read_result(
        store,
        sid,
        generation,
        tool_call_id="call-2",
        path="src/first.py",
        output="second contents",
    )
    _read_result(
        store,
        sid,
        generation,
        tool_call_id="call-1",
        path="src/second.py",
        output="first contents",
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    assert _pairs(_build(tmp_path, sid, generation, stop_seq)) == [
        ("src/second.py", "second contents"),
        ("src/first.py", "first contents"),
    ]


def test_reads_without_echoed_path_pair_fifo(tmp_path: Path):
    sid, generation = "cursor-session", "gen-read-fifo"
    store = Store(Config(root=tmp_path))
    _read_call(store, sid, generation, file_path="src/first.py")
    _read_call(store, sid, generation, file_path="src/second.py")
    _read_result(store, sid, generation, output="first contents")
    _read_result(store, sid, generation, output="second contents")
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    assert _pairs(_build(tmp_path, sid, generation, stop_seq)) == [
        ("src/first.py", "first contents"),
        ("src/second.py", "second contents"),
    ]


# --- subagents ---------------------------------------------------------------


def _subagent_stop(store: Store, sid: str, generation: str, **data) -> int:
    return _append(store, sid, "subagent_message", {"generation_id": generation, **data})


def _subagent_start(store: Store, sid: str, generation: str, **data) -> int:
    return _append(store, sid, "subagent_start", {"generation_id": generation, **data})


def _stored_event(store: Store, sid: str, seq: int) -> dict:
    return next(event for event in store.reader(sid).iter_events() if event["seq"] == seq)


def _resolve(store: Store, tmp_path: Path, sid: str, stop_seq: int):
    return resolve_subagent_export(
        session_dir(tmp_path, "cursor", sid), sid, _stored_event(store, sid, stop_seq)
    )


def _modern_lifecycle(
    store: Store,
    sid: str,
    *,
    parent_generation: str = "parent-gen",
    tool_call_id: str = "call-A",
    subagent_id: str = "child-A",
    start: dict | None = None,
    stop: dict | None = None,
) -> tuple[int, int, str]:
    start_data = {"subagent_id": subagent_id, "task": "Inspect auth", **(start or {})}
    if tool_call_id:
        start_data["tool_call_id"] = tool_call_id
    start_seq = _subagent_start(store, sid, parent_generation, **start_data)
    generation = cursor_subagent_generation_id(tool_call_id)
    stop_seq = _subagent_stop(
        store, sid, parent_generation, subagent_id=subagent_id, **(stop or {})
    )
    return start_seq, stop_seq, generation


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)


def _only_subagent(turn) -> dict:
    assert len(turn["subagents"]) == 1
    return turn["subagents"][0]


class TestModernSubagentTurn:
    def test_start_stop_build_one_aggregate_call(self, tmp_path: Path):
        sid = "modern-basic"
        store = Store(Config(root=tmp_path))
        start_seq, stop_seq, child_gen = _modern_lifecycle(store, sid)

        resolved = _resolve(store, tmp_path, sid, stop_seq)

        assert resolved is not None
        turn = resolved.turn
        assert turn["turn_id"] == str(start_seq)
        assert turn["turn_span_id"] == str(turn_span_id("cursor", sid, start_seq))
        assert turn["input_message"] == "Inspect auth"
        assert turn["subagents"] == []
        assert len(turn["llm_calls"]) == 1
        assert turn["llm_calls"][0]["call_id"] == child_gen

    def test_task_and_owned_response_take_precedence(self, tmp_path: Path):
        sid, call_id = "modern-precedence", "call-precedence"
        store = Store(Config(root=tmp_path))
        transcript = tmp_path / "precedence.jsonl"
        transcript.write_text(
            "\n".join(
                json.dumps(
                    {
                        "role": role,
                        "message": {"content": [{"type": "text", "text": text}]},
                    }
                )
                for role, text in (
                    ("user", "transcript task"),
                    ("assistant", "transcript response"),
                )
            )
        )
        _subagent_start(
            store, sid, "parent", subagent_id="child", tool_call_id=call_id, task="hook task"
        )
        child_gen = cursor_subagent_generation_id(call_id)
        _append(
            store, sid, "assistant_message", {"generation_id": child_gen, "text": "hook response"}
        )
        stop_seq = _subagent_stop(
            store,
            sid,
            "parent",
            subagent_id="child",
            summary="summary",
            agent_transcript_path=str(transcript),
        )

        turn = _resolve(store, tmp_path, sid, stop_seq).turn
        assert (turn["input_message"], turn["output_message"]) == ("hook task", "hook response")

    def test_transcript_text_is_fallback(self, tmp_path: Path):
        transcript = tmp_path / "child.jsonl"
        transcript.write_text(
            "\n".join(
                json.dumps({"role": role, "message": {"content": [{"type": "text", "text": text}]}})
                for role, text in (
                    ("user", "transcript input"),
                    ("assistant", "first"),
                    ("assistant", "last output"),
                )
            )
        )
        sid = "modern-transcript"
        store = Store(Config(root=tmp_path))
        _subagent_start(store, sid, "parent", subagent_id="child", tool_call_id="call-t")
        stop_seq = _subagent_stop(
            store, sid, "parent", subagent_id="child", agent_transcript_path=str(transcript)
        )

        turn = _resolve(store, tmp_path, sid, stop_seq).turn
        assert (turn["input_message"], turn["output_message"]) == (
            "transcript input",
            "last output",
        )

    def test_summary_is_final_output_fallback(self, tmp_path: Path):
        sid = "modern-summary"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(store, sid, stop={"summary": "done"})
        assert _resolve(store, tmp_path, sid, stop_seq).turn["output_message"] == "done"

    def test_missing_transcript_does_not_drop_hooks(self, tmp_path: Path):
        sid, call_id = "modern-missing-transcript", "call-missing"
        store = Store(Config(root=tmp_path))
        _subagent_start(
            store, sid, "parent", subagent_id="child", tool_call_id=call_id, task="task"
        )
        child_gen = cursor_subagent_generation_id(call_id)
        _read_call(store, sid, child_gen, tool_use_id="read-1", path="a.py")
        _read_result(store, sid, child_gen, tool_use_id="read-1", output="contents")
        stop_seq = _subagent_stop(
            store,
            sid,
            "parent",
            subagent_id="child",
            agent_transcript_path=str(tmp_path / "missing"),
        )

        turn = _resolve(store, tmp_path, sid, stop_seq).turn
        assert [tool["name"] for tool in turn["llm_calls"][0]["tool_calls"]] == ["read_file"]

    def test_lifecycle_timestamps_bound_turn_and_call(self, tmp_path: Path):
        sid = "modern-time"
        store = Store(Config(root=tmp_path))
        start_seq, stop_seq, _ = _modern_lifecycle(store, sid)
        start_event, stop_event = (
            _stored_event(store, sid, start_seq),
            _stored_event(store, sid, stop_seq),
        )

        turn = _resolve(store, tmp_path, sid, stop_seq).turn
        assert (turn["start_ts"], turn["end_ts"]) == (start_event["ts"], stop_event["ts"])
        assert (turn["llm_calls"][0]["start_ts"], turn["llm_calls"][0]["end_ts"]) == (
            start_event["ts"],
            stop_event["ts"],
        )

    def test_actual_token_fields_are_preserved(self, tmp_path: Path):
        sid = "modern-usage"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(
            store, sid, stop={"input_tokens": 10, "outputTokens": 3, "cache_read_tokens": 4}
        )
        assert _resolve(store, tmp_path, sid, stop_seq).turn["llm_calls"][0]["usage"] == {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
        }

    def test_absent_tokens_produce_empty_usage(self, tmp_path: Path):
        sid = "modern-no-usage"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(store, sid)
        assert _resolve(store, tmp_path, sid, stop_seq).turn["llm_calls"][0]["usage"] == {}

    def test_model_and_provider_precedence(self, tmp_path: Path):
        sid = "modern-model"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(
            store,
            sid,
            start={"subagent_model": "claude-sonnet-4", "model": "gpt-5"},
            stop={"model": "gemini-2.5"},
        )
        call = _resolve(store, tmp_path, sid, stop_seq).turn["llm_calls"][0]
        assert (call["model"], call["provider"]) == ("claude-sonnet-4", "anthropic")

    @pytest.mark.parametrize(
        ("status", "expected"),
        [("completed", "completed"), ("failed", "errored"), ("aborted", "interrupted")],
    )
    def test_completed_errored_and_interrupted_statuses(
        self, tmp_path: Path, status: str, expected: str
    ):
        sid = f"modern-status-{status}"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(store, sid, stop={"status": status})
        assert _resolve(store, tmp_path, sid, stop_seq).turn["status"] == expected

    def test_attributes_exclude_transcript_and_routing_fields(self, tmp_path: Path):
        sid = "modern-attrs"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(
            store,
            sid,
            start={
                "subagent_type": "explore",
                "description": "start description",
                "subagent_model": "gpt-5",
                "is_parallel_worker": True,
                "message_count": 1,
            },
            stop={
                "description": "stop description",
                "message_count": 9,
                "tool_call_count": 4,
                "loop_count": 2,
                "git_branch": "feature",
                "error_message": "boom",
                "modified_files": ["a.py"],
                "agent_transcript_path": "/secret/path",
                "conversation_id": "private",
            },
        )
        attrs = _resolve(store, tmp_path, sid, stop_seq).turn["attributes"]
        assert attrs == {
            "cursor.subagent.id": "child-A",
            "cursor.subagent.type": "explore",
            "cursor.subagent.description": "start description",
            "cursor.subagent.model": "gpt-5",
            "cursor.subagent.git_branch": "feature",
            "cursor.subagent.error_message": "boom",
            "cursor.subagent.message_count": 9,
            "cursor.subagent.tool_call_count": 4,
            "cursor.subagent.loop_count": 2,
            "cursor.subagent.is_parallel_worker": True,
            "cursor.subagent.modified_files": ["a.py"],
        }
        assert not any("transcript" in key or "conversation" in key for key in attrs)


class TestSubagentToolOwnership:
    def _with_tools(self, tmp_path: Path, events: list[tuple[str, str, dict]]):
        sid, call_id = "ownership", "call-owner"
        store = Store(Config(root=tmp_path))
        _subagent_start(
            store, sid, "parent", subagent_id="child", tool_call_id=call_id, task="work"
        )
        child_gen = cursor_subagent_generation_id(call_id)
        for event_type, generation, data in events:
            _append(
                store,
                sid,
                event_type,
                {"generation_id": child_gen if generation == "child" else generation, **data},
            )
        stop_seq = _subagent_stop(store, sid, "parent", subagent_id="child")
        return _resolve(store, tmp_path, sid, stop_seq).turn

    def test_exact_derived_generation_owns_tool(self, tmp_path: Path):
        turn = self._with_tools(
            tmp_path,
            [
                ("tool_call", "child", {"tool_name": "read_file", "path": "owned"}),
                ("tool_result", "child", {"tool_name": "read_file", "output": "yes"}),
            ],
        )
        assert [tool["name"] for tool in turn["llm_calls"][0]["tool_calls"]] == ["read_file"]

    def test_parent_generation_tool_is_excluded(self, tmp_path: Path):
        turn = self._with_tools(
            tmp_path,
            [
                ("tool_call", "parent", {"tool_name": "shell", "command": "no"}),
                ("tool_result", "parent", {"tool_name": "shell", "output": "no"}),
            ],
        )
        assert turn["llm_calls"][0]["tool_calls"] == []

    def test_sibling_generation_tool_is_excluded(self, tmp_path: Path):
        sid = "ownership-concurrent"
        store = Store(Config(root=tmp_path))
        gen_a = cursor_subagent_generation_id("call-A")
        gen_b = cursor_subagent_generation_id("call-B")
        _subagent_start(store, sid, "parent", subagent_id="A", tool_call_id="call-A", task="A")
        _subagent_start(store, sid, "parent", subagent_id="B", tool_call_id="call-B", task="B")
        _read_call(store, sid, gen_a, tool_use_id="read-A", path="a.py")
        _shell_call(store, sid, gen_b, tool_use_id="shell-B", command="echo B")
        _read_result(store, sid, gen_a, tool_use_id="read-A", output="A")
        _shell_result(store, sid, gen_b, tool_use_id="shell-B", output="B")
        stop_b = _subagent_stop(store, sid, "parent", subagent_id="B")
        stop_a = _subagent_stop(store, sid, "parent", subagent_id="A")

        tools_a = _resolve(store, tmp_path, sid, stop_a).turn["llm_calls"][0]["tool_calls"]
        tools_b = _resolve(store, tmp_path, sid, stop_b).turn["llm_calls"][0]["tool_calls"]

        assert [(tool["name"], tool["tool_call_id"]) for tool in tools_a] == [
            ("read_file", "read-A")
        ]
        assert [(tool["name"], tool["tool_call_id"]) for tool in tools_b] == [("shell", "shell-B")]
        assert gen_a != gen_b

    def test_matching_generation_outside_lifecycle_is_excluded(self, tmp_path: Path):
        sid, call_id = "ownership-bounds", "call-bounds"
        child_gen = cursor_subagent_generation_id(call_id)
        store = Store(Config(root=tmp_path))
        _read_call(store, sid, child_gen, path="before")
        _subagent_start(
            store, sid, "parent", subagent_id="child", tool_call_id=call_id, task="work"
        )
        stop_seq = _subagent_stop(store, sid, "parent", subagent_id="child")
        _read_result(store, sid, child_gen, output="after")
        assert _resolve(store, tmp_path, sid, stop_seq).turn["llm_calls"][0]["tool_calls"] == []

    def test_generic_pre_post_pair_preserves_arguments_result_and_id(self, tmp_path: Path):
        turn = self._with_tools(
            tmp_path,
            [
                (
                    "tool_call",
                    "child",
                    {
                        "tool_name": "Task",
                        "tool_use_id": "task-1",
                        "tool_input": {"prompt": "delegate"},
                    },
                ),
                (
                    "tool_result",
                    "child",
                    {
                        "tool_name": "Task",
                        "tool_use_id": "task-1",
                        "tool_output": {"result": "done"},
                    },
                ),
            ],
        )
        tool = turn["llm_calls"][0]["tool_calls"][0]
        assert tool["tool_call_id"] == "task-1"
        assert tool["attributes"]["gen_ai.tool.call.arguments"] == {"prompt": "delegate"}
        assert tool["attributes"]["gen_ai.tool.call.result"] == {"result": "done"}

    def test_post_without_pre_creates_self_timed_tool(self, tmp_path: Path):
        turn = self._with_tools(
            tmp_path,
            [
                (
                    "tool_result",
                    "child",
                    {
                        "tool_name": "mcp.search",
                        "tool_use_id": "mcp-1",
                        "tool_output": "found",
                        "duration": 25,
                    },
                )
            ],
        )
        tool = turn["llm_calls"][0]["tool_calls"][0]
        assert tool["name"] == "mcp.search"
        assert tool["tool_call_id"] == "mcp-1"
        assert _parse(tool["end_ts"]) > _parse(tool["start_ts"])

    def test_transcript_tool_use_does_not_create_tool_span(self, tmp_path: Path):
        transcript = tmp_path / "tools.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "name": "shell", "input": {"command": "no"}}
                        ]
                    },
                }
            )
        )
        sid = "ownership-transcript"
        store = Store(Config(root=tmp_path))
        _subagent_start(
            store, sid, "parent", subagent_id="child", tool_call_id="call-t", task="work"
        )
        stop_seq = _subagent_stop(
            store, sid, "parent", subagent_id="child", agent_transcript_path=str(transcript)
        )
        assert _resolve(store, tmp_path, sid, stop_seq).turn["llm_calls"][0]["tool_calls"] == []


class TestSubagentExportResolution:
    def test_task_id_is_preferred_parent(self, tmp_path: Path):
        sid = "resolve-task"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(store, sid, tool_call_id="task-parent")
        resolved = _resolve(store, tmp_path, sid, stop_seq)
        assert (resolved.tool_call_id, resolved.parent_turn_seq) == ("task-parent", None)

    def test_missing_task_id_resolves_start_generation_turn(self, tmp_path: Path):
        sid, generation = "resolve-turn", "parent"
        store = Store(Config(root=tmp_path))
        turn_seq = _append(
            store, sid, "user_message", {"generation_id": generation, "prompt": "go"}
        )
        _, stop_seq, _ = _modern_lifecycle(
            store, sid, parent_generation=generation, tool_call_id=""
        )
        resolved = _resolve(store, tmp_path, sid, stop_seq)
        assert (resolved.tool_call_id, resolved.parent_turn_seq) == ("", turn_seq)

    def test_bogus_parent_generation_is_unresolved(self, tmp_path: Path):
        sid = "resolve-bogus"
        store = Store(Config(root=tmp_path))
        _, stop_seq, _ = _modern_lifecycle(store, sid, parent_generation=sid, tool_call_id="")
        assert _resolve(store, tmp_path, sid, stop_seq).parent_turn_seq is None

    def test_background_stop_ignores_later_user_turn(self, tmp_path: Path):
        sid = "resolve-background"
        store = Store(Config(root=tmp_path))
        turn_a = _append(store, sid, "user_message", {"generation_id": "a", "prompt": "A"})
        _subagent_start(store, sid, "a", subagent_id="child", task="work")
        _append(store, sid, "user_message", {"generation_id": "b", "prompt": "B"})
        stop_seq = _subagent_stop(store, sid, "b", subagent_id="child")
        assert _resolve(store, tmp_path, sid, stop_seq).parent_turn_seq == turn_a

    def test_duplicate_stop_returns_none(self, tmp_path: Path):
        sid = "resolve-duplicate"
        store = Store(Config(root=tmp_path))
        _, _, _ = _modern_lifecycle(store, sid)
        duplicate = _subagent_stop(store, sid, "parent", subagent_id="child-A")
        assert _resolve(store, tmp_path, sid, duplicate) is None

    def test_unmatched_stop_returns_none(self, tmp_path: Path):
        sid = "resolve-unmatched"
        store = Store(Config(root=tmp_path))
        stop_seq = _subagent_stop(store, sid, "parent", subagent_id="missing")
        assert _resolve(store, tmp_path, sid, stop_seq) is None

    def test_reused_subagent_id_resolves_target_stop_by_sequence(self, tmp_path: Path):
        sid = "resolve-reused"
        store = Store(Config(root=tmp_path))
        _subagent_start(store, sid, "parent", subagent_id="same", tool_call_id="call-1", task="one")
        first_stop = _subagent_stop(store, sid, "parent", subagent_id="same")
        second_start = _subagent_start(
            store, sid, "parent", subagent_id="same", tool_call_id="call-2", task="two"
        )
        second_stop = _subagent_stop(store, sid, "parent", subagent_id="same")
        first = _resolve(store, tmp_path, sid, first_stop)
        second = _resolve(store, tmp_path, sid, second_stop)
        assert first.tool_call_id == "call-1"
        assert (second.tool_call_id, second.turn["turn_id"]) == ("call-2", str(second_start))

    def test_resumed_subagent_uses_resume_task_as_new_start(self, tmp_path: Path):
        sid = "resolve-resume"
        store = Store(Config(root=tmp_path))
        _modern_lifecycle(store, sid, subagent_id="call-1", tool_call_id="call-1")
        resume_seq = _append(
            store,
            sid,
            "tool_call",
            {
                "generation_id": "parent",
                "tool_name": "Task",
                "tool_use_id": "call-2",
                "tool_input": {"resume": "agent-uuid", "prompt": "Continue the review"},
            },
        )
        resumed_generation = cursor_subagent_generation_id("call-2")
        _append(
            store,
            sid,
            "assistant_message",
            {"generation_id": resumed_generation, "text": "Review complete"},
        )
        resumed_stop = _subagent_stop(store, sid, "parent", subagent_id="call-2")

        resolved = _resolve(store, tmp_path, sid, resumed_stop)

        assert resolved is not None
        assert resolved.tool_call_id == "call-2"
        assert resolved.turn["turn_id"] == str(resume_seq)
        assert resolved.turn["input_message"] == "Continue the review"
        assert resolved.turn["output_message"] == "Review complete"
        assert resolved.turn["llm_calls"][0]["call_id"] == resumed_generation


class TestSubagentNesting:
    def test_nested_task_and_child_use_distinct_exact_generations(self, tmp_path: Path):
        sid, outer_call, nested_call = "nested", "call-outer", "call-nested"
        outer_gen = cursor_subagent_generation_id(outer_call)
        nested_gen = cursor_subagent_generation_id(nested_call)
        store = Store(Config(root=tmp_path))
        _subagent_start(
            store, sid, "parent", subagent_id="outer", tool_call_id=outer_call, task="outer"
        )
        _append(
            store,
            sid,
            "tool_call",
            {
                "generation_id": outer_gen,
                "tool_name": "Task",
                "tool_use_id": nested_call,
                "tool_input": {"prompt": "nested"},
            },
        )
        _subagent_start(
            store, sid, outer_gen, subagent_id="nested", tool_call_id=nested_call, task="nested"
        )
        _read_call(store, sid, nested_gen, tool_use_id="nested-read", path="nested.py")
        _read_result(store, sid, nested_gen, tool_use_id="nested-read", output="nested contents")
        nested_stop = _subagent_stop(store, sid, outer_gen, subagent_id="nested")
        _append(
            store,
            sid,
            "tool_result",
            {
                "generation_id": outer_gen,
                "tool_name": "Task",
                "tool_use_id": nested_call,
                "tool_output": "done",
            },
        )
        outer_stop = _subagent_stop(store, sid, "parent", subagent_id="outer")

        outer = _resolve(store, tmp_path, sid, outer_stop).turn
        nested = _resolve(store, tmp_path, sid, nested_stop).turn
        assert outer_gen != nested_gen
        assert outer["llm_calls"][0]["tool_calls"] == []
        assert [tool["tool_call_id"] for tool in nested["llm_calls"][0]["tool_calls"]] == [
            "nested-read"
        ]
        assert outer["subagents"] == nested["subagents"] == []


class TestLegacySubagentCompatibility:
    def test_stop_duration_start_fallback(self, tmp_path: Path):
        sid, generation = "legacy-duration", "gen"
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
        sub_seq = _subagent_stop(
            store, sid, generation, subagent_id="legacy", task="Work", duration_ms=45_000
        )
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
        leaf = _only_subagent(_build(tmp_path, sid, generation, stop_seq))
        assert leaf["end_ts"] == _stored_event(store, sid, sub_seq)["ts"]
        assert (_parse(leaf["end_ts"]) - _parse(leaf["start_ts"])).total_seconds() == 45

    def test_summary_only_leaf_construction(self, tmp_path: Path):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "cursor-subagent-stop.json").read_text()
        )["data"]
        sid, generation = fixture["conversation_id"], fixture["generation_id"]
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
        sub_seq = _subagent_stop(
            store,
            sid,
            generation,
            **{key: value for key, value in fixture.items() if key != "generation_id"},
        )
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
        leaf = _only_subagent(_build(tmp_path, sid, generation, stop_seq))
        assert leaf["turn_span_id"] == str(turn_span_id("cursor", sid, sub_seq))
        assert leaf["input_message"] == fixture["task"] and leaf["output_message"] == ""
        assert leaf["llm_calls"] == []
        assert leaf["attributes"]["cursor.subagent.model"] == fixture["model"]

    def test_exact_parent_generation_filtering(self, tmp_path: Path):
        sid = "legacy-filter"
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": "new", "prompt": "delegate"})
        _subagent_stop(store, sid, "old", subagent_id="old", task="old")
        _subagent_stop(store, sid, "new", subagent_id="new", task="new")
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": "new"})
        assert (
            _only_subagent(_build(tmp_path, sid, "new", stop_seq))["attributes"][
                "cursor.subagent.id"
            ]
            == "new"
        )

    def test_one_time_parent_embedding_and_modern_exclusion(self, tmp_path: Path):
        sid, generation = "legacy-once", "parent"
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
        _subagent_stop(store, sid, generation, subagent_id="legacy", task="legacy")
        _subagent_start(
            store, sid, generation, subagent_id="modern", tool_call_id="call-modern", task="modern"
        )
        _subagent_stop(store, sid, generation, subagent_id="modern")
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
        turn = _build(tmp_path, sid, generation, stop_seq)
        assert [leaf["attributes"]["cursor.subagent.id"] for leaf in turn["subagents"]] == [
            "legacy"
        ]

    def test_duplicate_modern_stop_is_not_embedded_as_legacy(self, tmp_path: Path):
        sid, generation = "modern-duplicate-parent", "parent"
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
        _modern_lifecycle(store, sid, parent_generation=generation)
        _subagent_stop(store, sid, generation, subagent_id="child-A")
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

        assert _build(tmp_path, sid, generation, stop_seq)["subagents"] == []

    @pytest.mark.parametrize(
        ("status", "expected"),
        [("failure", "errored"), ("cancelled", "interrupted"), ("completed", "completed")],
    )
    def test_legacy_status_mapping(self, tmp_path: Path, status: str, expected: str):
        sid, generation = f"legacy-{status}", "parent"
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
        _subagent_stop(store, sid, generation, subagent_id="legacy", task="work", status=status)
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
        assert _only_subagent(_build(tmp_path, sid, generation, stop_seq))["status"] == expected


def test_build_turn_without_user_message_returns_none(tmp_path: Path):
    sid, generation = "cursor-session", "gen-tools-only"
    store = Store(Config(root=tmp_path))
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": generation, "tool_name": "shell", "command": "ls"},
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    assert (
        build_turn(
            session_dir_=session_dir(tmp_path, "cursor", sid),
            session_id=sid,
            generation_id=generation,
            stop_seq=stop_seq,
        )
        is None
    )


def test_build_turn_skips_bogus_stop_generation_id(tmp_path: Path):
    sid = "cursor-session"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": "real-gen", "prompt": "hi"})
    stop_seq = _append(
        store,
        sid,
        "turn_stop",
        {"generation_id": sid, "model": "cursor-grok-4.6-medium", "input_tokens": 4},
    )

    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id=sid,
        stop_seq=stop_seq,
    )
    assert turn is not None
    assert turn["input_message"] == "hi"
    assert turn["llm_calls"][0]["call_id"] == "real-gen"
    assert turn["llm_calls"][0]["usage"]["input_tokens"] == 4


def test_hook_reconstructed_subagent_carries_its_model_into_the_agent_name() -> None:
    """The exporter reads a callless subagent's model off `cursor.subagent.model`.

    Cursor is genuinely multi-provider, so a subagent dispatched to a different
    model than its parent has to register as its own agent in Logfire.
    """
    from thirdeye.otel_export import _agent_name, _turn_model
    from thirdeye.platforms.cursor.tracing import subagent_turn_from_event

    event = {
        "seq": 7,
        "ts": "2026-01-01T00:00:05.000Z",
        "data": {
            "subagent_id": "sa1",
            "subagent_model": "claude-sonnet-4",
            "task": "review it",
            "status": "completed",
        },
    }
    turn = subagent_turn_from_event("s1", event)

    assert turn["llm_calls"] == []
    assert _agent_name("cursor", _turn_model(turn)) == "cursor[claude-sonnet-4]"
