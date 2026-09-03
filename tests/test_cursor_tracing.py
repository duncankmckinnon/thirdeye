from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.interactions import canonical_interactions
from thirdeye.platforms.cursor.subagents import cursor_subagent_generation_id
from thirdeye.platforms.cursor.tracing import (
    build_turn,
    resolve_subagent_export,
    tool_calls_for_generation,
    usage_from_payload,
)
from thirdeye.span_ids import interaction_span_id, turn_span_id
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


# --- tool reconstruction -----------------------------------------------------

_TOOL_TS = "2026-09-02T12:00:00.000Z"
_TOOL_GEN = "gen-reconstruct"
_TOOL_SID = "cursor-session"


def _tool_event(seq: int, event_type: str, ts: str = _TOOL_TS, **data) -> dict:
    return {
        "seq": seq,
        "t": event_type,
        "ts": ts,
        "data": {"generation_id": _TOOL_GEN, **data},
    }


def _reconstructed_tools(events: list[dict]):
    return tool_calls_for_generation(events, _TOOL_SID, _TOOL_GEN)


def _tool_attrs(tool: dict) -> dict:
    return tool["attributes"]


class TestToolReconstruction:
    def test_paired_span_preserves_raw_payloads_and_event_seqs(self):
        call = _tool_event(
            2,
            "tool_call",
            tool_name="StrReplace",
            cursor_tool_family="edit",
            tool_use_id="edit-1",
            tool_input={"path": "src/a.py", "old_string": "x", "new_string": "y"},
        )
        result = _tool_event(
            5,
            "tool_result",
            tool_name="StrReplace",
            cursor_tool_family="edit",
            tool_use_id="edit-1",
            tool_output={"edits": [{"path": "src/a.py"}], "applied": True},
            exit_code=0,
        )

        tool = _reconstructed_tools([call, result])[0]
        attrs = _tool_attrs(tool)

        assert attrs["thirdeye.tool.call.payload"] == call["data"]
        assert attrs["thirdeye.tool.result.payload"] == result["data"]
        assert attrs["thirdeye.event.call_seq"] == 2
        assert attrs["thirdeye.event.result_seq"] == 5
        assert attrs["gen_ai.tool.call.arguments"] == {
            "path": "src/a.py",
            "old_string": "x",
            "new_string": "y",
        }
        assert attrs["gen_ai.tool.call.result"] == {
            "edits": [{"path": "src/a.py"}],
            "applied": True,
        }

    def test_semantic_arguments_use_single_present_candidate(self):
        call = _tool_event(
            1,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            command="pytest -q",
        )
        result = _tool_event(
            2,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            command="pytest -q",
            output="passed",
        )

        attrs = _tool_attrs(_reconstructed_tools([call, result])[0])

        assert attrs["gen_ai.tool.call.arguments"] == "pytest -q"
        assert attrs["gen_ai.tool.call.result"] == "passed"

    def test_semantic_arguments_honor_input_key(self):
        call = _tool_event(1, "tool_call", tool_name="custom", input={"query": "find tests"})
        result = _tool_event(
            2,
            "tool_result",
            tool_name="custom",
            response={"matches": 3},
        )

        attrs = _tool_attrs(_reconstructed_tools([call, result])[0])

        assert attrs["gen_ai.tool.call.arguments"] == {"query": "find tests"}
        assert attrs["gen_ai.tool.call.result"] == {"matches": 3}

    def test_semantic_arguments_combine_all_present_candidates(self):
        call = _tool_event(
            1,
            "tool_call",
            tool_name="ApplyPatch",
            cursor_tool_family="edit",
            tool_input={"path": "src/a.py"},
            command="apply",
            file_path="src/a.py",
        )
        result = _tool_event(
            2,
            "tool_result",
            tool_name="ApplyPatch",
            cursor_tool_family="edit",
            tool_output={"applied": True},
            output="done",
            diff="--- a\n+++ b",
        )

        attrs = _tool_attrs(_reconstructed_tools([call, result])[0])

        assert attrs["gen_ai.tool.call.arguments"] == {
            "tool_input": {"path": "src/a.py"},
            "command": "apply",
            "file_path": "src/a.py",
        }
        assert attrs["gen_ai.tool.call.result"] == {
            "tool_output": {"applied": True},
            "output": "done",
            "diff": "--- a\n+++ b",
        }

    def test_reverse_explicit_id_completion_pairs_correctly(self):
        result = _tool_event(
            1,
            "tool_result",
            tool_name="Task",
            tool_use_id="call-reverse",
            tool_output={"status": "done"},
        )
        call = _tool_event(
            2,
            "tool_call",
            tool_name="Task",
            tool_use_id="call-reverse",
            tool_input={"prompt": "delegate"},
        )

        tool = _reconstructed_tools([result, call])[0]

        assert tool["tool_call_id"] == "call-reverse"
        attrs = _tool_attrs(tool)
        assert attrs["thirdeye.tool.call.payload"] == call["data"]
        assert attrs["thirdeye.tool.result.payload"] == result["data"]
        assert attrs["thirdeye.event.call_seq"] == 2
        assert attrs["thirdeye.event.result_seq"] == 1
        assert attrs["gen_ai.tool.call.arguments"] == {"prompt": "delegate"}
        assert attrs["gen_ai.tool.call.result"] == {"status": "done"}

    def test_explicit_id_overrides_contradictory_signature(self):
        call_one = _tool_event(
            1,
            "tool_call",
            tool_name="read_file",
            cursor_tool_family="read",
            tool_use_id="call-1",
            path="src/first.py",
        )
        call_two = _tool_event(
            2,
            "tool_call",
            tool_name="read_file",
            cursor_tool_family="read",
            tool_use_id="call-2",
            path="src/second.py",
        )
        result_two = _tool_event(
            3,
            "tool_result",
            tool_name="read_file",
            cursor_tool_family="read",
            tool_use_id="call-2",
            path="src/first.py",
            output="second contents",
        )
        result_one = _tool_event(
            4,
            "tool_result",
            tool_name="read_file",
            cursor_tool_family="read",
            tool_use_id="call-1",
            path="src/second.py",
            output="first contents",
        )

        tools = _reconstructed_tools([call_one, call_two, result_two, result_one])
        by_id = {tool["tool_call_id"]: tool for tool in tools}

        assert _tool_attrs(by_id["call-1"])["gen_ai.tool.call.arguments"] == "src/first.py"
        assert _tool_attrs(by_id["call-1"])["gen_ai.tool.call.result"] == "first contents"
        assert _tool_attrs(by_id["call-2"])["gen_ai.tool.call.arguments"] == "src/second.py"
        assert _tool_attrs(by_id["call-2"])["gen_ai.tool.call.result"] == "second contents"
        assert _tool_attrs(by_id["call-1"])["thirdeye.event.result_seq"] == 4
        assert _tool_attrs(by_id["call-2"])["thirdeye.event.result_seq"] == 3

    def test_explicit_id_mismatch_does_not_steal_family_call(self):
        call = _tool_event(
            1,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            command="only-call",
        )
        result = _tool_event(
            2,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            tool_use_id="missing-call-id",
            output="orphan-out",
        )

        tools = _reconstructed_tools([call, result])

        assert len(tools) == 2
        unmatched_call = next(
            tool for tool in tools if _tool_attrs(tool).get("thirdeye.tool.unmatched") == "call"
        )
        unmatched_result = next(
            tool for tool in tools if _tool_attrs(tool).get("thirdeye.tool.unmatched") == "result"
        )
        assert unmatched_call["tool_call_id"] == f"{_TOOL_GEN}:shell:1"
        assert unmatched_result["tool_call_id"] == f"{_TOOL_GEN}:shell:result:2"
        assert "thirdeye.tool.result.payload" not in _tool_attrs(unmatched_call)
        assert "thirdeye.tool.call.payload" not in _tool_attrs(unmatched_result)

    def test_signature_pairing_retains_payload_metadata(self):
        call = _tool_event(
            1,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            command="make build",
        )
        result = _tool_event(
            2,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            command="make build",
            output="built",
            exit_code=0,
        )

        attrs = _tool_attrs(_reconstructed_tools([call, result])[0])

        assert attrs["gen_ai.tool.call.arguments"] == "make build"
        assert attrs["gen_ai.tool.call.result"] == "built"
        assert attrs["thirdeye.tool.call.payload"] == call["data"]
        assert attrs["thirdeye.tool.result.payload"] == result["data"]
        assert attrs["cursor.tool.exit_code"] == 0

    def test_fifo_pairing_retains_payload_metadata(self):
        call_first = _tool_event(
            1,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            command="first",
        )
        call_second = _tool_event(
            2,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            command="second",
        )
        result_first = _tool_event(
            3,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            output="out-first",
        )
        result_second = _tool_event(
            4,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            output="out-second",
        )

        tools = _reconstructed_tools([call_first, call_second, result_first, result_second])

        assert [
            (
                _tool_attrs(tool)["gen_ai.tool.call.arguments"],
                _tool_attrs(tool)["gen_ai.tool.call.result"],
                _tool_attrs(tool)["thirdeye.event.call_seq"],
                _tool_attrs(tool)["thirdeye.event.result_seq"],
            )
            for tool in tools
        ] == [
            ("first", "out-first", 1, 3),
            ("second", "out-second", 2, 4),
        ]

    def test_unmatched_call_emits_call_marker(self):
        call = _tool_event(
            7,
            "tool_call",
            tool_name="Grep",
            tool_use_id="grep-1",
            pattern="tool_calls_for_generation",
        )

        tool = _reconstructed_tools([call])[0]
        attrs = _tool_attrs(tool)

        assert attrs["thirdeye.tool.unmatched"] == "call"
        assert attrs["thirdeye.tool.call.payload"] == call["data"]
        assert attrs["thirdeye.event.call_seq"] == 7
        assert "thirdeye.tool.result.payload" not in attrs
        assert "thirdeye.event.result_seq" not in attrs

    def test_unmatched_result_emits_synthetic_id_and_result_marker(self):
        result = _tool_event(
            9,
            "tool_result",
            tool_name="mcp.search",
            tool_use_id="mcp-1",
            tool_output="found",
        )

        tool = _reconstructed_tools([result])[0]
        attrs = _tool_attrs(tool)

        assert tool["tool_call_id"] == f"{_TOOL_GEN}:mcp.search:result:9"
        assert attrs["thirdeye.tool.unmatched"] == "result"
        assert attrs["thirdeye.tool.result.payload"] == result["data"]
        assert attrs["thirdeye.event.result_seq"] == 9
        assert "thirdeye.tool.call.payload" not in attrs
        assert "thirdeye.event.call_seq" not in attrs

    def test_build_turn_surfaces_reconstructed_tool_payloads(self, tmp_path: Path):
        sid, generation = _TOOL_SID, _TOOL_GEN
        store = Store(Config(root=tmp_path))
        call_seq = _append(
            store,
            sid,
            "tool_call",
            {
                "generation_id": generation,
                "tool_name": "shell",
                "cursor_tool_family": "shell",
                "command": "echo hi",
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
                "command": "echo hi",
                "output": "hi",
            },
        )
        _append(store, sid, "user_message", {"generation_id": generation, "prompt": "run shell"})
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

        tool = _build(tmp_path, sid, generation, stop_seq)["llm_calls"][0]["tool_calls"][0]
        attrs = tool["attributes"]
        call_event = _stored_event(store, sid, call_seq)
        result_event = _stored_event(store, sid, result_seq)

        assert attrs["thirdeye.tool.call.payload"] == call_event["data"]
        assert attrs["thirdeye.tool.result.payload"] == result_event["data"]
        assert attrs["thirdeye.event.call_seq"] == call_seq
        assert attrs["thirdeye.event.result_seq"] == result_seq

    def test_build_turn_keeps_unmatched_tools_without_prompt(self, tmp_path: Path):
        """Tool-only generations still export their unmatched sides."""
        sid, generation = _TOOL_SID, _TOOL_GEN
        store = Store(Config(root=tmp_path))
        call_seq = _append(
            store,
            sid,
            "tool_call",
            {
                "generation_id": generation,
                "tool_name": "Grep",
                "tool_use_id": "grep-orphan",
                "pattern": "orphan",
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
        tool = turn["llm_calls"][0]["tool_calls"][0]
        assert tool["tool_call_id"] == "grep-orphan"
        assert _tool_attrs(tool)["thirdeye.tool.unmatched"] == "call"
        assert _tool_attrs(tool)["thirdeye.event.call_seq"] == call_seq

    def test_reverse_signature_completion_pairs_correctly(self):
        result = _tool_event(
            1,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            command="make build",
            output="built",
        )
        call = _tool_event(
            2,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            command="make build",
        )

        tools = _reconstructed_tools([result, call])

        assert len(tools) == 1
        attrs = _tool_attrs(tools[0])
        assert "thirdeye.tool.unmatched" not in attrs
        assert attrs["thirdeye.event.call_seq"] == 2
        assert attrs["thirdeye.event.result_seq"] == 1
        assert attrs["thirdeye.tool.call.payload"] == call["data"]
        assert attrs["thirdeye.tool.result.payload"] == result["data"]

    def test_reverse_completion_orders_span_timestamps(self):
        result = _tool_event(
            1,
            "tool_result",
            ts="2026-09-02T12:00:01.000Z",
            tool_name="Task",
            tool_use_id="call-reverse",
            tool_output={"status": "done"},
        )
        call = _tool_event(
            2,
            "tool_call",
            ts="2026-09-02T12:00:04.000Z",
            tool_name="Task",
            tool_use_id="call-reverse",
            tool_input={"prompt": "delegate"},
        )

        tool = _reconstructed_tools([result, call])[0]

        # The callbacks arrived reversed; the span still spans the observed
        # interval rather than ending before it starts.
        assert tool["start_ts"] == "2026-09-02T12:00:01.000Z"
        assert tool["end_ts"] == "2026-09-02T12:00:04.000Z"

    def test_forward_completion_keeps_call_and_result_timestamps(self):
        call = _tool_event(
            1,
            "tool_call",
            ts="2026-09-02T12:00:01.000Z",
            tool_name="shell",
            cursor_tool_family="shell",
            command="make build",
        )
        result = _tool_event(
            2,
            "tool_result",
            ts="2026-09-02T12:00:09.000Z",
            tool_name="shell",
            cursor_tool_family="shell",
            command="make build",
            output="built",
        )

        tool = _reconstructed_tools([call, result])[0]

        assert tool["start_ts"] == "2026-09-02T12:00:01.000Z"
        assert tool["end_ts"] == "2026-09-02T12:00:09.000Z"

    def test_unsigned_result_does_not_claim_explicit_id_call(self):
        call = _tool_event(
            1,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            tool_use_id="call-explicit",
            command="make build",
        )
        result = _tool_event(
            2,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            output="built",
        )

        tools = _reconstructed_tools([call, result])

        assert [_tool_attrs(tool).get("thirdeye.tool.unmatched") for tool in tools] == [
            "call",
            "result",
        ]
        assert tools[0]["tool_call_id"] == "call-explicit"
        assert tools[1]["tool_call_id"] == f"{_TOOL_GEN}:shell:result:2"

    def test_unsigned_result_does_not_reach_past_explicit_id_call(self):
        """Positional pairing never reorders: it stops at an explicit id it cannot match."""
        explicit_call = _tool_event(
            1,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            tool_use_id="call-explicit",
            command="explicit",
        )
        signature_call = _tool_event(
            2,
            "tool_call",
            tool_name="shell",
            cursor_tool_family="shell",
            command="signed",
        )
        result = _tool_event(
            3,
            "tool_result",
            tool_name="shell",
            cursor_tool_family="shell",
            output="built",
        )

        tools = _reconstructed_tools([explicit_call, signature_call, result])

        assert [_tool_attrs(tool).get("thirdeye.tool.unmatched") for tool in tools] == [
            "call",
            "call",
            "result",
        ]
        assert [tool["tool_call_id"] for tool in tools] == [
            "call-explicit",
            f"{_TOOL_GEN}:shell:2",
            f"{_TOOL_GEN}:shell:result:3",
        ]

    def test_unmatched_call_with_explicit_id_preserves_call_id(self):
        call = _tool_event(
            3,
            "tool_call",
            tool_name="Read",
            tool_use_id="read-explicit",
            path="src/main.py",
        )

        tool = _reconstructed_tools([call])[0]

        assert tool["tool_call_id"] == "read-explicit"
        assert _tool_attrs(tool)["thirdeye.tool.unmatched"] == "call"

    def test_instant_tool_keeps_one_payload_on_both_sides(self):
        """Preserved: an instant callback is a complete call and result at once."""
        event = _tool_event(
            1,
            "tool_result",
            tool_name="Grep",
            tool_use_id="grep-instant",
            cursor_instant=True,
            tool_input={"pattern": "TODO"},
            tool_output='{"matches": 3}',
            duration=25,
        )

        tools = _reconstructed_tools([event])
        attrs = _tool_attrs(tools[0])

        assert len(tools) == 1
        assert tools[0]["tool_call_id"] == "grep-instant"
        assert "thirdeye.tool.unmatched" not in attrs
        assert attrs["thirdeye.tool.call.payload"] == event["data"]
        assert attrs["thirdeye.tool.result.payload"] == event["data"]
        assert attrs["thirdeye.event.call_seq"] == 1
        assert attrs["thirdeye.event.result_seq"] == 1
        # Embedded JSON is decoded rather than pre-stringified.
        assert attrs["gen_ai.tool.call.result"] == {"matches": 3}
        assert tools[0]["start_ts"] < tools[0]["end_ts"]

    def test_pairing_tolerates_unusable_timestamps(self):
        """Preserved fail-open behavior: unparseable or absent `ts` never raises."""
        malformed = _reconstructed_tools(
            [
                _tool_event(1, "tool_result", ts="not-a-timestamp", command="c", output="o"),
                _tool_event(2, "tool_call", ts="also-not-a-timestamp", command="c"),
            ]
        )
        missing = _reconstructed_tools(
            [
                _tool_event(1, "tool_call", ts="", command="c"),
                _tool_event(2, "tool_result", ts="", command="c", output="o"),
            ]
        )

        assert malformed[0]["start_ts"] == "also-not-a-timestamp"
        assert malformed[0]["end_ts"] == "not-a-timestamp"
        assert (missing[0]["start_ts"], missing[0]["end_ts"]) == ("", "")


# --- turn reconstruction -----------------------------------------------------


def _expected_interaction_attributes(interaction) -> dict:
    return {
        "thirdeye.interaction.kind": interaction.kind,
        "thirdeye.interaction.payload": interaction.payload,
        "thirdeye.interaction.correlation_id": interaction.correlation_id,
        "thirdeye.interaction.source_type": interaction.source_type,
        "thirdeye.interaction.source_seq": interaction.source_seq,
        "thirdeye.interaction.timestamp": interaction.ts,
        "thirdeye.interaction.generation_id": interaction.generation_id,
        "thirdeye.interaction.duplicate_seqs": list(interaction.duplicate_seqs),
    }


def _interaction_by_id(turn, interaction_id: str) -> dict:
    interactions = turn.get("interactions") or []
    return next(item for item in interactions if item["interaction_id"] == interaction_id)


class TestTurnReconstruction:
    def test_second_turn_input_includes_prior_turn_context(self, tmp_path: Path):
        sid = "two-turn-session"
        gen_one, gen_two = "gen-turn-1", "gen-turn-2"
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": gen_one, "prompt": "first question"})
        _append(store, sid, "assistant_thought", {"generation_id": gen_one, "text": "plan one"})
        _shell_call(store, sid, gen_one, tool_use_id="shell-1", command="ls")
        _shell_result(store, sid, gen_one, tool_use_id="shell-1", output="README.md")
        _append(
            store,
            sid,
            "assistant_message",
            {"generation_id": gen_one, "text": "answer one"},
        )
        _append(store, sid, "turn_stop", {"generation_id": gen_one})
        _append(store, sid, "user_message", {"generation_id": gen_two, "prompt": "second question"})
        _append(store, sid, "assistant_thought", {"generation_id": gen_two, "text": "plan two"})
        _read_call(store, sid, gen_two, tool_use_id="read-1", path="src/main.py")
        _read_result(store, sid, gen_two, tool_use_id="read-1", output="contents")
        _append(
            store,
            sid,
            "assistant_message",
            {"generation_id": gen_two, "text": "answer two"},
        )
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": gen_two})

        turn = build_turn(
            session_dir_=session_dir(tmp_path, "cursor", sid),
            session_id=sid,
            generation_id=gen_two,
            stop_seq=stop_seq,
        )

        assert turn is not None
        assert turn["input_message"] == "second question"
        assert turn["output_message"] == "answer two"
        call = turn["llm_calls"][0]
        assert call["input_messages"] == [
            {"role": "user", "parts": [{"type": "text", "content": "first question"}]},
            {
                "role": "assistant",
                "thirdeye.kind": "reasoning",
                "parts": [{"type": "text", "content": "plan one"}],
            },
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "shell-1",
                        "name": "shell",
                        "arguments": "ls",
                    }
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "shell-1",
                        "response": "README.md",
                    }
                ],
            },
            {"role": "assistant", "parts": [{"type": "text", "content": "answer one"}]},
            {"role": "user", "parts": [{"type": "text", "content": "second question"}]},
            {
                "role": "assistant",
                "thirdeye.kind": "reasoning",
                "parts": [{"type": "text", "content": "plan two"}],
            },
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "read-1",
                        "name": "read_file",
                        "arguments": "src/main.py",
                    }
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "read-1",
                        "response": "contents",
                    }
                ],
            },
        ]
        assert call["output_messages"] == [
            {"role": "assistant", "parts": [{"type": "text", "content": "answer two"}]},
        ]

    def test_final_assistant_message_is_output_not_input(self, tmp_path: Path):
        sid, generation = "single-turn-session", "gen-single"
        store = Store(Config(root=tmp_path))
        _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
        _append(store, sid, "assistant_thought", {"generation_id": generation, "text": "think"})
        response_seq = _append(
            store,
            sid,
            "assistant_message",
            {"generation_id": generation, "text": "done"},
        )
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

        turn = build_turn(
            session_dir_=session_dir(tmp_path, "cursor", sid),
            session_id=sid,
            generation_id=generation,
            stop_seq=stop_seq,
        )

        assert turn is not None
        call = turn["llm_calls"][0]
        assert call["input_messages"] == [
            {"role": "user", "parts": [{"type": "text", "content": "go"}]},
            {
                "role": "assistant",
                "thirdeye.kind": "reasoning",
                "parts": [{"type": "text", "content": "think"}],
            },
        ]
        assert call["output_messages"] == [
            {"role": "assistant", "parts": [{"type": "text", "content": "done"}]},
        ]
        assert response_seq == stop_seq - 1

    def test_build_turn_emits_exact_interaction_recovery_records(self, tmp_path: Path):
        sid, generation = "recovery-session", "gen-recovery"
        store = Store(Config(root=tmp_path))
        turn_seq = _append(
            store,
            sid,
            "user_message",
            {"generation_id": generation, "prompt": "ship it", "flags": {"urgent": True}},
        )
        _append(
            store,
            sid,
            "assistant_thought",
            {"generation_id": generation, "text": "plan", "model": "claude-4"},
        )
        call_seq = _shell_call(
            store,
            sid,
            generation,
            tool_use_id="shell-1",
            command="pytest -q",
        )
        result_seq = _shell_result(
            store,
            sid,
            generation,
            tool_use_id="shell-1",
            output="passed",
        )
        response_seq = _append(
            store,
            sid,
            "assistant_message",
            {"generation_id": generation, "text": "all green"},
        )
        stop_seq = _append(
            store,
            sid,
            "turn_stop",
            {"generation_id": generation, "input_tokens": 12, "output_tokens": 3},
        )
        events = list(store.reader(sid).iter_events())
        expected_turn_span_id = str(turn_span_id("cursor", sid, turn_seq))
        expected = {
            item.interaction_id: item
            for item in canonical_interactions(events, generation_id=generation, through_seq=stop_seq)
            if item.kind not in {"tool_call", "tool_result"}
        }

        turn = build_turn(
            session_dir_=session_dir(tmp_path, "cursor", sid),
            session_id=sid,
            generation_id=generation,
            stop_seq=stop_seq,
        )

        assert turn is not None
        interactions = turn.get("interactions") or []
        assert {item["interaction_id"] for item in interactions} == set(expected)
        assert all("tool_call" not in item["kind"] for item in interactions)
        assert all("tool_result" not in item["kind"] for item in interactions)

        for record in interactions:
            canonical = expected[record["interaction_id"]]
            assert record["kind"] == canonical.kind
            assert record["span_id"] == str(
                interaction_span_id("cursor", sid, canonical.interaction_id)
            )
            assert record["parent_span_id"] == expected_turn_span_id
            assert record["start_ts"] == canonical.ts
            assert record["end_ts"] == canonical.ts
            assert record["attributes"] == _expected_interaction_attributes(canonical)

        user_canonical = next(item for item in expected.values() if item.kind == "user_message")
        user_record = _interaction_by_id(turn, user_canonical.interaction_id)
        stop_record = next(item for item in interactions if item["kind"] == "lifecycle")
        assert user_record["attributes"]["thirdeye.interaction.payload"] == {
            "generation_id": generation,
            "prompt": "ship it",
            "flags": {"urgent": True},
        }
        assert stop_record["attributes"]["thirdeye.interaction.source_type"] == "turn_stop"
        assert stop_record["attributes"]["thirdeye.interaction.source_seq"] == stop_seq
        assert {call_seq, result_seq, response_seq}.isdisjoint(
            item["attributes"]["thirdeye.interaction.source_seq"] for item in interactions
        )

    def test_reasoning_duplicate_produces_one_recovery_record(self, tmp_path: Path):
        sid, generation = "dedupe-session", "gen-dedupe"
        store = Store(Config(root=tmp_path))
        turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
        _append(
            store,
            sid,
            "assistant_thought",
            {"generation_id": generation, "text": "plan", "model": "claude-4"},
        )
        _append(
            store,
            sid,
            "assistant_thought",
            {"generation_id": generation, "text": "plan", "model": "gpt-5"},
        )
        _append(
            store,
            sid,
            "assistant_message",
            {"generation_id": generation, "text": "done"},
        )
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
        events = list(store.reader(sid).iter_events())
        reasoning = next(
            item
            for item in canonical_interactions(events, generation_id=generation, through_seq=stop_seq)
            if item.kind == "reasoning"
        )

        turn = build_turn(
            session_dir_=session_dir(tmp_path, "cursor", sid),
            session_id=sid,
            generation_id=generation,
            stop_seq=stop_seq,
        )

        assert turn is not None
        reasoning_records = [
            item for item in (turn.get("interactions") or []) if item["kind"] == "reasoning"
        ]
        assert len(reasoning_records) == 1
        record = reasoning_records[0]
        assert record["interaction_id"] == reasoning.interaction_id
        assert record["attributes"]["thirdeye.interaction.duplicate_seqs"] == list(
            reasoning.duplicate_seqs
        )
        assert record["attributes"]["thirdeye.interaction.payload"] == reasoning.payload

    def test_recovery_records_parent_to_deterministic_turn_span_id(self, tmp_path: Path):
        sid, generation = "parent-session", "gen-parent"
        store = Store(Config(root=tmp_path))
        turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "hi"})
        stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})
        expected_parent = str(turn_span_id("cursor", sid, turn_seq))

        turn = build_turn(
            session_dir_=session_dir(tmp_path, "cursor", sid),
            session_id=sid,
            generation_id=generation,
            stop_seq=stop_seq,
        )

        assert turn is not None
        assert turn["turn_span_id"] == expected_parent
        interactions = turn.get("interactions") or []
        assert interactions
        assert all(item["parent_span_id"] == expected_parent for item in interactions)


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
        child_gen = cursor_subagent_generation_id("call-owner")
        tool = turn["llm_calls"][0]["tool_calls"][0]
        attrs = tool["attributes"]
        assert tool["name"] == "mcp.search"
        assert tool["tool_call_id"] == (
            f"{child_gen}:mcp.search:result:{attrs['thirdeye.event.result_seq']}"
        )
        assert attrs["thirdeye.tool.unmatched"] == "result"
        assert attrs["thirdeye.tool.result.payload"]["tool_use_id"] == "mcp-1"
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


def test_build_turn_without_user_message_keeps_unmatched_tool(tmp_path: Path):
    """A tool-only generation is still a turn: its unmatched call stays visible."""
    sid, generation = "cursor-session", "gen-tools-only"
    store = Store(Config(root=tmp_path))
    _append(
        store,
        sid,
        "tool_call",
        {"generation_id": generation, "tool_name": "shell", "command": "ls"},
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    turn = build_turn(
        session_dir_=session_dir(tmp_path, "cursor", sid),
        session_id=sid,
        generation_id=generation,
        stop_seq=stop_seq,
    )

    assert turn is not None
    assert turn["input_message"] == ""
    tool = turn["llm_calls"][0]["tool_calls"][0]
    assert tool["attributes"]["thirdeye.tool.unmatched"] == "call"
    assert tool["attributes"]["gen_ai.tool.call.arguments"] == "ls"


def test_build_turn_without_llm_content_returns_none(tmp_path: Path):
    sid, generation = "cursor-session", "gen-empty"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "reasoning", {"generation_id": generation, "text": "thinking"})
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
