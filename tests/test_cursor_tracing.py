from __future__ import annotations

from pathlib import Path

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.tracing import build_turn, usage_from_payload
from thirdeye.span_ids import turn_span_id
from thirdeye.store import Store


def _append(store: Store, sid: str, event_type: str, data: dict) -> int:
    return store.append_event(
        session_id=sid, platform="cursor", cwd="/repo", t=event_type, data=data
    )


def test_usage_includes_cache_buckets_in_otel_input_total():
    assert usage_from_payload(
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_tokens": 20,
            "cache_write_tokens": 3,
        }
    ) == {
        "input_tokens": 33,
        "output_tokens": 4,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 3,
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
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_tokens": 20,
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
    assert call["usage"]["input_tokens"] == 30
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
