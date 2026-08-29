from __future__ import annotations

import json
from datetime import datetime
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


# --- subagent leaves ---------------------------------------------------------
#
# Cursor's `subagentStop` callback is the only signal a subagent ever produced:
# it fires once, after the child has finished, and carries no record of the
# child's own model calls or tools. These build the same payload the hook
# persists as a `subagent_message` event.


def _subagent_stop(store: Store, sid: str, generation: str, **data) -> int:
    return _append(store, sid, "subagent_message", {"generation_id": generation, **data})


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)


def _only_subagent(turn) -> dict:
    assert len(turn["subagents"]) == 1
    return turn["subagents"][0]


def test_captured_subagent_stop_builds_one_leaf(tmp_path: Path):
    fixture_path = Path(__file__).parent / "fixtures" / "cursor-subagent-stop.json"
    captured = json.loads(fixture_path.read_text())
    data = captured["data"]
    sid = data["conversation_id"]
    generation = data["generation_id"]
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
    sub_seq = _append(store, sid, captured["t"], data)
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    leaf = _only_subagent(_build(tmp_path, sid, generation, stop_seq))

    assert leaf["turn_span_id"] == str(turn_span_id("cursor", sid, sub_seq))
    assert leaf["input_message"] == data["task"]
    assert leaf["status"] == "completed"
    assert leaf["attributes"] == {
        "cursor.subagent.id": data["subagent_id"],
        "cursor.subagent.type": data["subagent_type"],
        "cursor.subagent.description": data["description"],
        "cursor.subagent.message_count": data["message_count"],
        "cursor.subagent.tool_call_count": data["tool_call_count"],
        "cursor.subagent.loop_count": data["loop_count"],
    }
    assert (_parse(leaf["end_ts"]) - _parse(leaf["start_ts"])).total_seconds() == 16.635


def test_subagent_leaf_uses_task_and_empty_output(tmp_path: Path):
    """The dispatched task is the leaf's input; Cursor reports no child output."""
    sid, generation = "cursor-session", "gen-subagent"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "explore auth"})
    _subagent_stop(
        store,
        sid,
        generation,
        subagent_id="agent-1",
        task="Inspect the authentication flow",
        summary="Located the relevant middleware",
        status="completed",
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    leaf = _only_subagent(_build(tmp_path, sid, generation, stop_seq))

    assert leaf["input_message"] == "Inspect the authentication flow"
    assert leaf["output_message"] == ""
    assert leaf["llm_calls"] == []
    assert leaf["permission_requests"] == []
    assert leaf["subagents"] == []


def test_subagent_leaf_start_precedes_end_by_duration_ms(tmp_path: Path):
    sid, generation = "cursor-session", "gen-duration"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
    sub_seq = _subagent_stop(
        store, sid, generation, subagent_id="agent-1", task="Work", duration_ms=45_000
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    leaf = _only_subagent(_build(tmp_path, sid, generation, stop_seq))

    events = {
        event["seq"]: event for event in Store(Config(root=tmp_path)).reader(sid).iter_events()
    }
    assert leaf["end_ts"] == events[sub_seq]["ts"]
    assert (_parse(leaf["end_ts"]) - _parse(leaf["start_ts"])).total_seconds() == 45.0


def test_subagent_leaf_preserves_type_id_description_and_counts(tmp_path: Path):
    sid, generation = "cursor-session", "gen-attrs"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
    _subagent_stop(
        store,
        sid,
        generation,
        subagent_id="agent-1",
        subagent_type="generalPurpose",
        description="Exploring authentication",
        task="Inspect the authentication flow",
        message_count=12,
        tool_call_count=8,
        loop_count=2,
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    leaf = _only_subagent(_build(tmp_path, sid, generation, stop_seq))

    assert leaf["attributes"] == {
        "cursor.subagent.id": "agent-1",
        "cursor.subagent.type": "generalPurpose",
        "cursor.subagent.description": "Exploring authentication",
        "cursor.subagent.message_count": 12,
        "cursor.subagent.tool_call_count": 8,
        "cursor.subagent.loop_count": 2,
    }


def test_failed_subagent_maps_to_errored(tmp_path: Path):
    sid, generation = "cursor-session", "gen-failed"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
    for index, status in enumerate(("error", "failed", "failure"), start=1):
        _subagent_stop(
            store,
            sid,
            generation,
            subagent_id=f"agent-{index}",
            task="Work",
            status=status,
        )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    turn = _build(tmp_path, sid, generation, stop_seq)

    assert [leaf["status"] for leaf in turn["subagents"]] == ["errored"] * 3


def test_successful_subagent_maps_to_completed(tmp_path: Path):
    """Only an explicitly failed status errors; anything else stays completed."""
    sid, generation = "cursor-session", "gen-ok"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
    _subagent_stop(store, sid, generation, subagent_id="agent-1", task="Work", status="completed")
    _subagent_stop(store, sid, generation, subagent_id="agent-2", task="More", status="aborted")
    _subagent_stop(store, sid, generation, subagent_id="agent-3", task="Even more")
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    turn = _build(tmp_path, sid, generation, stop_seq)

    assert [leaf["status"] for leaf in turn["subagents"]] == ["completed"] * 3


def test_subagent_from_other_generation_is_ignored(tmp_path: Path):
    sid = "cursor-session"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": "new", "prompt": "delegate"})
    _subagent_stop(store, sid, "old", subagent_id="agent-old", task="Stale work")
    _subagent_stop(store, sid, "new", subagent_id="agent-new", task="Current work")
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": "new"})

    leaf = _only_subagent(_build(tmp_path, sid, "new", stop_seq))

    assert leaf["attributes"]["cursor.subagent.id"] == "agent-new"


def test_turn_without_subagent_keeps_empty_list(tmp_path: Path):
    sid, generation = "cursor-session", "gen-plain"
    store = Store(Config(root=tmp_path))
    _append(store, sid, "user_message", {"generation_id": generation, "prompt": "no delegation"})
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation, "model": "gpt-5"})

    assert _build(tmp_path, sid, generation, stop_seq)["subagents"] == []


def test_subagent_turn_id_is_cursor_scoped_and_distinct_from_parent(tmp_path: Path):
    sid, generation = "shared-session", "gen-ids"
    store = Store(Config(root=tmp_path))
    turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "go"})
    sub_seq = _subagent_stop(store, sid, generation, subagent_id="agent-1", task="Work")
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation})

    turn = _build(tmp_path, sid, generation, stop_seq)
    leaf = _only_subagent(turn)

    assert leaf["turn_span_id"] == str(turn_span_id("cursor", sid, sub_seq))
    assert leaf["turn_span_id"] != str(turn_span_id("claude", sid, sub_seq))
    assert (
        leaf["turn_span_id"] != turn["turn_span_id"] == str(turn_span_id("cursor", sid, turn_seq))
    )
    assert leaf["turn_id"] != turn["turn_id"]


def test_subagent_with_session_id_generation_attaches_via_turn_window(tmp_path: Path):
    """Cursor ``subagentStop`` often repeats the conversation id as ``generation_id``."""
    sid, generation = "cursor-session", "gen-live"
    store = Store(Config(root=tmp_path))
    turn_seq = _append(store, sid, "user_message", {"generation_id": generation, "prompt": "delegate"})
    _append(store, sid, "tool_call", {"generation_id": generation, "tool_name": "shell", "command": "ls"})
    sub_seq = _append(
        store,
        sid,
        "subagent_message",
        {
            "generation_id": sid,
            "subagent_id": "agent-1",
            "subagent_type": "explore",
            "task": "smoke test",
            "duration_ms": 1000,
        },
    )
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": generation, "model": "composer-2.5"})

    turn = _build(tmp_path, sid, generation, stop_seq)

    assert turn is not None
    assert len(turn["subagents"]) == 1
    assert turn["subagents"][0]["attributes"]["cursor.subagent.id"] == "agent-1"
    assert turn["subagents"][0]["turn_span_id"] == str(turn_span_id("cursor", sid, sub_seq))
    assert turn["turn_span_id"] == str(turn_span_id("cursor", sid, turn_seq))


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
    stop_seq = _append(store, sid, "turn_stop", {"generation_id": sid})

    assert (
        build_turn(
            session_dir_=session_dir(tmp_path, "cursor", sid),
            session_id=sid,
            generation_id=sid,
            stop_seq=stop_seq,
        )
        is None
    )
