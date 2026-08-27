from __future__ import annotations

from pathlib import Path

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.tracing import build_turn, usage_from_payload
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
