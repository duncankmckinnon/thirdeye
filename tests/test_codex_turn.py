from __future__ import annotations

import json
from pathlib import Path

from thirdeye.platforms.codex.turn import extract_turn_codex


def _frame(ts: str, outer: str, payload: dict) -> str:
    return json.dumps({"timestamp": ts, "type": outer, "payload": payload})


def test_extracts_exact_turn_deduplicates_usage_and_pairs_tools(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    lines = [
        _frame("2026-01-01T00:00:00Z", "turn_context", {"turn_id": "t1", "model": "gpt-5"}),
        _frame("2026-01-01T00:00:01Z", "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _frame("2026-01-01T00:00:02Z", "event_msg", {"type": "user_message", "message": "fix it"}),
        _frame(
            "2026-01-01T00:00:03Z",
            "response_item",
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "exec_command",
                "arguments": '{"cmd":"pytest"}',
            },
        ),
        _frame(
            "2026-01-01T00:00:04Z",
            "response_item",
            {"type": "function_call_output", "call_id": "c1", "output": "passed"},
        ),
        _frame(
            "2026-01-01T00:00:05Z",
            "event_msg",
            {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"total_tokens": 15},
                    "last_token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 4,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 2,
                    },
                },
            },
        ),
        # Repeat report for the same call: it must not double the turn total.
        _frame(
            "2026-01-01T00:00:06Z",
            "event_msg",
            {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"total_tokens": 15},
                    "last_token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 4,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 2,
                    },
                },
            },
        ),
        _frame("2026-01-01T00:00:07Z", "event_msg", {"type": "agent_message", "message": "done"}),
        _frame(
            "2026-01-01T00:00:08Z",
            "event_msg",
            {"type": "task_complete", "turn_id": "t1", "last_agent_message": "done"},
        ),
        _frame("2026-01-01T00:00:09Z", "turn_context", {"turn_id": "t2", "model": "gpt-5"}),
    ]
    path.write_text("\n".join(lines) + "\n")

    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert turn["turn_id"] == "t1"
    assert turn["start_ts"] == "2026-01-01T00:00:01Z"
    assert turn["end_ts"] == "2026-01-01T00:00:08Z"
    assert turn["user_prompt"] == "fix it"
    assert turn["assistant_output"] == "done"
    assert turn["status"] == "completed"
    assert len(turn["calls"]) == 2

    first = turn["calls"][0]
    assert first["call_id"] == "t1:0"
    assert first["provider"] == "openai"
    assert first["model"] == "gpt-5"
    assert first["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 4,
        "reasoning_output_tokens": 2,
    }
    assert first["tool_calls"] == [
        {
            "tool_call_id": "c1",
            "name": "exec_command",
            "start_ts": "2026-01-01T00:00:03Z",
            "end_ts": "2026-01-01T00:00:04Z",
            "attributes": {"arguments": '{"cmd":"pytest"}', "result": "passed"},
        }
    ]

    second = turn["calls"][1]
    assert second["call_id"] == "t1:1"
    assert second["usage"] == {}
    assert second["tool_calls"] == []


def test_missing_turn_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(_frame("2026-01-01T00:00:00Z", "turn_context", {"turn_id": "t1"}))
    assert extract_turn_codex(str(path), "other") is None


def test_reconstructs_per_call_messages_and_tool_result_input(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                _frame("2026-01-01T00:00:00Z", "turn_context", {"turn_id": "t1", "model": "gpt-5"}),
                _frame(
                    "2026-01-01T00:00:01Z",
                    "event_msg",
                    {"type": "user_message", "message": "inspect"},
                ),
                _frame(
                    "2026-01-01T00:00:02Z",
                    "response_item",
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "I should read"}],
                    },
                ),
                _frame(
                    "2026-01-01T00:00:03Z",
                    "response_item",
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "read",
                        "arguments": {"path": "a.py"},
                    },
                ),
                _frame(
                    "2026-01-01T00:00:04Z",
                    "response_item",
                    {"type": "function_call_output", "call_id": "c1", "output": "contents"},
                ),
                _frame(
                    "2026-01-01T00:00:05Z",
                    "event_msg",
                    {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {"total_tokens": 12},
                            "last_token_usage": {"input_tokens": 10, "output_tokens": 2},
                        },
                    },
                ),
                _frame(
                    "2026-01-01T00:00:06Z",
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "found it"}],
                    },
                ),
                _frame(
                    "2026-01-01T00:00:07Z",
                    "event_msg",
                    {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {"total_tokens": 20},
                            "last_token_usage": {"input_tokens": 6, "output_tokens": 2},
                        },
                    },
                ),
            ]
        )
        + "\n"
    )

    calls = extract_turn_codex(str(path), "t1")["calls"]
    assert len(calls) == 2
    assert calls[0]["input_messages"][0]["parts"][0]["content"] == "inspect"
    assert [part["type"] for part in calls[0]["output_messages"][0]["parts"]] == [
        "reasoning",
        "tool_call",
    ]
    assert calls[0]["tool_calls"][0]["tool_call_id"] == "c1"
    assert calls[1]["input_messages"][0]["parts"] == [
        {"type": "tool_call_response", "id": "c1", "response": "contents"}
    ]
    assert calls[1]["output_messages"][0]["parts"][0]["content"] == "found it"


def test_extracts_mcp_and_image_tools_and_provider_timing(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                _frame("2026-01-01T00:00:00Z", "turn_context", {"turn_id": "t1", "model": "gpt-5"}),
                _frame(
                    "2026-01-01T00:00:01Z",
                    "event_msg",
                    {"type": "task_started", "turn_id": "t1", "started_at": 1767225600},
                ),
                _frame(
                    "2026-01-01T00:00:03Z",
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "call_id": "m1",
                        "invocation": {"server": "files", "tool": "read", "arguments": {"p": "a"}},
                        "duration": {"secs": 2, "nanos": 0},
                        "result": {"Ok": "content"},
                    },
                ),
                _frame(
                    "2026-01-01T00:00:04Z",
                    "response_item",
                    {
                        "type": "image_generation_call",
                        "call_id": "i1",
                        "prompt": "a fox",
                        "status": "completed",
                    },
                ),
                _frame(
                    "2026-01-01T00:00:05Z",
                    "event_msg",
                    {"type": "task_complete", "turn_id": "t1", "completed_at": 1767225605},
                ),
            ]
        )
        + "\n"
    )
    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert turn["start_ts"] == "2026-01-01T00:00:00.000Z"
    assert turn["end_ts"] == "2026-01-01T00:00:05.000Z"
    tool_calls = turn["calls"][0]["tool_calls"]
    assert [tool["name"] for tool in tool_calls] == ["files.read", "image_generation"]
    assert tool_calls[0]["start_ts"] == "2026-01-01T00:00:01.000Z"


def test_synthesized_timestamps_use_millisecond_format(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                _frame("fallback", "turn_context", {"turn_id": "t1", "model": "gpt-5"}),
                _frame(
                    "fallback",
                    "event_msg",
                    {"type": "task_started", "turn_id": "t1", "started_at": 0},
                ),
                _frame(
                    "1970-01-01T00:00:01.000Z",
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "call_id": "m1",
                        "invocation": {"server": "files", "tool": "read"},
                        "duration": {"secs": 0, "nanos": 876544000},
                    },
                ),
                _frame(
                    "fallback",
                    "event_msg",
                    {"type": "task_complete", "turn_id": "t1", "completed_at": 0.123456},
                ),
            ]
        )
        + "\n"
    )

    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert turn["start_ts"] == "1970-01-01T00:00:00.000Z"
    assert turn["end_ts"] == "1970-01-01T00:00:00.123Z"
    assert turn["calls"][0]["tool_calls"][0]["start_ts"] == "1970-01-01T00:00:00.123Z"


def test_turn_aborted_marks_interrupted_and_keeps_partial_call(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                _frame("2026-01-01T00:00:00Z", "turn_context", {"turn_id": "t1", "model": "gpt-5"}),
                _frame(
                    "2026-01-01T00:00:01Z", "event_msg", {"type": "task_started", "turn_id": "t1"}
                ),
                _frame(
                    "2026-01-01T00:00:02Z",
                    "event_msg",
                    {"type": "user_message", "message": "fix it"},
                ),
                _frame(
                    "2026-01-01T00:00:03Z",
                    "response_item",
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "exec_command",
                        "arguments": '{"cmd":"pytest"}',
                    },
                ),
                _frame(
                    "2026-01-01T00:00:04Z", "event_msg", {"type": "turn_aborted", "turn_id": "t1"}
                ),
            ]
        )
        + "\n"
    )
    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert turn["status"] == "interrupted"
    assert turn["end_ts"] == "2026-01-01T00:00:04Z"
    # The in-flight call and its tool call are still captured, not discarded.
    assert len(turn["calls"]) == 1
    assert turn["calls"][0]["tool_calls"][0]["tool_call_id"] == "c1"


def test_turn_aborted_real_payload_shape_marks_interrupted(tmp_path: Path) -> None:
    """The synthetic ``turn_aborted`` frame above only carries ``turn_id``.
    A genuine frame pulled from a real Codex rollout on this machine
    (2026-08-20, codex-cli 0.14x) also carries ``reason``, ``started_at``,
    ``completed_at``, and ``duration_ms`` -- this proves the extra fields a
    real payload has don't break detection.
    """
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                _frame("2026-01-01T00:00:00Z", "turn_context", {"turn_id": "t1", "model": "gpt-5"}),
                _frame(
                    "2026-01-01T00:00:01Z", "event_msg", {"type": "task_started", "turn_id": "t1"}
                ),
                _frame(
                    "2026-01-01T00:00:02Z",
                    "event_msg",
                    {"type": "user_message", "message": "fix it"},
                ),
                json.dumps(
                    {
                        "timestamp": "2026-08-20T18:20:40.773Z",
                        "ordinal": 45,
                        "type": "event_msg",
                        "payload": {
                            "type": "turn_aborted",
                            "turn_id": "t1",
                            "reason": "interrupted",
                            "started_at": 1787250009,
                            "completed_at": 1787250040,
                            "duration_ms": 31592,
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert turn["status"] == "interrupted"
    assert turn["end_ts"] == "2026-08-20T18:20:40.773Z"
