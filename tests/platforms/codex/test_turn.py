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


def test_web_search_end_is_self_contained_and_produces_a_tool_call(tmp_path: Path) -> None:
    """Real Codex rollouts never emit a matching ``response_item/web_search_call``
    for a search -- ``event_msg/web_search_end`` carries everything (call_id,
    query, action, results) by itself, the same way ``mcp_tool_call_end`` does.
    Checked against 34 real local rollouts (18 ``web_search_end`` frames, 0
    ``web_search_call`` frames) before writing this test.
    """
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
                    "2026-01-01T00:00:03.000Z",
                    "event_msg",
                    {
                        "type": "web_search_end",
                        "call_id": "exec-e558476c-83b3-40c0-afb4-ed9cc0d2a715",
                        "query": "site:example.com docs",
                        "action": {"type": "search", "queries": ["site:example.com docs"]},
                        "results": [
                            {"type": "text_result", "title": "Example", "url": "https://x"}
                        ],
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
    tool_calls = turn["calls"][0]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "web_search"
    assert tool_calls[0]["tool_call_id"] == "exec-e558476c-83b3-40c0-afb4-ed9cc0d2a715"
    assert tool_calls[0]["start_ts"] == "2026-01-01T00:00:03.000Z"
    assert tool_calls[0]["attributes"]["arguments"] == "site:example.com docs"


def test_web_search_end_open_page_action_is_named_open_page(tmp_path: Path) -> None:
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
                        "type": "web_search_end",
                        "call_id": "exec-1",
                        "query": "https://example.com/page",
                        "action": {"type": "open_page", "url": "https://example.com/page"},
                        "results": [],
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
    assert turn["calls"][0]["tool_calls"][0]["name"] == "open_page"


def test_multiple_web_searches_in_one_turn_do_not_cross_contaminate(tmp_path: Path) -> None:
    """Each ``web_search_end`` is self-contained, so two searches back-to-back
    (or, before the fix this test guards, interleaved with an unrelated
    ``response_item/web_search_call`` from a different call) can never mix up
    each other's call_id/query the way a single shared correlation slot would.
    """
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
                    "2026-01-01T00:00:02Z",
                    "event_msg",
                    {
                        "type": "web_search_end",
                        "call_id": "exec-a",
                        "query": "query a",
                        "action": {"type": "search"},
                        "results": [],
                    },
                ),
                _frame(
                    "2026-01-01T00:00:03Z",
                    "event_msg",
                    {
                        "type": "web_search_end",
                        "call_id": "exec-b",
                        "query": "query b",
                        "action": {"type": "search"},
                        "results": [],
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
    tool_calls = turn["calls"][0]["tool_calls"]
    assert [tc["tool_call_id"] for tc in tool_calls] == ["exec-a", "exec-b"]
    assert [tc["attributes"]["arguments"] for tc in tool_calls] == ["query a", "query b"]


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


def test_response_item_user_message_supplies_prompt_when_no_user_message_event(
    tmp_path: Path,
) -> None:
    """Newer Codex CLI builds (checked against local rollouts from 2026-08-20)
    stop emitting ``event_msg/user_message`` entirely -- the only record of
    what the user typed is a ``response_item/message`` frame with
    ``role: "user"`` (the OpenAI Responses API shape also used for the
    assistant's reply). Before this fix, ``user_prompt`` stayed empty for
    every such turn, so Logfire showed an ``agent-turn`` span with an output
    but no input for the majority of real Codex sessions.
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
                    "response_item",
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "is claude set up too?"}],
                    },
                ),
                _frame(
                    "2026-01-01T00:00:03Z",
                    "event_msg",
                    {"type": "agent_message", "message": "yes it is"},
                ),
                _frame(
                    "2026-01-01T00:00:04Z",
                    "event_msg",
                    {"type": "task_complete", "turn_id": "t1", "last_agent_message": "yes it is"},
                ),
            ]
        )
        + "\n"
    )

    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert turn["user_prompt"] == "is claude set up too?"
    assert turn["calls"][0]["input_messages"][0]["parts"][0]["content"] == "is claude set up too?"


def test_response_item_user_message_skips_synthetic_environment_context(tmp_path: Path) -> None:
    """Codex also injects environment/context and other ambient state as its
    own ``response_item/message`` frames with ``role: "user"`` (verified
    against real rollouts: ``<environment_context>``, ``<recommended_plugins>``,
    ``<task-notification>``, and ``<in-app-browser-context>`` all appear this
    way). These are not something the user typed, so they must not become
    ``user_prompt`` even when they arrive before the real prompt.
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
                    "response_item",
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "<environment_context>\n  <cwd>/x</cwd>\n</environment_context>",
                            }
                        ],
                    },
                ),
                _frame(
                    "2026-01-01T00:00:03Z",
                    "response_item",
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "fix the bug"}],
                    },
                ),
                _frame(
                    "2026-01-01T00:00:04Z",
                    "event_msg",
                    {"type": "task_complete", "turn_id": "t1", "last_agent_message": "done"},
                ),
            ]
        )
        + "\n"
    )

    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert turn["user_prompt"] == "fix the bug"


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


def test_turn_context_model_reaches_the_exported_agent_name(tmp_path: Path) -> None:
    """Codex names its model once per turn, on `turn_context`, not per call.

    The exporter reads the model off the turn's calls, so this checks the
    single per-turn value actually lands on every call it stamps.
    """
    from thirdeye.otel_export import _agent_name, _turn_model

    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                _frame(
                    "2026-01-01T00:00:00Z", "turn_context", {"turn_id": "t1", "model": "gpt-5.6"}
                ),
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
                        "arguments": "{}",
                    },
                ),
                _frame(
                    "2026-01-01T00:00:04Z",
                    "response_item",
                    {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                ),
                _frame(
                    "2026-01-01T00:00:06Z", "event_msg", {"type": "task_complete", "turn_id": "t1"}
                ),
            ]
        )
        + "\n"
    )

    turn = extract_turn_codex(str(path), "t1")
    assert turn is not None
    assert {call["model"] for call in turn["calls"]} == {"gpt-5.6"}
    # `calls` is what the tracing layer hands the exporter as `llm_calls`.
    assert _agent_name("codex", _turn_model({"llm_calls": turn["calls"]})) == "codex[gpt-5.6]"
