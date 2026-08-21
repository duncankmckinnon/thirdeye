from __future__ import annotations

import json

from thirdeye.tracing import (
    LlmCallSpanDict,
    PermissionRequestSpanDict,
    ToolCallSpanDict,
    TurnSpanDict,
    TurnStatus,
    UsageDict,
)
from thirdeye.tracing import model as tracing_model


def test_reexports_match_module() -> None:
    assert tracing_model.TurnSpanDict is TurnSpanDict
    assert tracing_model.LlmCallSpanDict is LlmCallSpanDict
    assert tracing_model.ToolCallSpanDict is ToolCallSpanDict
    assert tracing_model.PermissionRequestSpanDict is PermissionRequestSpanDict
    assert tracing_model.UsageDict is UsageDict


def test_all_matches_re_exported_names() -> None:
    import thirdeye.tracing as pkg

    assert set(pkg.__all__) == {
        "LlmCallSpanDict",
        "PermissionRequestSpanDict",
        "ToolCallSpanDict",
        "TurnSpanDict",
        "TurnStatus",
        "UsageDict",
    }
    for name in pkg.__all__:
        assert hasattr(pkg, name)


def test_usage_dict_is_total_false() -> None:
    usage: UsageDict = {}
    assert usage == {}
    usage = {"input_tokens": 1, "output_tokens": 2}
    assert usage["input_tokens"] == 1


def test_tool_call_span_dict_shape() -> None:
    tool_call: ToolCallSpanDict = {
        "tool_call_id": "tc-1",
        "name": "Bash",
        "start_ts": "2026-01-01T00:00:00.000Z",
        "end_ts": "2026-01-01T00:00:01.000Z",
        "attributes": {"cmd": "ls"},
    }
    assert json.loads(json.dumps(tool_call)) == tool_call


def test_llm_call_span_dict_shape() -> None:
    tool_call: ToolCallSpanDict = {
        "tool_call_id": "tc-1",
        "name": "Bash",
        "start_ts": "2026-01-01T00:00:00.000Z",
        "end_ts": "2026-01-01T00:00:01.000Z",
        "attributes": {},
    }
    llm_call: LlmCallSpanDict = {
        "call_id": "call-1",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "start_ts": "2026-01-01T00:00:00.000Z",
        "end_ts": "2026-01-01T00:00:02.000Z",
        "input_messages": [{"role": "user", "content": "hi"}],
        "output_messages": [{"role": "assistant", "content": "hello"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "tool_calls": [tool_call],
    }
    assert json.loads(json.dumps(llm_call)) == llm_call


def test_permission_request_span_dict_shape() -> None:
    req: PermissionRequestSpanDict = {
        "ts": "2026-01-01T00:00:00.000Z",
        "tool_name": "Bash",
        "attributes": {"decision": "allow"},
    }
    assert json.loads(json.dumps(req)) == req


def test_turn_status_literal_values() -> None:
    completed: TurnStatus = "completed"
    interrupted: TurnStatus = "interrupted"
    errored: TurnStatus = "errored"
    assert {completed, interrupted, errored} == {"completed", "interrupted", "errored"}


def test_turn_span_dict_is_json_roundtrippable_and_recursive() -> None:
    subagent: TurnSpanDict = {
        "turn_id": "sub-1",
        "start_ts": "2026-01-01T00:00:00.000Z",
        "end_ts": "2026-01-01T00:00:01.000Z",
        "input_message": "",
        "output_message": "",
        "status": "completed",
        "llm_calls": [],
        "permission_requests": [],
        "subagents": [],
        "attributes": {"display_name": "researcher"},
    }
    turn: TurnSpanDict = {
        "turn_id": "turn-1",
        "start_ts": "2026-01-01T00:00:00.000Z",
        "end_ts": "2026-01-01T00:00:05.000Z",
        "input_message": "do the thing",
        "output_message": "done",
        "status": "completed",
        "llm_calls": [],
        "permission_requests": [],
        "subagents": [subagent],
        "attributes": {},
    }
    round_tripped = json.loads(json.dumps(turn))
    assert round_tripped == turn
    assert round_tripped["subagents"][0]["turn_id"] == "sub-1"


def test_interrupted_turn_allows_empty_output_message() -> None:
    turn: TurnSpanDict = {
        "turn_id": "turn-2",
        "start_ts": "2026-01-01T00:00:00.000Z",
        "end_ts": "2026-01-01T00:00:01.000Z",
        "input_message": "do the thing",
        "output_message": "",
        "status": "interrupted",
        "llm_calls": [],
        "permission_requests": [],
        "subagents": [],
        "attributes": {},
    }
    assert turn["status"] == "interrupted"
    assert turn["output_message"] == ""
