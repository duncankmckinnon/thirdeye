"""Reconstruct Cursor generations as Thirdeye turns.

Cursor hooks are point-in-time JSON callbacks. This adapter keeps their raw
payloads in the local event log, pairs before/after tool callbacks, and emits
the generic turn model consumed by the OTel GenAI Logfire exporter.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from thirdeye.reader import SessionReader
from thirdeye.span_ids import turn_span_id
from thirdeye.tracing.model import ToolCallSpanDict, TurnSpanDict, UsageDict


def _data(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("data")
    return value if isinstance(value, dict) else {}


def _text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _integer(data: dict[str, Any], snake: str, camel: str) -> int | None:
    value = data.get(snake)
    if value is None:
        value = data.get(camel)
    if value in (None, "", "--"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _structured(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return _structured(value)
    return None


def _start_ts(event: dict[str, Any]) -> str:
    end_ts = str(event.get("ts") or "")
    duration = _integer(_data(event), "duration", "durationMs")
    if not end_ts or duration is None or duration < 0:
        return end_ts
    try:
        normalized = end_ts[:-1] + "+00:00" if end_ts.endswith("Z") else end_ts
        started = datetime.fromisoformat(normalized) - timedelta(milliseconds=duration)
        return started.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except ValueError:
        return end_ts


def _provider(model: str) -> str:
    normalized = model.lower()
    if "claude" in normalized:
        return "anthropic"
    if "gemini" in normalized:
        return "gcp.gemini"
    if "deepseek" in normalized:
        return "deepseek"
    if "grok" in normalized:
        return "x_ai"
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return ""


def usage_from_payload(data: dict[str, Any]) -> UsageDict:
    uncached = _integer(data, "input_tokens", "inputTokens")
    cache_read = _integer(data, "cache_read_tokens", "cacheReadTokens")
    cache_write = _integer(data, "cache_write_tokens", "cacheWriteTokens")
    output = _integer(data, "output_tokens", "outputTokens")
    usage: UsageDict = {}
    if uncached is not None:
        # OTel requires input_tokens to include cached and cache-creation tokens.
        usage["input_tokens"] = uncached + (cache_read or 0) + (cache_write or 0)
    if output is not None:
        usage["output_tokens"] = output
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    return usage


def _tool_span(
    *,
    session_id: str,
    generation_id: str,
    name: str,
    start: dict[str, Any],
    end: dict[str, Any],
) -> ToolCallSpanDict:
    start_data, end_data = _data(start), _data(end)
    call_id = _text(
        start_data,
        "tool_call_id",
        "toolCallId",
        "tool_use_id",
        "toolUseId",
        "call_id",
        "callId",
    ) or _text(
        end_data,
        "tool_call_id",
        "toolCallId",
        "tool_use_id",
        "toolUseId",
        "call_id",
        "callId",
    )
    call_id = call_id or f"{generation_id}:{name}:{start.get('seq', 0)}"
    arguments = _value(
        start_data,
        "tool_input",
        "toolInput",
        "arguments",
        "command",
        "file_path",
        "filePath",
        "path",
    )
    result = _value(
        end_data,
        "tool_output",
        "toolOutput",
        "result",
        "output",
        "stdout",
        "response",
        "edits",
        "diff",
    )
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": name,
        "gen_ai.tool.call.id": call_id,
    }
    if arguments:
        attributes["gen_ai.tool.call.arguments"] = arguments
    if result:
        attributes["gen_ai.tool.call.result"] = result
    exit_code = end_data.get("exit_code", end_data.get("exitCode"))
    if exit_code is not None:
        attributes["cursor.tool.exit_code"] = exit_code
    return {
        "tool_call_id": call_id,
        "name": name,
        "start_ts": (
            _start_ts(end) if start is end else str(start.get("ts") or end.get("ts") or "")
        ),
        "end_ts": str(end.get("ts") or start.get("ts") or ""),
        "attributes": attributes,
    }


def tool_calls_for_generation(
    events: list[dict[str, Any]], session_id: str, generation_id: str
) -> list[ToolCallSpanDict]:
    open_tools: dict[str, list[dict[str, Any]]] = {}
    completed: list[ToolCallSpanDict] = []
    for event in events:
        event_type = str(event.get("t") or "")
        data = _data(event)
        name = _text(data, "tool_name", "toolName", "name", "tool") or "unknown"
        family = str(data.get("cursor_tool_family") or name)
        if event_type == "tool_call" and not data.get("cursor_instant"):
            open_tools.setdefault(family, []).append(event)
            continue
        if event_type == "tool_result" and not data.get("cursor_instant"):
            stack = open_tools.get(family) or []
            start = stack.pop() if stack else event
            completed.append(
                _tool_span(
                    session_id=session_id,
                    generation_id=generation_id,
                    name=name,
                    start=start,
                    end=event,
                )
            )
            continue
        if event_type in {"tool_call", "tool_result"}:
            completed.append(
                _tool_span(
                    session_id=session_id,
                    generation_id=generation_id,
                    name=name,
                    start=event,
                    end=event,
                )
            )
    return completed


def build_turn(
    *, session_dir_: Path, session_id: str, generation_id: str, stop_seq: int
) -> TurnSpanDict | None:
    all_events = list(SessionReader(session_dir_).iter_events(seq_range=(0, stop_seq + 1)))
    events = [
        event
        for event in all_events
        if str(_data(event).get("generation_id") or _data(event).get("generationId") or "")
        == generation_id
    ]
    if not events:
        return None
    prompt_event = next((event for event in events if event.get("t") == "user_message"), None)
    response_events = [event for event in events if event.get("t") == "assistant_message"]
    stop_event = next(
        (event for event in reversed(events) if event.get("t") == "turn_stop"), events[-1]
    )
    response_event = response_events[-1] if response_events else None
    prompt_data = _data(prompt_event or {})
    response_data = _data(response_event or {})
    stop_data = _data(stop_event)
    prompt = _text(prompt_data, "prompt", "input", "text")
    response = _text(response_data, "text", "response", "output")
    model = _text(stop_data, "model", "model_name") or _text(response_data, "model", "model_name")
    usage = usage_from_payload(stop_data)
    start_event = prompt_event or events[0]
    start_ts = str(start_event.get("ts") or "")
    end_ts = str(stop_event.get("ts") or start_ts)
    tools = tool_calls_for_generation(events, session_id, generation_id)
    # Tools already dispatched live keep the same deterministic parent IDs and
    # must not be emitted again with the completed turn.
    from thirdeye.platforms.cursor.live_spans import committed_tool_call_ids

    committed_tools = committed_tool_call_ids(session_dir_, generation_id)
    tools = [tool for tool in tools if tool["tool_call_id"] not in committed_tools]
    llm_calls = []
    if prompt or response or model or usage:
        llm_calls.append(
            {
                "call_id": generation_id,
                "provider": _provider(model),
                "model": model,
                "start_ts": start_ts,
                "end_ts": str((response_event or stop_event).get("ts") or end_ts),
                "input_messages": (
                    [{"role": "user", "parts": [{"type": "text", "content": prompt}]}]
                    if prompt
                    else []
                ),
                "output_messages": (
                    [{"role": "assistant", "parts": [{"type": "text", "content": response}]}]
                    if response
                    else []
                ),
                "usage": usage,
                "tool_calls": tools,
            }
        )
    turn_seq = int(start_event.get("seq") or 0)
    status_value = _text(stop_data, "status", "reason").lower()
    status = "errored" if status_value in {"error", "failed", "failure"} else "completed"
    return {
        "turn_id": str(turn_seq),
        "turn_span_id": str(turn_span_id(session_id, turn_seq)),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "input_message": prompt,
        "output_message": response,
        "status": status,
        "llm_calls": llm_calls,
        "permission_requests": [],
        "subagents": [],
        "attributes": {"cursor.generation.id": generation_id},
    }
