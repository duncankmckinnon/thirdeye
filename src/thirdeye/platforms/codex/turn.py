"""Reconstruct one completed Codex turn from its rollout JSONL.

Codex's notify payload is intentionally small. The rollout is the source of
truth for turn timing, messages, usage, and tool execution, so
``extract_turn_codex`` walks it to group frames into one entry per model
inference (``calls``, each already shaped as a
``thirdeye.tracing.model.LlmCallSpanDict`` with its own ``ToolCallSpanDict``
children) instead of trying to correlate the many repeated ``token_count``
frames to individual responses some other way. ``platforms/codex/tracing.py``
adapts this dict's shape into a full ``TurnSpanDict``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _unix_seconds_to_iso(value: Any, fallback: str) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")
    return fallback


def _subtract_duration(ts: str, duration: Any) -> str:
    if not ts or not isinstance(duration, dict):
        return ts
    try:
        end = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        delta = timedelta(
            seconds=float(duration.get("secs", 0)) + float(duration.get("nanos", 0)) / 1e9
        )
        return (end - delta).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError):
        return ts


# Renames a Codex rollout usage block's own key names onto UsageDict's, per
# thirdeye.tracing.model.UsageDict.
_USAGE_RENAME = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cached_input_tokens": "cache_read_input_tokens",
    "cache_write_input_tokens": "cache_creation_input_tokens",
    "reasoning_output_tokens": "reasoning_output_tokens",
}


def _usage_dict(raw: dict[str, Any]) -> dict[str, int]:
    return {
        target: raw[source]
        for source, target in _USAGE_RENAME.items()
        if isinstance(raw.get(source), int)
    }


def _tool_call_dicts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tool_call_id": entry["call_id"],
            "name": entry["name"],
            "start_ts": entry["start_ts"],
            "end_ts": entry["end_ts"],
            "attributes": {"arguments": entry["arguments"], "result": entry["result"]},
        }
        for entry in entries
    ]


def _llm_call_dict(
    *,
    turn_id: str,
    call_index: int,
    model: str,
    start_ts: str,
    end_ts: str,
    input_parts: list[dict[str, Any]],
    output_parts: list[dict[str, Any]],
    usage: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "call_id": f"{turn_id}:{call_index}",
        "provider": "openai",
        "model": model,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "input_messages": (
            [{"role": "user" if call_index == 0 else "tool", "parts": input_parts}]
            if input_parts
            else []
        ),
        "output_messages": ([{"role": "assistant", "parts": output_parts}] if output_parts else []),
        "usage": _usage_dict(usage),
        "tool_calls": _tool_call_dicts(tools),
    }


def extract_turn_codex(rollout_path: str, turn_id: str) -> dict[str, Any] | None:
    path = Path(rollout_path)
    if not turn_id or not path.is_file():
        return None

    in_turn = False
    status = "completed"
    start_ts = ""
    end_ts = ""
    model = ""
    user_prompt = ""
    assistant_output = ""
    usage_by_total: dict[int, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    input_parts: list[dict[str, Any]] = []
    output_parts: list[dict[str, Any]] = []
    call_tools: list[dict[str, Any]] = []
    next_input_parts: list[dict[str, Any]] = []
    call_start_ts = ""
    last_output_ts = ""
    pending: dict[str, dict[str, Any]] = {}
    pending_search: dict[str, Any] | None = None

    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None

    for line in lines:
        try:
            frame = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        outer = frame.get("type")
        payload = frame.get("payload") if isinstance(frame.get("payload"), dict) else {}
        subtype = payload.get("type")
        ts = str(frame.get("timestamp") or "")

        if outer == "turn_context":
            frame_turn = str(payload.get("turn_id") or "")
            if in_turn and frame_turn and frame_turn != turn_id:
                break
            if frame_turn == turn_id:
                in_turn = True
                start_ts = start_ts or ts
                model = str(payload.get("model") or model)
            continue

        if outer == "event_msg" and subtype == "task_started":
            frame_turn = str(payload.get("turn_id") or "")
            if in_turn and frame_turn and frame_turn != turn_id:
                break
            if frame_turn == turn_id:
                in_turn = True
                start_ts = _unix_seconds_to_iso(payload.get("started_at"), ts or start_ts)
            continue
        if not in_turn:
            continue

        if outer == "event_msg" and subtype == "task_complete":
            frame_turn = str(payload.get("turn_id") or turn_id)
            if frame_turn == turn_id:
                assistant_output = str(payload.get("last_agent_message") or assistant_output)
                end_ts = _unix_seconds_to_iso(payload.get("completed_at"), ts or end_ts)
            continue
        if outer == "event_msg" and subtype == "turn_aborted":
            frame_turn = str(payload.get("turn_id") or turn_id)
            if frame_turn == turn_id:
                status = "interrupted"
                end_ts = ts or end_ts
                break
            continue
        if outer == "event_msg" and subtype == "user_message":
            user_prompt = str(payload.get("message") or user_prompt)
            if user_prompt and not input_parts:
                input_parts.append({"type": "text", "content": user_prompt})
            start_ts = start_ts or ts
            call_start_ts = call_start_ts or ts
            continue
        if outer == "event_msg" and subtype == "agent_message":
            assistant_output = str(payload.get("message") or assistant_output)
            end_ts = ts or end_ts
            # Newer rollouts also carry a response_item/message for this text.
            # Keep this as a fallback for older schemas without duplicating it.
            if assistant_output and not any(
                part.get("type") == "text" and part.get("content") == assistant_output
                for part in output_parts
            ):
                output_parts.append({"type": "text", "content": assistant_output})
                last_output_ts = ts or last_output_ts
            continue
        if outer == "event_msg" and subtype == "token_count":
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            last = info.get("last_token_usage")
            total = info.get("total_token_usage")
            watermark = total.get("total_tokens") if isinstance(total, dict) else None
            if isinstance(last, dict) and isinstance(watermark, int):
                # Codex repeats the latest count.  Last-wins by cumulative
                # watermark preserves one delta per actual inference call.
                is_new_call = watermark not in usage_by_total
                usage_by_total[watermark] = last
                end_ts = ts or end_ts
                if is_new_call and (output_parts or call_tools):
                    calls.append(
                        _llm_call_dict(
                            turn_id=turn_id,
                            call_index=len(calls),
                            model=model,
                            start_ts=call_start_ts or start_ts or ts,
                            end_ts=last_output_ts or ts,
                            input_parts=input_parts,
                            output_parts=output_parts,
                            usage=dict(last),
                            tools=call_tools,
                        )
                    )
                    input_parts = next_input_parts
                    output_parts = []
                    call_tools = []
                    next_input_parts = []
                    call_start_ts = ts
                    last_output_ts = ""
            continue

        if outer == "response_item" and subtype == "message" and payload.get("role") == "assistant":
            for block in payload.get("content") or []:
                if not isinstance(block, dict):
                    continue
                text = block.get("text") or block.get("content")
                if text and not any(
                    part.get("type") == "text" and part.get("content") == text
                    for part in output_parts
                ):
                    output_parts.append({"type": "text", "content": text})
            last_output_ts = ts or last_output_ts
            continue
        if outer == "response_item" and subtype == "reasoning":
            summaries = payload.get("summary") or []
            texts = []
            for item in summaries:
                if isinstance(item, dict) and (item.get("text") or item.get("content")):
                    texts.append(str(item.get("text") or item.get("content")))
                elif isinstance(item, str):
                    texts.append(item)
            if texts:
                output_parts.append({"type": "reasoning", "content": "\n".join(texts)})
                last_output_ts = ts or last_output_ts
            continue

        if outer == "response_item" and subtype in {
            "function_call",
            "custom_tool_call",
            "local_shell_call",
        }:
            call_id = str(payload.get("call_id") or "")
            entry = {
                "name": str(payload.get("name") or subtype),
                "call_id": call_id,
                "arguments": payload.get("arguments", payload.get("input", "")),
                "result": "",
                "start_ts": ts,
                "end_ts": ts,
            }
            call_tools.append(entry)
            output_parts.append(
                {
                    "type": "tool_call",
                    "id": call_id,
                    "name": entry["name"],
                    "arguments": entry["arguments"],
                }
            )
            last_output_ts = ts or last_output_ts
            if call_id:
                pending[call_id] = entry
            continue
        if outer == "response_item" and subtype == "image_generation_call":
            call_id = str(payload.get("call_id") or payload.get("id") or "")
            entry = {
                "name": "image_generation",
                "call_id": call_id,
                "arguments": payload.get("prompt") or payload.get("input") or "",
                "result": payload.get("status") or payload.get("result") or "",
                "start_ts": ts,
                "end_ts": ts,
            }
            call_tools.append(entry)
            output_parts.append(
                {
                    "type": "tool_call",
                    "id": call_id,
                    "name": entry["name"],
                    "arguments": entry["arguments"],
                }
            )
            last_output_ts = ts or last_output_ts
            continue
        if outer == "event_msg" and subtype == "mcp_tool_call_end":
            invocation = (
                payload.get("invocation") if isinstance(payload.get("invocation"), dict) else {}
            )
            server = str(invocation.get("server") or "mcp")
            tool_name = str(invocation.get("tool") or "tool")
            entry = {
                "name": f"{server}.{tool_name}",
                "call_id": str(payload.get("call_id") or ""),
                "arguments": invocation.get("arguments") or {},
                "result": payload.get("result") or "",
                "start_ts": _subtract_duration(ts, payload.get("duration")),
                "end_ts": ts,
            }
            call_tools.append(entry)
            output_parts.append(
                {
                    "type": "tool_call",
                    "id": entry["call_id"],
                    "name": entry["name"],
                    "arguments": entry["arguments"],
                }
            )
            last_output_ts = entry["start_ts"] or last_output_ts
            continue
        if outer == "response_item" and subtype in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            call_id = str(payload.get("call_id") or "")
            entry = pending.get(call_id)
            if entry is not None:
                entry["result"] = payload.get("output", "")
                entry["end_ts"] = ts or entry["start_ts"]
                next_input_parts.append(
                    {
                        "type": "tool_call_response",
                        "id": call_id,
                        "response": payload.get("output", ""),
                    }
                )
            continue
        if outer == "event_msg" and subtype == "web_search_end":
            pending_search = {"call_id": str(payload.get("call_id") or ""), "ts": ts}
            continue
        if outer == "response_item" and subtype == "web_search_call":
            action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
            action_type = str(action.get("type") or "search")
            entry = {
                "name": "open_page" if action_type == "open_page" else "web_search",
                "call_id": (pending_search or {}).get("call_id", ""),
                "arguments": action.get("url") or action.get("query") or "",
                "result": payload.get("status", ""),
                "start_ts": (pending_search or {}).get("ts") or ts,
                "end_ts": ts,
            }
            call_tools.append(entry)
            output_parts.append(
                {
                    "type": "tool_call",
                    "id": entry["call_id"],
                    "name": entry["name"],
                    "arguments": entry["arguments"],
                }
            )
            last_output_ts = entry["start_ts"] or last_output_ts
            pending_search = None

    if not in_turn:
        return None

    # Some older/trimmed rollouts omit the final token_count frame. Preserve
    # their final visible assistant output as a call even without usage.
    if output_parts or call_tools:
        calls.append(
            _llm_call_dict(
                turn_id=turn_id,
                call_index=len(calls),
                model=model,
                start_ts=call_start_ts or start_ts or end_ts,
                end_ts=last_output_ts or end_ts or start_ts,
                input_parts=input_parts,
                output_parts=output_parts,
                usage={},
                tools=call_tools,
            )
        )

    return {
        "turn_id": turn_id,
        "start_ts": start_ts or end_ts,
        "end_ts": end_ts or start_ts,
        "user_prompt": user_prompt,
        "assistant_output": assistant_output,
        "status": status,
        "calls": calls,
    }
