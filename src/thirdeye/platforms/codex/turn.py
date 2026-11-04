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

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from thirdeye.platforms.codex.rollout import iter_frames


def _message_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        str(block.get("text") or block.get("content"))
        for block in payload.get("content") or []
        if isinstance(block, dict) and (block.get("text") or block.get("content"))
    )


def _normalize_messages(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use response items when available, otherwise modern message events.

    Called on one turn only. Matching counts preserve repeated messages while
    suppressing the duplicate item_completed representation, even when a token
    report separates the two records. Other item kinds retain their raw shape.
    """
    responses = Counter(
        (p.get("role"), p.get("phase"), _message_text(p))
        for f in frames
        if f.get("type") == "response_item"
        and isinstance(p := f.get("payload"), dict)
        and p.get("type") == "message"
    )
    result = []
    seen_ids = set()
    for frame in frames:
        p = frame.get("payload")
        if not isinstance(p, dict):
            continue
        item = p.get("item")
        if (
            frame.get("type") == "event_msg"
            and p.get("type") == "item_completed"
            and isinstance(item, dict)
            and item.get("type") in {"AgentMessage", "UserMessage"}
        ):
            item_id = item.get("id")
            if item_id and item_id in seen_ids:
                continue
            if item_id:
                seen_ids.add(item_id)
            message = {
                **item,
                "type": "message",
                "role": "assistant" if item["type"] == "AgentMessage" else "user",
            }
            key = (message["role"], message.get("phase"), _message_text(message))
            if responses[key]:
                responses[key] -= 1
                continue
            frame = {**frame, "type": "response_item", "payload": message}
        result.append(frame)
    return result


def _iso_ms(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def _unix_seconds_to_iso(value: Any, fallback: str) -> str:
    if isinstance(value, (int, float)):
        return _iso_ms(datetime.fromtimestamp(value, tz=UTC))
    return fallback


def _subtract_duration(ts: str, duration: Any) -> str:
    if not ts or not isinstance(duration, dict):
        return ts
    try:
        end = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        delta = timedelta(
            seconds=float(duration.get("secs", 0)) + float(duration.get("nanos", 0)) / 1e9
        )
        return _iso_ms(end - delta)
    except (TypeError, ValueError, OverflowError):
        return ts


# Codex also injects environment/context and other ambient state as its own
# `response_item`/`message` frames with `role: "user"` (the OpenAI Responses
# API shape), not something the person actually typed. Verified against real
# rollouts: these are the only tag names seen wrapping such synthetic content.
_SYNTHETIC_USER_TAGS = (
    "<environment_context>",
    "<recommended_plugins>",
    "<task-notification>",
    "<in-app-browser-context",
)

# AGENTS.md arrives as an untagged heading over an <INSTRUCTIONS> block.
_AGENTS_MD_PREFIX = "# AGENTS.md instructions"


def _strip_synthetic_prefix(text: str) -> str:
    """Drop leading synthetic blocks, keeping any prompt the user typed after them.

    The Codex app prepends ``<in-app-browser-context>`` to the typed prompt in
    the same message, so a synthetic opening tag does not make the whole
    message synthetic. An unclosed block is treated as entirely synthetic.
    """
    text = text.strip()
    while text.startswith((*_SYNTHETIC_USER_TAGS, _AGENTS_MD_PREFIX)):
        if text.startswith(_AGENTS_MD_PREFIX):
            close = "</INSTRUCTIONS>"
        else:
            close = "</" + text[1:].split(">", 1)[0].split(None, 1)[0] + ">"
        end = text.find(close)
        if end == -1:
            return ""
        text = text[end + len(close) :].strip()
    return text


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

    try:
        frames = [frame for _, frame in iter_frames(path, 0)]
    except OSError:
        return None

    # Select the turn before normalizing overlapping message representations.
    selected = []
    selecting = False
    for frame in frames:
        p = frame.get("payload")
        if not isinstance(p, dict):
            continue
        if frame.get("type") == "turn_context" or (
            frame.get("type") == "event_msg" and p.get("type") == "task_started"
        ):
            identity = p.get("turn_id")
            if selecting and identity and identity != turn_id:
                break
            selecting = selecting or identity == turn_id
        if selecting:
            selected.append(frame)

    for frame in _normalize_messages(selected):
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
            text = str(payload.get("message") or "")
            if payload.get("phase") != "commentary":
                assistant_output = text or assistant_output
            end_ts = ts or end_ts
            # Newer rollouts also carry a response_item/message for this text.
            # Keep this as a fallback for older schemas without duplicating it.
            if text and not any(
                part.get("type") == "text" and part.get("content") == text for part in output_parts
            ):
                output_parts.append({"type": "text", "content": text})
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

        if outer == "response_item" and subtype == "message" and payload.get("role") == "user":
            texts = [
                str(block.get("text") or block.get("content"))
                for block in payload.get("content") or []
                if isinstance(block, dict) and (block.get("text") or block.get("content"))
            ]
            joined = _strip_synthetic_prefix("\n".join(texts))
            if joined:
                user_prompt = joined
                if not input_parts:
                    input_parts.append({"type": "text", "content": joined})
                call_start_ts = call_start_ts or ts
            continue

        if outer == "response_item" and subtype == "message" and payload.get("role") == "assistant":
            if payload.get("phase") == "final_answer":
                assistant_output = _message_text(payload) or assistant_output
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
            # Self-contained, like mcp_tool_call_end: real Codex rollouts
            # carry call_id/query/action/results directly on this frame and
            # never emit a matching response_item/web_search_call to pair it
            # with (confirmed against 34 real local rollouts -- 18
            # web_search_end frames, 0 web_search_call frames).
            action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
            action_type = str(action.get("type") or "search")
            entry = {
                "name": "open_page" if action_type == "open_page" else "web_search",
                "call_id": str(payload.get("call_id") or ""),
                "arguments": payload.get("query") or action.get("url") or "",
                "result": payload.get("results") or "",
                "start_ts": ts,
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
            last_output_ts = ts or last_output_ts

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
