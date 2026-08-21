"""Reconstruct one Claude Code turn — completed or interrupted — from
thirdeye's own local event store plus the Claude transcript, for handoff to
`thirdeye.otel_export.export_turn`.

The event store alone carries tool calls, permission requests, and subagent
boundaries (paired by `tool_use_id`); the transcript alone carries the actual
LLM call content (messages, usage). `build_turn` is the one place that joins
the two into a single `TurnSpanDict`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.platforms.claude.usage import extract_calls_from_transcript
from thirdeye.reader import SessionReader
from thirdeye.tracing.model import (
    LlmCallSpanDict,
    PermissionRequestSpanDict,
    ToolCallSpanDict,
    TurnSpanDict,
)

_CORRELATED_TYPES = frozenset(
    {
        "tool_call",
        "tool_result",
        "permission_request",
        "permission_denied",
        "subagent_start",
        "subagent_message",
    }
)


def _tool_use_id(data: Any) -> str | None:
    tu_id = data.get("tool_use_id") if isinstance(data, dict) else None
    return str(tu_id) if tu_id else None


def _pair_tool_calls(events: list[dict[str, Any]]) -> list[ToolCallSpanDict]:
    """Pair `tool_call`/`tool_result` events by `tool_use_id`, mirroring
    `web/routes/sessions.py`'s `_pair_events`. A call with no result yet
    (still running, or the turn was interrupted before it finished) is
    simply omitted — there's nothing to attach to an LLM call's tool_calls.
    """
    open_calls: dict[str, dict[str, Any]] = {}
    pairs: list[ToolCallSpanDict] = []
    for ev in events:
        data = ev.get("data") or {}
        if ev.get("t") == "tool_call":
            tu_id = _tool_use_id(data)
            if tu_id:
                open_calls[tu_id] = ev
        elif ev.get("t") == "tool_result":
            tu_id = _tool_use_id(data)
            call_ev = open_calls.pop(tu_id, None) if tu_id else None
            if call_ev is None:
                continue
            call_data = call_ev.get("data") or {}
            pairs.append(
                {
                    "tool_call_id": tu_id,
                    "name": str(call_data.get("tool_name") or ""),
                    "start_ts": str(call_ev.get("ts") or ""),
                    "end_ts": str(ev.get("ts") or ""),
                    "attributes": {**call_data, **data},
                }
            )
    return pairs


def _pair_permission_events(events: list[dict[str, Any]]) -> list[PermissionRequestSpanDict]:
    out: list[PermissionRequestSpanDict] = []
    for ev in events:
        if ev.get("t") not in {"permission_request", "permission_denied"}:
            continue
        data = ev.get("data") or {}
        out.append(
            {
                "ts": str(ev.get("ts") or ""),
                "tool_name": str(data.get("tool_name") or ""),
                "attributes": dict(data),
            }
        )
    return out


def _attach_tool_calls(
    llm_calls: list[LlmCallSpanDict], tool_calls: list[ToolCallSpanDict]
) -> None:
    """Attach each tool call to the LLM call whose `output_messages` requested
    it — a `tool_call` message part carries the same `tool_use_id` Claude
    generated, which is also the `id` thirdeye's own tool_call/tool_result
    events are keyed on.
    """
    by_id = {tc["tool_call_id"]: tc for tc in tool_calls}
    for call in llm_calls:
        attached: list[ToolCallSpanDict] = []
        for message in call["output_messages"]:
            for part in message.get("parts", []):
                if part.get("type") != "tool_call":
                    continue
                tc = by_id.get(str(part.get("id") or ""))
                if tc is not None:
                    attached.append(tc)
        call["tool_calls"] = attached


def _build_subagent_turn(start_ev: dict[str, Any], stop_ev: dict[str, Any]) -> TurnSpanDict:
    start_data = start_ev.get("data") or {}
    stop_data = stop_ev.get("data") or {}
    # `hooks.subagent_stop` stores the SubagentStop payload's
    # `agent_transcript_path` under this renamed key specifically so it
    # survives `_STRIP_KEYS` (which otherwise strips every
    # `agent_transcript_path` as noise) — see that function's own comment.
    transcript_path = stop_data.get("agent_transcript")
    llm_calls, _ = extract_calls_from_transcript(transcript_path, 0)
    return {
        "turn_id": str(start_ev.get("seq")),
        "start_ts": str(start_ev.get("ts") or ""),
        "end_ts": str(stop_ev.get("ts") or ""),
        "input_message": str(start_data.get("prompt") or start_data.get("description") or ""),
        "output_message": str(stop_data.get("result") or stop_data.get("output") or ""),
        # A subagent only ever shows up here once its own SubagentStop has
        # actually fired, so it completed by definition — Claude Code has no
        # documented "subagent was interrupted" signal to check instead.
        "status": "completed",
        "llm_calls": llm_calls,
        "permission_requests": [],
        "subagents": [],
        "attributes": {
            k: v
            for k, v in start_data.items()
            if k not in {"prompt", "description", "tool_use_id"}
        },
    }


def _pair_subagents(events: list[dict[str, Any]]) -> list[TurnSpanDict]:
    by_id: dict[str, dict[str, Any]] = {}
    fifo: list[dict[str, Any]] = []
    subagents: list[TurnSpanDict] = []
    for ev in events:
        data = ev.get("data") or {}
        if ev.get("t") == "subagent_start":
            tu_id = _tool_use_id(data)
            if tu_id:
                by_id[tu_id] = ev
            else:
                fifo.append(ev)
        elif ev.get("t") == "subagent_message":
            tu_id = _tool_use_id(data)
            start_ev = by_id.pop(tu_id, None) if tu_id else None
            if start_ev is None and fifo:
                start_ev = fifo.pop(0)
            if start_ev is None:
                continue
            subagents.append(_build_subagent_turn(start_ev, ev))
    return subagents


def _fallback_output_message(llm_calls: list[LlmCallSpanDict]) -> str:
    """Claude Code's Stop hook payload doesn't document a reliable
    final-response field; when the caller didn't have one to pass in, the
    last text part of the last LLM call's own output is the next best
    source of truth for what the user actually saw.
    """
    for call in reversed(llm_calls):
        for message in reversed(call["output_messages"]):
            for part in reversed(message.get("parts", [])):
                if part.get("type") == "text" and part.get("content"):
                    return str(part["content"])
    return ""


def build_turn(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    stop_seq: int,
    stop_ts: str,
    transcript_path: str | None,
    final_response: str,
) -> TurnSpanDict | None:
    from thirdeye.platforms.claude.hooks import _open_turn_path

    try:
        marker = json.loads(_open_turn_path(session_dir_).read_text())
        turn_seq = int(marker["turn_seq"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    events = list(
        SessionReader(session_dir_).iter_events(
            types=_CORRELATED_TYPES, seq_range=(turn_seq, stop_seq)
        )
    )
    tool_calls = _pair_tool_calls(events)
    permission_requests = _pair_permission_events(events)
    subagents = _pair_subagents(events)

    offset = int(marker.get("transcript_offset", 0))
    llm_calls, _ = extract_calls_from_transcript(
        transcript_path or marker.get("transcript_path"), offset
    )
    _attach_tool_calls(llm_calls, tool_calls)

    return {
        "turn_id": str(turn_seq),
        "start_ts": str(marker.get("start_ts") or ""),
        "end_ts": stop_ts,
        "input_message": str(marker.get("prompt") or ""),
        "output_message": final_response or _fallback_output_message(llm_calls),
        "status": "completed",
        "llm_calls": llm_calls,
        "permission_requests": permission_requests,
        "subagents": subagents,
        "attributes": {},
    }
