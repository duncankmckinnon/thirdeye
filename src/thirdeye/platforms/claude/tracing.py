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
    """Mirrors `web/routes/sessions.py`'s `_pair_events`. A call with no
    result yet (still running, or the turn was interrupted first) is
    omitted — there's nothing to attach it to.
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


def _build_subagent_turn(
    start_ev: dict[str, Any], stop_ev: dict[str, Any], task_ev: dict[str, Any] | None
) -> TurnSpanDict:
    start_data = start_ev.get("data") or {}
    stop_data = stop_ev.get("data") or {}
    # `hooks.subagent_stop` renames the SubagentStop payload's
    # `agent_transcript_path` to `agent_transcript` so it survives
    # `_STRIP_KEYS`, which otherwise strips every `agent_transcript_path`.
    transcript_path = stop_data.get("agent_transcript")
    llm_calls, _ = extract_calls_from_transcript(transcript_path, 0)

    # SubagentStart/SubagentStop carry no prompt/result text of their own —
    # the task text lives on the `Task` tool invocation that launched this
    # subagent, correlated positionally (see `_pair_subagents`).
    task_input = (task_ev.get("data") or {}).get("tool_input") if task_ev else None
    task_input = task_input if isinstance(task_input, dict) else {}

    return {
        "turn_id": str(start_ev.get("seq")),
        "start_ts": str(start_ev.get("ts") or ""),
        "end_ts": str(stop_ev.get("ts") or ""),
        "input_message": str(task_input.get("prompt") or task_input.get("description") or ""),
        "output_message": str(stop_data.get("last_assistant_message") or ""),
        # A subagent only appears here once its own SubagentStop has fired,
        # so it completed by definition; Claude Code has no "interrupted
        # subagent" signal to check for instead.
        "status": "completed",
        "llm_calls": llm_calls,
        "permission_requests": [],
        "subagents": [],
        "attributes": {k: v for k, v in start_data.items() if k != "agent_id"},
    }


def _pair_subagents(events: list[dict[str, Any]]) -> list[TurnSpanDict]:
    """Starts and stops are paired by `agent_id`, which SubagentStart and
    SubagentStop both carry — parallel subagents can finish out of order, so
    FIFO pairing would mismatch them. The originating `Task` tool call has no
    shared id with either hook, so it's matched positionally: Claude Code
    issues each Task's PreToolUse before that subagent's own SubagentStart,
    in the same relative order, so the oldest not-yet-claimed Task tool_call
    is that start's likely source.
    """
    starts_by_agent: dict[str, dict[str, Any]] = {}
    task_by_agent: dict[str, dict[str, Any]] = {}
    pending_tasks: list[dict[str, Any]] = []
    subagents: list[TurnSpanDict] = []
    for ev in events:
        data = ev.get("data") or {}
        t = ev.get("t")
        if t == "tool_call" and data.get("tool_name") == "Task":
            pending_tasks.append(ev)
        elif t == "subagent_start":
            agent_id = data.get("agent_id")
            if not agent_id:
                continue
            starts_by_agent[str(agent_id)] = ev
            if pending_tasks:
                task_by_agent[str(agent_id)] = pending_tasks.pop(0)
        elif t == "subagent_message":
            agent_id = data.get("agent_id")
            start_ev = starts_by_agent.pop(str(agent_id), None) if agent_id else None
            if start_ev is None:
                continue
            task_ev = task_by_agent.pop(str(agent_id), None)
            subagents.append(_build_subagent_turn(start_ev, ev, task_ev))
    return subagents


def _fallback_output_message(llm_calls: list[LlmCallSpanDict]) -> str:
    """Used when the caller has no `final_response` of its own (the
    interruption catch-all never has one) — the last text part of the last
    LLM call's output is the next best source of what the user actually saw.
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
