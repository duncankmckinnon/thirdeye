from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.platforms.claude.usage import extract_calls_from_transcript
from thirdeye.reader import SessionReader
from thirdeye.span_ids import turn_span_id
from thirdeye.tracing.model import (
    LlmCallSpanDict,
    PermissionRequestSpanDict,
    ToolCallSpanDict,
    TurnSpanDict,
    TurnStatus,
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
_TOOL_AND_PERMISSION_TYPES = frozenset(
    {"tool_call", "tool_result", "permission_request", "permission_denied"}
)
_READ_MARKER = object()


def _unmatched_user_message(session_dir_: Path, stop_seq: int) -> dict[str, Any] | None:
    """Return the open user event immediately before the proving event."""
    candidate = None
    for event in SessionReader(session_dir_).iter_events(
        types={"user_message", "assistant_message", "error"},
        seq_range=(0, stop_seq),
    ):
        if event.get("t") == "user_message":
            candidate = event
        else:
            candidate = None
    return candidate


def _tool_use_id(data: Any) -> str | None:
    tu_id = data.get("tool_use_id") if isinstance(data, dict) else None
    return str(tu_id) if tu_id else None


def _pair_tool_calls(events: list[dict[str, Any]]) -> list[ToolCallSpanDict]:
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


def _first_input_text(llm_calls: list[LlmCallSpanDict]) -> str:
    if not llm_calls:
        return ""
    for message in llm_calls[0]["input_messages"]:
        for part in message.get("parts", []):
            if part.get("type") == "text" and part.get("content"):
                return str(part["content"])
    return ""


def _fallback_output_message(llm_calls: list[LlmCallSpanDict]) -> str:
    for call in reversed(llm_calls):
        for message in reversed(call["output_messages"]):
            for part in reversed(message.get("parts", [])):
                if part.get("type") == "text" and part.get("content"):
                    return str(part["content"])
    return ""


def _subagent_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    pending_tasks: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for ev in events:
        data = ev.get("data") or {}
        if ev.get("t") == "tool_call" and data.get("tool_name") == "Task":
            pending_tasks.append(ev)
            continue
        agent_id = data.get("agent_id")
        if not agent_id:
            continue
        agent_id = str(agent_id)
        if ev.get("t") == "subagent_start":
            starts[agent_id] = ev
            if pending_tasks:
                tasks[agent_id] = pending_tasks.pop(0)
        elif ev.get("t") == "subagent_message":
            start_ev = starts.pop(agent_id, None)
            if start_ev is None:
                continue
            windows.append(
                {
                    "start": int(start_ev["seq"]),
                    "stop": int(ev["seq"]),
                    "agent_id": agent_id,
                    "start_ev": start_ev,
                    "stop_ev": ev,
                    "task_ev": tasks.pop(agent_id, None),
                }
            )
    windows.sort(key=lambda w: w["start"])
    return windows


def _owning_agent(seq: int, windows: list[dict[str, Any]]) -> str | None:
    # Claude Code doesn't tag a subagent's own tool/permission hook events
    # with its agent_id, so the smallest seq-range window containing the
    # event is the best available proxy. This is exact for sequential or
    # properly-nested subagents; genuinely interleaved concurrent subagents
    # sharing overlapping windows can still be misattributed to a sibling.
    owner = None
    owner_width = None
    for w in windows:
        if w["start"] <= seq <= w["stop"]:
            width = w["stop"] - w["start"]
            if owner_width is None or width < owner_width:
                owner, owner_width = w["agent_id"], width
    return owner


def _partition_events(
    events: list[dict[str, Any]], windows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    top_level: list[dict[str, Any]] = []
    owned: dict[str, list[dict[str, Any]]] = {w["agent_id"]: [] for w in windows}
    for ev in events:
        if ev.get("t") not in _TOOL_AND_PERMISSION_TYPES:
            continue
        owner = _owning_agent(int(ev["seq"]), windows)
        if owner is None:
            top_level.append(ev)
        else:
            owned[owner].append(ev)
    return top_level, owned


def _build_subagent_turn(
    start_ev: dict[str, Any],
    stop_ev: dict[str, Any],
    tool_calls: list[ToolCallSpanDict],
    permission_requests: list[PermissionRequestSpanDict],
    task_ev: dict[str, Any] | None,
    *,
    session_id: str,
) -> TurnSpanDict:
    start_data = start_ev.get("data") or {}
    stop_data = stop_ev.get("data") or {}
    # `hooks.subagent_stop` renames the SubagentStop payload's
    # `agent_transcript_path` to `agent_transcript` so it survives `_STRIP_KEYS`.
    llm_calls = extract_calls_from_transcript(stop_data.get("agent_transcript"), 0).calls
    _attach_tool_calls(llm_calls, tool_calls)
    task_input = (task_ev.get("data") or {}).get("tool_input") if task_ev else None
    task_input = task_input if isinstance(task_input, dict) else {}
    return {
        "turn_id": str(start_ev.get("seq")),
        "turn_span_id": str(turn_span_id(session_id, int(start_ev["seq"]))),
        "start_ts": str(start_ev.get("ts") or ""),
        "end_ts": str(stop_ev.get("ts") or ""),
        # The subagent's own transcript opens with its task prompt as a plain
        # user message — more reliable than correlating back to the Task tool
        # call that spawned it, which shares no id with SubagentStart/Stop.
        "input_message": _first_input_text(llm_calls)
        or str(task_input.get("prompt") or task_input.get("description") or ""),
        "output_message": str(stop_data.get("last_assistant_message") or ""),
        # A subagent only appears here once its own SubagentStop has fired,
        # so it completed by definition; Claude Code has no "interrupted
        # subagent" signal to check instead.
        "status": "completed",
        "llm_calls": llm_calls,
        "permission_requests": permission_requests,
        "subagents": [],
        "attributes": {k: v for k, v in start_data.items() if k != "agent_id"},
    }


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
    status: TurnStatus = "completed",
    marker_snapshot: Mapping[str, Any] | None | object = _READ_MARKER,
) -> TurnSpanDict | None:
    from thirdeye.platforms.claude.hooks import _read_open_turn

    marker = _read_open_turn(session_dir_) if marker_snapshot is _READ_MARKER else marker_snapshot
    trusted_cursor = marker is not None
    if marker is not None:
        turn_seq = int(marker["turn_seq"])

    if marker is None:
        user_message = _unmatched_user_message(session_dir_, stop_seq)
        if user_message is None:
            return None
        try:
            turn_seq = int(user_message["seq"])
        except (ValueError, TypeError, KeyError):
            return None
        user_data = user_message.get("data") or {}
        marker = {
            "turn_seq": turn_seq,
            "turn_span_id": str(turn_span_id(session_id, turn_seq)),
            "start_ts": str(user_message.get("ts") or ""),
            "prompt": str(user_data.get("prompt") or ""),
            "prompt_id": user_data.get("prompt_id"),
            "transcript_path": None,
            "transcript_offset": 0,
            "last_frame_ts": None,
        }

    try:
        derived_turn_span_id = str(int(marker["turn_span_id"]))
    except (ValueError, TypeError, KeyError):
        derived_turn_span_id = str(turn_span_id(session_id, turn_seq))

    events = list(
        SessionReader(session_dir_).iter_events(
            types=_CORRELATED_TYPES, seq_range=(turn_seq, stop_seq)
        )
    )
    windows = _subagent_windows(events)
    top_level_events, owned = _partition_events(events, windows)

    subagents = [
        _build_subagent_turn(
            w["start_ev"],
            w["stop_ev"],
            _pair_tool_calls(owned[w["agent_id"]]),
            _pair_permission_events(owned[w["agent_id"]]),
            w["task_ev"],
            session_id=session_id,
        )
        for w in windows
    ]

    tool_calls = _pair_tool_calls(top_level_events)
    permission_requests = _pair_permission_events(top_level_events)

    if trusted_cursor:
        parsed_calls = extract_calls_from_transcript(
            transcript_path or marker.get("transcript_path"),
            int(marker["transcript_offset"]),
            initial_prev_ts=marker.get("last_frame_ts"),
            incremental=False,
        )
        # A `message.id` split across a `user` frame reopens as a second group
        # after the frame that closed it. When that trailing fragment arrives
        # with no further PostToolUse to consume it, this parse is the one that
        # sees it — but live emission already exported a chat span under the
        # same deterministic id, and re-exporting would double-count its
        # tokens. Read the ids straight off the marker mapping rather than via
        # `hooks`, which imports this module.
        committed = {
            call_id
            for call_id in (marker.get("committed_call_ids") or [])
            if isinstance(call_id, str)
        }
        llm_calls = [call for call in parsed_calls.calls if call["call_id"] not in committed]
    else:
        # A session transcript is append-only across turns. Without the marker's
        # byte cursor, parsing it would attach all historical calls to this turn.
        llm_calls = []
    _attach_tool_calls(llm_calls, tool_calls)

    # The interruption catch-all has no real final_response and must not
    # backfill one from the transcript — an empty output is the correct,
    # honest signal for a turn that never actually finished.
    output_message = final_response
    if status == "completed" and not output_message:
        output_message = _fallback_output_message(llm_calls)

    return {
        "turn_id": str(turn_seq),
        "turn_span_id": derived_turn_span_id,
        "start_ts": str(marker.get("start_ts") or ""),
        "end_ts": stop_ts,
        "input_message": str(marker.get("prompt") or ""),
        "output_message": output_message,
        "status": status,
        "llm_calls": llm_calls,
        "permission_requests": permission_requests,
        "subagents": subagents,
        "attributes": {},
    }
