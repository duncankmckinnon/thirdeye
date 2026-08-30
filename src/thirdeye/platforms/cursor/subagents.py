"""Pure lifecycle, request-ID, and transcript primitives for Cursor subagents.

Cursor never hands a child execution's request ID to any hook that could
report it: `subagentStart` runs with parent context and carries the dispatching
Task `tool_call_id`, while the child's own tool hooks carry an internal
`subagentRequestId` as their `generation_id`. The installed local-agent runtime
builds that ID deterministically from the dispatching Task call ID with a fixed
SHA-256 digest mutation (`J6("subagent-request-" + parentTaskToolCallId)`), so
thirdeye reproduces the construction here and attributes child events by exact
`generation_id` equality only -- never by time overlap, nearest message, shared
`conversation_id`, tool name, FIFO position, or transcript order.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CursorTranscriptSummary:
    input_text: str
    output_text: str


@dataclass(frozen=True)
class CursorSubagentWindow:
    start_event: dict[str, Any] | None
    stop_event: dict[str, Any]
    subagent_id: str
    tool_call_id: str
    generation_id: str


def _data(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("data")
    return value if isinstance(value, dict) else {}


def _string(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _seq(event: dict[str, Any]) -> int:
    try:
        return int(event.get("seq") or 0)
    except (TypeError, ValueError):
        return 0


def _tool_call_id(event: dict[str, Any]) -> str:
    return _string(_data(event), "tool_call_id", "toolCallId", "tool_use_id", "toolUseId")


def _resume_start(event: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Return the resumed child id and a lifecycle-compatible start event.

    Cursor IDE deliberately skips ``subagentStart`` for resumed Task calls. Its
    preceding ``preToolUse`` callback still carries ``tool_input.resume`` and a
    fresh Task call id, so use that durable tool event as the resumed run's
    boundary. A plain repeated stop remains distinguishable because it has no
    intervening resume dispatch.
    """
    if event.get("t") != "tool_call":
        return "", None
    data = _data(event)
    if _string(data, "tool_name", "toolName", "name", "tool").lower() != "task":
        return "", None
    tool_input = data.get("tool_input", data.get("toolInput"))
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            return "", None
    if not isinstance(tool_input, dict):
        return "", None
    subagent_id = _string(tool_input, "resume")
    if not subagent_id:
        return "", None

    start_data = dict(data)
    start_data["subagent_id"] = subagent_id
    start_data["tool_call_id"] = _tool_call_id(event)
    start_data["cursor_resume"] = True
    for source, target in (
        ("prompt", "task"),
        ("description", "description"),
        ("subagent_type", "subagent_type"),
        ("model", "subagent_model"),
    ):
        value = tool_input.get(source)
        if value not in (None, ""):
            start_data[target] = value
    return subagent_id, {**event, "data": start_data}


def cursor_subagent_generation_id(tool_call_id: str) -> str:
    if not tool_call_id:
        return ""
    digest = bytearray(hashlib.sha256(f"subagent-request-{tool_call_id}".encode()).digest())
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    value = digest.hex()
    return "-".join((value[0:8], value[8:12], value[12:16], value[16:20], value[20:32]))


def _cursor_subagent_lifecycle(
    events: list[dict[str, Any]],
) -> tuple[list[CursorSubagentWindow], set[int]]:
    ordered = sorted(events, key=_seq)
    open_starts: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    modern_completed: set[str] = set()
    modern_stop_seqs: set[int] = set()
    windows: list[CursorSubagentWindow] = []

    for event in ordered:
        event_type = str(event.get("t") or "")
        data = _data(event)
        subagent_id = _string(data, "subagent_id", "subagentId", "agent_id", "agentId")
        if event_type == "tool_call":
            resumed_id, resume_start = _resume_start(event)
            if resumed_id and resume_start is not None:
                modern_completed.discard(resumed_id)
                open_starts[resumed_id].append(resume_start)
            continue
        if event_type == "subagent_start":
            if subagent_id:
                modern_completed.discard(subagent_id)
                # Cursor IDE omits this callback on resume, while Cursor CLI
                # versions may still emit it. Replace the synthetic Task
                # boundary when both runtimes' signals are present so one
                # resumed invocation always creates exactly one window.
                tool_call_id = _tool_call_id(event)
                for index, queued in enumerate(open_starts[subagent_id]):
                    queued_data = _data(queued)
                    if queued_data.get("cursor_resume") is not True:
                        continue
                    if tool_call_id and _tool_call_id(queued) != tool_call_id:
                        continue
                    del open_starts[subagent_id][index]
                    break
                open_starts[subagent_id].append(event)
            continue
        if event_type != "subagent_message":
            continue
        if subagent_id and open_starts[subagent_id]:
            start = open_starts[subagent_id].popleft()
            modern_completed.add(subagent_id)
            modern_stop_seqs.add(_seq(event))
            tool_call_id = _tool_call_id(start)
            windows.append(
                CursorSubagentWindow(
                    start_event=start,
                    stop_event=event,
                    subagent_id=subagent_id,
                    tool_call_id=tool_call_id,
                    generation_id=cursor_subagent_generation_id(tool_call_id),
                )
            )
        elif subagent_id and subagent_id in modern_completed:
            modern_stop_seqs.add(_seq(event))
        else:
            windows.append(
                CursorSubagentWindow(
                    start_event=None,
                    stop_event=event,
                    subagent_id=subagent_id,
                    tool_call_id="",
                    generation_id="",
                )
            )

    windows.sort(key=lambda window: _seq(window.stop_event))
    return windows, modern_stop_seqs


def cursor_subagent_windows(events: list[dict[str, Any]]) -> list[CursorSubagentWindow]:
    return _cursor_subagent_lifecycle(events)[0]


def modern_subagent_stop_seqs(events: list[dict[str, Any]]) -> set[int]:
    """Return paired and duplicate stops owned by a modern lifecycle."""
    return _cursor_subagent_lifecycle(events)[1]


def window_for_stop(
    events: list[dict[str, Any]], stop_event: dict[str, Any]
) -> CursorSubagentWindow | None:
    target_seq = _seq(stop_event)
    for window in cursor_subagent_windows(events):
        if _seq(window.stop_event) == target_seq:
            return window
    return None


def events_for_subagent(
    events: list[dict[str, Any]], window: CursorSubagentWindow
) -> list[dict[str, Any]]:
    if window.start_event is None or not window.generation_id:
        return []
    start_seq = _seq(window.start_event)
    stop_seq = _seq(window.stop_event)
    return [
        event
        for event in events
        if start_seq < _seq(event) <= stop_seq
        and _string(_data(event), "generation_id", "generationId") == window.generation_id
    ]


def read_cursor_transcript(path: str | None) -> CursorTranscriptSummary:
    if not path:
        return CursorTranscriptSummary("", "")
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return CursorTranscriptSummary("", "")

    input_text = ""
    output_text = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        role = record.get("role")
        if role not in ("user", "assistant"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            block_text = block.get("text")
            if not isinstance(block_text, str):
                continue
            block_text = block_text.strip()
            if block_text:
                parts.append(block_text)
        if not parts:
            continue
        joined = "\n".join(parts)
        if role == "user":
            if not input_text:
                input_text = joined
        else:
            output_text = joined

    return CursorTranscriptSummary(input_text, output_text)
