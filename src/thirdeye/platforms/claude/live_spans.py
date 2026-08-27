from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.otel_export import export_spans
from thirdeye.paths import otel_state_path
from thirdeye.platforms.claude.hooks import (
    _advance_turn_cursor,
    _locked_open_turn,
    _read_open_turn_unlocked,
    committed_call_ids,
    committed_tool_use_ids,
)
from thirdeye.platforms.claude.tracing import _pair_tool_calls
from thirdeye.platforms.claude.usage import extract_calls_from_transcript
from thirdeye.reader import SessionReader
from thirdeye.span_ids import chat_span_id, tool_span_id, trace_id_for_session
from thirdeye.tracing.model import LlmCallSpanDict, ToolCallSpanDict
from thirdeye.usage.errlog import log_capture_error

_PLATFORM = "claude"
_TOOL_EVENT_TYPES = frozenset({"tool_call", "tool_result"})


def _log_skip(config: Config, session_id: str, tool_use_id: str, reason: str) -> None:
    """Every early return here assumes Stop-time reconstruction recovers
    the span later. That assumption has been observed not to hold in
    practice, and with nothing recorded at the skip site, a genuinely lost
    span looked identical to one correctly deferred. Never raises.
    """
    try:
        log_capture_error(
            thirdeye_home=config.root,
            phase="emit_live_spans_skipped",
            level="info",
            platform=_PLATFORM,
            session_id=session_id,
            message=f"reason={reason} tool_use_id={tool_use_id!r}",
        )
    except Exception:
        pass


def _requesting_call_id(calls: list[LlmCallSpanDict], tool_use_id: str) -> str | None:
    for call in calls:
        for message in call["output_messages"]:
            for part in message.get("parts", []):
                if part.get("type") == "tool_call" and str(part.get("id") or "") == tool_use_id:
                    return call["call_id"]
    return None


def _scan_requesting_call_id(transcript_path: str | None, tool_use_id: str) -> str | None:
    if not transcript_path:
        return None
    path = Path(transcript_path)
    if not path.is_file():
        return None

    with path.open("rb") as transcript:
        for raw in transcript:
            try:
                frame = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(frame, dict) or frame.get("type") != "assistant":
                continue
            message = frame.get("message")
            if not isinstance(message, dict):
                continue
            call_id = message.get("id") or frame.get("requestId") or frame.get("uuid")
            content = message.get("content")
            if not call_id or not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and str(block.get("id") or "") == tool_use_id
                ):
                    return str(call_id)
    return None


def _paired_tool_call(
    session_dir_: Path, *, turn_seq: int, tool_use_id: str
) -> ToolCallSpanDict | None:
    events = list(
        SessionReader(session_dir_).iter_events(
            types=_TOOL_EVENT_TYPES,
            seq_range=(turn_seq, 2**63 - 1),
        )
    )
    for tool_call in reversed(_pair_tool_calls(events)):
        if tool_call["tool_call_id"] == tool_use_id:
            return tool_call
    return None


def _trace_id(session_dir_: Path, session_id: str) -> int:
    try:
        state = json.loads(otel_state_path(session_dir_).read_text())
        return int(state["trace_id"], 16)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return trace_id_for_session(session_id)


def _chat_span(
    session_id: str, turn_id: int, turn_seq: int, call: LlmCallSpanDict
) -> dict[str, Any]:
    model = call.get("model") or ""
    return {
        "name": f"chat {model}" if model else "chat",
        "span_id": chat_span_id(session_id, call["call_id"]),
        "parent_span_id": turn_id,
        # A live span's `agent-turn` parent is not exported until Stop, so
        # until then it has no parent row to inherit turn identity from. Carry
        # it on the span itself, or the span cannot be attributed to a turn
        # while the turn it belongs to is still running.
        "turn_seq": turn_seq,
        "turn_span_id": str(turn_id),
        "start_ts": call["start_ts"],
        "end_ts": call["end_ts"],
        "attributes": call,
    }


def _tool_span(
    session_id: str,
    tool_use_id: str,
    parent_span_id: int,
    turn_id: int,
    turn_seq: int,
    tool_call: ToolCallSpanDict,
) -> dict[str, Any]:
    return {
        "name": f"tool: {tool_call['name']}",
        "tool_name": tool_call["name"],
        "tool_call_id": tool_use_id,
        "span_id": tool_span_id(session_id, tool_use_id),
        "parent_span_id": parent_span_id,
        "turn_seq": turn_seq,
        "turn_span_id": str(turn_id),
        "start_ts": tool_call["start_ts"],
        "end_ts": tool_call["end_ts"],
        "attributes": tool_call["attributes"],
    }


def _emit_live_spans(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    tool_use_id: str,
) -> None:
    if not config.logfire.enabled or not config.logfire.token:
        return

    with _locked_open_turn(session_dir_, fcntl.LOCK_EX):
        marker = _read_open_turn_unlocked(session_dir_)
        if marker is None:
            _log_skip(config, session_id, tool_use_id, "no_open_turn_marker")
            return

        transcript_path = marker["transcript_path"]
        parsed = extract_calls_from_transcript(
            transcript_path,
            marker["transcript_offset"],
            initial_prev_ts=marker["last_frame_ts"],
            incremental=True,
        )
        requesting_call_id = _requesting_call_id(parsed.calls, tool_use_id)
        if requesting_call_id is None:
            requesting_call_id = _scan_requesting_call_id(transcript_path, tool_use_id)
        if requesting_call_id is None:
            _log_skip(config, session_id, tool_use_id, "requesting_call_id_not_found")
            return

        tool_call = _paired_tool_call(
            session_dir_, turn_seq=marker["turn_seq"], tool_use_id=tool_use_id
        )
        if tool_call is None:
            _log_skip(config, session_id, tool_use_id, "tool_call_not_paired")
            return

        turn_id = int(marker["turn_span_id"])
        turn_seq = int(marker["turn_seq"])
        # A call already exported by an earlier hook must not be exported
        # again: a `message.id` split across a `user` frame reopens as a second
        # group deriving the same deterministic span id, which Logfire stores
        # as a second row rather than an update, double-counting its tokens.
        already_committed = set(committed_call_ids(marker))
        fresh_calls = [call for call in parsed.calls if call["call_id"] not in already_committed]
        spans = [_chat_span(session_id, turn_id, turn_seq, call) for call in fresh_calls]

        already_committed_tools = set(committed_tool_use_ids(marker))
        newly_committed_tool_use_ids: list[str] = []
        # A parallel tool-dispatch message can fragment across multiple
        # transcript lines sharing one call_id (Claude Code writes each
        # tool_use block as its own frame, and a dispatch acknowledgement
        # between two of them can close and reopen the group). Whichever
        # hook call happens to be the one that commits a given call_id is
        # that whole group's only remaining chance to attach its OTHER
        # tool_use parts too: `_advance_turn_cursor` below moves the
        # transcript cursor past everything this parse consumed, and the
        # cursor never revisits bytes it has already advanced past -- a
        # sibling tool_use_id left for "later" here has no later.
        for call in fresh_calls:
            for message in call["output_messages"]:
                for part in message.get("parts", []):
                    if part.get("type") != "tool_call":
                        continue
                    sibling_id = str(part.get("id") or "")
                    if (
                        not sibling_id
                        or sibling_id == tool_use_id
                        or sibling_id in already_committed_tools
                    ):
                        continue
                    sibling_tool_call = _paired_tool_call(
                        session_dir_, turn_seq=turn_seq, tool_use_id=sibling_id
                    )
                    if sibling_tool_call is None:
                        continue  # not locally recorded yet -- Stop-time may still recover it
                    spans.append(
                        _tool_span(
                            session_id,
                            sibling_id,
                            chat_span_id(session_id, call["call_id"]),
                            turn_id,
                            turn_seq,
                            sibling_tool_call,
                        )
                    )
                    newly_committed_tool_use_ids.append(sibling_id)

        # A hook process can invoke this more than once for the same
        # tool_use_id (e.g. `post_tool_use` firing twice for one tool call);
        # both derive the same deterministic `tool_span_id`, so re-exporting
        # would double-count it the same way an uncommitted chat span would.
        tool_already_committed = tool_use_id in already_committed_tools
        if not tool_already_committed:
            spans.append(
                _tool_span(
                    session_id,
                    tool_use_id,
                    chat_span_id(session_id, requesting_call_id),
                    turn_id,
                    turn_seq,
                    tool_call,
                )
            )
            newly_committed_tool_use_ids.append(tool_use_id)
        if not spans:
            _log_skip(config, session_id, tool_use_id, "no_new_spans_already_committed")
            return
        exported = export_spans(
            config,
            session_dir_,
            session_id,
            _PLATFORM,
            cwd,
            _trace_id(session_dir_, session_id),
            spans,
        )
        if exported is False:
            _log_skip(config, session_id, tool_use_id, "export_spans_failed")
            return
        _advance_turn_cursor(
            session_dir_,
            expected_turn_seq=marker["turn_seq"],
            offset=parsed.offset,
            last_frame_ts=parsed.last_frame_ts,
            newly_committed_call_ids=[call["call_id"] for call in fresh_calls],
            newly_committed_tool_use_ids=newly_committed_tool_use_ids or None,
        )


def emit_live_spans(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    tool_use_id: str,
) -> None:
    try:
        _emit_live_spans(config, session_dir_, session_id, cwd, tool_use_id)
    except Exception as exc:
        try:
            log_capture_error(
                thirdeye_home=config.root,
                phase="emit_live_spans_failed",
                level="error",
                platform=_PLATFORM,
                session_id=session_id,
                error=exc,
                message=f"tool_use_id={tool_use_id!r}",
            )
        except Exception:
            pass
