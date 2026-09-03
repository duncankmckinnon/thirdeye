"""Emit completed Cursor tool operations before the generation stops.

Cursor only reports authoritative model usage at ``stop``, so chat and agent
spans remain Stop-time exports. Completed tools can be sent immediately and
parented to deterministic chat/turn IDs; Logfire joins them to those parents
when the completed turn arrives a moment later.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.otel_export import export_spans
from thirdeye.paths import otel_state_path
from thirdeye.platforms.cursor.interactions import CanonicalInteraction, canonical_interactions
from thirdeye.platforms.cursor.subagents import cursor_subagent_generation_id
from thirdeye.platforms.cursor.tracing import (
    resolve_turn_seq,
    tool_calls_for_generation,
)
from thirdeye.reader import SessionReader
from thirdeye.span_ids import (
    chat_span_id,
    interaction_span_id,
    tool_span_id,
    trace_id_for_session,
    turn_span_id,
)
from thirdeye.usage.errlog import log_capture_error

_PLATFORM = "cursor"
# Tool call IDs are arbitrary Cursor strings and may start with "i:". Use a
# null-byte sentinel so interaction entries cannot collide with legacy tools.
_INTERACTION_STATE_PREFIX = "\x00thirdeye:cursor:interaction:"
_INTERACTION_DUP_SUFFIX = "\x00d\x00"


def _interaction_state_entry(interaction_id: str, duplicate_count: int = 0) -> str:
    entry = f"{_INTERACTION_STATE_PREFIX}{interaction_id}"
    if duplicate_count:
        return f"{entry}{_INTERACTION_DUP_SUFFIX}{duplicate_count}"
    return entry


def _parse_interaction_state_entry(entry: str) -> tuple[str, int] | None:
    if not entry.startswith(_INTERACTION_STATE_PREFIX):
        return None
    body = entry[len(_INTERACTION_STATE_PREFIX) :]
    if _INTERACTION_DUP_SUFFIX in body:
        interaction_id, duplicate_count = body.rsplit(_INTERACTION_DUP_SUFFIX, 1)
        return interaction_id, int(duplicate_count)
    return body, 0


def _parse_committed_state(entries: list[str]) -> tuple[set[str], dict[str, int]]:
    tool_ids: set[str] = set()
    interaction_dup_counts: dict[str, int] = {}
    for entry in entries:
        parsed = _parse_interaction_state_entry(entry)
        if parsed is not None:
            interaction_id, duplicate_count = parsed
            interaction_dup_counts[interaction_id] = duplicate_count
            continue
        tool_ids.add(entry)
    return tool_ids, interaction_dup_counts


def _trace_id(session_dir_: Path, session_id: str) -> int:
    try:
        state = json.loads(otel_state_path(session_dir_).read_text())
        return int(state["trace_id"], 16)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return trace_id_for_session(_PLATFORM, session_id)


def _state_path(session_dir_: Path) -> Path:
    return session_dir_ / "cursor-live-state.json"


def _lock_path(session_dir_: Path) -> Path:
    return session_dir_ / "cursor-live-state.lock"


@contextlib.contextmanager
def _locked(session_dir_: Path) -> Iterator[None]:
    path = _lock_path(session_dir_)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_state(session_dir_: Path) -> dict[str, list[str]]:
    try:
        raw = json.loads(_state_path(session_dir_).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(generation): [str(value) for value in values if isinstance(value, str)]
        for generation, values in raw.items()
        if isinstance(values, list)
    }


def _write_state(session_dir_: Path, state: dict[str, list[str]]) -> None:
    path = _state_path(session_dir_)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, separators=(",", ":")))
    os.replace(tmp, path)


def committed_tool_call_ids(session_dir_: Path, generation_id: str) -> set[str]:
    with _locked(session_dir_):
        tool_ids, _ = _parse_committed_state(_read_state(session_dir_).get(generation_id, []))
        return tool_ids


def _interaction_span_attributes(interaction: CanonicalInteraction) -> dict[str, Any]:
    return {
        "thirdeye.interaction.kind": interaction.kind,
        "thirdeye.interaction.payload": interaction.payload,
        "thirdeye.interaction.correlation_id": interaction.correlation_id,
        "thirdeye.interaction.source_type": interaction.source_type,
        "thirdeye.interaction.source_seq": interaction.source_seq,
        "thirdeye.interaction.generation_id": interaction.generation_id,
        **(
            {"thirdeye.interaction.duplicate_seqs": list(interaction.duplicate_seqs)}
            if interaction.duplicate_seqs
            else {}
        ),
    }


def _interaction_span_dict(
    interaction: CanonicalInteraction, *, session_id: str, turn_seq: int, turn_id: int
) -> dict[str, Any]:
    return {
        "name": (
            interaction.kind
            if interaction.kind == "reasoning"
            else f"interaction: {interaction.kind}"
        ),
        "span_id": interaction_span_id(_PLATFORM, session_id, interaction.interaction_id),
        "parent_span_id": turn_id,
        "turn_seq": turn_seq,
        "turn_span_id": str(turn_id),
        "start_ts": interaction.ts,
        "end_ts": interaction.ts,
        "attributes": _interaction_span_attributes(interaction),
    }


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _event_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _generation_events(
    session_dir_: Path, generation_id: str, through_seq: int
) -> list[dict[str, Any]]:
    return [
        event
        for event in SessionReader(session_dir_).iter_events(seq_range=(0, through_seq + 1))
        if str((event.get("data") or {}).get("generation_id") or "") == generation_id
    ]


def _context_turn_seq(
    events: list[dict[str, Any]], session_id: str, generation_id: str, through_seq: int
) -> int | None:
    turn_seq = resolve_turn_seq(
        events,
        generation_id=generation_id,
        session_id=session_id,
        through_seq=through_seq,
    )
    if turn_seq is not None:
        return turn_seq
    # Nested Task calls run under a child generation with no user_message in
    # the parent session. Anchor their identity attributes to the lifecycle
    # event whose Task id deterministically created that generation.
    for event in reversed(events):
        if int(event.get("seq") or 0) > through_seq:
            continue
        if event.get("t") not in {"subagent_start", "tool_call"}:
            continue
        data = _event_data(event)
        if event.get("t") == "tool_call":
            name = _event_text(data, "tool_name", "toolName", "name", "tool")
            if name.lower() != "task":
                continue
        call_id = _event_text(data, "tool_call_id", "toolCallId", "tool_use_id", "toolUseId")
        if cursor_subagent_generation_id(call_id) == generation_id:
            return int(event.get("seq") or 0)
    return None


def _emit_task_parent_span(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    tool_call_id: str,
    through_seq: int,
) -> None:
    """Emit the deterministic Task parent even when Cursor sends no post hook."""
    if not config.logfire.enabled or not config.logfire.token or not tool_call_id:
        return
    with _locked(session_dir_):
        all_events = list(SessionReader(session_dir_).iter_events(seq_range=(0, through_seq + 1)))
        task_event: dict[str, Any] | None = None
        for event in reversed(all_events):
            if event.get("t") != "tool_call":
                continue
            data = _event_data(event)
            name = _event_text(data, "tool_name", "toolName", "name", "tool")
            call_id = _event_text(data, "tool_call_id", "toolCallId", "tool_use_id", "toolUseId")
            if name.lower() == "task" and call_id == tool_call_id:
                task_event = event
                break
        if task_event is None:
            return

        task_data = _event_data(task_event)
        generation_id = _event_text(task_data, "generation_id", "generationId")
        commit_key = generation_id or f"task:{tool_call_id}"
        state = _read_state(session_dir_)
        committed = set(state.get(commit_key, []))
        if tool_call_id in committed:
            return
        turn_seq = None
        if generation_id:
            turn_seq = _context_turn_seq(all_events, session_id, generation_id, through_seq)
        if turn_seq is None:
            turn_seq = int(task_event.get("seq") or 0) or None
        if turn_seq is None:
            log_capture_error(
                thirdeye_home=config.root,
                phase="emit_cursor_task_parent_skipped",
                level="warn",
                platform=_PLATFORM,
                session_id=session_id,
                message=f"no turn anchor for tool_call_id={tool_call_id!r}",
            )
            return
        if not generation_id:
            log_capture_error(
                thirdeye_home=config.root,
                phase="emit_cursor_task_parent_ungenerated",
                level="warn",
                platform=_PLATFORM,
                session_id=session_id,
                message=f"emitting Task span without generation_id tool_call_id={tool_call_id!r}",
            )

        timestamp = str(task_event.get("ts") or "")
        tool_input = task_data.get("tool_input", task_data.get("toolInput"))
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "Task",
            "gen_ai.tool.call.id": tool_call_id,
        }
        if tool_input not in (None, ""):
            attributes["gen_ai.tool.call.arguments"] = tool_input
        parent_id = (
            chat_span_id(_PLATFORM, session_id, generation_id)
            if generation_id
            else turn_span_id(_PLATFORM, session_id, turn_seq)
        )
        span = {
            "name": "tool: Task",
            "tool_name": "Task",
            "tool_call_id": tool_call_id,
            "span_id": tool_span_id(_PLATFORM, session_id, tool_call_id),
            "parent_span_id": parent_id,
            "turn_seq": turn_seq,
            "turn_span_id": str(turn_span_id(_PLATFORM, session_id, turn_seq)),
            "start_ts": timestamp,
            "end_ts": timestamp,
            "attributes": attributes,
        }
        if not export_spans(
            config,
            session_dir_,
            session_id,
            _PLATFORM,
            cwd,
            _trace_id(session_dir_, session_id),
            [span],
        ):
            return
        state[commit_key] = sorted(committed | {tool_call_id})
        _write_state(session_dir_, state)


def emit_task_parent_span(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    tool_call_id: str,
    through_seq: int,
) -> None:
    """Fail-open wrapper for the Task span needed by a detached child turn."""
    try:
        _emit_task_parent_span(config, session_dir_, session_id, cwd, tool_call_id, through_seq)
    except Exception as exc:
        try:
            log_capture_error(
                thirdeye_home=config.root,
                phase="emit_cursor_task_parent_failed",
                level="error",
                platform=_PLATFORM,
                session_id=session_id,
                error=exc,
                message=f"tool_call_id={tool_call_id!r}",
            )
        except Exception:
            pass


def _emit_live_tools(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    generation_id: str,
    through_seq: int,
) -> None:
    if not config.logfire.enabled or not config.logfire.token:
        return
    with _locked(session_dir_):
        state = _read_state(session_dir_)
        committed_entries = state.get(generation_id, [])
        committed_tools, _ = _parse_committed_state(committed_entries)
        events = _generation_events(session_dir_, generation_id, through_seq)
        if not events:
            return
        fresh = [
            tool
            for tool in tool_calls_for_generation(events, session_id, generation_id)
            if tool["tool_call_id"] not in committed_tools
        ]
        if not fresh:
            return
        all_events = list(SessionReader(session_dir_).iter_events(seq_range=(0, through_seq + 1)))
        turn_seq = resolve_turn_seq(
            all_events,
            generation_id=generation_id,
            session_id=session_id,
            through_seq=through_seq,
        )
        if turn_seq is None:
            return
        turn_id = turn_span_id(_PLATFORM, session_id, turn_seq)
        parent_id = chat_span_id(_PLATFORM, session_id, generation_id)
        spans = [
            {
                "name": f"tool: {tool['name']}",
                "tool_name": tool["name"],
                "tool_call_id": tool["tool_call_id"],
                "span_id": tool_span_id(_PLATFORM, session_id, tool["tool_call_id"]),
                "parent_span_id": parent_id,
                "turn_seq": turn_seq,
                "turn_span_id": str(turn_id),
                "start_ts": tool["start_ts"],
                "end_ts": tool["end_ts"],
                "attributes": tool["attributes"],
            }
            for tool in fresh
        ]
        if not export_spans(
            config,
            session_dir_,
            session_id,
            _PLATFORM,
            cwd,
            _trace_id(session_dir_, session_id),
            spans,
        ):
            return
        state[generation_id] = sorted(
            set(committed_entries) | {tool["tool_call_id"] for tool in fresh}
        )
        _write_state(session_dir_, state)


def emit_live_tools(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    generation_id: str,
    through_seq: int,
) -> None:
    try:
        _emit_live_tools(config, session_dir_, session_id, cwd, generation_id, through_seq)
    except Exception as exc:
        try:
            log_capture_error(
                thirdeye_home=config.root,
                phase="emit_live_spans_failed",
                level="error",
                platform=_PLATFORM,
                session_id=session_id,
                error=exc,
                message=f"generation_id={generation_id!r}",
            )
        except Exception:
            pass


def committed_interaction_dup_counts(session_dir_: Path, generation_id: str) -> dict[str, int]:
    with _locked(session_dir_):
        _, interaction_dup_counts = _parse_committed_state(
            _read_state(session_dir_).get(generation_id, [])
        )
        return interaction_dup_counts


def committed_interaction_ids(session_dir_: Path, generation_id: str) -> set[str]:
    return set(committed_interaction_dup_counts(session_dir_, generation_id))


def _emit_live_interactions(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    generation_id: str,
    through_seq: int,
) -> None:
    if not config.logfire.enabled or not config.logfire.token:
        return
    with _locked(session_dir_):
        state = _read_state(session_dir_)
        committed_entries = state.get(generation_id, [])
        committed_tools, committed_interactions = _parse_committed_state(committed_entries)
        all_events = list(SessionReader(session_dir_).iter_events(seq_range=(0, through_seq + 1)))
        interactions = canonical_interactions(
            all_events, generation_id=generation_id, through_seq=through_seq
        )
        if not interactions and not all_events:
            return
        turn_seq = resolve_turn_seq(
            all_events,
            generation_id=generation_id,
            session_id=session_id,
            through_seq=through_seq,
        )
        if turn_seq is None:
            return
        turn_id = turn_span_id(_PLATFORM, session_id, turn_seq)
        export_interactions = [
            interaction
            for interaction in interactions
            if interaction.kind in ("reasoning", "assistant_message")
            and (
                interaction.interaction_id not in committed_interactions
                or len(interaction.duplicate_seqs)
                > committed_interactions[interaction.interaction_id]
            )
        ]
        tools = tool_calls_for_generation(
            _generation_events(session_dir_, generation_id, through_seq), session_id, generation_id
        )
        fresh_tools = [
            tool
            for tool in tools
            if tool["tool_call_id"] not in committed_tools
        ]

        interaction_spans = [
            _interaction_span_dict(
                interaction, session_id=session_id, turn_seq=turn_seq, turn_id=turn_id
            )
            for interaction in export_interactions
        ]

        chat_id = chat_span_id(_PLATFORM, session_id, generation_id)
        tool_spans = [
            {
                "name": f"tool: {tool['name']}",
                "tool_name": tool["name"],
                "tool_call_id": tool["tool_call_id"],
                "span_id": tool_span_id(_PLATFORM, session_id, tool["tool_call_id"]),
                "parent_span_id": chat_id,
                "turn_seq": turn_seq,
                "turn_span_id": str(turn_id),
                "start_ts": tool["start_ts"],
                "end_ts": tool["end_ts"],
                "attributes": tool["attributes"],
            }
            for tool in fresh_tools
        ]

        spans = interaction_spans + tool_spans
        if not spans:
            return

        if not export_spans(
            config,
            session_dir_,
            session_id,
            _PLATFORM,
            cwd,
            _trace_id(session_dir_, session_id),
            spans,
        ):
            return
        updated_entries = set(committed_entries)
        for interaction in export_interactions:
            updated_entries.discard(interaction.interaction_id)
            updated_entries.discard(
                _interaction_state_entry(
                    interaction.interaction_id,
                    committed_interactions.get(interaction.interaction_id, 0),
                )
            )
            updated_entries.add(
                _interaction_state_entry(
                    interaction.interaction_id, len(interaction.duplicate_seqs)
                )
            )
        state[generation_id] = sorted(
            updated_entries | {tool["tool_call_id"] for tool in fresh_tools}
        )
        _write_state(session_dir_, state)


def emit_live_interactions(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    generation_id: str,
    through_seq: int,
) -> None:
    try:
        _emit_live_interactions(config, session_dir_, session_id, cwd, generation_id, through_seq)
    except Exception as exc:
        try:
            log_capture_error(
                thirdeye_home=config.root,
                phase="emit_live_interactions_failed",
                level="error",
                platform=_PLATFORM,
                session_id=session_id,
                error=exc,
                message=f"generation_id={generation_id!r}",
            )
        except Exception:
            pass
