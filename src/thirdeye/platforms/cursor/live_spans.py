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
from thirdeye.platforms.cursor.tracing import tool_calls_for_generation
from thirdeye.reader import SessionReader
from thirdeye.span_ids import chat_span_id, tool_span_id, trace_id_for_session, turn_span_id
from thirdeye.usage.errlog import log_capture_error

_PLATFORM = "cursor"


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
        return set(_read_state(session_dir_).get(generation_id, []))


def _generation_events(
    session_dir_: Path, generation_id: str, through_seq: int
) -> list[dict[str, Any]]:
    return [
        event
        for event in SessionReader(session_dir_).iter_events(seq_range=(0, through_seq + 1))
        if str((event.get("data") or {}).get("generation_id") or "") == generation_id
    ]


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
        committed = set(state.get(generation_id, []))
        events = _generation_events(session_dir_, generation_id, through_seq)
        if not events:
            return
        fresh = [
            tool
            for tool in tool_calls_for_generation(events, session_id, generation_id)
            if tool["tool_call_id"] not in committed
        ]
        if not fresh:
            return
        # Match ``build_turn`` so live children join the same deterministic
        # agent-turn even if another generation-scoped hook preceded submit.
        start_event = next(
            (event for event in events if event.get("t") == "user_message"), events[0]
        )
        turn_seq = int(start_event.get("seq") or 0)
        turn_id = turn_span_id(session_id, turn_seq)
        parent_id = chat_span_id(session_id, generation_id)
        spans = [
            {
                "name": f"tool: {tool['name']}",
                "tool_name": tool["name"],
                "tool_call_id": tool["tool_call_id"],
                "span_id": tool_span_id(session_id, tool["tool_call_id"]),
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
            trace_id_for_session(session_id),
            spans,
        ):
            return
        state[generation_id] = sorted(committed | {tool["tool_call_id"] for tool in fresh})
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
