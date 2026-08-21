"""Adapt ``extract_turn_codex``'s reshaped dict (turn.py) into a full
``thirdeye.tracing.model.TurnSpanDict`` for ``thirdeye.otel_export.export_turn``.

Permission requests and subagent invocations arrive on a separate live stream
(``hooks_json.py``, uncorrelated with the rollout's own seq numbers) rather
than in the rollout itself, so they're pulled from the local event store
instead, bounded two ways: by **seq range** (this turn's own "agent_turn"
event fence vs. the previous one -- notify() always appends exactly one per
call, so this is a reliable per-turn boundary even on a different axis from
the rollout's own seq numbers), then by **timestamp** within that window
against the turn's own ``[start_ts, end_ts]`` (the seq bound alone only
excludes *other turns'* events, not this turn's own events that fall outside
the rollout-derived span).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from thirdeye.reader import SessionReader
from thirdeye.tracing.model import PermissionRequestSpanDict, TurnSpanDict


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)


def _in_range(ts: str, start_ts: str, end_ts: str) -> bool:
    try:
        return _parse_ts(start_ts) <= _parse_ts(ts) <= _parse_ts(end_ts)
    except ValueError:
        return False


def _turn_seq_lower_bound(session_dir_: Path, upto_seq: int) -> int:
    lo = 0
    for event in SessionReader(session_dir_).iter_events(
        types={"agent_turn"}, seq_range=(0, upto_seq)
    ):
        seq = event.get("seq")
        if isinstance(seq, int):
            lo = seq + 1
    return lo


def _permission_requests_in_range(
    session_dir_: Path, seq_lo: int, seq_hi: int, start_ts: str, end_ts: str
) -> list[PermissionRequestSpanDict]:
    out: list[PermissionRequestSpanDict] = []
    reader = SessionReader(session_dir_)
    for event in reader.iter_events(types={"permission_request"}, seq_range=(seq_lo, seq_hi)):
        ts = str(event.get("ts") or "")
        if not _in_range(ts, start_ts, end_ts):
            continue
        data = event.get("data") or {}
        out.append(
            {
                "ts": ts,
                "tool_name": str(data.get("tool_name") or ""),
                "attributes": {k: v for k, v in data.items() if k != "tool_name"},
            }
        )
    return out


def _subagent_turn(session_id: str, start: dict[str, Any], stop: dict[str, Any]) -> TurnSpanDict:
    start_data = start.get("data") or {}
    stop_data = stop.get("data") or {}
    input_message = start_data.get("prompt") or start_data.get("description") or ""
    output_message = stop_data.get("message") or stop_data.get("output") or ""
    return {
        "turn_id": f"subagent:{session_id}:{start.get('seq')}",
        "start_ts": str(start.get("ts") or ""),
        "end_ts": str(stop.get("ts") or ""),
        "input_message": str(input_message),
        "status": "completed",
        "output_message": str(output_message),
        "llm_calls": [],
        "permission_requests": [],
        "subagents": [],
        "attributes": {},
    }


def _subagents_in_range(
    session_dir_: Path, session_id: str, seq_lo: int, seq_hi: int, start_ts: str, end_ts: str
) -> list[TurnSpanDict]:
    reader = SessionReader(session_dir_)
    events = [
        event
        for event in reader.iter_events(
            types={"subagent_start", "subagent_message"}, seq_range=(seq_lo, seq_hi)
        )
        if _in_range(str(event.get("ts") or ""), start_ts, end_ts)
    ]

    # Subagent invocations nest like a call stack -- a Task tool invoked from
    # inside a running subagent nests under it -- and these payloads carry no
    # id shared between a start and its stop, so pairing is LIFO: a stop
    # closes the most recently opened, still-unclosed start. This is what
    # makes interleaved starts A, B followed by stops B, A pair correctly as
    # A->A / B->B instead of crossing into A->B / B->A.
    stack: list[dict[str, Any]] = []
    subagents: list[TurnSpanDict] = []
    for event in events:
        if event.get("t") == "subagent_start":
            stack.append(event)
        elif stack:
            subagents.append(_subagent_turn(session_id, stack.pop(), event))
        # A stop with nothing open on the stack has no start in this turn's
        # own window to pair with -- nothing to nest it under, so it's
        # dropped rather than fabricating a start for it.

    subagents.sort(key=lambda subagent: subagent["start_ts"])
    return subagents


def build_turn(
    *,
    session_dir_: Path,
    session_id: str,
    seq: int,
    turn: dict[str, Any],
) -> TurnSpanDict:
    start_ts = turn["start_ts"]
    end_ts = turn["end_ts"]
    seq_lo = _turn_seq_lower_bound(session_dir_, seq)
    return {
        "turn_id": turn["turn_id"],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "input_message": turn["user_prompt"],
        "output_message": turn["assistant_output"],
        "status": turn["status"],
        "llm_calls": turn["calls"],
        "permission_requests": _permission_requests_in_range(
            session_dir_, seq_lo, seq, start_ts, end_ts
        ),
        "subagents": _subagents_in_range(session_dir_, session_id, seq_lo, seq, start_ts, end_ts),
        "attributes": {},
    }
