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
    attributes = {
        k: v
        for k, v in {**start_data, **stop_data}.items()
        if k not in {"prompt", "description", "message", "output"}
    }
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
        "attributes": attributes,
    }


def _subagent_id(event: dict[str, Any]) -> str | None:
    """Return the stable child identity used by Codex hook payloads, if any.

    Accept the spelling variants used elsewhere by Codex so tracing remains
    compatible across CLI/app hook schema revisions.  ``thread_id`` is a
    useful fallback for implementations that identify the child by its own
    thread rather than an explicit agent id.
    """
    data = event.get("data") or {}
    for key in ("agent_id", "agent-id", "agentId", "thread_id", "thread-id", "threadId"):
        value = data.get(key)
        if value is not None and str(value):
            return str(value)
    return None


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

    # Modern Codex payloads identify the child. Pair by that identity because
    # concurrently dispatched siblings may finish in any order. Older
    # identifier-less payloads retain the historical LIFO fallback.
    stack: list[dict[str, Any]] = []
    starts_by_id: dict[str, list[dict[str, Any]]] = {}
    subagents: list[TurnSpanDict] = []
    for event in events:
        if event.get("t") == "subagent_start":
            agent_id = _subagent_id(event)
            if agent_id is None:
                stack.append(event)
            else:
                starts_by_id.setdefault(agent_id, []).append(event)
        else:
            agent_id = _subagent_id(event)
            starts = starts_by_id.get(agent_id, []) if agent_id is not None else []
            if starts:
                start = starts.pop(0)
                if not starts:
                    starts_by_id.pop(agent_id, None)
                subagents.append(_subagent_turn(session_id, start, event))
            elif agent_id is None and stack:
                subagents.append(_subagent_turn(session_id, stack.pop(), event))
        # An unmatched stop has no start in this turn's own window to pair
        # with, so it is dropped rather than fabricating a child span.

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
