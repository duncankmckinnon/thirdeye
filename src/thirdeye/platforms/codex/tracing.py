"""Adapt ``extract_turn_codex``'s reshaped dict (turn.py) into a full
``thirdeye.tracing.model.TurnSpanDict`` for ``thirdeye.otel_export.export_turn``.

Permission requests and subagent invocations arrive on a separate live stream
(``platforms/codex/hooks_json.py``, uncorrelated with the rollout's own seq
numbers) rather than in the rollout itself, so they can't be read off ``turn``
directly. The only axis both streams share is wall-clock time, so they're
pulled from the local event store by filtering to the turn's own
``[start_ts, end_ts]`` window instead.
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


def _permission_requests_in_range(
    session_dir_: Path, start_ts: str, end_ts: str
) -> list[PermissionRequestSpanDict]:
    out: list[PermissionRequestSpanDict] = []
    for event in SessionReader(session_dir_).iter_events(types={"permission_request"}):
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


def _subagents_in_range(session_dir_: Path, start_ts: str, end_ts: str) -> list[TurnSpanDict]:
    events = SessionReader(session_dir_).iter_events(
        types={"subagent_start", "subagent_message"}
    )
    starts: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    for event in events:
        if not _in_range(str(event.get("ts") or ""), start_ts, end_ts):
            continue
        (starts if event.get("t") == "subagent_start" else stops).append(event)

    subagents: list[TurnSpanDict] = []
    for start, stop in zip(starts, stops):
        start_data = start.get("data") or {}
        stop_data = stop.get("data") or {}
        input_message = start_data.get("prompt") or start_data.get("description") or ""
        output_message = stop_data.get("message") or stop_data.get("output") or ""
        subagents.append(
            {
                "turn_id": f"subagent:{start.get('seq')}:{stop.get('seq')}",
                "start_ts": str(start.get("ts") or start_ts),
                "end_ts": str(stop.get("ts") or end_ts),
                "input_message": str(input_message),
                "output_message": str(output_message),
                "status": "completed",
                "llm_calls": [],
                "permission_requests": [],
                "subagents": [],
                "attributes": {},
            }
        )
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
    return {
        "turn_id": turn["turn_id"],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "input_message": turn["user_prompt"],
        "output_message": turn["assistant_output"],
        "status": turn["status"],
        "llm_calls": turn["calls"],
        "permission_requests": _permission_requests_in_range(session_dir_, start_ts, end_ts),
        "subagents": _subagents_in_range(session_dir_, start_ts, end_ts),
        "attributes": {},
    }
