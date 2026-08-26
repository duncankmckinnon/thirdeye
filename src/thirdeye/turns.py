from __future__ import annotations

import json
from typing import Any

from thirdeye.meta import SessionMeta
from thirdeye.store import Store


def _record(meta: SessionMeta, events: list[dict[str, Any]]) -> dict[str, Any]:
    first = events[0]
    last = events[-1]
    data = first.get("data") if isinstance(first.get("data"), dict) else {}
    explicit_id = data.get("turn_id") or data.get("turn-id") or data.get("turnId")
    turn_id = str(explicit_id if explicit_id is not None else first.get("seq", ""))
    return {
        "id": f"{meta.session_id}:{turn_id}",
        "turn_id": turn_id,
        "session_id": meta.session_id,
        "platform": meta.platform,
        "cwd": meta.cwd,
        "start_seq": first.get("seq"),
        "end_seq": last.get("seq"),
        "start_ts": first.get("ts"),
        "end_ts": last.get("ts"),
        "events": events,
    }


def session_turns(meta: SessionMeta, store: Store) -> list[dict[str, Any]]:
    """Return durable top-level turn slices reconstructed from stored events."""
    events = list(store.reader(meta.session_id).iter_events())
    if meta.platform == "codex":
        starts = [i for i, event in enumerate(events) if event.get("t") == "agent_turn"]
        return [
            _record(meta, events[start : starts[n + 1] if n + 1 < len(starts) else len(events)])
            for n, start in enumerate(starts)
        ]

    turns: list[dict[str, Any]] = []
    current: list[dict[str, Any]] | None = None
    for event in events:
        event_type = event.get("t")
        if event_type == "user_message":
            if current:
                turns.append(_record(meta, current))
            current = [event]
        elif current is not None:
            current.append(event)
            if event_type == "assistant_message":
                turns.append(_record(meta, current))
                current = None
    if current:
        turns.append(_record(meta, current))
    return turns


def filter_turns(
    sessions: list[SessionMeta],
    store: Store,
    turn_id: str | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    turns = [turn for meta in sessions for turn in session_turns(meta, store)]
    if turn_id is not None:
        turns = [turn for turn in turns if turn["id"] == turn_id or turn["turn_id"] == turn_id]
    terms = [term.strip().lower() for term in (query or "").split(",") if term.strip()]
    if terms:
        turns = [
            turn
            for turn in turns
            if all(
                term in json.dumps(turn, default=str, ensure_ascii=False).lower() for term in terms
            )
        ]
    return turns
