"""Pure semantic projection for V1 Copilot archives."""

from __future__ import annotations

from typing import Any

from .events import normalize_records
from .turns import build_turns
from .types import SemanticProjection, SourceRecord


def build_semantics(
    records: list[SourceRecord], prior_state: dict[str, Any]
) -> tuple[SemanticProjection, dict[str, Any]]:
    """Replay immutable source records into semantic events and main turns.

    ``prior_state`` is intentionally not treated as evidence.  It is only a
    retained description of still-open interactions for incremental callers;
    replaying the complete archive always produces the authoritative result.
    """
    events = normalize_records(records)
    turns, call_candidates, pending, semantic_state = build_turns(records)
    old_open = prior_state.get("open_interactions") if isinstance(prior_state, dict) else None
    if isinstance(old_open, dict):
        for key, item in old_open.items():
            if key not in semantic_state["open_interactions"] and isinstance(item, dict):
                # State cannot prove a new association, but retaining an
                # unfinished prior partition avoids silently claiming closure.
                semantic_state["open_interactions"][key] = item
    return ({"events": events, "turns": turns, "call_candidates": call_candidates, "pending": pending, "diagnostics": []}, semantic_state)
