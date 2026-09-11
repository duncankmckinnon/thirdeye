"""Pure semantic projection for V1 Copilot archives."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .events import normalize_records
from .turns import build_turns
from .types import ProjectionDiagnostic, SemanticProjection, SourceRecord


def build_semantics(
    records: list[SourceRecord], prior_state: dict[str, Any]
) -> tuple[SemanticProjection, dict[str, Any]]:
    """Replay immutable source records into semantic events and main turns.

    ``prior_state`` is not evidence.  It only retains still-open interactions
    whose ``source_ids`` are absent from this partition.  Replaying the
    complete archive with an empty prior state is always authoritative.
    """
    events = normalize_records(records)
    turns, call_candidates, pending, diagnostics, semantic_state = build_turns(
        records, prior_state if isinstance(prior_state, dict) else {}
    )
    diagnostics = [*_auxiliary_diagnostics(events), *diagnostics]
    return (
        {
            "events": events,
            "turns": turns,
            "call_candidates": call_candidates,
            "pending": pending,
            "diagnostics": diagnostics,
        },
        deepcopy(semantic_state),
    )


def _auxiliary_diagnostics(events: list[dict[str, Any]]) -> list[ProjectionDiagnostic]:
    found: list[ProjectionDiagnostic] = []
    for event in events:
        if event.get("kind") != "auxiliary_model_call":
            continue
        attributes = event.get("attributes") or {}
        found.append(
            {
                "code": "auxiliary_excluded_from_main",
                "severity": "info",
                "message": "title-generation model.* record classified auxiliary",
                "source_ids": list(event.get("source_ids") or []),
                "details": {
                    "classification": event.get("classification"),
                    "model": attributes.get("model"),
                },
            }
        )
    return found
