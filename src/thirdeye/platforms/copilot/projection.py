"""Pure composition of Copilot semantic, accounting, and attribution views."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .attribution import join_usage
from .tracing import build_semantics
from .types import Attribution, Projection, ProjectionDiagnostic, SourceRecord
from .usage import build_accounting


def _find_turn(turns: list[dict[str, Any]], turn_id: str) -> dict[str, Any] | None:
    for turn in turns:
        if turn.get("turn_id") == turn_id:
            return turn
        found = _find_turn(turn.get("subagents") or [], turn_id)
        if found is not None:
            return found
    return None


def _accounting_call(attribution: Attribution, row: object) -> dict[str, Any] | None:
    if row is None or not hasattr(row, "to_dict"):
        return None
    return {
        "accounting_id": attribution["logical_call_id"],
        "usage": row.to_dict(),
        "attribution_status": attribution["status"],
        "agent_id": attribution["agent_id"],
        "call_id": attribution["call_id"] if attribution["status"] == "matched" else None,
        "attributes": {
            "logical_call_id": attribution["logical_call_id"],
            "usage_source_id": attribution["usage_source_id"],
            "join_kind": attribution["join_kind"],
            "evidence": list(attribution["evidence"]),
        },
    }


def build_projection(
    records: list[SourceRecord], prior_state: dict[str, Any]
) -> tuple[Projection, dict[str, Any]]:
    """Build a local projection without mutating either source projection."""

    semantic, semantic_state = build_semantics(records, prior_state)
    accounting, accounting_state = build_accounting(records, prior_state)
    attributions = join_usage(semantic, accounting)
    turns = deepcopy(semantic["turns"])
    rows = {row.call_id: row for row in accounting["usage_rows"]}
    pending = [*semantic["pending"]]
    diagnostics: list[ProjectionDiagnostic] = [*semantic["diagnostics"], *accounting["diagnostics"]]

    for attribution in attributions:
        row = rows.get(attribution["logical_call_id"])
        accounting_call = _accounting_call(attribution, row)
        if attribution["status"] == "matched":
            inferred = attribution["join_kind"] == "inferred"
            diagnostics.append(
                {
                    "code": "inferred_join" if inferred else "capability_gap",
                    "severity": "info",
                    "message": (
                        "usage joined by uniquely consistent evidence"
                        if inferred
                        else "usage joined to an assistant call"
                    ),
                    "source_ids": [attribution["usage_source_id"]],
                    "details": {
                        "logical_call_id": attribution["logical_call_id"],
                        "call_id": attribution["call_id"],
                    },
                }
            )
        else:
            pending.append(
                {
                    "id": f"pending:usage:{attribution['logical_call_id']}",
                    "kind": "unmatched_usage",
                    "reason": f"usage attribution is {attribution['status']}",
                    "source_ids": [attribution["usage_source_id"]],
                    "evidence": list(attribution["evidence"]),
                }
            )
        if accounting_call is None or attribution["stored_turn_id"] is None:
            continue
        owner = _find_turn(turns, attribution["stored_turn_id"])
        if owner is not None:
            owner.setdefault("accounting_calls", []).append(accounting_call)

    return (
        {
            "normalized_events": semantic["events"],
            "turns": turns,
            "usage_rows": accounting["usage_rows"],
            "attributions": attributions,
            "pending": pending,
            "diagnostics": diagnostics,
        },
        {"semantic_state": semantic_state, "accounting_state": accounting_state},
    )
