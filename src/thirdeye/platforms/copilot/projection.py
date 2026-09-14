"""Pure composition of Copilot semantic, accounting, and attribution views."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .attribution import join_usage
from .tracing import build_semantics
from .types import AccountingCandidate, Attribution, Projection, ProjectionDiagnostic, SourceRecord
from .usage import build_accounting


def _unwrap_semantic_state(prior_state: dict[str, Any]) -> dict[str, Any]:
    """Accept flat tracing state or a nested ProjectionState envelope."""

    if isinstance(prior_state.get("open_interactions"), dict):
        return prior_state
    nested = prior_state.get("semantic_state")
    if isinstance(nested, dict):
        return nested
    return {}


def _find_turn(turns: list[dict[str, Any]], identity: str) -> dict[str, Any] | None:
    for turn in turns:
        if turn.get("turn_id") == identity:
            return turn
        if any(call.get("call_id") == identity for call in turn.get("llm_calls") or []):
            return turn
        found = _find_turn(turn.get("subagents") or [], identity)
        if found is not None:
            return found
    return None


def _copilot_supplemental_export_attributes(
    candidate: AccountingCandidate | None,
) -> dict[str, Any]:
    """Copy native Copilot billing/latency onto export attributes, never as USD."""

    metrics = (candidate or {}).get("supplemental_metrics") or {}
    attributes: dict[str, Any] = {}
    if "total_nano_aiu" in metrics:
        attributes["copilot.billing.nano_aiu"] = metrics["total_nano_aiu"]
    if "duration_ms" in metrics:
        attributes["copilot.latency.duration_ms"] = metrics["duration_ms"]
    if "output_ttft_ms" in metrics:
        attributes["copilot.latency.output_ttft_ms"] = metrics["output_ttft_ms"]
    if "time_to_first_token_ms" in metrics:
        attributes["copilot.latency.time_to_first_token_ms"] = metrics["time_to_first_token_ms"]
    if "inter_token_latency_ms" in metrics:
        attributes["copilot.latency.inter_token_latency_ms"] = metrics["inter_token_latency_ms"]
    return attributes


def _accounting_call(
    attribution: Attribution, row: object, candidate: AccountingCandidate | None
) -> dict[str, Any] | None:
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
            **_copilot_supplemental_export_attributes(candidate),
        },
    }


def build_projection(
    records: list[SourceRecord], prior_state: dict[str, Any]
) -> tuple[Projection, dict[str, Any]]:
    """Build a local projection without mutating either source projection."""

    semantic, semantic_state = build_semantics(records, _unwrap_semantic_state(prior_state))
    accounting, accounting_state = build_accounting(records, prior_state)
    attributions = join_usage(semantic, accounting)
    turns = deepcopy(semantic["turns"])
    rows = {row.call_id: row for row in accounting["usage_rows"]}
    candidates = {item["logical_call_id"]: item for item in accounting["candidates"]}
    pending = [*semantic["pending"]]
    diagnostics: list[ProjectionDiagnostic] = [*semantic["diagnostics"], *accounting["diagnostics"]]

    for attribution in attributions:
        row = rows.get(attribution["logical_call_id"])
        accounting_call = _accounting_call(
            attribution, row, candidates.get(attribution["logical_call_id"])
        )
        if attribution["status"] == "matched":
            if attribution["join_kind"] == "inferred":
                diagnostics.append(
                    {
                        "code": "inferred_join",
                        "severity": "info",
                        "message": "usage joined by uniquely consistent evidence",
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
        if accounting_call is None:
            continue
        owner = None
        if attribution["status"] == "matched" and attribution["call_id"] is not None:
            owner = _find_turn(turns, attribution["call_id"])
        elif attribution["stored_turn_id"] is not None:
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
            "accounting_candidates": accounting["candidates"],
        },
        {"semantic_state": semantic_state, "accounting_state": accounting_state},
    )
