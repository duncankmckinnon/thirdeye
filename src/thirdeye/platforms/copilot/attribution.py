"""Defensible joins between Copilot transcript calls and database usage.

SQLite usage rows deliberately do not share an assistant-message identifier
with the transcript.  This module therefore treats a join as an assertion
that must be supported by every available identity signal; a convenient
timestamp is never evidence.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import (
    AccountingCandidate,
    AccountingProjection,
    Attribution,
    CallCandidate,
    SemanticProjection,
)

_DIRECT_ID_FIELDS = (
    "provider_call_id",
    "assistant_message_id",
    "native_call_id",
    "native_message_id",
    "message_id",
)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _direct_ids(candidate: dict[str, Any], *, semantic: bool) -> set[str]:
    """Read future native linkage fields without making one up today."""

    result = {
        value
        for field in _DIRECT_ID_FIELDS
        if (value := _string(candidate.get(field))) is not None
    }
    # A future accounting producer may carry the semantic durable call ID.
    # The current accounting shape uses logical_call_id instead, so this does
    # not turn a database identity into an accidental match.
    call_id = _string(candidate.get("call_id"))
    if call_id and (semantic or call_id.startswith("copilot:call:")):
        result.add(call_id)
    return result


def _root_candidate(
    candidate: CallCandidate, by_tool: dict[str, list[CallCandidate]]
) -> CallCandidate | None:
    """Follow a child task's parent tool back to its main interaction."""

    current = candidate
    seen: set[str] = set()
    while current.get("agent_id") is not None:
        call_id = current["call_id"]
        if call_id in seen:
            return None
        seen.add(call_id)
        parent_tool = _string(current.get("parent_tool_call_id"))
        parents = by_tool.get(parent_tool or "", [])
        if len(parents) != 1:
            return None
        current = parents[0]
    return current


def _interaction_indexes(
    calls: list[CallCandidate],
) -> tuple[dict[str, int], dict[str, CallCandidate]]:
    """Build only the verified main interaction order used by DB turn_index."""

    by_tool: dict[str, list[CallCandidate]] = defaultdict(list)
    for call in calls:
        for tool_id in call.get("tool_call_ids", []):
            if isinstance(tool_id, str) and tool_id:
                by_tool[tool_id].append(call)

    indexes: dict[str, int] = {}
    roots: dict[str, CallCandidate] = {}
    for call in calls:
        root = _root_candidate(call, by_tool)
        if root is None:
            continue
        interaction = _string(root.get("interaction_id"))
        if interaction is None:
            continue
        # A main interaction appears in archive order.  Repeated model cycles
        # intentionally retain its first position; this is not a bare turnId.
        if interaction not in indexes:
            indexes[interaction] = len(indexes)
            roots[interaction] = root
    return indexes, roots


def _root_for(call: CallCandidate, calls: list[CallCandidate]) -> CallCandidate | None:
    by_tool: dict[str, list[CallCandidate]] = defaultdict(list)
    for item in calls:
        for tool_id in item.get("tool_call_ids", []):
            if isinstance(tool_id, str) and tool_id:
                by_tool[tool_id].append(item)
    return _root_candidate(call, by_tool)


def _finish_matches(usage: AccountingCandidate, call: CallCandidate) -> bool:
    reason = _string(usage.get("finish_reason"))
    if reason is None:
        return True
    if reason == "tool_calls":
        return bool(call.get("tool_call_ids"))
    if reason in {"stop", "length", "content_filter"}:
        return not call.get("tool_call_ids") and bool(call.get("finish_evidence"))
    return False


def _initiator_matches(
    usage: AccountingCandidate,
    call: CallCandidate,
    candidates: list[CallCandidate],
) -> bool:
    initiator = (usage.get("supplemental_metrics") or {}).get("initiator")
    if initiator is None:
        return True
    if not isinstance(initiator, str):
        return False
    same_interaction = [
        item
        for item in candidates
        if item.get("interaction_id") == call.get("interaction_id")
        and item.get("agent_id") == call.get("agent_id")
        and item.get("parent_tool_call_id") == call.get("parent_tool_call_id")
    ]
    call_order = same_interaction.index(call)
    if initiator == "user":
        return call_order == 0
    if initiator == "sub-agent":
        return call.get("agent_id") is not None
    if initiator == "agent":
        return call_order > 0 and call.get("agent_id") is None
    return False


def _turn_for_usage(
    usage: AccountingCandidate, indexes: dict[str, int], roots: dict[str, CallCandidate]
) -> tuple[str | None, str | None]:
    """Return (main interaction, main stored turn) only when DB index is sound."""

    index = usage.get("turn_index")
    if not isinstance(index, int) or isinstance(index, bool):
        return None, None
    matches = [interaction for interaction, value in indexes.items() if value == index]
    if len(matches) != 1:
        return None, None
    interaction = matches[0]
    root = roots[interaction]
    return interaction, _string(root.get("stored_turn_id"))


def _base_attribution(
    usage: AccountingCandidate, stored_turn_id: str | None
) -> Attribution:
    return {
        "usage_source_id": usage["usage_source_id"],
        "logical_call_id": usage["logical_call_id"],
        "stored_turn_id": stored_turn_id,
        "agent_id": usage.get("agent_id"),
        "call_id": None,
        "status": "pending",
        "join_kind": None,
        "evidence": [f"logical_call_id:{usage['logical_call_id']}"],
    }


def join_usage(semantic: SemanticProjection, accounting: AccountingProjection) -> list[Attribution]:
    """Join each accounting candidate at most once, retaining uncertainty.

    Direct provider/message identifiers win when future source formats expose
    them.  Current CLI rows are inferred only after their main interaction is
    established from main order and child parent-tool lineage.
    """

    calls = list(semantic.get("call_candidates") or [])
    indexes, roots = _interaction_indexes(calls)
    preliminary: list[tuple[Attribution, CallCandidate | None]] = []

    for usage in accounting.get("candidates") or []:
        interaction, stored_turn_id = _turn_for_usage(usage, indexes, roots)
        result = _base_attribution(usage, stored_turn_id)
        direct = _direct_ids(usage, semantic=False)
        direct_matches = [call for call in calls if direct & _direct_ids(call, semantic=True)]
        if direct_matches:
            if len(direct_matches) != 1:
                result.update(
                    status="conflicting",
                    evidence=[
                        *(f"call_id:{call['call_id']}" for call in direct_matches),
                        "join_conflict:competing_direct_ids",
                    ],
                )
                preliminary.append((result, None))
                continue
            call = direct_matches[0]
            result.update(
                stored_turn_id=_string((_root_for(call, calls) or call).get("stored_turn_id")),
                agent_id=call.get("agent_id"),
                call_id=call["call_id"],
                status="matched",
                join_kind="direct",
                evidence=["join_kind:direct", f"call_id:{call['call_id']}"],
            )
            preliminary.append((result, call))
            continue

        if interaction is None:
            result["evidence"].extend(
                [
                    f"turn_index:{usage.get('turn_index')}",
                    "delayed_row:true",
                ]
            )
            preliminary.append((result, None))
            continue

        possible = [
            call
            for call in calls
            if (_root_for(call, calls) or call).get("interaction_id") == interaction
            and call.get("agent_id") == usage.get("agent_id")
            and call.get("parent_tool_call_id") == usage.get("parent_tool_call_id")
            and call.get("model") == usage.get("model")
            and _finish_matches(usage, call)
            and _initiator_matches(usage, call, calls)
        ]
        if len(possible) != 1:
            if possible:
                result.update(
                    status="ambiguous",
                    evidence=[
                        *(f"call_id:{call['call_id']}" for call in possible),
                        f"interaction_id:{interaction}",
                        f"model:{usage.get('model')}",
                        f"turn_index:{usage.get('turn_index')}",
                    ],
                )
            else:
                result["evidence"].extend(
                    [
                        f"interaction_id:{interaction}",
                        f"turn_index:{usage.get('turn_index')}",
                        "delayed_row:true",
                    ]
                )
            preliminary.append((result, None))
            continue

        call = possible[0]
        order = calls.index(call)
        initiator = (usage.get("supplemental_metrics") or {}).get("initiator")
        result.update(
            call_id=call["call_id"],
            status="matched",
            join_kind="inferred",
            evidence=[
                "join_kind:inferred",
                f"interaction_id:{interaction}",
                f"turn_index:{usage.get('turn_index')}",
                f"agent_id:{usage.get('agent_id')}",
                f"parent_tool_call_id:{usage.get('parent_tool_call_id')}",
                f"model:{usage.get('model')}",
                f"finish:{usage.get('finish_reason')}",
                f"initiator:{initiator}",
                f"order:{order}",
                f"call_id:{call['call_id']}",
            ],
        )
        preliminary.append((result, call))

    claimed: dict[str, list[Attribution]] = defaultdict(list)
    for attribution, call in preliminary:
        if call is not None and attribution["status"] == "matched":
            claimed[call["call_id"]].append(attribution)
    for call_id, entries in claimed.items():
        if len(entries) < 2:
            continue
        for entry in entries:
            entry.update(
                call_id=None,
                status="conflicting",
                join_kind=None,
                evidence=[f"call_id:{call_id}", "join_conflict:multiple_usage_rows"],
            )
    return [attribution for attribution, _ in preliminary]
