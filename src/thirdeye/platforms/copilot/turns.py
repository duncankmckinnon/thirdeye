"""Explicit-identity Copilot interaction and recursive child-tree assembly.

Callers that partition an archive must include every record belonging to
still-open interactions.  ``prior_state`` only retains open keys whose
``source_ids`` are absent from the current partition; a complete archive
replay with ``prior_state={}`` is always the authoritative result.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from thirdeye.tracing.model import LlmCallSpanDict, ToolCallSpanDict, TurnSpanDict

from .events import (
    agent_id,
    interaction_id,
    record_data,
    record_type,
    source_reference,
    tool_call_id,
)
from .types import CallCandidate, PendingItem, ProjectionDiagnostic, SourceRecord

_ABORT_TYPES = frozenset({"assistant.abort", "session.abort", "abort"})
_PUBLIC_CANDIDATE_KEYS = (
    "call_id",
    "stored_turn_id",
    "interaction_id",
    "agent_id",
    "parent_tool_call_id",
    "model",
    "source_ids",
    "source_references",
    "start_ts",
    "end_ts",
    "tool_call_ids",
    "finish_evidence",
)


def _source_key(record: SourceRecord) -> str:
    return record["source_id"].split("/", 1)[0]


def _turn_id(record: SourceRecord, interaction: str, agent: str | None) -> str:
    base = f"copilot:turn:{_source_key(record)}:{record['native_session_id']}:{interaction}"
    if agent:
        return f"{base}:{agent}"
    return base


def _key(interaction: str, agent: str | None) -> str:
    return f"{interaction}|{agent or 'main'}"


def _public_candidate(raw: dict[str, Any]) -> CallCandidate:
    return {key: raw[key] for key in _PUBLIC_CANDIDATE_KEYS}  # type: ignore[return-value]


def _text_parts(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "content": text}] if text else []


def _reasoning_parts(summary: str | None) -> list[dict[str, Any]]:
    if not isinstance(summary, str) or not summary:
        return []
    return [{"type": "reasoning", "content": summary}]


def _missing_identity(record: SourceRecord, reason: str, evidence: str) -> PendingItem:
    return {
        "id": f"pending:identity:{record['source_id']}",
        "kind": "missing_identity",
        "reason": reason,
        "source_ids": [record["source_id"]],
        "evidence": [evidence],
    }


def _incomplete_tool(record: SourceRecord, call: str, reason: str) -> PendingItem:
    return {
        "id": f"pending:tool:{call}",
        "kind": "incomplete_tool_pair",
        "reason": reason,
        "source_ids": [record["source_id"]],
        "evidence": [f"tool_call_id:{call}"],
    }


def _collect_nested_turn_ids(turns: list[TurnSpanDict]) -> set[str]:
    found: set[str] = set()
    stack = list(turns)
    while stack:
        turn = stack.pop()
        found.add(turn["turn_id"])
        stack.extend(turn.get("subagents") or [])
    return found


def build_turns(
    records: list[SourceRecord],
    prior_state: dict[str, Any] | None = None,
) -> tuple[
    list[TurnSpanDict],
    list[CallCandidate],
    list[PendingItem],
    list[ProjectionDiagnostic],
    dict[str, Any],
]:
    """Build completed main turns and recursive child spans from transcript IDs.

    ``prior_state`` is not evidence.  Unfinished interactions keep the
    ``OpenInteractionState`` shape; reconstructing a partition still requires
    the records that belong to those interactions.
    """
    interactions: dict[str, dict[str, Any]] = {}
    active_native_turns: dict[tuple[str | None, str], str] = {}
    child_parent: dict[str, tuple[str, str]] = {}
    tool_owner: dict[str, str] = {}
    calls: dict[str, dict[str, Any]] = {}
    tool_spans: dict[str, ToolCallSpanDict] = {}
    pending: list[PendingItem] = []
    diagnostics: list[ProjectionDiagnostic] = []

    def ensure(record: SourceRecord, interaction: str, agent: str | None) -> dict[str, Any]:
        key = _key(interaction, agent)
        if key not in interactions:
            interactions[key] = {
                "key": key,
                "interaction": interaction,
                "agent": agent,
                "turn_id": _turn_id(record, interaction, agent),
                "source_ids": [],
                "start_ts": record.get("ts"),
                "end_ts": None,
                "input": "",
                "output": "",
                "calls": [],
                "permission_requests": [],
                "status": "completed",
                "complete": False,
                "saw_final_answer": False,
            }
        item = interactions[key]
        item["source_ids"].append(record["source_id"])
        if item["start_ts"] is None:
            item["start_ts"] = record.get("ts")
        return item

    def remember_parent(child: str | None, parent_call: str | None) -> None:
        if child and parent_call and parent_call in tool_owner:
            child_parent[child] = (tool_owner[parent_call], parent_call)

    def close_previous_for_agent(agent: str | None, new_interaction: str, ts: str | None) -> None:
        for item in interactions.values():
            if item["agent"] != agent or item["interaction"] == new_interaction:
                continue
            if item["complete"]:
                continue
            item["complete"] = True
            item["end_ts"] = item["end_ts"] or ts

    def complete_item(item: dict[str, Any], ts: str | None) -> None:
        item["complete"] = True
        item["end_ts"] = ts or item["end_ts"]

    def attach_finish(
        item: dict[str, Any], record: SourceRecord, kind: str
    ) -> dict[str, Any] | None:
        last_id = item["calls"][-1] if item["calls"] else None
        last_call = calls.get(last_id) if last_id else None
        if last_call is None:
            return None
        last_call["end_ts"] = record.get("ts")
        if record["source_id"] not in last_call["source_ids"]:
            last_call["source_ids"].append(record["source_id"])
        last_call["source_references"].append(source_reference(record, "finish"))
        last_call["finish_evidence"].append(
            {"source_id": record["source_id"], "kind": kind, "value": None}
        )
        return last_call

    def owner_for_turn_end(record: SourceRecord, native_turn: Any, agent: str | None) -> str | None:
        if native_turn is None:
            return None
        turn_key = str(native_turn)
        bound = active_native_turns.get((agent, turn_key))
        if bound is not None:
            return bound
        matches: list[str] = []
        for candidate in calls.values():
            if candidate.get("native_turn_id") != turn_key:
                continue
            if candidate["finish_evidence"]:
                continue
            if candidate["agent_id"] != agent:
                continue
            key = candidate["interaction_key"]
            if key not in matches:
                matches.append(key)
        if len(matches) == 1:
            return matches[0]
        return None

    for record in records:
        if record.get("source_kind") != "transcript":
            continue
        native_type = record_type(record)
        data = record_data(record)
        agent = agent_id(record)
        interaction = interaction_id(record)

        if native_type == "subagent.started":
            remember_parent(agent, tool_call_id(record))
            continue

        if native_type == "subagent.completed":
            child = agent
            parent_call = tool_call_id(record)
            remember_parent(child, parent_call)
            if child:
                for item in interactions.values():
                    if item["agent"] == child and not item["complete"]:
                        complete_item(item, record.get("ts"))
            continue

        if native_type == "user.message":
            if interaction is None:
                pending.append(
                    _missing_identity(
                        record,
                        "user message has no interactionId",
                        "native_type:user.message",
                    )
                )
                continue
            close_previous_for_agent(agent, interaction, record.get("ts"))
            item = ensure(record, interaction, agent)
            item["input"] = str(data.get("content") or item["input"])
            continue

        if native_type == "assistant.turn_start":
            if interaction is None:
                pending.append(
                    _missing_identity(
                        record,
                        "assistant turn has no interactionId",
                        "native_type:assistant.turn_start",
                    )
                )
                continue
            item = ensure(record, interaction, agent)
            native_turn = data.get("turnId")
            if native_turn is not None:
                active_native_turns[(agent, str(native_turn))] = item["key"]
            continue

        if native_type == "assistant.message":
            if interaction is None:
                pending.append(
                    _missing_identity(
                        record,
                        "assistant message has no interactionId",
                        "native_type:assistant.message",
                    )
                )
                continue
            item = ensure(record, interaction, agent)
            remember_parent(agent, data.get("parentToolCallId"))
            call_id = f"copilot:call:{record['source_id']}"
            requests = (
                data.get("toolRequests") if isinstance(data.get("toolRequests"), list) else []
            )
            requested_ids = [
                str(request["toolCallId"])
                for request in requests
                if isinstance(request, dict) and request.get("toolCallId")
            ]
            content = data.get("content") if isinstance(data.get("content"), str) else ""
            reasoning = data.get("reasoningSummary") or data.get("intentionSummary")
            native_turn = data.get("turnId")
            candidate = {
                "call_id": call_id,
                "stored_turn_id": item["turn_id"],
                "interaction_id": interaction,
                "agent_id": agent,
                "parent_tool_call_id": data.get("parentToolCallId"),
                "model": data.get("model"),
                "source_ids": [record["source_id"]],
                "source_references": [source_reference(record, "assistant_message")],
                "start_ts": record.get("ts"),
                "end_ts": None,
                "tool_call_ids": requested_ids,
                "finish_evidence": [],
                "interaction_key": item["key"],
                "native_turn_id": str(native_turn) if native_turn is not None else None,
                "content": content,
                "reasoning_summary": reasoning if isinstance(reasoning, str) else None,
                "final_answer": data.get("phase") == "final_answer",
            }
            calls[call_id] = candidate
            item["calls"].append(call_id)
            if content:
                item["output"] = content
            if candidate["final_answer"]:
                item["saw_final_answer"] = True
                item["status"] = "completed"
            for request in requests:
                if not isinstance(request, dict) or not request.get("toolCallId"):
                    continue
                call = str(request["toolCallId"])
                tool_owner[call] = item["key"]
                tool_spans[call] = {
                    "tool_call_id": call,
                    "name": str(request.get("name") or ""),
                    "start_ts": str(record.get("ts") or ""),
                    "end_ts": "",
                    "attributes": {
                        "arguments": deepcopy(request.get("arguments")),
                        "intention_summary": request.get("intentionSummary"),
                        "request_source_id": record["source_id"],
                    },
                }
            continue

        if native_type == "tool.execution_start":
            call = tool_call_id(record)
            if call and call in tool_spans:
                span = tool_spans[call]
                span["start_ts"] = str(record.get("ts") or span["start_ts"])
                span["attributes"].update(
                    {
                        "arguments": deepcopy(data.get("arguments")),
                        "execution_start_source_id": record["source_id"],
                    }
                )
            elif call:
                pending.append(
                    _incomplete_tool(
                        record,
                        call,
                        "tool execution start has no requesting assistant message",
                    )
                )
            continue

        if native_type in {
            "tool.execution_complete",
            "tool.execution_error",
            "tool.execution_failure",
        }:
            call = tool_call_id(record)
            if call and call in tool_spans:
                span = tool_spans[call]
                span["end_ts"] = str(record.get("ts") or "")
                span["attributes"].update(
                    {
                        "result": deepcopy(data.get("result")),
                        "success": data.get("success"),
                        "execution_result_source_id": record["source_id"],
                    }
                )
            elif call:
                pending.append(
                    _incomplete_tool(
                        record,
                        call,
                        "tool result has no requesting assistant message",
                    )
                )
            continue

        if native_type in {"permission.request", "permissionRequest"}:
            if interaction is None:
                continue
            item = ensure(record, interaction, agent)
            item["permission_requests"].append(
                {
                    "ts": str(record.get("ts") or ""),
                    "tool_name": str(data.get("toolName") or ""),
                    "attributes": {
                        "arguments": deepcopy(data.get("toolArgs")),
                        "source_id": record["source_id"],
                    },
                }
            )
            continue

        if native_type in {"permission.decision", "permissionDecision"}:
            if interaction is None:
                continue
            item = ensure(record, interaction, agent)
            decision = data.get("decision")
            tool_name = data.get("toolName")
            for request in reversed(item["permission_requests"]):
                if tool_name and request["tool_name"] != tool_name:
                    continue
                request["attributes"]["decision"] = decision
                request["attributes"]["decision_source_id"] = record["source_id"]
                break
            continue

        is_abort = native_type in _ABORT_TYPES
        is_error = (
            bool(native_type)
            and "error" in native_type.lower()
            and not native_type.startswith("model.")
            and not native_type.startswith("tool.")
        )
        if is_abort or is_error:
            native_turn = data.get("turnId")
            owner = (
                active_native_turns.pop((agent, str(native_turn)), None)
                if native_turn is not None
                else None
            )
            if owner is None and interaction is not None:
                owner = (
                    _key(interaction, agent) if _key(interaction, agent) in interactions else None
                )
            if owner and owner in interactions:
                item = interactions[owner]
                item["status"] = "interrupted" if is_abort else "errored"
                item["source_ids"].append(record["source_id"])
                attach_finish(item, record, "abort" if is_abort else "error")
                # Abort closes the user turn.  An error leaves it open so a
                # later model cycle in the same interaction can retry.
                if is_abort:
                    complete_item(item, record.get("ts"))
            continue

        if native_type == "assistant.turn_end":
            native_turn = data.get("turnId")
            owner = owner_for_turn_end(record, native_turn, agent)
            if native_turn is not None:
                active_native_turns.pop((agent, str(native_turn)), None)
            if owner is None:
                pending.append(
                    {
                        "id": f"pending:identity:{record['source_id']}",
                        "kind": "incomplete_tool_pair",
                        "reason": "assistant.turn_end has no matching open model cycle",
                        "source_ids": [record["source_id"]],
                        "evidence": [f"turn_id:{native_turn}"],
                    }
                )
                continue
            item = interactions[owner]
            item["source_ids"].append(record["source_id"])
            item["end_ts"] = record.get("ts")
            last_call = attach_finish(item, record, "assistant_turn_end")
            if (
                last_call is not None
                and not last_call["tool_call_ids"]
                and last_call.get("final_answer")
            ):
                complete_item(item, record.get("ts"))
            continue

        if native_type in {"session.shutdown", "session.end", "session.close"}:
            for item in interactions.values():
                if item["complete"]:
                    continue
                if item["saw_final_answer"] or item["status"] != "completed":
                    complete_item(item, record.get("ts"))
            continue

    def call_span(candidate: dict[str, Any], user_input: str) -> LlmCallSpanDict:
        attached = [tool_spans[tool] for tool in candidate["tool_call_ids"] if tool in tool_spans]
        output_parts = _text_parts(str(candidate.get("content") or "")) + _reasoning_parts(
            candidate.get("reasoning_summary")
        )
        return {
            "call_id": candidate["call_id"],
            "provider": "unknown",
            "model": str(candidate.get("model") or "unknown"),
            "start_ts": str(candidate.get("start_ts") or ""),
            "end_ts": str(candidate.get("end_ts") or candidate.get("start_ts") or ""),
            "input_messages": (
                [{"role": "user", "parts": _text_parts(user_input)}] if user_input else []
            ),
            "output_messages": (
                [{"role": "assistant", "parts": output_parts}] if output_parts else []
            ),
            "usage": {},
            "tool_calls": attached,
        }

    spans: dict[str, TurnSpanDict] = {}
    for key, item in interactions.items():
        if not item["complete"]:
            continue
        spans[key] = {
            "turn_id": item["turn_id"],
            "start_ts": str(item["start_ts"] or ""),
            "end_ts": str(item["end_ts"] or item["start_ts"] or ""),
            "input_message": item["input"],
            "output_message": item["output"],
            "status": item["status"],
            "llm_calls": [call_span(calls[call], item["input"]) for call in item["calls"]],
            "permission_requests": item["permission_requests"],
            "subagents": [],
            "attributes": {
                "interaction_id": item["interaction"],
                "agent_id": item["agent"],
            },
            "accounting_calls": [],
        }

    for child, (parent_key, parent_call) in child_parent.items():
        child_items = [
            item for item in interactions.values() if item["agent"] == child and item["complete"]
        ]
        for item in child_items:
            child_span = spans.get(item["key"])
            parent_span = spans.get(parent_key)
            if child_span is not None and parent_span is not None:
                child_span["attributes"]["parent_tool_call_id"] = parent_call
                parent_span["subagents"].append(child_span)

    emitted_ids = _collect_nested_turn_ids(
        [span for key, span in spans.items() if interactions[key]["agent"] is None]
    )
    for item in interactions.values():
        if not item["complete"] or item["agent"] is None:
            continue
        if item["turn_id"] in emitted_ids:
            continue
        pending.append(
            {
                "id": f"pending:identity:{item['turn_id']}",
                "kind": "missing_identity",
                "reason": "completed child interaction has no resolved parent tool call",
                "source_ids": item["source_ids"],
                "evidence": [
                    f"interaction_id:{item['interaction']}",
                    f"agent_id:{item['agent']}",
                ],
            }
        )
        diagnostics.append(
            {
                "code": "capability_gap",
                "severity": "warning",
                "message": "completed child interaction has no resolved parent tool call",
                "source_ids": item["source_ids"],
                "details": {
                    "interaction_id": item["interaction"],
                    "agent_id": item["agent"],
                    "stored_turn_id": item["turn_id"],
                },
            }
        )
        for call_id in item["calls"]:
            calls[call_id]["stored_turn_id"] = None

    open_state: dict[str, Any] = {}
    current_source_ids = {record["source_id"] for record in records}
    for item in interactions.values():
        if item["complete"]:
            continue
        pending_tools = [
            tool
            for call in item["calls"]
            for tool in calls[call]["tool_call_ids"]
            if not tool_spans.get(tool, {}).get("end_ts")
        ]
        has_finish = any(calls[call]["finish_evidence"] for call in item["calls"])
        reason = (
            "user interaction has no completed final assistant turn"
            if has_finish
            else "user interaction has no assistant.turn_end"
        )
        open_state[item["key"]] = {
            "interaction_id": item["interaction"],
            "agent_id": item["agent"],
            "stored_turn_id": item["turn_id"],
            "source_ids": item["source_ids"],
            "last_event_source_id": item["source_ids"][-1] if item["source_ids"] else None,
            "start_ts": item["start_ts"],
            "pending_tool_call_ids": pending_tools,
        }
        pending.append(
            {
                "id": f"pending:{item['turn_id']}",
                "kind": "open_interaction",
                "reason": reason,
                "source_ids": item["source_ids"],
                "evidence": [f"interaction_id:{item['interaction']}"],
            }
        )
        diagnostics.append(
            {
                "code": "open_interaction",
                "severity": "info",
                "message": reason,
                "source_ids": item["source_ids"],
                "details": {"interaction_id": item["interaction"]},
            }
        )

    prior = prior_state.get("open_interactions") if isinstance(prior_state, dict) else None
    if isinstance(prior, dict):
        for key, prior_item in prior.items():
            if key in open_state or not isinstance(prior_item, dict):
                continue
            prior_sources = prior_item.get("source_ids") or []
            if prior_sources and any(
                source_id in current_source_ids for source_id in prior_sources
            ):
                continue
            open_state[key] = deepcopy(prior_item)

    main_turns = [span for key, span in spans.items() if interactions[key]["agent"] is None]
    main_turns.sort(key=lambda turn: (turn["start_ts"] == "", turn["start_ts"]))
    call_candidates = [_public_candidate(raw) for raw in calls.values()]
    return main_turns, call_candidates, pending, diagnostics, {"open_interactions": open_state}
