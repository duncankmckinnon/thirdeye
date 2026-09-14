"""Explicit-identity Copilot interaction and recursive child-tree assembly.

Callers that partition an archive must include every record belonging to
still-open interactions.  ``prior_state`` retains those open keys, including
completed nested child spans, when their ``source_ids`` are absent from the
current partition.  A complete archive replay with ``prior_state={}`` is
always the authoritative result.
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
    unknown_source_schema,
)
from .types import CallCandidate, PendingItem, ProjectionDiagnostic, SourceRecord

_ABORT_TYPES = frozenset({"assistant.abort", "session.abort", "abort"})
_PUBLIC_CANDIDATE_KEYS = (
    "call_id",
    "stored_turn_id",
    "interaction_id",
    "agent_id",
    "parent_tool_call_id",
    "native_turn_id",
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
    active_native_turns: dict[tuple[str | None, str, str], str] = {}
    child_parent: dict[str, tuple[str, str]] = {}
    tool_owner: dict[str, str] = {}
    calls: dict[str, dict[str, Any]] = {}
    tool_spans: dict[str, ToolCallSpanDict] = {}
    pending: list[PendingItem] = []
    diagnostics: list[ProjectionDiagnostic] = []
    prior = prior_state if isinstance(prior_state, dict) else {}
    raw_open = prior.get("open_interactions")
    prior_open = raw_open if isinstance(raw_open, dict) else {}

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

    def bind_native_turn(
        item: dict[str, Any], agent: str | None, interaction: str | None, native_turn: Any
    ) -> None:
        if interaction is None or native_turn is None:
            return
        active_native_turns[(agent, interaction, str(native_turn))] = item["key"]

    def complete_item(item: dict[str, Any], ts: str | None) -> None:
        item["complete"] = True
        item["end_ts"] = ts or item["end_ts"]

    def attach_finish(
        item: dict[str, Any], record: SourceRecord, kind: str, native_turn: Any
    ) -> dict[str, Any] | None:
        turn_key = str(native_turn) if native_turn is not None else None
        matches: list[dict[str, Any]] = []
        if turn_key is not None:
            for call_id in item["calls"]:
                candidate = calls[call_id]
                if candidate["finish_evidence"]:
                    continue
                if candidate.get("native_turn_id") == turn_key:
                    matches.append(candidate)
        if turn_key is None or len(matches) != 1:
            pending.append(
                {
                    "id": f"pending:identity:{record['source_id']}",
                    "kind": "missing_identity",
                    "reason": f"{kind} has no uniquely matching open model cycle",
                    "source_ids": [record["source_id"]],
                    "evidence": [f"turn_id:{native_turn}"],
                }
            )
            return None
        last_call = matches[0]
        last_call["end_ts"] = record.get("ts")
        if record["source_id"] not in last_call["source_ids"]:
            last_call["source_ids"].append(record["source_id"])
        last_call["source_references"].append(source_reference(record, "finish"))
        last_call["finish_evidence"].append(
            {"source_id": record["source_id"], "kind": kind, "value": None}
        )
        return last_call

    def owner_for_native_turn(
        agent: str | None, native_turn: Any, interaction: str | None
    ) -> str | None:
        if native_turn is None:
            return None
        turn_key = str(native_turn)
        if interaction is not None:
            return active_native_turns.get((agent, interaction, turn_key))
        matches = [
            key
            for (bound_agent, _bound_ix, bound_turn), key in active_native_turns.items()
            if bound_agent == agent and bound_turn == turn_key
        ]
        unique = list(dict.fromkeys(matches))
        if len(unique) == 1:
            return unique[0]
        return None

    def drop_native_turn(agent: str | None, native_turn: Any, owner: str | None) -> None:
        if native_turn is None or owner is None:
            return
        turn_key = str(native_turn)
        for bind_key, bind_owner in list(active_native_turns.items()):
            if bind_owner == owner and bind_key[0] == agent and bind_key[2] == turn_key:
                active_native_turns.pop(bind_key, None)

    for record in records:
        if record.get("source_kind") != "transcript":
            continue
        if unknown_source_schema(record):
            payload = record.get("payload")
            version = payload.get("schema_version") if isinstance(payload, dict) else None
            pending.append(
                {
                    "id": f"pending:unknown-version:{record['source_id']}",
                    "kind": "missing_source_capability",
                    "reason": "unknown transcript schema_version",
                    "source_ids": [record["source_id"]],
                    "evidence": [f"schema_version:{version}"],
                }
            )
            diagnostics.append(
                {
                    "code": "capability_gap",
                    "severity": "warning",
                    "message": "unknown transcript schema_version; raw payload retained",
                    "source_ids": [record["source_id"]],
                    "details": {"schema_version": version},
                }
            )
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
            bind_native_turn(item, agent, interaction, data.get("turnId"))
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
            reasoning = data.get("reasoningSummary")
            native_turn = data.get("turnId")
            bind_native_turn(item, agent, interaction, native_turn)
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
            owner = owner_for_native_turn(agent, native_turn, interaction)
            if owner is None and interaction is not None:
                owner = (
                    _key(interaction, agent) if _key(interaction, agent) in interactions else None
                )
            drop_native_turn(agent, native_turn, owner)
            if owner and owner in interactions:
                item = interactions[owner]
                item["status"] = "interrupted" if is_abort else "errored"
                item["source_ids"].append(record["source_id"])
                attach_finish(item, record, "abort" if is_abort else "error", native_turn)
                # Abort closes the user turn.  An error leaves it open so a
                # later model cycle in the same interaction can retry.
                if is_abort:
                    complete_item(item, record.get("ts"))
            continue

        if native_type == "assistant.turn_end":
            native_turn = data.get("turnId")
            owner = owner_for_native_turn(agent, native_turn, interaction)
            drop_native_turn(agent, native_turn, owner)
            if owner is None:
                pending.append(
                    {
                        "id": f"pending:identity:{record['source_id']}",
                        "kind": "missing_identity",
                        "reason": "assistant.turn_end has no matching open model cycle",
                        "source_ids": [record["source_id"]],
                        "evidence": [f"turn_id:{native_turn}"],
                    }
                )
                continue
            item = interactions[owner]
            item["source_ids"].append(record["source_id"])
            item["end_ts"] = record.get("ts")
            last_call = attach_finish(item, record, "assistant_turn_end", native_turn)
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
            if child_span is None:
                continue
            child_span["attributes"]["parent_tool_call_id"] = parent_call
            parent_span = spans.get(parent_key)
            if parent_span is not None:
                parent_span["subagents"].append(child_span)

    def nested_children_for(parent_key: str) -> list[TurnSpanDict]:
        found: list[TurnSpanDict] = []
        seen: set[str] = set()
        for child, (pkey, _parent_call) in child_parent.items():
            if pkey != parent_key:
                continue
            for item in interactions.values():
                if item["agent"] != child or not item["complete"]:
                    continue
                child_span = spans.get(item["key"])
                if child_span is None or child_span["turn_id"] in seen:
                    continue
                found.append(deepcopy(child_span))
                seen.add(child_span["turn_id"])
        prior_item = prior_open.get(parent_key) if isinstance(prior_open, dict) else None
        if isinstance(prior_item, dict):
            for child in prior_item.get("nested_children") or []:
                if not isinstance(child, dict):
                    continue
                turn_id = child.get("turn_id")
                if not isinstance(turn_id, str) or turn_id in seen:
                    continue
                found.append(deepcopy(child))
                seen.add(turn_id)
        return found

    for key, span in spans.items():
        existing = {child["turn_id"] for child in span.get("subagents") or []}
        for child in nested_children_for(key):
            if child["turn_id"] in existing:
                continue
            span.setdefault("subagents", []).append(child)
            existing.add(child["turn_id"])

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
        retained: dict[str, Any] = {
            "interaction_id": item["interaction"],
            "agent_id": item["agent"],
            "stored_turn_id": item["turn_id"],
            "source_ids": item["source_ids"],
            "last_event_source_id": item["source_ids"][-1] if item["source_ids"] else None,
            "start_ts": item["start_ts"],
            "pending_tool_call_ids": pending_tools,
        }
        nested = nested_children_for(item["key"])
        if nested:
            retained["nested_children"] = nested
        open_state[item["key"]] = retained
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

    seen_incomplete_tools = {
        item["id"] for item in pending if item["kind"] == "incomplete_tool_pair"
    }
    for candidate in calls.values():
        for tool in candidate["tool_call_ids"]:
            span = tool_spans.get(tool)
            if span and span.get("end_ts"):
                continue
            pending_id = f"pending:tool:{tool}"
            if pending_id in seen_incomplete_tools:
                continue
            source_ids = list(candidate["source_ids"][:1])
            request_source = (span or {}).get("attributes", {}).get("request_source_id")
            if request_source:
                source_ids = [request_source]
            pending.append(
                {
                    "id": pending_id,
                    "kind": "incomplete_tool_pair",
                    "reason": "tool request has no execution result",
                    "source_ids": source_ids,
                    "evidence": [f"tool_call_id:{tool}"],
                }
            )
            seen_incomplete_tools.add(pending_id)

    if isinstance(prior_open, dict):
        for key, prior_item in prior_open.items():
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
