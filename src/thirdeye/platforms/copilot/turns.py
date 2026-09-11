"""Explicit-identity Copilot interaction and recursive child-tree assembly."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from thirdeye.tracing.model import LlmCallSpanDict, ToolCallSpanDict, TurnSpanDict

from .events import agent_id, interaction_id, record_data, record_type, tool_call_id
from .types import CallCandidate, PendingItem, SourceRecord, SourceReference


def _source_key(record: SourceRecord) -> str | None:
    source_id = record["source_id"]
    if source_id.startswith("hook/") or source_id.startswith("copilot-db:"):
        return None
    return source_id.split("/", 1)[0] or None


def _turn_id(record: SourceRecord, interaction: str) -> str:
    source_key = _source_key(record)
    if source_key is None:
        return f"copilot:turn:unknown:{record['native_session_id']}:{interaction}"
    return f"copilot:turn:{source_key}:{record['native_session_id']}:{interaction}"


def _reference(record: SourceRecord, role: str) -> SourceReference:
    return {"source_id": record["source_id"], "source_kind": record["source_kind"], "role": role}  # type: ignore[typeddict-item]


def _key(interaction: str, agent: str | None) -> str:
    return f"{interaction}|{agent or 'main'}"


def build_turns(records: list[SourceRecord]) -> tuple[list[TurnSpanDict], list[CallCandidate], list[PendingItem], dict[str, Any]]:
    """Build completed main turns and recursive child spans from transcript IDs only."""
    interactions: dict[str, dict[str, Any]] = {}
    active_native_turns: dict[tuple[str | None, str], str] = {}
    child_parent: dict[str, tuple[str, str]] = {}
    tool_owner: dict[str, str] = {}
    calls: dict[str, dict[str, Any]] = {}
    tool_spans: dict[str, ToolCallSpanDict] = {}
    pending: list[PendingItem] = []

    def ensure(record: SourceRecord, interaction: str, agent: str | None) -> dict[str, Any]:
        key = _key(interaction, agent)
        if key not in interactions:
            interactions[key] = {
                "key": key, "interaction": interaction, "agent": agent, "turn_id": _turn_id(record, interaction),
                "source_ids": [], "start_ts": None, "end_ts": None, "input": "", "output": "", "calls": [],
                "permission_requests": [], "status": "completed", "complete": False, "parent": None,
            }
        item = interactions[key]
        item["source_ids"].append(record["source_id"])
        item["start_ts"] = item["start_ts"] or record.get("ts")
        return item

    for record in records:
        if record.get("source_kind") != "transcript":
            continue
        native_type = record_type(record)
        data = record_data(record)
        agent = agent_id(record)
        interaction = interaction_id(record)

        if native_type == "subagent.started":
            child = agent_id(record)
            parent_call = tool_call_id(record)
            if child and parent_call and parent_call in tool_owner:
                child_parent[child] = (tool_owner[parent_call], parent_call)
            continue
        if native_type == "user.message":
            if interaction is None:
                pending.append({"id": f"pending:identity:{record['source_id']}", "kind": "missing_identity", "reason": "user message has no interactionId", "source_ids": [record["source_id"]], "evidence": ["native_type:user.message"]})
                continue
            item = ensure(record, interaction, agent)
            item["input"] = str(data.get("content") or item["input"])
            continue
        if native_type == "assistant.turn_start":
            if interaction is None:
                pending.append({"id": f"pending:identity:{record['source_id']}", "kind": "missing_identity", "reason": "assistant turn has no interactionId", "source_ids": [record["source_id"]], "evidence": ["native_type:assistant.turn_start"]})
                continue
            item = ensure(record, interaction, agent)
            native_turn = data.get("turnId")
            if native_turn is not None:
                active_native_turns[(agent, str(native_turn))] = item["key"]
            continue
        if native_type == "assistant.message":
            if interaction is None:
                pending.append({"id": f"pending:identity:{record['source_id']}", "kind": "missing_identity", "reason": "assistant message has no interactionId", "source_ids": [record["source_id"]], "evidence": ["native_type:assistant.message"]})
                continue
            item = ensure(record, interaction, agent)
            call_id = f"copilot:call:{record['source_id']}"
            requests = data.get("toolRequests") if isinstance(data.get("toolRequests"), list) else []
            requested_ids = [str(request["toolCallId"]) for request in requests if isinstance(request, dict) and request.get("toolCallId")]
            candidate = {"call_id": call_id, "stored_turn_id": item["turn_id"], "interaction_id": interaction, "agent_id": agent, "parent_tool_call_id": data.get("parentToolCallId"), "model": data.get("model"), "source_ids": [record["source_id"]], "source_references": [_reference(record, "assistant_message")], "start_ts": record.get("ts"), "end_ts": None, "tool_call_ids": requested_ids, "finish_evidence": []}
            calls[call_id] = candidate
            item["calls"].append(call_id)
            content = data.get("content")
            if isinstance(content, str) and content:
                item["output"] = content
            for request in requests:
                if not isinstance(request, dict) or not request.get("toolCallId"):
                    continue
                call = str(request["toolCallId"])
                tool_owner[call] = item["key"]
                tool_spans[call] = {"tool_call_id": call, "name": str(request.get("name") or ""), "start_ts": str(record.get("ts") or ""), "end_ts": "", "attributes": {"arguments": deepcopy(request.get("arguments")), "intention_summary": request.get("intentionSummary"), "request_source_id": record["source_id"]}}
            continue
        if native_type == "tool.execution_start":
            call = tool_call_id(record)
            if call and call in tool_spans:
                span = tool_spans[call]
                span["start_ts"] = str(record.get("ts") or span["start_ts"])
                span["attributes"].update({"arguments": deepcopy(data.get("arguments")), "execution_start_source_id": record["source_id"]})
            elif call:
                pending.append({"id": f"pending:tool:{call}", "kind": "incomplete_tool_pair", "reason": "tool execution start has no requesting assistant message", "source_ids": [record["source_id"]], "evidence": [f"tool_call_id:{call}"]})
            continue
        if native_type == "tool.execution_complete":
            call = tool_call_id(record)
            if call and call in tool_spans:
                span = tool_spans[call]
                span["end_ts"] = str(record.get("ts") or "")
                span["attributes"].update({"result": deepcopy(data.get("result")), "success": data.get("success"), "execution_result_source_id": record["source_id"]})
            elif call:
                pending.append({"id": f"pending:tool:{call}", "kind": "incomplete_tool_pair", "reason": "tool result has no requesting assistant message", "source_ids": [record["source_id"]], "evidence": [f"tool_call_id:{call}"]})
            continue
        if native_type in {"assistant.abort", "session.abort", "abort"} or (native_type and "error" in native_type.lower()):
            native_turn = data.get("turnId")
            owner = active_native_turns.get((agent, str(native_turn))) if native_turn is not None else None
            if owner and owner in interactions:
                interactions[owner]["status"] = "interrupted" if native_type in {"assistant.abort", "session.abort", "abort"} else "errored"
                interactions[owner]["end_ts"] = record.get("ts")
            continue
        if native_type == "assistant.turn_end":
            native_turn = data.get("turnId")
            owner = active_native_turns.pop((agent, str(native_turn)), None) if native_turn is not None else None
            if owner is None:
                pending.append({"id": f"pending:identity:{record['source_id']}", "kind": "missing_identity", "reason": "assistant.turn_end cannot be assigned without agent and active turn identity", "source_ids": [record["source_id"]], "evidence": [f"turn_id:{native_turn}"]})
                continue
            item = interactions[owner]
            item["source_ids"].append(record["source_id"])
            item["end_ts"] = record.get("ts")
            # A turn end completes a model cycle.  It only completes the user
            # interaction when the cycle contains the final answer (or was
            # explicitly aborted/errored); tool cycles remain open.
            last_call = calls.get(item["calls"][-1]) if item["calls"] else None
            if last_call is not None:
                last_call["end_ts"] = record.get("ts")
                last_call["source_ids"].append(record["source_id"])
                last_call["source_references"].append(_reference(record, "finish"))
                last_call["finish_evidence"].append({"source_id": record["source_id"], "kind": "assistant_turn_end", "value": None})
                if not last_call["tool_call_ids"] and item["output"]:
                    item["complete"] = True
            continue

    def call_span(candidate: dict[str, Any]) -> LlmCallSpanDict:
        attached = [tool_spans[tool] for tool in candidate["tool_call_ids"] if tool in tool_spans]
        return {"call_id": candidate["call_id"], "provider": "unknown", "model": str(candidate.get("model") or "unknown"), "start_ts": str(candidate.get("start_ts") or ""), "end_ts": str(candidate.get("end_ts") or candidate.get("start_ts") or ""), "input_messages": [], "output_messages": [], "usage": {}, "tool_calls": attached}

    spans: dict[str, TurnSpanDict] = {}
    for key, item in interactions.items():
        if not item["complete"]:
            continue
        spans[key] = {"turn_id": item["turn_id"], "start_ts": str(item["start_ts"] or ""), "end_ts": str(item["end_ts"] or item["start_ts"] or ""), "input_message": item["input"], "output_message": item["output"], "status": item["status"], "llm_calls": [call_span(calls[call]) for call in item["calls"]], "permission_requests": item["permission_requests"], "subagents": [], "attributes": {"interaction_id": item["interaction"], "agent_id": item["agent"]}, "accounting_calls": []}

    for child, (parent_key, parent_call) in child_parent.items():
        child_items = [item for item in interactions.values() if item["agent"] == child and item["complete"]]
        for item in child_items:
            child_span = spans.get(item["key"])
            parent_span = spans.get(parent_key)
            if child_span is not None and parent_span is not None:
                child_span["attributes"]["parent_tool_call_id"] = parent_call
                parent_span["subagents"].append(child_span)

    open_state: dict[str, Any] = {}
    for item in interactions.values():
        if item["complete"]:
            continue
        open_state[item["key"]] = {"interaction_id": item["interaction"], "agent_id": item["agent"], "stored_turn_id": item["turn_id"], "source_ids": item["source_ids"], "last_event_source_id": item["source_ids"][-1] if item["source_ids"] else None, "start_ts": item["start_ts"], "pending_tool_call_ids": [tool for call in item["calls"] for tool in calls[call]["tool_call_ids"] if not tool_spans.get(tool, {}).get("end_ts")]}
        pending.append({"id": f"pending:{item['turn_id']}", "kind": "open_interaction", "reason": "user interaction has no completed final assistant turn", "source_ids": item["source_ids"], "evidence": [f"interaction_id:{item['interaction']}"]})

    main_turns = [span for key, span in spans.items() if interactions[key]["agent"] is None]
    main_turns.sort(key=lambda turn: turn["start_ts"])
    call_candidates: list[CallCandidate] = list(calls.values())  # type: ignore[assignment]
    return main_turns, call_candidates, pending, {"open_interactions": open_state}
