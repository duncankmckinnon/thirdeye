"""Lossless semantic normalization for archived Copilot source records.

This module deliberately does not correlate records across sources.  In
particular, a hook observation has no invocation ID in the observed corpus,
so it remains an observation instead of becoming a guessed tool span.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .types import NormalizedEvent, SourceRecord, SourceReference, SourceReferenceRole


def record_type(record: SourceRecord) -> str | None:
    """Return the native event name carried by an archived record."""
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return None
    value = payload.get("type") if record.get("source_kind") != "hook" else payload.get("event")
    return value if isinstance(value, str) and value else None


def record_data(record: SourceRecord) -> dict[str, Any]:
    """Return the native data mapping without rewriting the raw record."""
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return {}
    if record.get("source_kind") == "hook":
        data = payload.get("hook_payload")
    else:
        data = payload.get("data")
    return data if isinstance(data, dict) else {}


def agent_id(record: SourceRecord) -> str | None:
    payload = record.get("payload", {})
    if isinstance(payload, dict) and isinstance(payload.get("agentId"), str):
        return payload["agentId"]
    data = record_data(record)
    for key in ("agentId", "agent_id"):
        value = data.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def interaction_id(record: SourceRecord) -> str | None:
    value = record_data(record).get("interactionId")
    return str(value) if value is not None and str(value) else None


def tool_call_id(record: SourceRecord) -> str | None:
    data = record_data(record)
    for key in ("toolCallId", "tool_call_id"):
        value = data.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _reference(record: SourceRecord, role: SourceReferenceRole) -> SourceReference:
    return {"source_id": record["source_id"], "source_kind": record["source_kind"], "role": role}


def _event(
    record: SourceRecord,
    kind: str,
    role: SourceReferenceRole,
    *,
    classification: str = "main",
    suffix: str = "",
    attributes: dict[str, Any] | None = None,
) -> NormalizedEvent:
    return {
        "id": f"copilot:event:{record['source_id']}{suffix}",
        "kind": kind,  # type: ignore[typeddict-item]
        "classification": classification,  # type: ignore[typeddict-item]
        "ts": record.get("ts"),
        "source_ids": [record["source_id"]],
        "source_references": [_reference(record, role)],
        "attributes": attributes or {},
    }


def _identity_attributes(record: SourceRecord) -> dict[str, Any]:
    data = record_data(record)
    return {
        "interaction_id": interaction_id(record),
        "agent_id": agent_id(record),
        "parent_tool_call_id": data.get("parentToolCallId"),
        "turn_id": data.get("turnId"),
    }


def normalize_record(record: SourceRecord) -> list[NormalizedEvent]:
    """Normalize one record while retaining a direct pointer to its evidence."""
    native_type = record_type(record)
    data = record_data(record)
    identity = _identity_attributes(record)
    if native_type is None:
        return [_event(record, "unknown", "hook", attributes={"raw_payload": deepcopy(record.get("payload"))})]

    if native_type == "user.message":
        attrs = {**identity, "delivery": data.get("delivery"), "source": data.get("source")}
        return [_event(record, "user_prompt", "user_prompt", attributes=attrs)]
    if native_type == "assistant.message":
        attrs = {**identity, "model": data.get("model"), "phase": data.get("phase")}
        events = [_event(record, "assistant_message", "assistant_message", attributes=attrs)]
        requests = data.get("toolRequests")
        if isinstance(requests, list):
            for request in requests:
                if not isinstance(request, dict) or not request.get("toolCallId"):
                    continue
                call_id = str(request["toolCallId"])
                events.append(
                    _event(
                        record,
                        "tool_request",
                        "tool_request",
                        suffix=f":tool:{call_id}",
                        attributes={
                            **identity,
                            "tool_call_id": call_id,
                            "name": request.get("name"),
                            "arguments": deepcopy(request.get("arguments")),
                            "intention_summary": request.get("intentionSummary"),
                        },
                    )
                )
        return events
    if native_type == "tool.execution_start":
        return [_event(record, "tool_execution_start", "tool_execution", attributes={**identity, "tool_call_id": tool_call_id(record), "name": data.get("toolName"), "arguments": deepcopy(data.get("arguments"))})]
    if native_type == "tool.execution_complete":
        kind = "tool_execution_complete" if data.get("success") is not False else "tool_execution_failure"
        return [_event(record, kind, "tool_result", attributes={**identity, "tool_call_id": tool_call_id(record), "success": data.get("success"), "result": deepcopy(data.get("result"))})]
    if native_type in {"permission.request", "permissionRequest"}:
        return [_event(record, "permission_request", "permission_request", attributes={**identity, "tool_name": data.get("toolName"), "arguments": deepcopy(data.get("toolArgs"))})]
    if native_type in {"permission.decision", "permissionDecision"}:
        return [_event(record, "permission_decision", "permission_decision", attributes={**identity, "decision": data.get("decision"), "tool_name": data.get("toolName")})]
    if native_type in {"notification", "session.notification", "hook.start", "hook.end"}:
        return [_event(record, "notification", "hook", attributes={**identity, "native_type": native_type, **deepcopy(data)})]
    if native_type in {"prompt.transformation", "prompt.transformed"}:
        return [_event(record, "prompt_transformation", "hook", attributes={**identity, **deepcopy(data)})]
    if native_type in {"assistant.abort", "session.abort", "abort"}:
        return [_event(record, "abort", "finish", attributes=identity)]
    if "error" in native_type.lower():
        return [_event(record, "error", "finish", attributes={**identity, "message": data.get("message") or data.get("error")})]
    if native_type in {"session.start", "session.resume"}:
        return [_event(record, "session_start", "hook", attributes=deepcopy(data))]
    if native_type in {"session.end", "session.shutdown", "session.close"}:
        kind = "session_shutdown" if native_type == "session.shutdown" else "session_end"
        return [_event(record, kind, "shutdown", classification="shutdown_validation" if kind == "session_shutdown" else "main", attributes=deepcopy(data))]
    if native_type in {"session.usage_checkpoint", "preCompact", "session.compaction", "context.compaction"}:
        return [_event(record, "compaction", "checkpoint" if native_type == "session.usage_checkpoint" else "compaction", classification="checkpoint" if native_type == "session.usage_checkpoint" else "main", attributes=deepcopy(data))]
    if native_type.startswith("model."):
        return [_event(record, "auxiliary_model_call", "auxiliary_model", classification="title_generation", attributes={"native_type": native_type, **deepcopy(data)})]
    if native_type == "subagent.started":
        return [_event(record, "subagent_started", "nested_child", attributes={**identity, "tool_call_id": tool_call_id(record), **deepcopy(data)})]
    if native_type == "subagent.completed":
        return [_event(record, "subagent_completed", "nested_child", attributes={**identity, "tool_call_id": tool_call_id(record), **deepcopy(data)})]

    hook_kinds = {
        "permissionRequest": ("permission_request", "permission_request"),
        "preToolUse": ("tool_execution_start", "hook"),
        "postToolUse": ("tool_execution_complete", "hook"),
        "postToolUseFailure": ("tool_execution_failure", "hook"),
        "notification": ("notification", "hook"),
        "userPromptSubmitted": ("prompt_transformation", "hook"),
        "agentStop": ("session_end", "finish"),
        "subagentStart": ("subagent_started", "nested_child"),
        "subagentStop": ("subagent_completed", "nested_child"),
        "sessionStart": ("session_start", "hook"),
        "sessionEnd": ("session_end", "hook"),
    }
    mapped = hook_kinds.get(native_type)
    if mapped is not None:
        kind, role = mapped
        attrs = {**identity, **deepcopy(data)}
        return [_event(record, kind, role, attributes=attrs)]
    return [_event(record, "unknown", "hook", attributes={"native_type": native_type, "raw_payload": deepcopy(record.get("payload"))})]


def normalize_records(records: list[SourceRecord]) -> list[NormalizedEvent]:
    """Normalize archive records in replay order without deduplicating evidence."""
    return [event for record in records for event in normalize_record(record)]
