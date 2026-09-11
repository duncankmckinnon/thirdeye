"""Behavioral tests for Copilot semantic event normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from thirdeye.platforms.copilot.events import (
    agent_id,
    interaction_id,
    normalize_record,
    normalize_records,
    record_data,
    record_type,
    tool_call_id,
)
from thirdeye.platforms.copilot.types import SourceRecord

FIXTURES = Path(__file__).parent / "fixtures"
RECON_CASES = FIXTURES / "reconciliation-cases"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
SOURCE_KEY = "a" * 64


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _transcript_record(
    *,
    source_id: str,
    native_type: str,
    data: dict[str, Any],
    ts: str = "2026-09-10T17:08:24.503Z",
    agent: str | None = None,
) -> SourceRecord:
    payload: dict[str, Any] = {
        "type": native_type,
        "data": data,
        "id": source_id.rsplit("/", 1)[-1],
        "timestamp": ts,
        "schema_version": 1,
    }
    if agent is not None:
        payload["agentId"] = agent
    return {
        "source_id": source_id,
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": ts,
        "observed_at": "2026-09-10T17:09:00.000Z",
        "payload": payload,
        "locator": {"file": "events.jsonl", "native_event_id": payload["id"]},
    }


def _hook_record(*, source_id: str, event: str, hook_payload: dict[str, Any]) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": "hook",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:24.500Z",
        "observed_at": "2026-09-10T17:09:00.000Z",
        "payload": {
            "schema_version": 1,
            "event": event,
            "hook_payload": hook_payload,
            "context": {},
        },
        "locator": {"observation_id": source_id.rsplit("/", 1)[-1], "event": event},
    }


def _event_kinds(record: SourceRecord) -> list[str]:
    return [event["kind"] for event in normalize_record(record)]


# --- record accessors ---


def test_record_type_uses_event_field_for_hooks():
    record = _hook_record(
        source_id=f"hook/{NATIVE_SESSION_ID}/obs-1",
        event="preToolUse",
        hook_payload={"toolName": "view"},
    )
    assert record_type(record) == "preToolUse"


def test_record_data_reads_hook_payload():
    record = _hook_record(
        source_id=f"hook/{NATIVE_SESSION_ID}/obs-1",
        event="permissionRequest",
        hook_payload={"toolName": "view", "toolArgs": {"path": "/tmp/a.txt"}},
    )
    assert record_data(record)["toolName"] == "view"


def test_identity_helpers_read_transcript_fields():
    record = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/msg-1",
        native_type="assistant.message",
        data={
            "interactionId": "interaction-a",
            "turnId": "0",
            "toolRequests": [{"toolCallId": "call_abc", "name": "view", "arguments": {}}],
        },
        agent="agent-child",
    )
    execution = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/tool-1",
        native_type="tool.execution_start",
        data={"toolCallId": "call_abc", "toolName": "view", "arguments": {}},
    )
    assert interaction_id(record) == "interaction-a"
    assert agent_id(record) == "agent-child"
    assert tool_call_id(execution) == "call_abc"


# --- user and assistant messages ---


def test_user_message_normalizes_delivery_and_source_metadata():
    record = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/user-1",
        native_type="user.message",
        data={
            "content": "hello",
            "interactionId": "ix-1",
            "turnId": "0",
            "delivery": "idle",
            "source": "agent-parent",
        },
    )
    events = normalize_record(record)
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "user_prompt"
    assert event["source_references"][0]["role"] == "user_prompt"
    assert event["attributes"]["delivery"] == "idle"
    assert event["attributes"]["source"] == "agent-parent"
    assert event["attributes"]["interaction_id"] == "ix-1"


def test_assistant_message_emits_tool_requests_with_stable_ids():
    source_id = f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/assistant-1"
    record = _transcript_record(
        source_id=source_id,
        native_type="assistant.message",
        data={
            "content": "",
            "model": "gpt-5.6-luna",
            "interactionId": "ix-1",
            "turnId": "0",
            "toolRequests": [
                {
                    "toolCallId": "call_alpha",
                    "name": "view",
                    "arguments": {"path": "/fixture/workspace/alpha.txt"},
                    "intentionSummary": "view alpha",
                },
                {
                    "toolCallId": "call_beta",
                    "name": "view",
                    "arguments": {"path": "/fixture/workspace/beta.txt"},
                },
            ],
        },
    )
    events = normalize_record(record)
    assert [event["kind"] for event in events] == [
        "assistant_message",
        "tool_request",
        "tool_request",
    ]
    tool_events = events[1:]
    assert tool_events[0]["id"] == f"copilot:event:{source_id}:tool:call_alpha"
    assert tool_events[1]["id"] == f"copilot:event:{source_id}:tool:call_beta"
    assert tool_events[0]["attributes"]["arguments"] == {"path": "/fixture/workspace/alpha.txt"}
    assert tool_events[0]["attributes"]["intention_summary"] == "view alpha"


# --- tool execution ---


def test_tool_execution_complete_and_failure_kinds():
    success = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/tool-ok",
        native_type="tool.execution_complete",
        data={"toolCallId": "call_ok", "success": True, "result": "17"},
    )
    failure = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/tool-bad",
        native_type="tool.execution_complete",
        data={"toolCallId": "call_bad", "success": False, "result": "denied"},
    )
    assert _event_kinds(success) == ["tool_execution_complete"]
    assert _event_kinds(failure) == ["tool_execution_failure"]
    assert normalize_record(failure)[0]["attributes"]["result"] == "denied"


def test_tool_execution_start_preserves_arguments():
    record = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/tool-start",
        native_type="tool.execution_start",
        data={
            "toolCallId": "call_start",
            "toolName": "view",
            "arguments": {"path": "/fixture/workspace/alpha.txt"},
        },
    )
    event = normalize_record(record)[0]
    assert event["kind"] == "tool_execution_start"
    assert event["attributes"]["name"] == "view"
    assert event["attributes"]["arguments"] == {"path": "/fixture/workspace/alpha.txt"}


# --- permission, compaction, lifecycle ---


@pytest.mark.parametrize(
    ("case_name", "expected_kind", "expected_classification"),
    [
        ("permission", "permission_request", "main"),
        ("compaction", "compaction", "checkpoint"),
        ("auxiliary_title_generation", "auxiliary_model_call", "title_generation"),
    ],
)
def test_reconciliation_case_event_normalization(
    case_name: str,
    expected_kind: str,
    expected_classification: str,
) -> None:
    case = _load_json(RECON_CASES / "cases.json")[case_name]
    projection, _ = _build_semantics_from_case_records(case["input_records"])
    matching = [event for event in projection["events"] if event["kind"] == expected_kind]
    assert matching, f"expected {expected_kind} in {case_name}"
    assert matching[0]["classification"] == expected_classification


def _build_semantics_from_case_records(records: list[SourceRecord]) -> tuple[dict[str, Any], dict[str, Any]]:
    from thirdeye.platforms.copilot.tracing import build_semantics

    return build_semantics(records, {})


def test_permission_hook_maps_without_becoming_tool_execution_role():
    case = _load_json(RECON_CASES / "cases.json")["permission"]
    event = normalize_record(case["input_records"][0])[0]
    assert event["kind"] == "permission_request"
    assert event["source_references"][0]["role"] == "permission_request"


def test_compaction_checkpoint_classification():
    case = _load_json(RECON_CASES / "cases.json")["compaction"]
    transcript = case["input_records"][0]
    event = normalize_record(transcript)[0]
    assert event["kind"] == "compaction"
    assert event["classification"] == "checkpoint"
    assert event["source_references"][0]["role"] == "checkpoint"


def test_session_shutdown_is_shutdown_validation():
    record = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/shutdown",
        native_type="session.shutdown",
        data={"totalNanoAiu": 123},
    )
    event = normalize_record(record)[0]
    assert event["kind"] == "session_shutdown"
    assert event["classification"] == "shutdown_validation"


def test_subagent_events_use_nested_child_role():
    record = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/sub-start",
        native_type="subagent.started",
        data={"toolCallId": "call_parent_task", "agentName": "explore"},
        agent="bf8cb9f3-2097-4db0-a3c8-78a2653b2106",
    )
    event = normalize_record(record)[0]
    assert event["kind"] == "subagent_started"
    assert event["source_references"][0]["role"] == "nested_child"


# --- unknown and hook observations ---


def test_unknown_native_type_is_preserved():
    record = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/future",
        native_type="future.event.v2",
        data={"feature": "beta"},
    )
    event = normalize_record(record)[0]
    assert event["kind"] == "unknown"
    assert event["attributes"]["native_type"] == "future.event.v2"
    assert event["attributes"]["raw_payload"]["type"] == "future.event.v2"


def test_hook_pretooluse_stays_hook_observation_not_transcript_execution():
    record = _hook_record(
        source_id=f"hook/{NATIVE_SESSION_ID}/obs-pre",
        event="preToolUse",
        hook_payload={"toolName": "view", "toolArgs": {"path": "/tmp/a.txt"}},
    )
    event = normalize_record(record)[0]
    assert event["kind"] == "tool_execution_start"
    assert event["source_references"][0]["role"] == "hook"


def test_normalize_records_replays_in_order_without_deduplication():
    records = [
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/a",
            native_type="user.message",
            data={"content": "one", "interactionId": "ix", "turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/b",
            native_type="user.message",
            data={"content": "two", "interactionId": "ix", "turnId": "0"},
        ),
    ]
    events = normalize_records(records)
    assert len(events) == 2
    assert events[0]["attributes"]["interaction_id"] == "ix"
    assert events[1]["source_ids"] == [records[1]["source_id"]]


def test_events_module_has_no_usage_or_export_imports() -> None:
    source = Path(__import__("thirdeye.platforms.copilot.events", fromlist=["__file__"]).__file__)
    text = source.read_text(encoding="utf-8")
    for forbidden in ("usage.py", "attribution", "projection_store", "export_state"):
        assert forbidden not in text
