"""Behavioral tests for Copilot semantic projection (build_semantics)."""

from __future__ import annotations

import ast
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.tracing import build_semantics
from thirdeye.platforms.copilot.transcript import read_transcript
from thirdeye.platforms.copilot.turns import build_turns
from thirdeye.platforms.copilot.types import SourceRecord

FIXTURES = Path(__file__).parent / "fixtures"
RECON_CASES = FIXTURES / "reconciliation-cases"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
CHILD_AGENT_ID = "bf8cb9f3-2097-4db0-a3c8-78a2653b2106"
SOURCE_KEY = "a" * 64


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _drain_cli_transcript(home: Path) -> list[SourceRecord]:
    session_dir = home / "session-state" / NATIVE_SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "events.jsonl", session_dir / "events.jsonl")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")
    paths = resolve_sources(home)
    cursor: dict[str, Any] = {}
    records: list[SourceRecord] = []
    while True:
        slice_ = read_transcript(paths, NATIVE_SESSION_ID, cursor)
        records.extend(slice_["records"])
        cursor = slice_["next_cursor"]
        if slice_["exhausted"]:
            break
    return [record for record in records if record["source_kind"] == "transcript"]


@pytest.fixture
def cli_transcript_records(tmp_path: Path) -> list[SourceRecord]:
    return _drain_cli_transcript(tmp_path / "copilot-home")


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


# --- observed CLI fixture ---


def test_cli_fixture_reconstructs_two_main_interactions_and_explore_child(
    cli_transcript_records: list[SourceRecord],
) -> None:
    projection, state = build_semantics(cli_transcript_records, {})
    turns = projection["turns"]

    assert len(turns) == 2
    assert [turn["output_message"] for turn in turns] == ["42", "42"]
    assert [turn["attributes"]["interaction_id"] for turn in turns] == [
        "6d2b89fd-a653-430c-b532-b0936d72eb42",
        "793d3703-6f4a-4814-8877-34a7325848ce",
    ]

    child_turn = turns[1]["subagents"][0]
    assert child_turn["output_message"] == "42"
    assert child_turn["attributes"]["agent_id"] == CHILD_AGENT_ID
    assert child_turn["attributes"]["parent_tool_call_id"] == "call_qx4FH5DADTeT1qVLb37HNpBk"
    assert child_turn["attributes"]["interaction_id"] == "7c0fa097-c0e2-48da-b2b6-fcfc1ad83a6b"

    assert projection["pending"] == []
    assert state["open_interactions"] == {}


def test_cli_fixture_counts_five_tools_and_six_call_candidates(
    cli_transcript_records: list[SourceRecord],
) -> None:
    projection, _ = build_semantics(cli_transcript_records, {})
    tool_requests = [event for event in projection["events"] if event["kind"] == "tool_request"]
    tool_starts = [
        event for event in projection["events"] if event["kind"] == "tool_execution_start"
    ]
    tool_completes = [
        event for event in projection["events"] if event["kind"] == "tool_execution_complete"
    ]

    assert len(tool_requests) == 5
    assert len(tool_starts) == 5
    assert len(tool_completes) == 5
    assert len(projection["call_candidates"]) == 6

    tool_ids = sorted(event["attributes"]["tool_call_id"] for event in tool_requests)
    assert tool_ids == [
        "call_Jeh7IbrUHaq4jVdxtyQCVrns",
        "call_YSSva4HCniiETlxdGGjcrHbh",
        "call_ayHplfzxjRFMTCpmTKEFhCSJ",
        "call_qx4FH5DADTeT1qVLb37HNpBk",
        "call_zZncCGtp1twgcL2eoFUNwInh",
    ]


def test_cli_fixture_replay_is_deterministic(cli_transcript_records: list[SourceRecord]) -> None:
    first, first_state = build_semantics(cli_transcript_records, {})
    second, _ = build_semantics(cli_transcript_records, first_state)
    third, _ = build_semantics(cli_transcript_records, {})

    assert first["events"] == third["events"]
    assert first["turns"] == third["turns"]
    assert first["call_candidates"] == third["call_candidates"]
    assert second["events"] == third["events"]


def _assert_expected_projection(projection: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("events", "turns", "call_candidates", "pending", "diagnostics"):
        if key not in expected:
            continue
        actual = projection[key]
        wanted = expected[key]
        if key == "events":
            by_id = {event["id"]: event for event in actual}
            for item in wanted:
                got = by_id[item["id"]]
                assert got["kind"] == item["kind"]
                assert got["classification"] == item["classification"]
                assert got["ts"] == item["ts"]
                assert got["source_ids"] == item["source_ids"]
                for attr_key, attr_value in item.get("attributes", {}).items():
                    assert got["attributes"].get(attr_key) == attr_value
            continue
        if key == "call_candidates":
            by_id = {item["call_id"]: item for item in actual}
            for item in wanted:
                got = by_id[item["call_id"]]
                for field, value in item.items():
                    assert got[field] == value, field
            continue
        assert actual == wanted


def test_build_semantics_does_not_import_usage_or_attribution_modules() -> None:
    forbidden = {"usage", "attribution", "projection_store", "export_state"}
    for module_name in ("tracing", "turns", "events"):
        source = Path(
            __import__(f"thirdeye.platforms.copilot.{module_name}", fromlist=["__file__"]).__file__
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
                imported.add(node.module)
        assert not (imported & forbidden), module_name


# --- interaction grouping and turn completion ---


def test_tool_cycle_does_not_complete_user_turn_without_final_answer() -> None:
    case = _load_json(RECON_CASES / "semantic-projection.json")
    projection, state = build_semantics(case["input_records"], {})
    _assert_expected_projection(projection, case["expected"])

    pending_kinds = Counter(item["kind"] for item in projection["pending"])
    assert pending_kinds["open_interaction"] == 1
    assert "missing_identity" not in pending_kinds

    open_key = "6d2b89fd-a653-430c-b532-b0936d72eb42|main"
    assert open_key in state["open_interactions"]
    assert state["open_interactions"][open_key]["pending_tool_call_ids"] == [
        "call_YSSva4HCniiETlxdGGjcrHbh",
        "call_ayHplfzxjRFMTCpmTKEFhCSJ",
    ]
    candidate = projection["call_candidates"][0]
    assert candidate["end_ts"] == "2026-09-10T17:08:24.593Z"
    assert candidate["finish_evidence"][0]["kind"] == "assistant_turn_end"


def test_partial_user_prompt_stays_open_with_pending_item() -> None:
    case = _load_json(RECON_CASES / "cases.json")["abort"]
    projection, state = build_semantics(case["input_records"], {})

    assert projection["turns"] == []
    assert any(item["kind"] == "open_interaction" for item in projection["pending"])
    assert "6d2b89fd-a653-430c-b532-b0936d72eb42|main" in state["open_interactions"]


def test_missing_interaction_id_creates_pending_not_fabricated_turn() -> None:
    record = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/user-no-ix",
        native_type="user.message",
        data={"content": "orphan prompt", "turnId": "0"},
    )
    _, _, pending, _, _ = build_turns([record])

    assert any(item["kind"] == "missing_identity" for item in pending)
    assert all(item["kind"] != "open_interaction" for item in pending)


def test_incomplete_tool_pair_when_execution_lacks_request() -> None:
    execution = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/exec-orphan",
        native_type="tool.execution_start",
        data={"toolCallId": "call_orphan", "toolName": "view", "arguments": {}},
    )
    _, _, pending, _, _ = build_turns([execution])

    assert pending == [
        {
            "id": "pending:tool:call_orphan",
            "kind": "incomplete_tool_pair",
            "reason": "tool execution start has no requesting assistant message",
            "source_ids": [execution["source_id"]],
            "evidence": ["tool_call_id:call_orphan"],
        }
    ]


def test_identical_concurrent_tools_pair_by_tool_call_id() -> None:
    case = _load_json(RECON_CASES / "cases.json")["identical_concurrent_tools"]
    projection, _ = build_semantics(case["input_records"], {})
    expected = case["expected"]

    start_events = [
        event for event in projection["events"] if event["kind"] == "tool_execution_start"
    ]
    assert sorted(event["id"] for event in start_events) == sorted(expected["event_ids"])
    assert sorted(event["attributes"]["tool_call_id"] for event in start_events) == sorted(
        expected["tool_call_ids"]
    )

    candidate = projection["call_candidates"][0]
    assert candidate["tool_call_ids"] == expected["tool_call_ids"]
    assert len({event["attributes"]["tool_call_id"] for event in start_events}) == 2


# --- nested child ownership ---


def test_nested_child_partial_records_do_not_create_main_turn_from_hook_session_id() -> None:
    case = _load_json(RECON_CASES / "cases.json")["nested_child"]
    projection, _ = build_semantics(case["input_records"], {})

    assert projection["turns"] == []
    assert {event["kind"] for event in projection["events"]} == {
        "subagent_started",
        "user_prompt",
        "prompt_transformation",
    }

    hook_only = [case["input_records"][2]]
    hook_projection, _ = build_semantics(hook_only, {})
    assert hook_projection["turns"] == []


def test_child_turn_nests_under_parent_when_full_fixture_replayed(
    cli_transcript_records: list[SourceRecord],
) -> None:
    projection, _ = build_semantics(cli_transcript_records, {})
    parent = next(
        turn
        for turn in projection["turns"]
        if turn["attributes"]["interaction_id"] == "793d3703-6f4a-4814-8877-34a7325848ce"
    )
    assert len(parent["subagents"]) == 1
    child = parent["subagents"][0]
    assert child["attributes"]["agent_id"] == CHILD_AGENT_ID
    assert child["attributes"]["parent_tool_call_id"] == "call_qx4FH5DADTeT1qVLb37HNpBk"
    assert all(turn["attributes"].get("agent_id") is None for turn in projection["turns"])


def test_prior_open_interaction_state_is_retained_when_replay_still_open() -> None:
    case = _load_json(RECON_CASES / "cases.json")["abort"]
    _, state = build_semantics(case["input_records"], {})
    prior_item = dict(state["open_interactions"]["6d2b89fd-a653-430c-b532-b0936d72eb42|main"])
    prior_item["note"] = "retained-from-incremental-caller"

    _, merged_state = build_semantics(
        case["input_records"], {"open_interactions": state["open_interactions"]}
    )
    retained = merged_state["open_interactions"]["6d2b89fd-a653-430c-b532-b0936d72eb42|main"]
    assert retained["interaction_id"] == prior_item["interaction_id"]
    assert retained["stored_turn_id"] == prior_item["stored_turn_id"]

    # Simulate a key only present in prior state (unfinished partition from earlier chunk).
    orphan_key = "orphan|main"
    prior = {
        "open_interactions": {
            orphan_key: {
                "interaction_id": "orphan-interaction",
                "agent_id": None,
                "stored_turn_id": f"copilot:turn:{SOURCE_KEY}:{NATIVE_SESSION_ID}:orphan-interaction",
                "source_ids": ["kept/source"],
                "last_event_source_id": "kept/source",
                "start_ts": "2026-09-10T17:08:00.000Z",
                "pending_tool_call_ids": [],
            }
        }
    }
    _, merged = build_semantics(case["input_records"], prior)
    assert orphan_key in merged["open_interactions"]


def test_prior_state_does_not_resurrect_completed_interaction(
    cli_transcript_records: list[SourceRecord],
) -> None:
    completed_key = "6d2b89fd-a653-430c-b532-b0936d72eb42|main"
    interaction_id = completed_key.split("|", 1)[0]
    user = next(
        record
        for record in cli_transcript_records
        if record["payload"].get("type") == "user.message"
        and record["payload"]["data"].get("interactionId") == interaction_id
    )
    prior = {
        "open_interactions": {
            completed_key: {
                "interaction_id": interaction_id,
                "agent_id": None,
                "stored_turn_id": (
                    f"copilot:turn:{SOURCE_KEY}:{NATIVE_SESSION_ID}:{interaction_id}"
                ),
                "source_ids": [user["source_id"]],
                "last_event_source_id": user["source_id"],
                "start_ts": user.get("ts"),
                "pending_tool_call_ids": [],
            }
        }
    }
    full, merged = build_semantics(cli_transcript_records, prior)
    assert completed_key not in merged["open_interactions"]
    assert any(turn["attributes"]["interaction_id"] == interaction_id for turn in full["turns"])


# --- semantic projection contract slice ---


def test_semantic_projection_fixture_normalizes_expected_events() -> None:
    case = _load_json(RECON_CASES / "semantic-projection.json")
    projection, _ = build_semantics(case["input_records"], {})
    _assert_expected_projection(projection, case["expected"])
    turn_end_id = (
        "copilot:event:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6/0080e44c-ad62-4288-b2b2-061ec2b73d80"
    )
    by_id = {event["id"]: event for event in projection["events"]}
    assert by_id[turn_end_id]["kind"] == "assistant_turn_end"


def test_auxiliary_model_events_do_not_create_call_candidates() -> None:
    case = _load_json(RECON_CASES / "cases.json")["auxiliary_title_generation"]
    assert case["observed"] is False
    projection, _ = build_semantics(case["input_records"], {})
    _assert_expected_projection(projection, case["expected"])
    assert projection["turns"] == []


def test_retry_case_is_database_only_and_emits_no_semantic_events() -> None:
    """The shared retry fixture is accounting-only; semantics has nothing to reconstruct."""
    case = _load_json(RECON_CASES / "cases.json")["retry"]
    assert case["observed"] is False
    assert all(record["source_kind"] == "database" for record in case["input_records"])
    projection, _ = build_semantics(case["input_records"], {})
    assert projection["events"] == []
    assert projection["turns"] == []
    assert projection["call_candidates"] == []
    assert projection["pending"] == []


def test_observed_versus_synthetic_cases_run_through_build_semantics() -> None:
    cases = _load_json(RECON_CASES / "cases.json")
    observed = {name for name, case in cases.items() if case["observed"]}
    synthetic = {name for name, case in cases.items() if not case["observed"]}
    assert observed == {"identical_concurrent_tools", "nested_child"}
    assert "retry" in synthetic
    assert "permission" in synthetic
    for name in ("identical_concurrent_tools", "permission", "abort"):
        projection, _ = build_semantics(cases[name]["input_records"], {})
        assert "events" in projection
        _assert_expected_projection(projection, cases[name]["expected"])
    nested = build_semantics(cases["nested_child"]["input_records"], {})[0]
    assert nested["turns"] == []


def test_completed_child_without_parent_link_is_pending_not_dropped() -> None:
    child = "child-agent-1"
    records = [
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/child-user",
            native_type="user.message",
            data={"content": "explore", "interactionId": "child-ix", "turnId": "0"},
            agent=child,
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/child-final",
            native_type="assistant.message",
            data={
                "content": "42",
                "model": "gpt-5.6-luna",
                "interactionId": "child-ix",
                "turnId": "0",
                "phase": "final_answer",
                "toolRequests": [],
            },
            agent=child,
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/child-end",
            native_type="assistant.turn_end",
            data={"turnId": "0"},
            agent=child,
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/child-stop",
            native_type="subagent.completed",
            data={"agentName": "explore"},
            agent=child,
        ),
    ]
    projection, _ = build_semantics(records, {})
    assert projection["turns"] == []
    assert any(item["kind"] == "missing_identity" for item in projection["pending"])
    assert any(item["code"] == "capability_gap" for item in projection["diagnostics"])
    assert projection["call_candidates"][0]["stored_turn_id"] is None


def test_abort_emits_interrupted_turn() -> None:
    records = [
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/abort-user",
            native_type="user.message",
            data={"content": "hello", "interactionId": "ix-abort", "turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/abort-start",
            native_type="assistant.turn_start",
            data={"turnId": "0", "interactionId": "ix-abort"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/abort-msg",
            native_type="assistant.message",
            data={
                "content": "partial",
                "model": "gpt-5.6-luna",
                "interactionId": "ix-abort",
                "turnId": "0",
                "toolRequests": [],
            },
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/abort-event",
            native_type="assistant.abort",
            data={"turnId": "0", "interactionId": "ix-abort"},
        ),
    ]
    projection, state = build_semantics(records, {})
    assert len(projection["turns"]) == 1
    assert projection["turns"][0]["status"] == "interrupted"
    assert projection["turns"][0]["output_message"] == "partial"
    assert state["open_interactions"] == {}


def test_intermediate_assistant_output_does_not_close_user_turn() -> None:
    records = [
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/mid-user",
            native_type="user.message",
            data={"content": "keep going", "interactionId": "ix-mid", "turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/mid-msg",
            native_type="assistant.message",
            data={
                "content": "working...",
                "model": "gpt-5.6-luna",
                "interactionId": "ix-mid",
                "turnId": "0",
                "toolRequests": [],
            },
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/mid-end",
            native_type="assistant.turn_end",
            data={"turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/mid-tool-msg",
            native_type="assistant.message",
            data={
                "content": "",
                "model": "gpt-5.6-luna",
                "interactionId": "ix-mid",
                "turnId": "1",
                "toolRequests": [{"toolCallId": "call_later", "name": "view", "arguments": {}}],
            },
        ),
    ]
    projection, state = build_semantics(records, {})
    assert projection["turns"] == []
    assert "ix-mid|main" in state["open_interactions"]


def test_turn_id_includes_agent_to_avoid_collision() -> None:
    shared = "shared-interaction"
    records = [
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/main-user",
            native_type="user.message",
            data={"content": "parent", "interactionId": shared, "turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/child-user",
            native_type="user.message",
            data={"content": "child", "interactionId": shared, "turnId": "0"},
            agent="agent-child",
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/main-final",
            native_type="assistant.message",
            data={
                "content": "done",
                "model": "gpt-5.6-luna",
                "interactionId": shared,
                "turnId": "0",
                "phase": "final_answer",
                "toolRequests": [],
            },
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/child-final",
            native_type="assistant.message",
            data={
                "content": "done",
                "model": "gpt-5.6-luna",
                "interactionId": shared,
                "turnId": "0",
                "phase": "final_answer",
                "toolRequests": [],
                "parentToolCallId": "call_parent",
            },
            agent="agent-child",
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/main-end",
            native_type="assistant.turn_end",
            data={"turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/child-end",
            native_type="assistant.turn_end",
            data={"turnId": "0"},
            agent="agent-child",
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/shutdown",
            native_type="session.shutdown",
            data={},
        ),
    ]
    main_id = f"copilot:turn:{SOURCE_KEY}:{NATIVE_SESSION_ID}:{shared}"
    child_id = f"{main_id}:agent-child"
    assert main_id != child_id
    projection, _ = build_semantics(records, {})
    assert projection["turns"][0]["turn_id"] == main_id
    stored = {candidate["stored_turn_id"] for candidate in projection["call_candidates"]}
    assert main_id in stored
    assert child_id not in stored


def test_permission_request_attaches_to_completed_turn() -> None:
    records = [
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/perm-user",
            native_type="user.message",
            data={"content": "read it", "interactionId": "ix-perm", "turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/perm-req",
            native_type="permission.request",
            data={
                "toolName": "view",
                "toolArgs": {"path": "/tmp/a.txt"},
                "interactionId": "ix-perm",
                "turnId": "0",
            },
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/perm-dec",
            native_type="permission.decision",
            data={"toolName": "view", "decision": "allow", "interactionId": "ix-perm"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/perm-final",
            native_type="assistant.message",
            data={
                "content": "ok",
                "model": "gpt-5.6-luna",
                "interactionId": "ix-perm",
                "turnId": "0",
                "phase": "final_answer",
                "toolRequests": [],
            },
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/perm-end",
            native_type="assistant.turn_end",
            data={"turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/perm-shutdown",
            native_type="session.shutdown",
            data={},
        ),
    ]
    projection, _ = build_semantics(records, {})
    assert len(projection["turns"]) == 1
    requests = projection["turns"][0]["permission_requests"]
    assert len(requests) == 1
    assert requests[0]["tool_name"] == "view"
    assert requests[0]["attributes"]["decision"] == "allow"


def test_semantic_retry_after_error_stays_same_interaction() -> None:
    records = [
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/retry-user",
            native_type="user.message",
            data={"content": "try again", "interactionId": "ix-retry", "turnId": "0"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/retry-err-msg",
            native_type="assistant.message",
            data={
                "content": "",
                "model": "gpt-5.6-luna",
                "interactionId": "ix-retry",
                "turnId": "0",
                "toolRequests": [],
            },
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/retry-error",
            native_type="assistant.error",
            data={"turnId": "0", "interactionId": "ix-retry", "message": "timeout"},
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/retry-final",
            native_type="assistant.message",
            data={
                "content": "42",
                "model": "gpt-5.6-luna",
                "interactionId": "ix-retry",
                "turnId": "1",
                "phase": "final_answer",
                "toolRequests": [],
            },
        ),
        _transcript_record(
            source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/retry-end",
            native_type="assistant.turn_end",
            data={"turnId": "1"},
        ),
    ]
    projection, _ = build_semantics(records, {})
    assert len(projection["turns"]) == 1
    assert projection["turns"][0]["status"] == "completed"
    assert projection["turns"][0]["output_message"] == "42"
    assert len(projection["call_candidates"]) == 2
    assert {candidate["interaction_id"] for candidate in projection["call_candidates"]} == {
        "ix-retry"
    }
