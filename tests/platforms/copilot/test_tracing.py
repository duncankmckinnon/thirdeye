"""Behavioral tests for Copilot semantic projection (build_semantics)."""

from __future__ import annotations

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
    tool_requests = [
        event for event in projection["events"] if event["kind"] == "tool_request"
    ]
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


def test_build_semantics_does_not_import_usage_or_attribution_modules() -> None:
    for module_name in ("tracing", "turns", "events"):
        source = Path(
            __import__(f"thirdeye.platforms.copilot.{module_name}", fromlist=["__file__"]).__file__
        )
        text = source.read_text(encoding="utf-8")
        for forbidden in ("usage.py", "attribution", "projection_store", "export_state"):
            assert forbidden not in text


# --- interaction grouping and turn completion ---


def test_tool_cycle_does_not_complete_user_turn_without_final_answer() -> None:
    case = _load_json(RECON_CASES / "semantic-projection.json")
    projection, state = build_semantics(case["input_records"], {})

    assert projection["turns"] == []
    pending_kinds = Counter(item["kind"] for item in projection["pending"])
    assert pending_kinds["open_interaction"] == 1
    assert pending_kinds["missing_identity"] == 1

    open_key = "6d2b89fd-a653-430c-b532-b0936d72eb42|main"
    assert open_key in state["open_interactions"]
    assert state["open_interactions"][open_key]["pending_tool_call_ids"] == [
        "call_YSSva4HCniiETlxdGGjcrHbh",
        "call_ayHplfzxjRFMTCpmTKEFhCSJ",
    ]


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
    _, _, pending, _ = build_turns([record])

    assert any(item["kind"] == "missing_identity" for item in pending)
    assert all(item["kind"] != "open_interaction" for item in pending)


def test_incomplete_tool_pair_when_execution_lacks_request() -> None:
    execution = _transcript_record(
        source_id=f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/exec-orphan",
        native_type="tool.execution_start",
        data={"toolCallId": "call_orphan", "toolName": "view", "arguments": {}},
    )
    _, _, pending, _ = build_turns([execution])

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


# --- prior state retention ---


def test_prior_open_interaction_state_is_retained_when_replay_still_open() -> None:
    case = _load_json(RECON_CASES / "cases.json")["abort"]
    _, state = build_semantics(case["input_records"], {})
    prior_item = dict(state["open_interactions"]["6d2b89fd-a653-430c-b532-b0936d72eb42|main"])
    prior_item["note"] = "retained-from-incremental-caller"

    _, merged_state = build_semantics(case["input_records"], {"open_interactions": state["open_interactions"]})
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


# --- semantic projection contract slice ---


def test_semantic_projection_fixture_normalizes_expected_events() -> None:
    case = _load_json(RECON_CASES / "semantic-projection.json")
    projection, _ = build_semantics(case["input_records"], {})
    expected_events = case["expected"]["events"]

    by_id = {event["id"]: event for event in projection["events"]}
    for expected in expected_events:
        actual = by_id[expected["id"]]
        assert actual["kind"] == expected["kind"]
        assert actual["classification"] == expected["classification"]
        assert actual["ts"] == expected["ts"]
        assert actual["source_ids"] == expected["source_ids"]
        for key, value in expected["attributes"].items():
            assert actual["attributes"].get(key) == value

    turn_end_id = (
        "copilot:event:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6/0080e44c-ad62-4288-b2b2-061ec2b73d80"
    )
    assert by_id[turn_end_id]["kind"] == "unknown"


def test_auxiliary_model_events_do_not_create_call_candidates() -> None:
    case = _load_json(RECON_CASES / "cases.json")["auxiliary_title_generation"]
    projection, _ = build_semantics(case["input_records"], {})

    assert projection["call_candidates"] == []
    assert projection["turns"] == []
    assert projection["events"][0]["classification"] == "title_generation"
