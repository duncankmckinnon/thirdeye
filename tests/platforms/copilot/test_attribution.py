"""Behavioral tests for Copilot usage attribution and projection composition."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from thirdeye.platforms.copilot.attribution import join_usage
from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.projection import build_projection
from thirdeye.platforms.copilot.tracing import build_semantics
from thirdeye.platforms.copilot.transcript import read_transcript
from thirdeye.platforms.copilot.types import (
    AccountingCandidate,
    AccountingProjection,
    CallCandidate,
    SemanticProjection,
    SourceRecord,
)
from thirdeye.platforms.copilot.usage import build_accounting
from thirdeye.usage.types import UsageRow

FIXTURES = Path(__file__).parent / "fixtures"
RECON = FIXTURES / "reconciliation-cases"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
CHILD_AGENT_ID = "bf8cb9f3-2097-4db0-a3c8-78a2653b2106"
SOURCE_KEY = "a" * 64
GENERATION = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
OBSERVED_AT = "2026-09-10T17:09:00.000Z"
INTERACTION_ONE = "6d2b89fd-a653-430c-b532-b0936d72eb42"
TURN_ONE = f"copilot:turn:{SOURCE_KEY}:{NATIVE_SESSION_ID}:{INTERACTION_ONE}"
CALL_A = f"copilot:call:{SOURCE_KEY}/{NATIVE_SESSION_ID}/a4a17e63-7ba5-422f-8ee9-b495be417328"
CALL_B = f"copilot:call:{SOURCE_KEY}/{NATIVE_SESSION_ID}/33cc6465-29e1-4a04-8bdb-00241474b4d2"
TS_CYCLE_0 = "2026-09-10T17:08:24.503Z"
TS_CYCLE_1 = "2026-09-10T17:08:25.624Z"
INTERACTION_TWO = "793d3703-6f4a-4814-8877-34a7325848ce"


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


def _usage_record(
    row: dict[str, Any],
    *,
    content_revision: str,
    generation: str = GENERATION,
    source_key: str = SOURCE_KEY,
    observed_at: str = OBSERVED_AT,
) -> SourceRecord:
    primary_key = row["id"]
    source_id = (
        f"copilot-db:{source_key}:{NATIVE_SESSION_ID}:assistant_usage_events:"
        f"{primary_key}:{content_revision}"
    )
    return {
        "source_id": source_id,
        "source_kind": "database",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": row.get("created_at"),
        "observed_at": observed_at,
        "payload": {"table": "assistant_usage_events", "row": row},
        "locator": {
            "database": "/example/.copilot/session-store.db",
            "table": "assistant_usage_events",
            "primary_key": primary_key,
            "content_revision": content_revision,
            "generation": generation,
        },
    }


def _source_key(records: list[SourceRecord]) -> str:
    for record in records:
        if record["source_kind"] == "transcript":
            return record["source_id"].split("/", 1)[0]
    pytest.fail("expected at least one transcript record")


def _substitute_source_key(value: str, source_key: str) -> str:
    return value.replace(SOURCE_KEY, source_key)


def _rewrite_source_key(value: Any, source_key: str) -> Any:
    if isinstance(value, str):
        return _substitute_source_key(value, source_key)
    if isinstance(value, list):
        return [_rewrite_source_key(item, source_key) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_source_key(item, source_key) for key, item in value.items()}
    return value


def _iter_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for turn in turns:
        found.append(turn)
        found.extend(_iter_turns(turn.get("subagents") or []))
    return found


def _accounting_calls(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for turn in _iter_turns(turns):
        found.extend(turn.get("accounting_calls") or [])
    return found


def _align_attribution(expected: dict[str, Any], source_key: str) -> dict[str, Any]:
    aligned = copy.deepcopy(expected)
    for field in ("call_id", "stored_turn_id", "logical_call_id", "usage_source_id"):
        if isinstance(aligned.get(field), str):
            aligned[field] = _substitute_source_key(aligned[field], source_key)
    aligned["evidence"] = [_substitute_source_key(item, source_key) for item in aligned["evidence"]]
    return aligned


def _six_call_records(*, source_key: str = SOURCE_KEY) -> list[SourceRecord]:
    rows = _load_json(FIXTURES / "assistant-usage-events.json")
    revisions = {
        call["row_id"]: call["usage_source_id"].rsplit(":", 1)[-1]
        for call in _load_json(RECON / "observed-six-calls.json")["calls"]
    }
    return [
        _usage_record(row, content_revision=revisions[row["id"]], source_key=source_key)
        for row in rows
    ]


def _metric_totals(usage_rows: list[UsageRow]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    for row in usage_rows:
        totals["input_tokens"] += row.input_tokens
        totals["output_tokens"] += row.output_tokens
        totals["cache_read_tokens"] += row.cache_read_input_tokens or 0
        totals["cache_write_tokens"] += row.cache_creation_input_tokens or 0
        totals["reasoning_tokens"] += row.reasoning_output_tokens or 0
    return totals


def _call_candidate(
    *,
    call_id: str,
    stored_turn_id: str | None = TURN_ONE,
    interaction_id: str = INTERACTION_ONE,
    agent_id: str | None = None,
    parent_tool_call_id: str | None = None,
    model: str = "gpt-5.6-luna",
    tool_call_ids: list[str] | None = None,
    finish_evidence: list[dict[str, Any]] | None = None,
    assistant_message_id: str | None = None,
    start_ts: str = TS_CYCLE_0,
    end_ts: str = "2026-09-10T17:08:24.593Z",
) -> CallCandidate:
    candidate: CallCandidate = {
        "call_id": call_id,
        "stored_turn_id": stored_turn_id,
        "interaction_id": interaction_id,
        "agent_id": agent_id,
        "parent_tool_call_id": parent_tool_call_id,
        "model": model,
        "source_ids": [call_id.rsplit("/", 1)[-1]],
        "source_references": [],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "tool_call_ids": tool_call_ids or [],
        "finish_evidence": finish_evidence or [],
    }
    if assistant_message_id is not None:
        candidate["assistant_message_id"] = assistant_message_id  # type: ignore[typeddict-unknown-key]
    return candidate


def _accounting_candidate(
    *,
    row_id: int,
    turn_index: int | None = 0,
    agent_id: str | None = None,
    parent_tool_call_id: str | None = None,
    model: str = "gpt-5.6-luna",
    finish_reason: str | None = "tool_calls",
    initiator: str | None = "user",
    revision: str = "sha256:68bf2ca8903d9bdfe15a9d61144ba8b9b0e352678680e4490f2259bd2f468f47",
    assistant_message_id: str | None = None,
) -> AccountingCandidate:
    logical_call_id = (
        f"copilot:usage:{SOURCE_KEY}:assistant_usage_events:"
        f"sha256%3Abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:{row_id}"
    )
    usage_source_id = (
        f"copilot-db:{SOURCE_KEY}:{NATIVE_SESSION_ID}:assistant_usage_events:{row_id}:{revision}"
    )
    supplemental: dict[str, Any] = {}
    if initiator is not None:
        supplemental["initiator"] = initiator
    candidate: AccountingCandidate = {
        "usage_source_id": usage_source_id,
        "logical_call_id": logical_call_id,
        "turn_index": turn_index,
        "agent_id": agent_id,
        "parent_tool_call_id": parent_tool_call_id,
        "model": model,
        "provider": None,
        "source_ids": [usage_source_id],
        "source_references": [],
        "revision": {
            "primary_key": row_id,
            "content_revision": revision,
            "generation": GENERATION,
        },
        "timestamp": "2026-09-10T17:08:24.498Z",
        "finish_reason": finish_reason,
        "supplemental_metrics": supplemental,
    }
    if assistant_message_id is not None:
        candidate["assistant_message_id"] = assistant_message_id  # type: ignore[typeddict-unknown-key]
    return candidate


def _semantic(*, calls: list[CallCandidate]) -> SemanticProjection:
    return {
        "events": [],
        "turns": [
            {
                "turn_id": TURN_ONE,
                "stored_turn_id": TURN_ONE,
                "status": "completed",
                "output_message": "42",
                "subagents": [],
                "attributes": {"interaction_id": INTERACTION_ONE},
            }
        ],
        "call_candidates": calls,
        "pending": [],
        "diagnostics": [],
    }


def _accounting(*, candidates: list[AccountingCandidate]) -> AccountingProjection:
    return {"usage_rows": [], "candidates": candidates, "diagnostics": []}


def _attribution_by_logical(attributions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["logical_call_id"]: item for item in attributions}


# --- observed six-call corpus ---


def test_six_call_corpus_joins_all_usage_rows_inferred(cli_transcript_records: list[SourceRecord]):
    source_key = _source_key(cli_transcript_records)
    records = cli_transcript_records + _six_call_records(source_key=source_key)
    projection, _state = build_projection(records, {})
    expected = _load_json(RECON / "observed-six-calls.json")["expected_attributions"]
    by_logical = _attribution_by_logical(projection["attributions"])

    assert len(projection["attributions"]) == 6
    assert all(item["status"] == "matched" for item in projection["attributions"])
    assert all(item["join_kind"] == "inferred" for item in projection["attributions"])
    for wanted in expected:
        aligned = _align_attribution(wanted, source_key)
        actual = by_logical[aligned["logical_call_id"]]
        assert actual["call_id"] == aligned["call_id"]
        assert actual["stored_turn_id"] == aligned["stored_turn_id"]
        assert actual["agent_id"] == aligned["agent_id"]
        assert actual["evidence"][0] == "join_kind:inferred"
        assert actual["evidence"][-1] == aligned["evidence"][-1]
        assert f"turn_index:{aligned['evidence'][2].split(':', 1)[1]}" in actual["evidence"]


def test_six_call_usage_totals_unchanged_by_attribution(cli_transcript_records: list[SourceRecord]):
    source_key = _source_key(cli_transcript_records)
    records = cli_transcript_records + _six_call_records(source_key=source_key)
    projection, _state = build_projection(records, {})
    assert len(projection["usage_rows"]) == 6
    assert _metric_totals(projection["usage_rows"])["input_tokens"] == 35396


def test_child_usage_maps_to_main_stored_turn(cli_transcript_records: list[SourceRecord]):
    source_key = _source_key(cli_transcript_records)
    records = cli_transcript_records + _six_call_records(source_key=source_key)
    projection, _state = build_projection(records, {})
    child = next(item for item in projection["attributions"] if item["agent_id"] == CHILD_AGENT_ID)
    main_turn_two = (
        f"copilot:turn:{source_key}:{NATIVE_SESSION_ID}:793d3703-6f4a-4814-8877-34a7325848ce"
    )
    assert child["stored_turn_id"] == main_turn_two
    assert child["status"] == "matched"


# --- contract fixtures via join_usage ---


def test_attribution_fixture_matched_inferred():
    examples = _load_json(RECON / "attributions.json")
    semantic, _ = build_semantics(
        _load_json(RECON / "semantic-projection.json")["input_records"], {}
    )
    accounting, _ = build_accounting(
        [
            _usage_record(
                _load_json(FIXTURES / "assistant-usage-events.json")[0],
                content_revision="sha256:68bf2ca8903d9bdfe15a9d61144ba8b9b0e352678680e4490f2259bd2f468f47",
            )
        ],
        {},
    )
    attributions = join_usage(semantic, accounting)
    assert attributions == [examples["matched_inferred"]]


def test_attribution_fixture_pending_without_matching_call():
    examples = _load_json(RECON / "attributions.json")
    case = _load_json(RECON / "cases.json")["late_row"]
    projection, _state = build_projection(case["input_records"], {})
    assert projection["attributions"] == [examples["pending"]]


# --- direct / native joins ---


def test_direct_join_wins_over_inferred_evidence():
    shared_message_id = "native-msg-abc"
    semantic = _semantic(
        calls=[
            _call_candidate(
                call_id=CALL_A,
                tool_call_ids=["tool-a"],
                assistant_message_id=shared_message_id,
            ),
            _call_candidate(
                call_id=CALL_B,
                tool_call_ids=[],
                finish_evidence=[{"source_id": "finish"}],
                start_ts=TS_CYCLE_1,
                end_ts=TS_CYCLE_1,
            ),
        ]
    )
    accounting = _accounting(
        candidates=[_accounting_candidate(row_id=13, assistant_message_id=shared_message_id)]
    )
    attributions = join_usage(semantic, accounting)
    assert attributions[0]["status"] == "matched"
    assert attributions[0]["join_kind"] == "direct"
    assert attributions[0]["call_id"] == CALL_A
    assert "join_kind:direct" in attributions[0]["evidence"]


def test_competing_direct_ids_are_conflicting():
    shared_message_id = "shared-native-id"
    semantic = _semantic(
        calls=[
            _call_candidate(call_id=CALL_A, assistant_message_id=shared_message_id),
            _call_candidate(call_id=CALL_B, assistant_message_id=shared_message_id),
        ]
    )
    accounting = _accounting(
        candidates=[_accounting_candidate(row_id=13, assistant_message_id=shared_message_id)]
    )
    attributions = join_usage(semantic, accounting)
    assert attributions[0]["status"] == "conflicting"
    assert attributions[0]["call_id"] is None
    assert f"call_id:{CALL_A}" in attributions[0]["evidence"]
    assert f"call_id:{CALL_B}" in attributions[0]["evidence"]
    assert not any(item.startswith("join_conflict:") for item in attributions[0]["evidence"])


# --- inferred / ambiguous / pending ---


def test_inferred_join_requires_unique_consistent_evidence():
    semantic = _semantic(
        calls=[
            _call_candidate(
                call_id=CALL_A,
                tool_call_ids=["tool-a"],
            ),
            _call_candidate(
                call_id=CALL_B,
                tool_call_ids=[],
                finish_evidence=[{"source_id": "finish-b"}],
                start_ts=TS_CYCLE_1,
                end_ts=TS_CYCLE_1,
            ),
        ]
    )
    accounting = _accounting(
        candidates=[
            _accounting_candidate(row_id=13, finish_reason="tool_calls", initiator="user"),
            _accounting_candidate(
                row_id=14,
                finish_reason="stop",
                initiator="agent",
            ),
        ]
    )
    attributions = join_usage(semantic, accounting)
    by_row = {item["logical_call_id"].rsplit(":", 1)[-1]: item for item in attributions}
    assert by_row["13"]["status"] == "matched"
    assert by_row["13"]["join_kind"] == "inferred"
    assert by_row["13"]["call_id"] == CALL_A
    assert by_row["14"]["status"] == "matched"
    assert by_row["14"]["call_id"] == CALL_B


def test_ambiguous_when_multiple_calls_fit_same_evidence():
    semantic = _semantic(
        calls=[
            _call_candidate(call_id=CALL_A, tool_call_ids=["tool-a"]),
            _call_candidate(call_id=CALL_B, tool_call_ids=["tool-b"], start_ts=TS_CYCLE_1),
        ]
    )
    accounting = _accounting(
        candidates=[
            _accounting_candidate(row_id=13, finish_reason=None, initiator=None),
        ]
    )
    attributions = join_usage(semantic, accounting)
    assert attributions[0]["status"] == "ambiguous"
    assert attributions[0]["call_id"] is None
    assert f"call_id:{CALL_A}" in attributions[0]["evidence"]
    assert f"call_id:{CALL_B}" in attributions[0]["evidence"]


def test_pending_when_turn_index_has_no_main_interaction():
    semantic = _semantic(calls=[_call_candidate(call_id=CALL_A, tool_call_ids=["tool-a"])])
    accounting = _accounting(candidates=[_accounting_candidate(row_id=13, turn_index=99)])
    attributions = join_usage(semantic, accounting)
    assert attributions[0]["status"] == "pending"
    assert attributions[0]["stored_turn_id"] is None
    assert attributions[0]["call_id"] is None
    assert "turn_index:99" in attributions[0]["evidence"]
    assert "delayed_row:true" not in attributions[0]["evidence"]


def test_late_row_case_stays_pending_until_semantics_catch_up():
    case = _load_json(RECON / "cases.json")["late_row"]
    projection, _state = build_projection(case["input_records"], {})
    attribution = projection["attributions"][0]
    assert attribution["status"] == "pending"
    assert attribution["call_id"] is None
    assert attribution["logical_call_id"] == case["expected"]["attributions"][0]["logical_call_id"]
    assert "delayed_row:true" not in attribution["evidence"]
    assert "turn_index:0" in attribution["evidence"]
    assert len(projection["usage_rows"]) == 1


def test_delayed_row_resolves_after_transcript_replay():
    case = _load_json(RECON / "cases.json")["late_row"]
    partial = case["input_records"]
    final_answer = _load_json(RECON / "cases.json")["ambiguous"]["input_records"][1]
    turn_end = {
        "source_id": (f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/4a386a37-ca7e-4ebc-a746-cdda20f2a4bb"),
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:25.626Z",
        "observed_at": OBSERVED_AT,
        "payload": {
            "type": "assistant.turn_end",
            "data": {"turnId": "1"},
            "id": "4a386a37-ca7e-4ebc-a746-cdda20f2a4bb",
            "timestamp": "2026-09-10T17:08:25.626Z",
            "parentId": "33cc6465-29e1-4a04-8bdb-00241474b4d2",
            "schema_version": 1,
        },
        "locator": {
            "file": "events.jsonl",
            "native_event_id": "4a386a37-ca7e-4ebc-a746-cdda20f2a4bb",
        },
    }
    full = partial + [final_answer, turn_end]
    pending_projection, _ = build_projection(partial, {})
    resolved_projection, _ = build_projection(full, {})
    assert pending_projection["attributions"][0]["status"] == "pending"
    assert resolved_projection["attributions"][0]["status"] == "matched"
    assert resolved_projection["attributions"][0]["call_id"].endswith(
        "33cc6465-29e1-4a04-8bdb-00241474b4d2"
    )


# --- conflicting joins ---


def test_two_usage_rows_claiming_one_call_become_conflicting():
    semantic = _semantic(
        calls=[
            _call_candidate(
                call_id=CALL_A,
                tool_call_ids=["tool-a"],
            ),
            _call_candidate(
                call_id=CALL_B,
                tool_call_ids=[],
                finish_evidence=[{"source_id": "finish-b"}],
                start_ts=TS_CYCLE_1,
                end_ts=TS_CYCLE_1,
            ),
        ]
    )
    accounting = _accounting(
        candidates=[
            _accounting_candidate(row_id=13, finish_reason="tool_calls", initiator="user"),
            _accounting_candidate(row_id=14, finish_reason="tool_calls", initiator="user"),
        ]
    )
    attributions = join_usage(semantic, accounting)
    assert all(item["status"] == "conflicting" for item in attributions)
    assert all(item["call_id"] is None for item in attributions)
    assert all(f"call_id:{CALL_A}" in item["evidence"] for item in attributions)
    assert all(
        not any(token.startswith("join_conflict:") for token in item["evidence"])
        for item in attributions
    )


# --- build_projection composition ---


def test_build_projection_attaches_accounting_calls_without_mutating_semantics(
    cli_transcript_records: list[SourceRecord],
):
    source_key = _source_key(cli_transcript_records)
    records = cli_transcript_records + _six_call_records(source_key=source_key)
    semantic_before, _ = build_semantics(records, {})
    projection, _state = build_projection(records, {})
    semantic_after, _ = build_semantics(records, {})

    assert semantic_before["turns"] == semantic_after["turns"]
    assert all(not turn.get("accounting_calls") for turn in semantic_before["turns"])

    attached = _accounting_calls(projection["turns"])
    assert len(attached) == 6
    matched = [item for item in projection["attributions"] if item["status"] == "matched"]
    by_turn = {turn["turn_id"]: turn for turn in _iter_turns(projection["turns"])}
    for attribution in matched:
        owner = next(
            turn
            for turn in _iter_turns(projection["turns"])
            if any(
                item.get("call_id") == attribution["call_id"]
                for item in turn.get("accounting_calls") or []
            )
        )
        if attribution["agent_id"] is None:
            assert owner["turn_id"] == attribution["stored_turn_id"]
        else:
            assert owner["turn_id"] != attribution["stored_turn_id"]
            assert owner in (by_turn[attribution["stored_turn_id"]].get("subagents") or [])
        assert attribution["call_id"] in [
            item["call_id"] for item in owner.get("accounting_calls") or []
        ]


def test_matched_child_usage_attaches_to_nested_turn(
    cli_transcript_records: list[SourceRecord],
):
    source_key = _source_key(cli_transcript_records)
    records = cli_transcript_records + _six_call_records(source_key=source_key)
    projection, _ = build_projection(records, {})
    child = next(item for item in projection["attributions"] if item["agent_id"] == CHILD_AGENT_ID)
    main = next(turn for turn in projection["turns"] if turn["turn_id"] == child["stored_turn_id"])
    nested = main["subagents"][0]
    nested_ids = [
        item["call_id"]
        for item in nested.get("accounting_calls") or []
        if item["attribution_status"] == "matched"
    ]
    main_ids = [
        item["call_id"] for item in main.get("accounting_calls") or [] if item.get("call_id")
    ]
    assert child["call_id"] in nested_ids
    assert child["call_id"] not in main_ids
    assert any(llm.get("call_id") == child["call_id"] for llm in nested.get("llm_calls") or [])


def test_retry_reordering_same_interaction_keeps_call_assignments():
    first = _call_candidate(call_id=CALL_A, tool_call_ids=["tool-a"], start_ts=TS_CYCLE_0)
    second = _call_candidate(
        call_id=CALL_B,
        tool_call_ids=[],
        finish_evidence=[{"source_id": "finish-b"}],
        start_ts=TS_CYCLE_1,
        end_ts=TS_CYCLE_1,
    )
    usage_user = _accounting_candidate(row_id=13, finish_reason="tool_calls", initiator="user")
    usage_agent = _accounting_candidate(row_id=14, finish_reason="stop", initiator="agent")
    forward = join_usage(
        _semantic(calls=[first, second]),
        _accounting(candidates=[usage_user, usage_agent]),
    )
    reversed_calls = join_usage(
        _semantic(calls=[second, first]),
        _accounting(candidates=[usage_agent, usage_user]),
    )
    by_forward = {item["logical_call_id"].rsplit(":", 1)[-1]: item for item in forward}
    by_reversed = {item["logical_call_id"].rsplit(":", 1)[-1]: item for item in reversed_calls}
    assert by_forward["13"]["call_id"] == by_reversed["13"]["call_id"] == CALL_A
    assert by_forward["14"]["call_id"] == by_reversed["14"]["call_id"] == CALL_B
    assert "order:0" in by_forward["13"]["evidence"]
    assert "order:1" in by_forward["14"]["evidence"]
    assert "order:0" in by_reversed["13"]["evidence"]
    assert "order:1" in by_reversed["14"]["evidence"]


def test_cyclic_child_does_not_consume_turn_index():
    child_a = _call_candidate(
        call_id=f"copilot:call:{SOURCE_KEY}/{NATIVE_SESSION_ID}/cycle-a",
        stored_turn_id=f"{TURN_ONE}:agent-a",
        interaction_id="child-cycle",
        agent_id="agent-a",
        parent_tool_call_id="tool-b",
        tool_call_ids=["tool-a"],
    )
    child_b = _call_candidate(
        call_id=f"copilot:call:{SOURCE_KEY}/{NATIVE_SESSION_ID}/cycle-b",
        stored_turn_id=f"{TURN_ONE}:agent-b",
        interaction_id="child-cycle",
        agent_id="agent-b",
        parent_tool_call_id="tool-a",
        tool_call_ids=["tool-b"],
        start_ts=TS_CYCLE_1,
    )
    main = _call_candidate(call_id=CALL_A, tool_call_ids=["tool-main"])
    attributions = join_usage(
        _semantic(calls=[child_a, child_b, main]),
        _accounting(candidates=[_accounting_candidate(row_id=13, turn_index=0)]),
    )
    assert attributions[0]["stored_turn_id"] == TURN_ONE
    assert attributions[0]["call_id"] == CALL_A
    assert attributions[0]["status"] == "matched"


def test_non_unique_parent_tool_does_not_consume_turn_index():
    child = _call_candidate(
        call_id=f"copilot:call:{SOURCE_KEY}/{NATIVE_SESSION_ID}/orphan-child",
        stored_turn_id=f"{TURN_ONE}:agent-a",
        interaction_id="child-orphan",
        agent_id="agent-a",
        parent_tool_call_id="shared-tool",
        tool_call_ids=["child-tool"],
    )
    main = _call_candidate(call_id=CALL_A, tool_call_ids=["shared-tool"])
    other = _call_candidate(
        call_id=CALL_B,
        stored_turn_id=f"copilot:turn:{SOURCE_KEY}:{NATIVE_SESSION_ID}:{INTERACTION_TWO}",
        interaction_id=INTERACTION_TWO,
        tool_call_ids=["shared-tool"],
        start_ts=TS_CYCLE_1,
    )
    attributions = join_usage(
        _semantic(calls=[child, main, other]),
        _accounting(candidates=[_accounting_candidate(row_id=13, turn_index=0)]),
    )
    assert attributions[0]["stored_turn_id"] == TURN_ONE
    assert attributions[0]["call_id"] == CALL_A
    assert attributions[0]["status"] == "matched"


def test_partial_turn_incremental_state_matches_full_archive():
    cases = _load_json(RECON / "cases.json")
    partial = cases["partial_turn"]["input_records"]
    later = cases["abort"]["input_records"]
    full = partial + later
    first, state = build_projection(partial, {})
    assert (
        "6d2b89fd-a653-430c-b532-b0936d72eb42|main" in state["semantic_state"]["open_interactions"]
    )
    incremental, inc_state = build_projection(later, state)
    from_flat, flat_state = build_projection(later, state["semantic_state"])
    complete, full_state = build_projection(full, {})
    assert incremental["turns"] == complete["turns"] == from_flat["turns"]
    assert (
        inc_state["semantic_state"]["open_interactions"].keys()
        == full_state["semantic_state"]["open_interactions"].keys()
        == flat_state["semantic_state"]["open_interactions"].keys()
    )
    assert (
        "6d2b89fd-a653-430c-b532-b0936d72eb42|main"
        in inc_state["semantic_state"]["open_interactions"]
    )
    assert first["turns"] == []


def test_direct_match_does_not_emit_join_capability_gap(
    cli_transcript_records: list[SourceRecord],
):
    source_key = _source_key(cli_transcript_records)
    records = cli_transcript_records + _six_call_records(source_key=source_key)
    projection, _ = build_projection(records, {})
    join_gaps = [
        item
        for item in projection["diagnostics"]
        if item["code"] == "capability_gap" and item.get("details", {}).get("logical_call_id")
    ]
    assert join_gaps == []


def test_accounting_rows_persist_when_attribution_is_ambiguous():
    user = {
        "source_id": f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/amb-user",
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:22.203Z",
        "observed_at": OBSERVED_AT,
        "payload": {
            "type": "user.message",
            "data": {
                "content": "two similar cycles",
                "interactionId": INTERACTION_ONE,
                "turnId": "0",
            },
            "id": "amb-user",
            "timestamp": "2026-09-10T17:08:22.203Z",
            "schema_version": 1,
        },
        "locator": {"file": "events.jsonl", "native_event_id": "amb-user"},
    }
    first_msg = {
        "source_id": f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/a4a17e63-7ba5-422f-8ee9-b495be417328",
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": TS_CYCLE_0,
        "observed_at": OBSERVED_AT,
        "payload": {
            "type": "assistant.message",
            "data": {
                "content": "",
                "model": "gpt-5.6-luna",
                "interactionId": INTERACTION_ONE,
                "turnId": "0",
                "toolRequests": [{"toolCallId": "tool-a", "name": "view", "arguments": {}}],
            },
            "id": "a4a17e63-7ba5-422f-8ee9-b495be417328",
            "timestamp": TS_CYCLE_0,
            "schema_version": 1,
        },
        "locator": {
            "file": "events.jsonl",
            "native_event_id": "a4a17e63-7ba5-422f-8ee9-b495be417328",
        },
    }
    first_end = {
        "source_id": f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/amb-end-0",
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:24.593Z",
        "observed_at": OBSERVED_AT,
        "payload": {
            "type": "assistant.turn_end",
            "data": {"turnId": "0"},
            "id": "amb-end-0",
            "timestamp": "2026-09-10T17:08:24.593Z",
            "schema_version": 1,
        },
        "locator": {"file": "events.jsonl", "native_event_id": "amb-end-0"},
    }
    second_msg = {
        "source_id": f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/33cc6465-29e1-4a04-8bdb-00241474b4d2",
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": TS_CYCLE_1,
        "observed_at": OBSERVED_AT,
        "payload": {
            "type": "assistant.message",
            "data": {
                "content": "",
                "model": "gpt-5.6-luna",
                "interactionId": INTERACTION_ONE,
                "turnId": "1",
                "toolRequests": [{"toolCallId": "tool-b", "name": "view", "arguments": {}}],
            },
            "id": "33cc6465-29e1-4a04-8bdb-00241474b4d2",
            "timestamp": TS_CYCLE_1,
            "schema_version": 1,
        },
        "locator": {
            "file": "events.jsonl",
            "native_event_id": "33cc6465-29e1-4a04-8bdb-00241474b4d2",
        },
    }
    second_end = {
        "source_id": f"{SOURCE_KEY}/{NATIVE_SESSION_ID}/amb-end-1",
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:25.700Z",
        "observed_at": OBSERVED_AT,
        "payload": {
            "type": "assistant.turn_end",
            "data": {"turnId": "1"},
            "id": "amb-end-1",
            "timestamp": "2026-09-10T17:08:25.700Z",
            "schema_version": 1,
        },
        "locator": {"file": "events.jsonl", "native_event_id": "amb-end-1"},
    }
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row["finish_reason"] = None
    row["initiator"] = None
    usage = _usage_record(
        row,
        content_revision="sha256:68bf2ca8903d9bdfe15a9d61144ba8b9b0e352678680e4490f2259bd2f468f47",
    )
    ambiguous, _ = build_projection([user, first_msg, first_end, second_msg, second_end, usage], {})
    matched, _ = build_projection([user, first_msg, first_end, usage], {})
    assert ambiguous["attributions"][0]["status"] == "ambiguous"
    assert matched["attributions"][0]["status"] == "matched"
    assert len(ambiguous["usage_rows"]) == len(matched["usage_rows"]) == 1
    assert _metric_totals(ambiguous["usage_rows"]) == _metric_totals(matched["usage_rows"])
    assert ambiguous["usage_rows"][0].input_tokens == 6452


def test_late_row_projection_keeps_usage_rows_while_attribution_pending():
    case = _load_json(RECON / "cases.json")["late_row"]
    projection, _ = build_projection(case["input_records"], {})
    assert projection["attributions"][0]["status"] == "pending"
    assert len(projection["usage_rows"]) == 1
    assert projection["usage_rows"][0].input_tokens == 6587


def test_unmatched_usage_with_known_turn_stays_on_main_without_call_id():
    abort = _load_json(RECON / "cases.json")["abort"]["input_records"]
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row["model"] = "not-the-transcript-model"
    usage = _usage_record(
        row,
        content_revision="sha256:68bf2ca8903d9bdfe15a9d61144ba8b9b0e352678680e4490f2259bd2f468f47",
    )
    projection, _ = build_projection(abort + [usage], {})
    assert projection["attributions"][0]["status"] == "pending"
    assert projection["attributions"][0]["stored_turn_id"] is not None
    assert projection["attributions"][0]["call_id"] is None
    attached = _accounting_calls(projection["turns"])
    assert len(attached) == 1
    assert attached[0]["call_id"] is None
    assert attached[0]["attribution_status"] == "pending"
    owner = next(turn for turn in projection["turns"] if turn.get("accounting_calls"))
    assert owner["turn_id"] == projection["attributions"][0]["stored_turn_id"]


def test_pending_usage_without_stored_turn_skips_accounting_calls():
    records = _six_call_records()[:1]
    projection, _ = build_projection(records, {})
    assert projection["attributions"][0]["stored_turn_id"] is None
    assert projection["attributions"][0]["status"] == "pending"
    assert len(projection["usage_rows"]) == 1
    assert projection["turns"] == []
    assert _accounting_calls(projection["turns"]) == []
    pending = [item for item in projection["pending"] if item["kind"] == "unmatched_usage"]
    assert len(pending) == 1


def test_matched_inferred_usage_emits_diagnostic(cli_transcript_records: list[SourceRecord]):
    source_key = _source_key(cli_transcript_records)
    records = cli_transcript_records + _six_call_records(source_key=source_key)
    projection, _ = build_projection(records, {})
    inferred = [item for item in projection["diagnostics"] if item["code"] == "inferred_join"]
    assert len(inferred) == 6


def test_unmatched_usage_emits_pending_item():
    case = _load_json(RECON / "cases.json")["late_row"]
    projection, _ = build_projection(case["input_records"], {})
    pending = [item for item in projection["pending"] if item["kind"] == "unmatched_usage"]
    assert len(pending) == 1
    assert pending[0]["reason"] == "usage attribution is pending"
    assert pending[0]["id"].startswith("pending:usage:")


def test_source_correction_replaces_attribution_target_after_replay(
    cli_transcript_records: list[SourceRecord],
):
    source_key = _source_key(cli_transcript_records)
    original = cli_transcript_records + _six_call_records(source_key=source_key)
    first, state = build_projection(original, {})
    original_row = next(row for row in first["usage_rows"] if row.call_id.endswith(":13"))
    original_join = next(
        item for item in first["attributions"] if item["logical_call_id"] == original_row.call_id
    )
    corrected = _rewrite_source_key(
        copy.deepcopy(_load_json(RECON / "cases.json")["revision"]["input_records"][1]),
        source_key,
    )
    second, _ = build_projection(original + [corrected], state)
    assert len(second["usage_rows"]) == 6
    replaced = next(row for row in second["usage_rows"] if row.call_id == original_row.call_id)
    joined = next(
        item for item in second["attributions"] if item["logical_call_id"] == original_row.call_id
    )
    assert replaced.output_tokens == 999
    assert original_row.output_tokens != 999
    assert joined["call_id"] == original_join["call_id"]
    assert joined["usage_source_id"] == corrected["source_id"]
    assert joined["logical_call_id"] == original_join["logical_call_id"]
    assert _metric_totals(second["usage_rows"])["output_tokens"] == (
        _metric_totals(first["usage_rows"])["output_tokens"] - original_row.output_tokens + 999
    )


@pytest.fixture
def cli_transcript_records(tmp_path: Path) -> list[SourceRecord]:
    return _drain_cli_transcript(tmp_path / "copilot-home")
