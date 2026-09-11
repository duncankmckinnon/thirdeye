"""Behavioral tests for Copilot database usage normalization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.types import SourceRecord
from thirdeye.platforms.copilot.usage import build_accounting
from thirdeye.usage.types import UsageRow

FIXTURES = Path(__file__).parent / "fixtures"
RECONCILIATION = FIXTURES / "reconciliation-cases"

NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
SOURCE_KEY = "a" * 64
GENERATION = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
CHILD_AGENT_ID = "bf8cb9f3-2097-4db0-a3c8-78a2653b2106"
OBSERVED_AT = "2026-09-10T17:09:00.000Z"

SHUTDOWN_TOTALS = {
    "input_tokens": 35396,
    "output_tokens": 328,
    "reasoning_tokens": 55,
    "cache_read_tokens": 23948,
    "cache_write_tokens": 11430,
    "total_nano_aiu": 373366000,
}

MAIN_AGENT_TOTALS = {
    "input_tokens": 26416,
    "output_tokens": 229,
    "cache_read_tokens": 19522,
    "cache_write_tokens": 6882,
    "reasoning_tokens": 39,
}

CHILD_AGENT_TOTALS = {
    "input_tokens": 8980,
    "output_tokens": 99,
    "cache_read_tokens": 4426,
    "cache_write_tokens": 4548,
    "reasoning_tokens": 16,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic_codes(projection: dict[str, Any]) -> set[str]:
    return {item["code"] for item in projection["diagnostics"]}


def _usage_records(records: list[SourceRecord]) -> list[SourceRecord]:
    return [
        record
        for record in records
        if record.get("source_kind") == "database"
        and isinstance(record.get("payload"), dict)
        and record["payload"].get("table") == "assistant_usage_events"
    ]


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


def _shutdown_record(*, source_id: str = "transcript/shutdown-1") -> SourceRecord:
    usage = _load_json(FIXTURES / "usage.json")
    return {
        "source_id": source_id,
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:50.000Z",
        "observed_at": OBSERVED_AT,
        "payload": {"type": "session.shutdown", "data": usage},
        "locator": {
            "file": "events.jsonl",
            "file_generation": "1-abc",
            "byte_offset": 0,
            "byte_length": 100,
            "native_event_id": "shutdown-1",
        },
    }


def _checkpoint_record(*, source_id: str = "transcript/checkpoint-1") -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": "transcript",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:25.700Z",
        "observed_at": OBSERVED_AT,
        "payload": {
            "type": "session.usage_checkpoint",
            "data": {
                "inputTokens": 26416,
                "outputTokens": 229,
                "cacheReadTokens": 19522,
                "cacheWriteTokens": 6882,
                "reasoningTokens": 39,
                "totalNanoAiu": 238814000,
            },
        },
        "locator": {
            "file": "events.jsonl",
            "file_generation": "1-abc",
            "byte_offset": 0,
            "byte_length": 100,
            "native_event_id": "checkpoint-1",
        },
    }


def _six_call_records() -> list[SourceRecord]:
    rows = _load_json(FIXTURES / "assistant-usage-events.json")
    revisions = {
        call["row_id"]: call["usage_source_id"].rsplit(":", 1)[-1]
        for call in _load_json(RECONCILIATION / "observed-six-calls.json")["calls"]
    }
    return [_usage_record(row, content_revision=revisions[row["id"]]) for row in rows]


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


def _nano_aiu_total(candidates: list[dict[str, Any]]) -> int:
    return sum(
        candidate["supplemental_metrics"]["total_nano_aiu"]
        for candidate in candidates
        if "total_nano_aiu" in candidate["supplemental_metrics"]
    )


# --- contract fixture ---


def test_accounting_projection_fixture_matches_build_accounting():
    case = _load_json(RECONCILIATION / "accounting-projection.json")
    projection, state = build_accounting(case["input_records"], {})
    expected = case["expected"]

    assert [row.to_dict() for row in projection["usage_rows"]] == expected["usage_rows"]
    assert projection["candidates"] == expected["candidates"]
    assert projection["diagnostics"] == expected["diagnostics"]
    assert set(state["logical_calls"]) == {expected["candidates"][0]["logical_call_id"]}


# --- observed six-call corpus ---


def test_six_calls_are_accounted_once_from_cli_rows():
    records = _six_call_records()
    projection, _state = build_accounting(records, {})

    assert len(projection["usage_rows"]) == 6
    assert len(projection["candidates"]) == 6
    assert len({row.call_id for row in projection["usage_rows"]}) == 6
    assert len({candidate["logical_call_id"] for candidate in projection["candidates"]}) == 6


def test_six_call_totals_match_observed_shutdown_fixture():
    records = _six_call_records()
    projection, _state = build_accounting(records, {})
    totals = _metric_totals(projection["usage_rows"])

    assert totals["input_tokens"] == SHUTDOWN_TOTALS["input_tokens"]
    assert totals["output_tokens"] == SHUTDOWN_TOTALS["output_tokens"]
    assert totals["cache_read_tokens"] == SHUTDOWN_TOTALS["cache_read_tokens"]
    assert totals["cache_write_tokens"] == SHUTDOWN_TOTALS["cache_write_tokens"]
    assert totals["reasoning_tokens"] == SHUTDOWN_TOTALS["reasoning_tokens"]
    assert _nano_aiu_total(projection["candidates"]) == SHUTDOWN_TOTALS["total_nano_aiu"]


def test_per_agent_totals_match_shutdown_agent_metrics():
    records = _six_call_records()
    projection, _state = build_accounting(records, {})
    by_call = {candidate["logical_call_id"]: candidate for candidate in projection["candidates"]}
    grouped: dict[str | None, list[UsageRow]] = {}
    for usage_row in projection["usage_rows"]:
        grouped.setdefault(by_call[usage_row.call_id]["agent_id"], []).append(usage_row)

    assert _metric_totals(grouped[None]) == MAIN_AGENT_TOTALS
    assert _metric_totals(grouped[CHILD_AGENT_ID]) == CHILD_AGENT_TOTALS


def test_candidates_preserve_agent_parent_and_turn_index_evidence():
    records = _six_call_records()
    projection, _state = build_accounting(records, {})
    by_row_id = {
        candidate["revision"]["primary_key"]: candidate for candidate in projection["candidates"]
    }

    main_turn_zero = [by_row_id[str(row_id)] for row_id in (13, 14)]
    assert all(item["turn_index"] == 0 for item in main_turn_zero)
    assert all(item["agent_id"] is None for item in main_turn_zero)

    child_calls = [by_row_id[str(row_id)] for row_id in (16, 17)]
    assert all(item["agent_id"] == CHILD_AGENT_ID for item in child_calls)
    assert all(
        item["parent_tool_call_id"] == "call_qx4FH5DADTeT1qVLb37HNpBk" for item in child_calls
    )


def test_shutdown_validation_passes_when_totals_match():
    records = _six_call_records() + [_shutdown_record()]
    projection, _state = build_accounting(records, {})
    assert "shutdown_total_mismatch" not in _diagnostic_codes(projection)


def test_database_reader_records_normalize_to_same_six_calls(tmp_path: Path):
    from tests.platforms.copilot.test_database import _collect_all, _write_database

    usage_rows = _load_json(FIXTURES / "assistant-usage-events.json")
    home = tmp_path / "copilot"
    _write_database(
        home,
        session_id=NATIVE_SESSION_ID,
        cwd="/tmp/probe",
        usage_rows=usage_rows,
    )
    db_records = _usage_records(_collect_all(resolve_sources(home), NATIVE_SESSION_ID))
    synthetic_records = _six_call_records()

    sample = db_records[0]
    source_key = sample["source_id"].split(":")[1]
    generation = sample["locator"]["generation"]
    aligned = [
        _usage_record(
            record["payload"]["row"],
            content_revision=record["locator"]["content_revision"],
            generation=generation,
            source_key=source_key,
        )
        for record in synthetic_records
    ]

    db_projection, _ = build_accounting(db_records, {})
    synthetic_projection, _ = build_accounting(aligned, {})

    assert [row.call_id for row in db_projection["usage_rows"]] == [
        row.call_id for row in synthetic_projection["usage_rows"]
    ]
    assert [candidate["logical_call_id"] for candidate in db_projection["candidates"]] == [
        candidate["logical_call_id"] for candidate in synthetic_projection["candidates"]
    ]
    assert _metric_totals(db_projection["usage_rows"]) == _metric_totals(
        synthetic_projection["usage_rows"]
    )


# --- provider handling ---


def test_unknown_provider_maps_to_unknown_in_usage_row():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    record = _usage_record(
        row,
        content_revision="sha256:68bf2ca8903d9bdfe15a9d61144ba8b9b0e352678680e4490f2259bd2f468f47",
    )
    projection, _state = build_accounting([record], {})

    assert projection["candidates"][0]["provider"] is None
    assert projection["usage_rows"][0].provider_name == "unknown"
    unknown = [item for item in projection["diagnostics"] if item["code"] == "unknown_provider"]
    assert len(unknown) == 1
    assert record["source_id"] in unknown[0]["source_ids"]


def test_explicit_provider_is_preserved():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row["provider"] = "openai"
    record = _usage_record(row, content_revision="sha256:provider-rev")
    projection, _state = build_accounting([record], {})

    assert projection["candidates"][0]["provider"] == "openai"
    assert projection["usage_rows"][0].provider_name == "openai"
    assert "unknown_provider" not in _diagnostic_codes(projection)


def test_provider_name_column_is_accepted():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row["provider_name"] = "anthropic"
    record = _usage_record(row, content_revision="sha256:provider-name-rev")
    projection, _state = build_accounting([record], {})

    assert projection["candidates"][0]["provider"] == "anthropic"
    assert projection["usage_rows"][0].provider_name == "anthropic"


# --- absent vs zero ---


def test_absent_cache_and_reasoning_tokens_stay_none_in_usage_row():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row.pop("cache_read_tokens", None)
    row.pop("cache_write_tokens", None)
    row.pop("reasoning_tokens", None)
    record = _usage_record(row, content_revision="sha256:absent-cache-rev")
    projection, _state = build_accounting([record], {})

    usage_row = projection["usage_rows"][0]
    assert usage_row.cache_read_input_tokens is None
    assert usage_row.cache_creation_input_tokens is None
    assert usage_row.reasoning_output_tokens is None
    serialized = usage_row.to_dict()
    assert "gen_ai.usage.cache_read.input_tokens" not in serialized
    assert "gen_ai.usage.cache_creation.input_tokens" not in serialized
    assert "gen_ai.usage.reasoning.output_tokens" not in serialized


# --- missing / partial rows ---


def test_missing_required_fields_keep_candidate_without_usage_row():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row.pop("output_tokens")
    record = _usage_record(row, content_revision="sha256:missing-output-rev")
    projection, _state = build_accounting([record], {})

    assert len(projection["candidates"]) == 1
    assert projection["usage_rows"] == []
    missing = [item for item in projection["diagnostics"] if item["code"] == "missing_usage_fields"]
    assert missing
    assert "output_tokens" in missing[0]["details"]["missing_fields"]
    supplemental = projection["candidates"][0]["supplemental_metrics"]
    assert "output_tokens" not in supplemental
    assert supplemental["input_tokens"] == row["input_tokens"]


def test_missing_timestamp_diagnostic_and_no_usage_row():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row.pop("created_at")
    record = _usage_record(row, content_revision="sha256:missing-ts-rev")
    record["ts"] = None
    projection, _state = build_accounting([record], {})

    assert projection["usage_rows"] == []
    missing = projection["diagnostics"][0]
    assert missing["code"] == "missing_usage_fields"
    assert "timestamp" in missing["details"]["missing_fields"]


# --- revisions and conflicts ---


def test_later_revision_replaces_earlier_for_same_logical_call():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    first = _usage_record(row, content_revision="sha256:first-revision")
    updated = copy.deepcopy(row)
    updated["output_tokens"] = 999
    second = _usage_record(updated, content_revision="sha256:second-revision")
    projection, state = build_accounting([first, second], {})

    assert len(projection["usage_rows"]) == 1
    assert projection["usage_rows"][0].output_tokens == 999
    assert projection["candidates"][0]["usage_source_id"] == second["source_id"]
    assert projection["candidates"][0]["source_ids"] == [first["source_id"], second["source_id"]]
    assert len(state["logical_calls"]) == 1
    stored = state["logical_calls"][projection["candidates"][0]["logical_call_id"]]
    assert stored["metrics_digest"].startswith("sha256:")
    assert stored["metrics_digest"] != ""


def test_later_revision_in_new_partition_keeps_prior_source_ids():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    first = _usage_record(row, content_revision="sha256:first-revision")
    updated = copy.deepcopy(row)
    updated["output_tokens"] = 999
    second = _usage_record(updated, content_revision="sha256:second-revision")
    _, state = build_accounting([first], {})
    projection, next_state = build_accounting([second], state)

    assert projection["candidates"][0]["source_ids"] == [first["source_id"], second["source_id"]]
    stored = next_state["logical_calls"][projection["candidates"][0]["logical_call_id"]]
    assert stored["source_ids"] == [first["source_id"], second["source_id"]]
    assert projection["usage_rows"][0].output_tokens == 999


def test_prior_usage_source_id_seeds_when_source_ids_absent():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    first = _usage_record(row, content_revision="sha256:first-revision")
    _, state = build_accounting([first], {})
    logical_id = next(iter(state["logical_calls"]))
    del state["logical_calls"][logical_id]["source_ids"]
    updated = copy.deepcopy(row)
    updated["output_tokens"] = 999
    second = _usage_record(updated, content_revision="sha256:second-revision")
    projection, _next_state = build_accounting([second], state)

    assert projection["candidates"][0]["source_ids"] == [first["source_id"], second["source_id"]]


def test_reused_row_id_across_generations_emits_warning():
    row_a = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row_b = copy.deepcopy(row_a)
    row_b["output_tokens"] = 50
    first = _usage_record(
        row_a,
        content_revision="sha256:gen-a-rev",
        generation="sha256:generation-a",
    )
    second = _usage_record(
        row_b,
        content_revision="sha256:gen-b-rev",
        generation="sha256:generation-b",
    )
    projection, state = build_accounting([first, second], {})

    assert "usage_row_id_reuse" in _diagnostic_codes(projection)
    reuse = next(item for item in projection["diagnostics"] if item["code"] == "usage_row_id_reuse")
    assert first["source_id"] in reuse["source_ids"]
    assert second["source_id"] in reuse["source_ids"]
    call_ids = [row.call_id for row in projection["usage_rows"]]
    assert len(call_ids) == 2
    assert call_ids[0] != call_ids[1]
    assert projection["usage_rows"][0].output_tokens == row_a["output_tokens"]
    assert projection["usage_rows"][1].output_tokens == 50
    assert projection["candidates"][0]["usage_source_id"] == first["source_id"]
    assert len(state["logical_calls"]) == 2


def test_incompatible_metrics_quarantine_logical_call():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row["cache_read_tokens"] = row["input_tokens"] + 1
    record = _usage_record(row, content_revision="sha256:incompatible-rev")
    projection, state = build_accounting([record], {})

    assert projection["usage_rows"] == []
    assert len(projection["candidates"]) == 1
    assert projection["candidates"][0]["usage_source_id"] == record["source_id"]
    assert "usage_revision_conflict" in _diagnostic_codes(projection)
    conflict = next(
        item for item in projection["diagnostics"] if item["code"] == "usage_revision_conflict"
    )
    assert record["source_id"] in conflict["source_ids"]
    assert state["logical_calls"]


def test_cache_read_plus_write_exceeding_input_quarantines():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row["input_tokens"] = 100
    row["cache_read_tokens"] = 80
    row["cache_write_tokens"] = 80
    record = _usage_record(row, content_revision="sha256:cache-sum-rev")
    projection, state = build_accounting([record], {})

    assert projection["usage_rows"] == []
    assert len(projection["candidates"]) == 1
    assert "usage_revision_conflict" in _diagnostic_codes(projection)
    logical_id = projection["candidates"][0]["logical_call_id"]
    assert state["logical_calls"][logical_id]["quarantined"] is True


# --- checkpoint / shutdown ---


def test_checkpoint_snapshot_is_not_additive():
    records = _six_call_records() + [_checkpoint_record()]
    projection, _state = build_accounting(records, {})

    assert len(projection["usage_rows"]) == 6
    assert "checkpoint_not_additive" in _diagnostic_codes(projection)


def test_shutdown_mismatch_emits_diagnostic():
    usage = _load_json(FIXTURES / "usage.json")
    usage["modelMetrics"]["gpt-5.6-luna"]["usage"]["inputTokens"] = 1
    shutdown = _shutdown_record()
    shutdown["payload"]["data"] = usage
    projection, _state = build_accounting(_six_call_records() + [shutdown], {})

    assert "shutdown_total_mismatch" in _diagnostic_codes(projection)
    mismatch = next(
        item for item in projection["diagnostics"] if item["code"] == "shutdown_total_mismatch"
    )
    assert mismatch["details"]["expected_input_tokens"] == 1
    assert mismatch["details"]["accounted_input_tokens"] == SHUTDOWN_TOTALS["input_tokens"]


def test_shutdown_float_nano_aiu_is_used_for_validation():
    usage = _load_json(FIXTURES / "usage.json")
    usage["modelMetrics"]["gpt-5.6-luna"]["totalNanoAiu"] = 1.0
    shutdown = _shutdown_record()
    shutdown["payload"]["data"] = usage
    projection, _state = build_accounting(_six_call_records() + [shutdown], {})

    mismatch = next(
        item for item in projection["diagnostics"] if item["code"] == "shutdown_total_mismatch"
    )
    assert mismatch["details"]["expected_total_nano_aiu"] == 1
    assert "capability_gap" not in _diagnostic_codes(projection)


def test_shutdown_agent_metrics_mismatch_is_reported():
    usage = _load_json(FIXTURES / "usage.json")
    usage["agentMetrics"]["main"]["modelMetrics"]["gpt-5.6-luna"]["usage"]["inputTokens"] = 1
    shutdown = _shutdown_record()
    shutdown["payload"]["data"] = usage
    projection, _state = build_accounting(_six_call_records() + [shutdown], {})

    mismatches = [
        item for item in projection["diagnostics"] if item["code"] == "shutdown_total_mismatch"
    ]
    assert any(
        item["details"].get("agent_id") == "main"
        and item["details"].get("expected_input_tokens") == 1
        for item in mismatches
    )


def test_unusable_shutdown_emits_capability_gap_instead_of_silent_skip():
    shutdown = _shutdown_record()
    shutdown["payload"]["data"] = {
        "modelMetrics": {"gpt-5.6-luna": {"usage": {"inputTokens": "not-a-number"}}}
    }
    projection, _state = build_accounting(_six_call_records() + [shutdown], {})

    gaps = [item for item in projection["diagnostics"] if item["code"] == "capability_gap"]
    assert gaps
    assert shutdown["source_id"] in gaps[0]["source_ids"]
    assert "shutdown_total_mismatch" not in _diagnostic_codes(projection)


def test_shutdown_data_not_a_dict_emits_capability_gap():
    shutdown = _shutdown_record()
    shutdown["payload"]["data"] = "not-a-dict"
    projection, _state = build_accounting(_six_call_records() + [shutdown], {})

    gaps = [item for item in projection["diagnostics"] if item["code"] == "capability_gap"]
    assert any(item["details"].get("reason") == "missing_shutdown_data" for item in gaps)
    assert shutdown["source_id"] in gaps[0]["source_ids"]
    assert "shutdown_total_mismatch" not in _diagnostic_codes(projection)


def test_unparseable_agent_metrics_emits_capability_gap():
    usage = _load_json(FIXTURES / "usage.json")
    usage["agentMetrics"] = {"main": "not-a-dict"}
    shutdown = _shutdown_record()
    shutdown["payload"]["data"] = usage
    projection, _state = build_accounting(_six_call_records() + [shutdown], {})

    gaps = [item for item in projection["diagnostics"] if item["code"] == "capability_gap"]
    assert any(
        item["details"].get("reason") == "unparseable_agent_metrics"
        and shutdown["source_id"] in item["source_ids"]
        for item in gaps
    )


def test_mixed_archive_does_not_charge_sessions_turns_or_title_calls():
    title = _load_json(RECONCILIATION / "cases.json")["auxiliary_title_generation"][
        "input_records"
    ][0]
    sessions_record: SourceRecord = {
        "source_id": (
            f"copilot-db:{SOURCE_KEY}:{NATIVE_SESSION_ID}:sessions:1:sha256:sessions-rev"
        ),
        "source_kind": "database",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": None,
        "observed_at": OBSERVED_AT,
        "payload": {
            "schema_version": 1,
            "table": "sessions",
            "row": {"id": NATIVE_SESSION_ID},
        },
        "locator": {
            "database": "/example/.copilot/session-store.db",
            "table": "sessions",
            "primary_key": NATIVE_SESSION_ID,
            "content_revision": "sha256:sessions-rev",
            "generation": GENERATION,
        },
    }
    records = _six_call_records() + [sessions_record, title]
    projection, _state = build_accounting(records, {})
    totals = _metric_totals(projection["usage_rows"])

    assert len(projection["usage_rows"]) == 6
    assert len(projection["candidates"]) == 6
    assert totals["input_tokens"] == SHUTDOWN_TOTALS["input_tokens"]
    assert totals["output_tokens"] == SHUTDOWN_TOTALS["output_tokens"]
    assert totals["cache_read_tokens"] == SHUTDOWN_TOTALS["cache_read_tokens"]
    assert totals["cache_write_tokens"] == SHUTDOWN_TOTALS["cache_write_tokens"]
    assert totals["reasoning_tokens"] == SHUTDOWN_TOTALS["reasoning_tokens"]
    assert _nano_aiu_total(projection["candidates"]) == SHUTDOWN_TOTALS["total_nano_aiu"]


# --- supplemental metrics ---


def test_supplemental_metrics_preserve_nano_aiu_separately():
    records = _six_call_records()
    projection, _state = build_accounting(records, {})
    candidate = projection["candidates"][0]

    assert "total_nano_aiu" in candidate["supplemental_metrics"]
    assert candidate["supplemental_metrics"]["total_nano_aiu"] == 174125000
    usage_row = projection["usage_rows"][0].to_dict()
    assert "total_nano_aiu" not in usage_row


# --- incremental replay ---


def test_incremental_replay_matches_full_archive():
    records = _six_call_records()
    full_projection, full_state = build_accounting(records, {})

    state: dict[str, Any] = {}
    incremental_projection = None
    for index in range(1, len(records) + 1):
        incremental_projection, state = build_accounting(records[:index], state)

    assert incremental_projection is not None
    assert [row.to_dict() for row in incremental_projection["usage_rows"]] == [
        row.to_dict() for row in full_projection["usage_rows"]
    ]
    assert incremental_projection["candidates"] == full_projection["candidates"]
    assert state["logical_calls"] == full_state["logical_calls"]


def test_late_arrival_adds_new_call_without_rerunning_agent():
    first_batch = _six_call_records()[:3]
    late_row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[3])
    late_record = _usage_record(
        late_row,
        content_revision="sha256:late-arrival-rev",
    )

    first_projection, state = build_accounting(first_batch, {})
    second_projection, _state = build_accounting(first_batch + [late_record], state)

    assert len(first_projection["usage_rows"]) == 3
    assert len(second_projection["usage_rows"]) == 4
    assert second_projection["candidates"][-1]["agent_id"] == CHILD_AGENT_ID


# --- prior state ---


def test_prior_state_logical_calls_are_preserved_for_unseen_ids():
    records = _six_call_records()[:1]
    _, state = build_accounting(records, {})
    inherited = copy.deepcopy(state)

    projection, next_state = build_accounting([], inherited)
    assert projection["usage_rows"] == []
    assert projection["candidates"] == []
    assert next_state["logical_calls"] == inherited["logical_calls"]


def test_nested_accounting_state_preserves_unseen_logical_calls():
    records = _six_call_records()[:1]
    _, state = build_accounting(records, {})
    nested = {"accounting_state": copy.deepcopy(state)}

    projection, next_state = build_accounting([], nested)
    assert projection["usage_rows"] == []
    assert next_state["logical_calls"] == state["logical_calls"]


def test_incomplete_locator_emits_capability_gap_and_keeps_the_source_id():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    record = _usage_record(row, content_revision="sha256:missing-generation-rev")
    del record["locator"]["generation"]
    projection, _state = build_accounting([record], {})

    assert projection["usage_rows"] == []
    assert projection["candidates"] == []
    gap = next(item for item in projection["diagnostics"] if item["code"] == "capability_gap")
    assert record["source_id"] in gap["source_ids"]


def test_invalid_source_identity_emits_capability_gap():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    record = _usage_record(row, content_revision="sha256:bad-source-rev")
    record["source_id"] = "not-a-copilot-db-identity"
    projection, _state = build_accounting([record], {})

    assert projection["usage_rows"] == []
    assert projection["candidates"] == []
    gap = next(item for item in projection["diagnostics"] if item["code"] == "capability_gap")
    assert record["source_id"] in gap["source_ids"]


def test_truncated_source_key_emits_capability_gap():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    record = _usage_record(row, content_revision="sha256:short-key-rev", source_key="abc")
    projection, _state = build_accounting([record], {})

    assert projection["usage_rows"] == []
    gap = next(item for item in projection["diagnostics"] if item["code"] == "capability_gap")
    assert record["source_id"] in gap["source_ids"]
    assert gap["details"]["reason"] == "source_id is not a 64-character copilot-db identity"


def test_later_incompatible_revision_conflicts_using_prior_metrics_digest():
    row = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    first = _usage_record(row, content_revision="sha256:first-digest-rev")
    _, state = build_accounting([first], {})
    logical_id = next(iter(state["logical_calls"]))
    prior_digest = state["logical_calls"][logical_id]["metrics_digest"]

    bad = copy.deepcopy(row)
    bad["cache_read_tokens"] = bad["input_tokens"] + 1
    second = _usage_record(bad, content_revision="sha256:second-digest-rev")
    projection, _next_state = build_accounting([second], state)

    conflict = next(
        item for item in projection["diagnostics"] if item["code"] == "usage_revision_conflict"
    )
    assert conflict["details"]["prior_metrics_digest"] == prior_digest
    assert conflict["details"]["metrics_digest"] != prior_digest
    assert projection["usage_rows"] == []
    assert len(projection["candidates"]) == 1
    assert projection["candidates"][0]["source_ids"] == [first["source_id"], second["source_id"]]
    assert first["source_id"] in conflict["source_ids"]
    assert second["source_id"] in conflict["source_ids"]


def test_disjoint_partition_does_not_false_mismatch_shutdown():
    records = _six_call_records()
    _first, state = build_accounting(records[:3], {})
    second, _next_state = build_accounting(records[3:] + [_shutdown_record()], state)

    assert "shutdown_total_mismatch" not in _diagnostic_codes(second)
    assert len(second["usage_rows"]) == 3


def test_row_id_reuse_is_detected_across_partitions():
    row_a = copy.deepcopy(_load_json(FIXTURES / "assistant-usage-events.json")[0])
    row_b = copy.deepcopy(row_a)
    row_b["output_tokens"] = 50
    first = _usage_record(
        row_a,
        content_revision="sha256:gen-a-rev",
        generation="sha256:generation-a",
    )
    second = _usage_record(
        row_b,
        content_revision="sha256:gen-b-rev",
        generation="sha256:generation-b",
    )
    _first_projection, state = build_accounting([first], {})
    projection, _next_state = build_accounting([second], state)

    assert "usage_row_id_reuse" in _diagnostic_codes(projection)
    reuse = next(item for item in projection["diagnostics"] if item["code"] == "usage_row_id_reuse")
    assert first["source_id"] in reuse["source_ids"]
    assert second["source_id"] in reuse["source_ids"]
    assert len(projection["usage_rows"]) == 1
    assert projection["usage_rows"][0].output_tokens == 50


def test_integer_valued_float_row_metrics_do_not_false_mismatch_shutdown():
    records = _six_call_records()
    for record in records:
        row = record["payload"]["row"]
        assert isinstance(row, dict)
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "total_nano_aiu",
        ):
            if isinstance(row.get(field), int):
                row[field] = float(row[field])
    projection, state = build_accounting(records + [_shutdown_record()], {})
    totals = _metric_totals(projection["usage_rows"])

    assert len(projection["usage_rows"]) == 6
    assert totals["input_tokens"] == SHUTDOWN_TOTALS["input_tokens"]
    assert totals["output_tokens"] == SHUTDOWN_TOTALS["output_tokens"]
    assert totals["cache_read_tokens"] == SHUTDOWN_TOTALS["cache_read_tokens"]
    assert totals["cache_write_tokens"] == SHUTDOWN_TOTALS["cache_write_tokens"]
    assert totals["reasoning_tokens"] == SHUTDOWN_TOTALS["reasoning_tokens"]
    assert state["accounted_metrics"]["total_nano_aiu"] == SHUTDOWN_TOTALS["total_nano_aiu"]
    assert "shutdown_total_mismatch" not in _diagnostic_codes(projection)
