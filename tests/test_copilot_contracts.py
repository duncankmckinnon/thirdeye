"""Frozen contract tests for Copilot V1 capture types and identity."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

import thirdeye.platforms.copilot as copilot_pkg
from thirdeye.platforms.copilot.constants import (
    CLI_HOOK_EVENT_ALIASES,
    CLI_HOOK_EVENTS,
    COPILOT_HOME_ENV,
    DISPLAY_NAME,
    EVENT_ENVELOPE_VERSION,
    HOOKS_DIRECTORY_NAME,
    OWNED_HOOK_FILENAME,
    PLATFORM_NAME,
    SCHEMA_VERSION,
    SOURCE_RECORD_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from thirdeye.platforms.copilot.identity import (
    SOURCE_KEY_PREFIX_LEN,
    resolve_sources,
    source_keys_share_stored_prefix,
    stored_session_id,
    validate_native_id,
)
from thirdeye.platforms.copilot.types import (
    SCHEMA_VERSION as TYPES_SCHEMA_VERSION,
)
from thirdeye.platforms.copilot.types import (
    SourceBatch,
    SourcePaths,
    SourceRecord,
    SourceSlice,
    SyncResult,
)

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
CLI_FIXTURE = FIXTURES / "cli-1.0.83"
V1_CASES = FIXTURES / "v1-cases"

NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
CHILD_AGENT_ID = "bf8cb9f3-2097-4db0-a3c8-78a2653b2106"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_trip(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


# --- constants ---


def test_platform_identity_constants():
    assert PLATFORM_NAME == "copilot"
    assert DISPLAY_NAME == "GitHub Copilot CLI"


def test_schema_version_is_frozen_at_one():
    for version in (
        SOURCE_SCHEMA_VERSION,
        SCHEMA_VERSION,
        SOURCE_RECORD_SCHEMA_VERSION,
        EVENT_ENVELOPE_VERSION,
        TYPES_SCHEMA_VERSION,
    ):
        assert version == 1


def test_constants_module_has_no_resolved_paths():
    import thirdeye.platforms.copilot.constants as constants

    for name in dir(constants):
        if name.startswith("_"):
            continue
        value = getattr(constants, name)
        if isinstance(value, str) and ("/" in value or "\\" in value):
            pytest.fail(f"constants.{name} looks like a resolved path: {value!r}")


def test_hook_install_vocabulary():
    assert HOOKS_DIRECTORY_NAME == "hooks"
    assert OWNED_HOOK_FILENAME == "thirdeye.json"
    assert COPILOT_HOME_ENV == "COPILOT_HOME"


def test_cli_hook_aliases_cover_observed_events():
    expected = {
        "sessionStart",
        "userPromptSubmitted",
        "preToolUse",
        "postToolUse",
        "agentStop",
        "subagentStart",
        "subagentStop",
        "sessionEnd",
    }
    assert expected <= set(CLI_HOOK_EVENT_ALIASES)
    assert CLI_HOOK_EVENTS == tuple(CLI_HOOK_EVENT_ALIASES)


def test_hook_aliases_map_to_snake_case():
    for camel, snake in CLI_HOOK_EVENT_ALIASES.items():
        assert camel[0].islower()
        assert snake == re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()


# --- package exports ---


def test_public_exports_match_contract_surface():
    assert set(copilot_pkg.__all__) == {
        "SourceBatch",
        "SourcePaths",
        "SourceRecord",
        "SourceSlice",
        "SyncResult",
        "resolve_sources",
        "stored_session_id",
        "validate_native_id",
    }


def test_copilot_package_does_not_import_usage_store():
    import thirdeye.platforms.copilot.identity as identity
    import thirdeye.platforms.copilot.types as types

    for module in (copilot_pkg, identity, types):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "UsageStore" not in source
        assert "usage_store" not in source


# --- TypedDict / JSON round-trip fixtures ---


def test_source_slice_fixture_round_trips():
    raw = _load_json(V1_CASES / "source-slice.json")
    restored = _round_trip(raw)
    assert restored == raw
    assert set(restored) == set(SourceSlice.__annotations__)

    slice_: SourceSlice = restored
    assert slice_["exhausted"] is False
    assert len(slice_["records"]) == 1
    record = slice_["records"][0]
    assert record["source_kind"] == "transcript"
    assert record["payload"]["schema_version"] == 1
    assert record["ts"] == "2026-09-10T17:08:24.000Z"
    assert record["observed_at"] == "2026-09-10T17:08:25.000Z"


def test_source_batch_fixture_round_trips():
    raw = _load_json(V1_CASES / "source-batch.json")
    restored = _round_trip(raw)
    assert restored == raw
    assert set(restored) == set(SourceBatch.__annotations__)
    assert "exhausted" not in restored

    batch: SourceBatch = restored
    assert batch["source_key"] == restored["source_key"]
    assert len(batch["records"]) == 1
    record = batch["records"][0]
    assert record["source_kind"] == "database"
    assert record["payload"]["schema_version"] == 1
    assert record["ts"] is None


def test_sync_result_shape_accepts_zero_counts():
    result: SyncResult = {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 0,
    }
    assert _round_trip(dict(result)) == result


def test_source_record_optional_ts_null_round_trips():
    record: SourceRecord = {
        "source_id": "key/session-a/row-1/rev-a",
        "source_kind": "database",
        "native_session_id": "session-a",
        "ts": None,
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": {"schema_version": 1, "table": "turns", "row": {"id": 1}},
        "locator": {"table": "turns", "primary_key": 1, "content_revision": "rev-a"},
    }
    restored = _round_trip(dict(record))
    assert restored["ts"] is None
    assert restored["payload"]["row"]["id"] == 1


# --- identity: resolve_sources ---


def test_resolve_sources_uses_explicit_home(tmp_path: Path):
    home = tmp_path / "copilot-home"
    home.mkdir()
    paths = resolve_sources(home)

    assert paths["home"] == str(home.resolve())
    assert paths["session_root"] == str((home / "session-state").resolve())
    assert paths["database"] == str((home / "session-store.db").resolve())
    assert len(paths["source_key"]) == 64


def test_resolve_sources_honors_copilot_home_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "from-env"
    home.mkdir()
    monkeypatch.setenv(COPILOT_HOME_ENV, str(home))
    paths = resolve_sources()
    assert paths["home"] == str(home.resolve())


def test_source_key_is_full_sha256_of_canonical_home(tmp_path: Path):
    home = tmp_path / "canonical"
    home.mkdir()
    paths = resolve_sources(home)
    normalized = os.path.normcase(str(home.resolve()))
    expected = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    assert paths["source_key"] == expected


def test_aliases_to_same_canonical_home_share_source_key(tmp_path: Path):
    home = tmp_path / "alias-target"
    home.mkdir()
    via_dot = tmp_path / "alias-target" / ".." / "alias-target"
    assert resolve_sources(home)["source_key"] == resolve_sources(via_dot)["source_key"]


def test_forged_source_key_is_rejected(tmp_path: Path):
    home = tmp_path / "copilot-home"
    home.mkdir()
    paths = resolve_sources(home)
    forged: SourcePaths = dict(paths)
    forged["source_key"] = "0" * 64
    with pytest.raises(ValueError, match="source_key does not match"):
        stored_session_id(forged, "session-a")


def test_session_root_outside_home_is_rejected(tmp_path: Path):
    home = tmp_path / "copilot-home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    paths = resolve_sources(home)
    forged: SourcePaths = dict(paths)
    forged["session_root"] = str(outside.resolve())
    with pytest.raises(ValueError, match="session_root escapes"):
        stored_session_id(forged, "session-a")


# --- identity: validate_native_id ---


@pytest.mark.parametrize(
    "native_id",
    [
        "",
        " ",
        " leading",
        "trailing ",
        ".",
        "..",
        "../escape",
        "..\\escape",
        "bad/id",
        "bad\\id",
        "bad:id",
        "bad\x00id",
        "bad\nid",
    ],
)
def test_validate_native_id_rejects_unsafe_values(native_id: str):
    with pytest.raises(ValueError):
        validate_native_id(native_id)


def test_validate_native_id_accepts_uuid_like_ids():
    validate_native_id(NATIVE_SESSION_ID)


# --- identity: stored_session_id ---


def test_stored_session_id_format(tmp_path: Path):
    (tmp_path / "home").mkdir()
    paths = resolve_sources(tmp_path / "home")
    native = "session-a"
    stored = stored_session_id(paths, native)
    assert stored == f"copilot-{paths['source_key'][:SOURCE_KEY_PREFIX_LEN]}-{native}"


def test_different_homes_do_not_merge_same_native_id(tmp_path: Path):
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    native = "shared-native-id"
    id_a = stored_session_id(resolve_sources(home_a), native)
    id_b = stored_session_id(resolve_sources(home_b), native)
    assert id_a != id_b


def test_stored_session_id_rejects_source_key_that_does_not_match_home(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    paths = resolve_sources(home)
    other_home = tmp_path / "other-home"
    other_home.mkdir()
    other = resolve_sources(other_home)
    forged: SourcePaths = dict(paths)
    forged["source_key"] = other["source_key"]
    with pytest.raises(ValueError, match="source_key does not match"):
        stored_session_id(forged, "session-a")


def test_source_key_prefix_collision_is_an_archive_reuse_contract():
    case = _load_json(V1_CASES / "source-key-prefix-collision.json")
    first, second = case["homes"]
    native = case["native_session_id"]

    assert first["source_key"] != second["source_key"]
    assert source_keys_share_stored_prefix(first["source_key"], second["source_key"])
    stored_a = f"copilot-{first['source_key'][:SOURCE_KEY_PREFIX_LEN]}-{native}"
    stored_b = f"copilot-{second['source_key'][:SOURCE_KEY_PREFIX_LEN]}-{native}"
    assert stored_a == stored_b == case["colliding_stored_session_id"]
    assert case["retained_metadata_source_key"] == first["source_key"]
    assert second["source_key"] != case["retained_metadata_source_key"]
    assert source_keys_share_stored_prefix(first["source_key"], first["source_key"]) is False
    with pytest.raises(ValueError, match="64-character"):
        source_keys_share_stored_prefix(first["source_key"], "too-short")


# --- observed cli-1.0.83 fixtures ---


def test_cli_fixture_files_exist():
    for name in (
        "README.md",
        "events.jsonl",
        "hooks.jsonl",
        "usage.json",
        "assistant-usage-events.json",
    ):
        assert (CLI_FIXTURE / name).is_file()


def test_cli_transcript_retains_seventy_six_events():
    lines = (CLI_FIXTURE / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 76
    for line in lines:
        event = json.loads(line)
        assert "type" in event


def test_cli_external_hooks_count_twenty():
    lines = (CLI_FIXTURE / "hooks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20


def test_cli_fixture_has_two_main_prompts_and_one_child_prompt():
    events = [
        json.loads(line)
        for line in (CLI_FIXTURE / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    user_messages = [event for event in events if event.get("type") == "user.message"]
    assert len(user_messages) == 3

    main_prompts = [
        event
        for event in user_messages
        if event.get("agentId") is None and event["data"].get("source") is None
    ]
    child_prompts = [event for event in user_messages if event.get("agentId") is not None]
    assert len(main_prompts) == 2
    assert len(child_prompts) == 1
    assert child_prompts[0]["agentId"] == CHILD_AGENT_ID


def test_cli_child_prompt_stop_hooks_use_child_agent_id_as_session_id():
    hooks = [
        json.loads(line)
        for line in (CLI_FIXTURE / "hooks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [
        json.loads(line)
        for line in (CLI_FIXTURE / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    child_prompts = [
        hook
        for hook in hooks
        if hook["registered_event"] == "userPromptSubmitted"
        and hook["payload"]["sessionId"] == CHILD_AGENT_ID
    ]
    child_stops = [
        hook
        for hook in hooks
        if hook["registered_event"] == "agentStop" and hook["payload"]["sessionId"] == CHILD_AGENT_ID
    ]
    assert len(child_prompts) == 1
    assert len(child_stops) == 1
    assert child_stops[0]["payload"]["transcriptPath"].endswith(
        f"{NATIVE_SESSION_ID}/events.jsonl"
    )

    parent_lifecycle = [
        hook
        for hook in hooks
        if hook["registered_event"] in {"subagentStart", "subagentStop"}
    ]
    assert parent_lifecycle
    assert all(hook["payload"]["sessionId"] == NATIVE_SESSION_ID for hook in parent_lifecycle)

    transcript_child_prompt = next(
        event
        for event in events
        if event.get("type") == "hook.start"
        and event["data"].get("hookType") == "userPromptSubmitted"
        and event["data"]["input"]["sessionId"] == CHILD_AGENT_ID
    )
    assert (
        transcript_child_prompt["data"]["input"]["sessionId"]
        == child_prompts[0]["payload"]["sessionId"]
    )


def test_cli_assistant_usage_events_fixture_has_six_rows():
    rows = _load_json(CLI_FIXTURE / "assistant-usage-events.json")
    assert len(rows) == 6
    assert {row["session_id"] for row in rows} == {NATIVE_SESSION_ID}


def test_cli_usage_totals_match_assistant_usage_events():
    usage = _load_json(CLI_FIXTURE / "usage.json")
    rows = _load_json(CLI_FIXTURE / "assistant-usage-events.json")

    total_nano = sum(row["total_nano_aiu"] for row in rows)
    assert total_nano == usage["totalNanoAiu"]

    for field, usage_key in (
        ("input_tokens", "inputTokens"),
        ("output_tokens", "outputTokens"),
        ("cache_read_tokens", "cacheReadTokens"),
        ("cache_write_tokens", "cacheWriteTokens"),
        ("reasoning_tokens", "reasoningTokens"),
    ):
        row_total = sum(row[field] for row in rows)
        model_usage = usage["modelMetrics"]["gpt-5.6-luna"]["usage"]
        assert row_total == model_usage[usage_key]


# --- synthetic v1-cases fixtures ---


def test_v1_trailing_json_fixture_has_complete_and_incomplete_tail():
    text = (V1_CASES / "trailing-json.jsonl").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) == 2
    json.loads(lines[0])
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[1])


def test_v1_trailing_utf8_hex_decodes_to_incomplete_tail():
    raw = bytes.fromhex((V1_CASES / "trailing-utf8.hex").read_text(encoding="utf-8").strip())
    complete_prefix, incomplete_tail = raw.split(b"\n", 1)
    json.loads(complete_prefix.decode("utf-8"))
    assert incomplete_tail.endswith(b"\xe2\x82")
    with pytest.raises(UnicodeDecodeError):
        incomplete_tail.decode("utf-8")


def test_v1_unknown_event_fields_fixture_is_lossless_json():
    line = (V1_CASES / "unknown-event-fields.jsonl").read_text(encoding="utf-8").strip()
    event = json.loads(line)
    assert event["data"]["future_field"]["nested"] == [1, True, {"opaque": "retain"}]
    assert event["top_level_future"] == "retain"


def test_v1_database_row_revisions_documents_reuse_and_change():
    rows = _load_json(V1_CASES / "database-row-revisions.json")
    assert len(rows) == 3
    assert rows[0]["primary_key"] == rows[1]["primary_key"] == rows[2]["primary_key"]
    assert rows[0]["content_revision"] != rows[1]["content_revision"]
    assert rows[1]["generation"] != rows[2]["generation"]


def test_v1_distinct_hook_observations_have_unique_ids_same_payload():
    observations = _load_json(V1_CASES / "distinct-hook-observations.json")
    assert len(observations) == 2
    assert observations[0]["payload"] == observations[1]["payload"]
    assert observations[0]["observation_id"] != observations[1]["observation_id"]


def test_v1_missing_event_id_fixture_has_no_native_id():
    event = json.loads((V1_CASES / "missing-event-id.jsonl").read_text(encoding="utf-8").strip())
    assert "id" not in event
