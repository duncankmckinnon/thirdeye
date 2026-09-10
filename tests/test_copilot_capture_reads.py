"""Compatibility coverage for generic CLI reads of raw Copilot evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.config import Config
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.types import SourceBatch, SourcePaths, SourceRecord
from thirdeye.store import Store
from thirdeye.turns import session_turns

NATIVE_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
SOURCE_TS = "2026-09-10T17:08:24.506Z"
OBSERVED_AT = "2026-09-10T17:08:25.000Z"
TOOL_PATH = "/fixture/workspace/alpha.txt"

PROMPT_PAYLOAD = {
    "type": "user.message",
    "id": "prompt-1",
    "timestamp": SOURCE_TS,
    "data": {"content": "Read alpha.txt and beta.txt with separate view calls."},
}
TOOL_PAYLOAD = {
    "type": "tool.execution_start",
    "id": "tool-1",
    "timestamp": SOURCE_TS,
    "data": {"toolName": "view", "arguments": {"path": TOOL_PATH}},
}
HOOK_PAYLOAD = {
    "event": "userPromptSubmitted",
    "hook_payload": {
        "sessionId": NATIVE_ID,
        "prompt": "Read only alpha.txt as an explore child.",
        "agentId": "child-agent-id",
    },
}


def _record(kind: str, source_id: str, payload: dict[str, Any]) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": kind,
        "native_session_id": NATIVE_ID,
        "ts": SOURCE_TS,
        "observed_at": OBSERVED_AT,
        "payload": payload,
        "locator": {"file": "events.jsonl", "file_generation": "fixture-gen", "byte_offset": 42},
    }


def _assert_versioned_envelope(event: dict[str, Any], *, source_kind: str, payload: dict[str, Any]) -> None:
    data = event["data"]
    assert data["schema_version"] == 1
    record = data["source_record"]
    assert record["source_kind"] == source_kind
    assert record["payload"] == payload
    assert "schema_version" not in record["payload"]


def _capture_synthetic_batch(config: Config, paths: SourcePaths) -> str:
    """Archive labeled synthetic SourceRecords for generic read compatibility."""
    records = [
        _record("transcript", f"{paths['source_key']}/{NATIVE_ID}/prompt-1", PROMPT_PAYLOAD),
        _record("transcript", f"{paths['source_key']}/{NATIVE_ID}/tool-1", TOOL_PAYLOAD),
        _record("hook", f"hook/{NATIVE_ID}/child-prompt-1", HOOK_PAYLOAD),
    ]
    batch: SourceBatch = {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/fixture/workspace",
        "records": records,
        "next_cursor": {"fixture": 1},
        "diagnostics": [],
    }
    commit_batch(config, paths, batch)
    return stored_session_id(paths, NATIVE_ID)


def test_store_lists_and_retains_raw_source_identity(tmp_path: Path) -> None:
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(tmp_path / "copilot-home")
    stored_id = _capture_synthetic_batch(config, paths)

    sessions = list(Store(config).list_sessions(platform="copilot"))
    assert [(session.session_id, session.cwd) for session in sessions] == [
        (stored_id, "/fixture/workspace")
    ]
    assert sessions[0].extra["copilot"]["native_session_id"] == NATIVE_ID

    events = list(Store(config).reader(stored_id).iter_events())
    assert [event["t"] for event in events] == [
        "copilot_transcript",
        "copilot_transcript",
        "copilot_hook",
    ]
    assert all(event["t"] not in {"user_message", "tool_call"} for event in events)
    assert events[0]["ts"] == SOURCE_TS
    _assert_versioned_envelope(events[0], source_kind="transcript", payload=PROMPT_PAYLOAD)
    _assert_versioned_envelope(events[1], source_kind="transcript", payload=TOOL_PAYLOAD)
    source_record = events[1]["data"]["source_record"]
    assert source_record["source_id"] == f"{paths['source_key']}/{NATIVE_ID}/tool-1"
    assert source_record["native_session_id"] == NATIVE_ID
    assert source_record["locator"]["byte_offset"] == 42
    _assert_versioned_envelope(events[2], source_kind="hook", payload=HOOK_PAYLOAD)
    assert events[2]["data"]["source_record"]["payload"]["hook_payload"]["agentId"] == "child-agent-id"


def test_generic_cli_reads_search_raw_copilot_content(tmp_path: Path) -> None:
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(tmp_path / "copilot-home")
    stored_id = _capture_synthetic_batch(config, paths)
    runner = CliRunner()
    env = {"THIRDEYE_HOME": str(config.root)}

    listed = runner.invoke(main, ["list", "--platform", "copilot"], env=env)
    shown = runner.invoke(main, ["show", stored_id, "--json"], env=env)
    events = runner.invoke(main, ["events", stored_id, "--json", "--no-findings"], env=env)
    prompt_search = runner.invoke(
        main, ["search", "separate view calls", "--platform", "copilot"], env=env
    )
    tool_search = runner.invoke(main, ["search", TOOL_PATH, "--platform", "copilot"], env=env)
    tailed = runner.invoke(main, ["tail", stored_id, "-n", "1", "--json"], env=env)

    assert listed.exit_code == shown.exit_code == events.exit_code == prompt_search.exit_code == 0
    assert tool_search.exit_code == tailed.exit_code == 0
    assert stored_id in listed.output
    assert NATIVE_ID in listed.output
    assert '"t":"copilot_transcript"' in shown.output
    assert '"type":"tool.execution_start"' in events.output
    assert '"schema_version":1' in events.output
    assert '"source_kind":"transcript"' in events.output
    assert "separate view calls" in prompt_search.output
    assert TOOL_PATH in tool_search.output
    assert '"t":"copilot_hook"' in tailed.output


def test_all_source_kinds_map_to_raw_event_types_without_projection(tmp_path: Path) -> None:
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(tmp_path / "copilot-home")
    transcript_payload = {"type": "user.message"}
    database_payload = {"table": "assistant_usage_events", "tokens": 42}
    hook_payload = {"event": "sessionStart", "hook_payload": {}}
    metadata_payload = {"file": "workspace.yaml", "cwd": "/fixture/workspace"}
    records = [
        _record("transcript", f"{paths['source_key']}/{NATIVE_ID}/tx", transcript_payload),
        _record("database", f"{paths['source_key']}/{NATIVE_ID}/db", database_payload),
        _record("hook", f"hook/{NATIVE_ID}/hk", hook_payload),
        _record("metadata", f"{paths['source_key']}/{NATIVE_ID}/meta", metadata_payload),
    ]
    batch: SourceBatch = {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/fixture/workspace",
        "records": records,
        "next_cursor": {"fixture": 1},
        "diagnostics": [],
    }
    commit_batch(config, paths, batch)
    stored_id = stored_session_id(paths, NATIVE_ID)

    events = list(Store(config).reader(stored_id).iter_events())
    assert [event["t"] for event in events] == [
        "copilot_transcript",
        "copilot_database",
        "copilot_hook",
        "copilot_metadata",
    ]
    _assert_versioned_envelope(events[0], source_kind="transcript", payload=transcript_payload)
    _assert_versioned_envelope(events[1], source_kind="database", payload=database_payload)
    _assert_versioned_envelope(events[2], source_kind="hook", payload=hook_payload)
    _assert_versioned_envelope(events[3], source_kind="metadata", payload=metadata_payload)
    assert all(event["t"] not in {"user_message", "tool_call", "assistant_message"} for event in events)


def test_copilot_sessions_are_not_sliced_into_eval_turns(tmp_path: Path) -> None:
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(tmp_path / "copilot-home")
    stored_id = _capture_synthetic_batch(config, paths)
    store = Store(config)
    meta = store.get_meta(stored_id)

    assert session_turns(meta, store) == []


def test_hook_prompt_content_is_searchable(tmp_path: Path) -> None:
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(tmp_path / "copilot-home")
    _capture_synthetic_batch(config, paths)
    runner = CliRunner()
    env = {"THIRDEYE_HOME": str(config.root)}

    hook_search = runner.invoke(
        main,
        ["search", "explore child", "--platform", "copilot"],
        env=env,
    )

    assert hook_search.exit_code == 0
    assert "explore child" in hook_search.output
