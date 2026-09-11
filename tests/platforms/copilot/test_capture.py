"""Behavioral tests for Copilot capture composition (sync, hooks, spool, archive)."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import thirdeye.platforms.copilot.archive as archive_mod
import thirdeye.platforms.copilot.capture as capture_mod
import thirdeye.platforms.copilot.state as state_mod
from thirdeye.config import Config
from thirdeye.meta import read_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.platforms.copilot.archive import commit_batch, load_cursor
from thirdeye.platforms.copilot.capture import (
    capture_session,
    iter_captured_records,
    record_hook,
    sync,
)
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.hook_payload import parse_hook
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.spool import enqueue_hook, read_spool
from thirdeye.platforms.copilot.types import SourceBatch, SourcePaths, SourceRecord, SyncResult

FIXTURES = Path(__file__).parent / "fixtures"
CLI_FIXTURE = FIXTURES
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
OBSERVED_AT = "2026-09-10T17:08:25.626Z"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(
    source_id: str,
    *,
    native_session_id: str = NATIVE_SESSION_ID,
    source_kind: str = "transcript",
) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "native_session_id": native_session_id,
        "ts": "2026-09-10T17:08:24.000Z",
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _batch(
    paths: SourcePaths,
    records: list[SourceRecord],
    *,
    native_session_id: str = NATIVE_SESSION_ID,
    next_cursor: dict[str, Any] | None = None,
    base_cursor: dict[str, Any] | None = None,
) -> SourceBatch:
    cursor: dict[str, Any] = dict(next_cursor or {"generation": 1})
    if base_cursor is not None:
        cursor["base_cursor"] = base_cursor
    return {
        "source_key": paths["source_key"],
        "native_session_id": native_session_id,
        "cwd": "/proj",
        "records": records,
        "next_cursor": cursor,
        "diagnostics": [],
    }


def _write_transcript(home: Path, native_id: str, *, events_path: Path | None = None) -> None:
    session_dir = home / "session-state" / native_id
    session_dir.mkdir(parents=True, exist_ok=True)
    source = events_path or (CLI_FIXTURE / "events.jsonl")
    shutil.copy(source, session_dir / "events.jsonl")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")


def _write_database(home: Path, *, session_id: str = NATIVE_SESSION_ID) -> None:
    home.mkdir(parents=True, exist_ok=True)
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, created_at TEXT);
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER,
                content TEXT,
                updated_at TEXT
            );
            CREATE TABLE assistant_usage_events (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER,
                agent_id TEXT,
                parent_tool_call_id TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                reasoning_tokens INTEGER,
                total_nano_aiu INTEGER,
                request_multiplier REAL,
                duration_ms INTEGER,
                time_to_first_token_ms REAL,
                output_ttft_ms REAL,
                inter_token_latency_ms REAL,
                initiator TEXT,
                api_endpoint TEXT,
                reasoning_effort TEXT,
                finish_reason TEXT,
                content_filter_triggered INTEGER,
                token_details_json TEXT,
                created_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions (id, cwd, created_at) VALUES (?, ?, ?)",
            (session_id, "/tmp/probe", "2026-09-10T17:08:00.000Z"),
        )
        connection.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, session_id, 0, "turn-0", "2026-09-10T17:08:10.000Z"),
        )
        connection.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (2, session_id, 1, "turn-1", "2026-09-10T17:08:11.000Z"),
        )
        for row in _load_json(CLI_FIXTURE / "assistant-usage-events.json"):
            columns = ", ".join(row)
            placeholders = ", ".join("?" for _ in row)
            connection.execute(
                f"INSERT INTO assistant_usage_events ({columns}) VALUES ({placeholders})",
                tuple(row.values()),
            )
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def copilot_env(tmp_path: Path) -> tuple[Config, SourcePaths]:
    home = tmp_path / "copilot-home"
    home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(home)
    return config, paths


def _hook_record(*, observation_id: str, session_id: str = NATIVE_SESSION_ID) -> SourceRecord:
    return parse_hook(
        "agentStop",
        {
            "sessionId": session_id,
            "timestamp": 1789060105626,
            "cwd": "/fixture/workspace",
            "stopReason": "end_turn",
        },
        {"env": {"WB_PLAN": "p"}},
        observed_at=OBSERVED_AT,
        observation_id=observation_id,
    )


@contextmanager
def _fault_at(point: str) -> Iterator[None]:
    def injector(name: str) -> None:
        if name == point:
            raise RuntimeError(f"injected fault at {point}")

    archive_mod._fault_injector = injector
    state_mod._fault_injector = injector
    try:
        yield
    finally:
        archive_mod._fault_injector = None
        state_mod._fault_injector = None


def _empty_result() -> SyncResult:
    return {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 0,
    }


def _composed_batch(
    paths: SourcePaths,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    records: list[SourceRecord] | None = None,
    transcript_exhausted: bool = True,
    database_exhausted: bool = True,
) -> SourceBatch:
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_SESSION_ID,
        "cwd": "/proj",
        "records": records or [_record("composed/1")],
        "next_cursor": {
            "transcript": {"byte_offset": 1},
            "database": {"database_offset": 1},
            "base_cursor": {},
            "transcript_exhausted": transcript_exhausted,
            "database_exhausted": database_exhausted,
        },
        "diagnostics": diagnostics or [],
    }


# --- module exports ---


def test_iter_captured_records_is_reexported_from_archive(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    commit_batch(config, paths, _batch(paths, [_record("key/a/exported")]))
    captured = list(iter_captured_records(config, stored))
    assert len(captured) == 1
    assert captured[0]["source_id"] == "key/a/exported"


def test_capture_module_has_no_usage_store_or_export_imports() -> None:
    source = Path(capture_mod.__file__).read_text(encoding="utf-8")
    assert "UsageStore" not in source
    assert "usage_store" not in source
    assert "logfire" not in source
    assert "otel" not in source.lower()


# --- sync semantics ---


def test_sync_empty_discovery_is_successful_noop(copilot_env: tuple[Config, SourcePaths]) -> None:
    config, paths = copilot_env
    assert sync(config, paths) == {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 0,
    }


def test_sync_missing_selected_session_returns_error(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    assert sync(config, paths, session_id="missing-session-id") == {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 1,
    }


def test_sync_discovers_hook_only_spool_session(copilot_env: tuple[Config, SourcePaths]) -> None:
    config, paths = copilot_env
    hook_only = "hook-only-session-00000001"
    record = _hook_record(observation_id="obs-hook-only", session_id=hook_only)
    enqueue_hook(config, paths, record)

    result = sync(config, paths, session_id=hook_only)
    # Missing transcript/database sources surface diagnostics without blocking the hook.
    assert result["records_written"] >= 1
    assert result["errors"] >= 1
    assert result["pending"] >= 1
    stored = stored_session_id(paths, hook_only)
    captured = list(iter_captured_records(config, stored))
    assert any(item["source_kind"] == "hook" for item in captured)
    assert read_spool(config, paths, hook_only) == []


# --- capture_session integration ---


def test_capture_session_commits_fixture_transcript_and_database(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)

    result = capture_session(config, paths, NATIVE_SESSION_ID)
    assert result["errors"] == 0
    assert result["records_written"] > 0

    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    kinds = {record["source_kind"] for record in captured}
    assert "transcript" in kinds
    assert "database" in kinds
    transcript_count = sum(1 for record in captured if record["source_kind"] == "transcript")
    assert transcript_count == 76


def test_capture_session_prepends_spool_records_before_source_records(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    hook = _hook_record(observation_id="obs-order")
    enqueue_hook(config, paths, hook)
    order: list[str] = []

    original_commit = capture_mod.commit_batch

    def tracking_commit(cfg: Config, p: SourcePaths, batch: SourceBatch) -> SyncResult:
        order.extend(record["source_kind"] for record in batch["records"])
        return original_commit(cfg, p, batch)

    monkeypatch.setattr(capture_mod, "commit_batch", tracking_commit)
    capture_session(config, paths, NATIVE_SESSION_ID)

    assert order[0] == "hook"


def test_capture_session_closes_when_spool_drains_start_then_end(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    names = iter(["a" * 32, "b" * 32])

    class _OrderedUUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    monkeypatch.setattr(
        "thirdeye.platforms.copilot.spool.uuid.uuid4",
        lambda: _OrderedUUID(next(names)),
    )
    start = parse_hook(
        "sessionStart",
        {
            "sessionId": NATIVE_SESSION_ID,
            "timestamp": 1789060102204,
            "cwd": "/fixture/workspace",
            "source": "new",
        },
        {},
        observed_at="2026-09-10T17:08:22.000Z",
        observation_id="spool-start",
    )
    end = parse_hook(
        "sessionEnd",
        {
            "sessionId": NATIVE_SESSION_ID,
            "timestamp": 1789060147875,
            "cwd": "/fixture/workspace",
            "reason": "user_exit",
        },
        {},
        observed_at="2026-09-10T17:08:47.000Z",
        observation_id="spool-end",
    )
    enqueue_hook(config, paths, start)
    enqueue_hook(config, paths, end)

    capture_session(config, paths, NATIVE_SESSION_ID)

    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    meta = read_meta(meta_path(session_dir(config.root, PLATFORM_NAME, stored)))
    assert meta is not None
    assert meta.status == "closed"
    assert meta.ended_at is not None


def test_capture_session_acks_spool_after_successful_commit(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    record = _hook_record(observation_id="obs-ack")
    enqueue_hook(config, paths, record)
    assert read_spool(config, paths, NATIVE_SESSION_ID)

    capture_session(config, paths, NATIVE_SESSION_ID)
    assert read_spool(config, paths, NATIVE_SESSION_ID) == []


def test_capture_session_does_not_ack_spool_when_commit_reports_errors(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    record = _hook_record(observation_id="obs-no-ack")
    enqueue_hook(config, paths, record)

    def failing_commit(_cfg: Config, _paths: SourcePaths, _batch: SourceBatch) -> SyncResult:
        return {
            "sessions": 1,
            "records_written": 0,
            "duplicate_records": 0,
            "pending": 1,
            "errors": 1,
        }

    monkeypatch.setattr(capture_mod, "commit_batch", failing_commit)
    capture_session(config, paths, NATIVE_SESSION_ID)
    assert len(read_spool(config, paths, NATIVE_SESSION_ID)) == 1


def test_capture_session_retries_once_after_stale_cursor_race(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)

    seed = capture_session(config, paths, NATIVE_SESSION_ID)
    assert seed["errors"] == 0
    assert load_cursor(config, paths, NATIVE_SESSION_ID) != {}

    commit_calls: list[int] = []
    original_commit = capture_mod.commit_batch

    def tracking_commit(cfg: Config, p: SourcePaths, batch: SourceBatch) -> SyncResult:
        commit_calls.append(len(batch["records"]))
        return original_commit(cfg, p, batch)

    monkeypatch.setattr(capture_mod, "commit_batch", tracking_commit)

    original_load = capture_mod.load_cursor

    def staged_load(cfg: Config, p: SourcePaths, native_id: str) -> dict[str, Any]:
        if not commit_calls:
            return {}
        return original_load(cfg, p, native_id)

    monkeypatch.setattr(capture_mod, "load_cursor", staged_load)
    result = capture_session(config, paths, NATIVE_SESSION_ID)

    assert len(commit_calls) == 2
    assert result["errors"] == 0


def test_sync_repeat_is_idempotent(copilot_env: tuple[Config, SourcePaths]) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)

    first = sync(config, paths, session_id=NATIVE_SESSION_ID)
    count_after_first = len(list(iter_captured_records(config, stored)))
    second = sync(config, paths, session_id=NATIVE_SESSION_ID)
    count_after_second = len(list(iter_captured_records(config, stored)))

    assert first["records_written"] > 0
    assert first["errors"] == 0
    assert second["records_written"] == 0
    assert second["errors"] == 0
    assert count_after_second == count_after_first


def test_late_database_rows_are_ingested_after_initial_transcript_only_sync(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)

    first = sync(config, paths, session_id=NATIVE_SESSION_ID)
    assert first["errors"] == 1
    assert first["pending"] == 1
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    before = list(iter_captured_records(config, stored))
    assert before
    assert not any(record["source_kind"] == "database" for record in before)

    _write_database(home, session_id=NATIVE_SESSION_ID)
    second = sync(config, paths, session_id=NATIVE_SESSION_ID)
    assert second["errors"] == 0
    assert second["records_written"] > 0

    after = list(iter_captured_records(config, stored))
    assert any(record["source_kind"] == "database" for record in after)


def test_one_unavailable_source_does_not_block_the_other(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)

    def fail_database(_paths: SourcePaths, _native: str, _cursor: dict[str, Any]) -> Any:
        raise OSError("database locked")

    with patch("thirdeye.platforms.copilot.sources.read_database", side_effect=fail_database):
        result = capture_session(config, paths, NATIVE_SESSION_ID)

    assert result["records_written"] > 0
    assert result["errors"] >= 1
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    assert any(
        record["source_kind"] == "transcript" for record in iter_captured_records(config, stored)
    )


# --- record_hook ---


def test_record_hook_enqueues_then_captures(copilot_env: tuple[Config, SourcePaths]) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)

    result = record_hook(
        config,
        paths,
        "agentStop",
        {
            "sessionId": NATIVE_SESSION_ID,
            "timestamp": 1789060105626,
            "cwd": "/fixture/workspace",
            "stopReason": "end_turn",
        },
        {"env": {"WB_PLAN": "probe"}},
    )

    assert result["errors"] == 0
    assert result["records_written"] >= 1
    assert read_spool(config, paths, NATIVE_SESSION_ID) == []
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    kinds = {record["source_kind"] for record in iter_captured_records(config, stored)}
    assert "hook" in kinds
    assert "transcript" in kinds


def test_record_hook_uses_caller_context_not_importer_environment(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    captured_context: dict[str, Any] = {}

    def capture_parse(
        event: str,
        payload: dict,
        context: dict,
        *,
        observed_at: str,
        observation_id: str,
    ) -> SourceRecord:
        captured_context.update(context)
        return parse_hook(
            event, payload, context, observed_at=observed_at, observation_id=observation_id
        )

    monkeypatch.setattr(capture_mod, "parse_hook", capture_parse)
    monkeypatch.setattr(capture_mod, "capture_session", lambda *_args, **_kwargs: _empty_result())

    record_hook(
        config,
        paths,
        "sessionStart",
        {"sessionId": NATIVE_SESSION_ID, "timestamp": 1, "cwd": "/x"},
        {"env": {"CUSTOM": "from-caller"}},
    )
    assert captured_context == {"env": {"CUSTOM": "from-caller"}}


def test_sync_full_fixture_writes_expected_record_volume(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)

    result = sync(config, paths, session_id=NATIVE_SESSION_ID)
    assert result["errors"] == 0
    assert result["records_written"] > 80

    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    assert len(captured) == result["records_written"]


def test_capture_session_reports_pending_when_transcript_exceeds_reader_limit(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    session_dir = home / "session-state" / NATIVE_SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    events = "".join(f'{{"id":"evt-{index}"}}\n' for index in range(1001))
    (session_dir / "events.jsonl").write_text(events, encoding="utf-8")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")
    _write_database(home, session_id=NATIVE_SESSION_ID)

    result = capture_session(config, paths, NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    assert result["pending"] >= 1
    assert len(captured) < 1001 + 20
    assert result["records_written"] == len(captured)


def test_sync_does_not_chase_growing_database_source(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    reads = {"n": 0}

    def growing_database(
        _paths: SourcePaths, _native: str, cursor: dict[str, Any], **_kwargs: Any
    ) -> Any:
        reads["n"] += 1
        if reads["n"] > 80:
            raise AssertionError("sync chased a growing database source")
        offset = 0
        if isinstance(cursor, dict) and isinstance(cursor.get("database_offset"), int):
            offset = cursor["database_offset"]
        return {
            "records": [
                _record(f"db/{offset + index}", source_kind="database") for index in range(1000)
            ],
            "next_cursor": {"database_generation": "gen-1", "database_offset": offset + 1000},
            "diagnostics": [],
            "cwd": None,
            "exhausted": False,
        }

    def empty_transcript(
        _paths: SourcePaths, _native: str, _cursor: dict[str, Any], **_kwargs: Any
    ) -> Any:
        return {
            "records": [],
            "next_cursor": {"byte_offset": 0, "snapshot_end": 0, "file_generation": "g"},
            "diagnostics": [],
            "cwd": None,
            "exhausted": True,
        }

    monkeypatch.setattr("thirdeye.platforms.copilot.sources.read_database", growing_database)
    monkeypatch.setattr("thirdeye.platforms.copilot.sources.read_transcript", empty_transcript)
    monkeypatch.setattr(
        "thirdeye.platforms.copilot.sources.discover_database_sessions",
        lambda _paths: [NATIVE_SESSION_ID],
    )
    monkeypatch.setattr(
        "thirdeye.platforms.copilot.sources.discover_transcripts", lambda _paths: []
    )

    result = sync(config, paths, session_id=NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = [
        record
        for record in iter_captured_records(config, stored)
        if record["source_kind"] == "database"
    ]

    assert reads["n"] <= 80
    assert result["sessions"] == 1
    assert 1000 <= len(captured) <= 40_000
    assert len(captured) == len({record["source_id"] for record in captured})


def test_sync_drains_paginated_database_snapshot(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.executemany(
            "INSERT INTO turns (session_id, turn_index, content, updated_at) VALUES (?, ?, ?, ?)",
            [
                (NATIVE_SESSION_ID, index, f"turn-{index}", "2026-09-10T17:08:10.000Z")
                for index in range(2, 1002)
            ],
        )
        connection.commit()
        baseline = connection.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ? UNION ALL "
            "SELECT COUNT(*) FROM turns WHERE session_id = ? UNION ALL "
            "SELECT COUNT(*) FROM assistant_usage_events WHERE session_id = ?",
            (NATIVE_SESSION_ID, NATIVE_SESSION_ID, NATIVE_SESSION_ID),
        ).fetchall()
        expected = sum(row[0] for row in baseline)
    finally:
        connection.close()

    result = sync(config, paths, session_id=NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    database_records = [record for record in captured if record["source_kind"] == "database"]

    assert result["errors"] == 0
    assert result["pending"] == 0
    assert len(database_records) == expected


def test_sync_drains_paginated_invocation_snapshot(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    session_dir = home / "session-state" / NATIVE_SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    events = "".join(f'{{"id":"evt-{index}"}}\n' for index in range(1001))
    (session_dir / "events.jsonl").write_text(events, encoding="utf-8")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")
    _write_database(home, session_id=NATIVE_SESSION_ID)

    result = sync(config, paths, session_id=NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    captured = list(iter_captured_records(config, stored))
    transcript_events = [record for record in captured if record["source_kind"] == "transcript"]
    assert result["pending"] == 0
    assert result["errors"] == 0
    assert len(transcript_events) == 1001
    assert result["sessions"] == 1


def test_capture_session_pending_follows_retry_batch_when_source_becomes_available(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    unavailable = _composed_batch(
        paths,
        diagnostics=[
            {
                "code": "copilot_database_missing",
                "message": "Copilot session database is not present",
            }
        ],
    )
    available = _composed_batch(paths, diagnostics=[], records=[_record("composed/retry")])
    reads = [unavailable, available]
    commits = 0

    def fake_read(_paths: SourcePaths, _native: str, _cursor: dict[str, Any]) -> SourceBatch:
        return reads.pop(0)

    def fake_commit(_cfg: Config, _paths: SourcePaths, _batch: SourceBatch) -> SyncResult:
        nonlocal commits
        commits += 1
        if commits == 1:
            return {
                "sessions": 1,
                "records_written": 2,
                "duplicate_records": 0,
                "pending": 1,
                "errors": 1,
            }
        return {
            "sessions": 1,
            "records_written": 1,
            "duplicate_records": 0,
            "pending": 0,
            "errors": 0,
        }

    def fake_load(_cfg: Config, _paths: SourcePaths, _native: str) -> dict[str, Any]:
        if commits == 0:
            return {}
        return {"transcript": {"byte_offset": 8}}

    monkeypatch.setattr(capture_mod, "read_batch", fake_read)
    monkeypatch.setattr(capture_mod, "commit_batch", fake_commit)
    monkeypatch.setattr(capture_mod, "load_cursor", fake_load)

    result = capture_session(config, paths, NATIVE_SESSION_ID)
    assert commits == 2
    assert result["pending"] == 0
    assert result["errors"] == 0
    assert result["records_written"] == 3


def test_capture_session_pending_follows_retry_batch_when_source_becomes_unavailable(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    available = _composed_batch(paths, diagnostics=[])
    unavailable = _composed_batch(
        paths,
        diagnostics=[
            {
                "code": "copilot_database_missing",
                "message": "Copilot session database is not present",
            }
        ],
    )
    reads = [available, unavailable]
    commits = 0

    def fake_read(_paths: SourcePaths, _native: str, _cursor: dict[str, Any]) -> SourceBatch:
        return reads.pop(0)

    def fake_commit(_cfg: Config, _paths: SourcePaths, _batch: SourceBatch) -> SyncResult:
        nonlocal commits
        commits += 1
        if commits == 1:
            return {
                "sessions": 1,
                "records_written": 0,
                "duplicate_records": 0,
                "pending": 1,
                "errors": 1,
            }
        return {
            "sessions": 1,
            "records_written": 1,
            "duplicate_records": 0,
            "pending": 0,
            "errors": 0,
        }

    def fake_load(_cfg: Config, _paths: SourcePaths, _native: str) -> dict[str, Any]:
        if commits == 0:
            return {}
        return {"transcript": {"byte_offset": 8}}

    monkeypatch.setattr(capture_mod, "read_batch", fake_read)
    monkeypatch.setattr(capture_mod, "commit_batch", fake_commit)
    monkeypatch.setattr(capture_mod, "load_cursor", fake_load)

    result = capture_session(config, paths, NATIVE_SESSION_ID)
    assert commits == 2
    assert result["pending"] >= 1
    assert result["errors"] >= 1
    assert result["records_written"] == 1


def test_capture_session_keeps_journal_recovery_counts_after_stale_retry(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)

    seed = commit_batch(
        config, paths, _batch(paths, [_record("key/a/seed")], next_cursor={"generation": 1})
    )
    assert seed["errors"] == 0

    with _fault_at("after_journal"):
        with pytest.raises(RuntimeError, match="injected fault"):
            commit_batch(
                config,
                paths,
                _batch(
                    paths,
                    [_record("key/a/journal-only")],
                    next_cursor={"generation": 2},
                    base_cursor={"generation": 1},
                ),
            )

    commit_calls: list[SyncResult] = []
    original_commit = capture_mod.commit_batch

    def tracking_commit(cfg: Config, p: SourcePaths, batch: SourceBatch) -> SyncResult:
        result = original_commit(cfg, p, batch)
        commit_calls.append(result)
        return result

    monkeypatch.setattr(capture_mod, "commit_batch", tracking_commit)

    original_load = capture_mod.load_cursor

    def staged_load(cfg: Config, p: SourcePaths, native_id: str) -> dict[str, Any]:
        if not commit_calls:
            return {"generation": 1}
        return original_load(cfg, p, native_id)

    monkeypatch.setattr(capture_mod, "load_cursor", staged_load)
    result = capture_session(config, paths, NATIVE_SESSION_ID)

    assert len(commit_calls) == 2
    assert commit_calls[0]["records_written"] >= 1
    assert result["records_written"] == (
        commit_calls[0]["records_written"] + commit_calls[1]["records_written"]
    )
    assert result["duplicate_records"] == (
        commit_calls[0]["duplicate_records"] + commit_calls[1]["duplicate_records"]
    )
    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_SESSION_ID)))
    assert any(record["source_id"] == "key/a/journal-only" for record in captured)


def test_sync_reports_discovery_failure_instead_of_empty_noop(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env

    def boom(_paths: SourcePaths) -> list[str]:
        raise OSError("database unreadable")

    with patch("thirdeye.platforms.copilot.sources.discover_database_sessions", boom):
        result = sync(config, paths)

    assert result["sessions"] == 0
    assert result["records_written"] == 0
    assert result["errors"] >= 1
    assert result["pending"] >= 1


def test_sync_selected_unreadable_database_is_not_reported_as_missing(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    Path(paths["database"]).write_text("this is not a sqlite database", encoding="utf-8")

    result = sync(config, paths, session_id=NATIVE_SESSION_ID)
    assert result != {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 1,
    }
    assert result["errors"] >= 1
    assert result["pending"] >= 1


def test_sync_incompatible_database_only_is_not_silent_noop(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    Path(paths["database"]).write_text("this is not a sqlite database", encoding="utf-8")

    result = sync(config, paths)
    assert result["errors"] >= 1
    assert result["pending"] >= 1
    assert result["sessions"] == 0
