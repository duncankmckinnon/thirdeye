"""Behavioral tests for Copilot source discovery and batch composition."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from thirdeye.platforms.copilot.database import discover_database_sessions
from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.sources import discover_sessions, read_batch, resolve_sources as reexported_resolve
from thirdeye.platforms.copilot.transcript import discover_transcripts
from thirdeye.platforms.copilot.types import SourcePaths, SourceRecord, SourceSlice

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
V1_SLICE = FIXTURES / "v1-cases" / "source-slice.json"
NATIVE_ID = "session-a"


def _paths(home: Path) -> SourcePaths:
    return resolve_sources(home)


def _write_transcript(home: Path, native_id: str, *, events: str = '{"id":"1"}\n', cwd: str | None = None) -> None:
    session_dir = home / "session-state" / native_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").write_text(events, encoding="utf-8")
    if cwd is not None:
        (session_dir / "workspace.yaml").write_text(f"cwd: {cwd}\n", encoding="utf-8")


def _write_database(
    home: Path,
    *,
    session_id: str = NATIVE_ID,
    cwd: str = "/db/workspace",
    extra_sessions: list[str] | None = None,
) -> None:
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
                created_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions (id, cwd, created_at) VALUES (?, ?, ?)",
            (session_id, cwd, "2026-09-10T17:08:00.000Z"),
        )
        connection.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, session_id, 1, "turn", "2026-09-10T17:08:10.000Z"),
        )
        for other in extra_sessions or ():
            connection.execute(
                "INSERT INTO sessions (id, cwd, created_at) VALUES (?, ?, ?)",
                (other, f"/tmp/{other}", "2026-09-10T17:08:00.000Z"),
            )
        connection.commit()
    finally:
        connection.close()


def _record(source_id: str, *, source_kind: str = "transcript") -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "native_session_id": NATIVE_ID,
        "ts": "2026-09-10T17:08:24.000Z",
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _slice(
    *,
    records: list[SourceRecord] | None = None,
    next_cursor: dict[str, Any] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    cwd: str | None = "/fixture/workspace",
    exhausted: bool = False,
) -> SourceSlice:
    return {
        "records": records or [],
        "next_cursor": next_cursor or {"byte_offset": 1},
        "diagnostics": diagnostics or [],
        "cwd": cwd,
        "exhausted": exhausted,
    }


# --- re-exports and discovery ---


def test_resolve_sources_is_reexported_from_identity(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    home.mkdir()
    assert reexported_resolve(home) == resolve_sources(home)


def test_discover_sessions_unions_transcript_and_database_sorted(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    _write_transcript(home, "session-b")
    _write_transcript(home, "session-a")
    _write_database(home, session_id="session-c", extra_sessions=["session-d"])

    assert discover_sessions(_paths(home)) == ["session-a", "session-b", "session-c", "session-d"]


def test_discover_sessions_returns_transcript_only_when_database_missing(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    _write_transcript(home, "session-z")
    assert discover_sessions(_paths(home)) == ["session-z"]


def test_discover_sessions_continues_when_one_discoverer_raises(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    _write_transcript(home, "session-a")

    def boom(_paths: SourcePaths) -> list[str]:
        raise OSError("database unreadable")

    with patch("thirdeye.platforms.copilot.sources.discover_database_sessions", boom):
        assert discover_sessions(_paths(home)) == ["session-a"]


def test_discover_sessions_continues_when_transcript_discover_raises(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    _write_database(home, session_id="session-db")

    def boom(_paths: SourcePaths) -> list[str]:
        raise ValueError("transcript layout invalid")

    with patch("thirdeye.platforms.copilot.sources.discover_transcripts", boom):
        assert discover_sessions(_paths(home)) == ["session-db"]


# --- read_batch composition ---


def test_read_batch_merges_transcript_and_database_records(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    _write_transcript(home, NATIVE_ID, events='{"id":"evt-1"}\n')
    _write_database(home, session_id=NATIVE_ID)
    paths = _paths(home)

    batch = read_batch(paths, NATIVE_ID, {})
    kinds = {record["source_kind"] for record in batch["records"]}
    assert "transcript" in kinds
    assert "database" in kinds
    assert batch["source_key"] == paths["source_key"]
    assert batch["native_session_id"] == NATIVE_ID


def test_read_batch_namespaces_reader_cursors_independently() -> None:
    transcript_cursor = {"byte_offset": 64, "file_generation": "gen-1"}
    database_cursor = {"row_id": 7, "table": "turns"}
    transcript = _slice(records=[_record("t/1")], next_cursor=transcript_cursor, cwd="/transcript/cwd")
    database = _slice(
        records=[_record("d/1", source_kind="database")],
        next_cursor=database_cursor,
        cwd="/database/cwd",
    )

    with (
        patch("thirdeye.platforms.copilot.sources.read_transcript", return_value=transcript),
        patch("thirdeye.platforms.copilot.sources.read_database", return_value=database),
    ):
        batch = read_batch(_paths(Path("/tmp/unused")), NATIVE_ID, {"transcript": {"stale": True}})

    assert batch["next_cursor"]["transcript"] == transcript_cursor
    assert batch["next_cursor"]["database"] == database_cursor
    assert batch["next_cursor"]["base_cursor"] == {"transcript": {"stale": True}}


def test_read_batch_passes_namespaced_cursors_to_each_reader(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    home.mkdir()
    paths = _paths(home)
    incoming = {
        "transcript": {"byte_offset": 10},
        "database": {"row_id": 3},
        "base_cursor": {"ignored": True},
    }
    seen: dict[str, dict[str, Any]] = {}

    def capture_transcript(_paths: SourcePaths, _native: str, cursor: dict[str, Any]) -> SourceSlice:
        seen["transcript"] = cursor
        return _slice(exhausted=True)

    def capture_database(_paths: SourcePaths, _native: str, cursor: dict[str, Any]) -> SourceSlice:
        seen["database"] = cursor
        return _slice(exhausted=True)

    with (
        patch("thirdeye.platforms.copilot.sources.read_transcript", side_effect=capture_transcript),
        patch("thirdeye.platforms.copilot.sources.read_database", side_effect=capture_database),
    ):
        read_batch(paths, NATIVE_ID, incoming)

    assert seen["transcript"] == {"byte_offset": 10}
    assert seen["database"] == {"row_id": 3}


def test_read_batch_surfaces_transcript_failure_as_diagnostic_while_database_progresses(
    tmp_path: Path,
) -> None:
    home = tmp_path / "copilot"
    _write_database(home, session_id=NATIVE_ID)
    paths = _paths(home)

    def fail_transcript(_paths: SourcePaths, _native: str, _cursor: dict[str, Any]) -> SourceSlice:
        raise OSError("events.jsonl locked")

    with patch("thirdeye.platforms.copilot.sources.read_transcript", side_effect=fail_transcript):
        batch = read_batch(paths, NATIVE_ID, {})

    assert any(item["code"] == "copilot_transcript_read_failed" for item in batch["diagnostics"])
    assert any(record["source_kind"] == "database" for record in batch["records"])


def test_read_batch_surfaces_database_failure_as_diagnostic_while_transcript_progresses(
    tmp_path: Path,
) -> None:
    home = tmp_path / "copilot"
    _write_transcript(home, NATIVE_ID)
    paths = _paths(home)

    def fail_database(_paths: SourcePaths, _native: str, _cursor: dict[str, Any]) -> SourceSlice:
        raise ValueError("database schema mismatch")

    with patch("thirdeye.platforms.copilot.sources.read_database", side_effect=fail_database):
        batch = read_batch(paths, NATIVE_ID, {})

    assert any(item["code"] == "copilot_database_read_failed" for item in batch["diagnostics"])
    assert any(record["source_kind"] == "transcript" for record in batch["records"])


def test_read_batch_prefers_transcript_cwd_over_database() -> None:
    transcript = _slice(cwd="/from/transcript")
    database = _slice(cwd="/from/database", records=[_record("d/1", source_kind="database")])

    with (
        patch("thirdeye.platforms.copilot.sources.read_transcript", return_value=transcript),
        patch("thirdeye.platforms.copilot.sources.read_database", return_value=database),
    ):
        batch = read_batch(_paths(Path("/tmp/unused")), NATIVE_ID, {})

    assert batch["cwd"] == "/from/transcript"


def test_read_batch_falls_back_to_database_cwd_when_transcript_missing() -> None:
    transcript = _slice(cwd=None)
    database = _slice(cwd="/from/database", records=[_record("d/1", source_kind="database")])

    with (
        patch("thirdeye.platforms.copilot.sources.read_transcript", return_value=transcript),
        patch("thirdeye.platforms.copilot.sources.read_database", return_value=database),
    ):
        batch = read_batch(_paths(Path("/tmp/unused")), NATIVE_ID, {})

    assert batch["cwd"] == "/from/database"


def test_read_batch_fixture_slice_shape_is_compatible() -> None:
    """The v1 source-slice fixture must compose into a SourceBatch boundary."""
    fixture = json.loads(V1_SLICE.read_text(encoding="utf-8"))
    transcript = _slice(
        records=fixture["records"],
        next_cursor=fixture["next_cursor"],
        diagnostics=fixture["diagnostics"],
        cwd=fixture["cwd"],
        exhausted=fixture["exhausted"],
    )
    database = _slice(exhausted=True)

    with (
        patch("thirdeye.platforms.copilot.sources.read_transcript", return_value=transcript),
        patch("thirdeye.platforms.copilot.sources.read_database", return_value=database),
    ):
        batch = read_batch(_paths(Path("/tmp/unused")), NATIVE_ID, {})

    assert len(batch["records"]) == 1
    assert batch["records"][0]["source_kind"] == "transcript"
    assert batch["diagnostics"] == []


def test_discover_sessions_delegates_to_underlying_readers(tmp_path: Path) -> None:
    home = tmp_path / "copilot"
    _write_transcript(home, "only-transcript")
    paths = _paths(home)
    assert discover_transcripts(paths) == ["only-transcript"]
    assert discover_database_sessions(paths) == []
    assert discover_sessions(paths) == ["only-transcript"]
