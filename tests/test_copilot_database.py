"""Behavioral tests for the Copilot SQLite database reader."""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import threading
import warnings
from pathlib import Path
from typing import Any

import pytest

from thirdeye.platforms.copilot.database import discover_database_sessions, read_database
from thirdeye.platforms.copilot.identity import resolve_sources

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
CLI_FIXTURE = FIXTURES / "cli-1.0.83"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _create_standard_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            cwd TEXT,
            created_at TEXT
        );
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


def _seed_session(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    cwd: str = "/tmp/workspace",
) -> None:
    connection.execute(
        "INSERT INTO sessions (id, cwd, created_at) VALUES (?, ?, ?)",
        (session_id, cwd, "2026-09-10T17:08:00.000Z"),
    )


def _write_database(
    home: Path,
    *,
    session_id: str = "session-a",
    cwd: str = "/tmp/workspace",
    turns: list[tuple[int, str]] | None = None,
    usage_rows: list[dict[str, Any]] | None = None,
    extra_sessions: list[str] | None = None,
    schema_sql: str | None = None,
    journal_mode: str = "WAL",
    seed_session: bool = True,
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        if schema_sql is not None:
            connection.executescript(schema_sql)
        else:
            _create_standard_schema(connection)
        if journal_mode:
            connection.execute(f"PRAGMA journal_mode={journal_mode}")
        if seed_session:
            _seed_session(connection, session_id, cwd=cwd)
            for other in extra_sessions or ():
                _seed_session(connection, other, cwd=f"/tmp/{other}")
        for turn_id, content in turns or ():
            connection.execute(
                "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (turn_id, session_id, turn_id, content, "2026-09-10T17:08:10.000Z"),
            )
        for row in usage_rows or ():
            columns = ", ".join(row)
            placeholders = ", ".join("?" for _ in row)
            connection.execute(
                f"INSERT INTO assistant_usage_events ({columns}) VALUES ({placeholders})",
                tuple(row.values()),
            )
        connection.commit()
    finally:
        connection.close()
    return database


def _paths(home: Path) -> dict[str, str]:
    return resolve_sources(home)


def _records_for_table(records: list[dict[str, Any]], table: str) -> list[dict[str, Any]]:
    return [record for record in records if record["payload"]["table"] == table]


def _collect_all(
    paths: dict[str, str],
    native_id: str,
    *,
    max_records: int = 1000,
) -> list[dict[str, Any]]:
    cursor: dict[str, Any] = {}
    collected: list[dict[str, Any]] = []
    for _ in range(10_000):
        slice_ = read_database(paths, native_id, cursor, max_records=max_records)
        collected.extend(slice_["records"])
        if slice_["exhausted"]:
            return collected
        cursor = slice_["next_cursor"]
    raise AssertionError("database reader did not exhaust within 10000 bounded reads")


# --- module boundaries ---


def test_database_module_has_no_forbidden_imports():
    import thirdeye.platforms.copilot.database as database

    source = Path(database.__file__).read_text(encoding="utf-8")
    assert "UsageStore" not in source
    assert "usage_store" not in source
    assert "from thirdeye.store" not in source
    assert "import thirdeye.store" not in source
    assert "transcript" not in source


# --- discovery ---


def test_discover_database_sessions_returns_sorted_unique_ids(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(
        home,
        session_id="session-b",
        extra_sessions=["session-a", "session-c"],
    )
    paths = _paths(home)
    assert discover_database_sessions(paths) == ["session-a", "session-b", "session-c"]


def test_discover_database_sessions_empty_when_database_missing(tmp_path: Path):
    home = tmp_path / "copilot"
    home.mkdir()
    assert discover_database_sessions(_paths(home)) == []


def test_discover_database_sessions_empty_when_sessions_table_missing(tmp_path: Path):
    home = tmp_path / "copilot"
    home.mkdir()
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id TEXT)")
    connection.commit()
    connection.close()
    assert discover_database_sessions(_paths(home)) == []


# --- missing / empty sources ---


def test_read_database_missing_file_reports_diagnostic(tmp_path: Path):
    home = tmp_path / "copilot"
    home.mkdir()
    paths = _paths(home)
    slice_ = read_database(paths, "session-a", {})
    assert slice_["records"] == []
    assert slice_["exhausted"] is True
    assert slice_["cwd"] is None
    assert slice_["diagnostics"] == [
        {
            "code": "copilot_database_missing",
            "message": "Copilot session database is not present",
            "path": paths["database"],
        }
    ]


def test_read_database_non_file_path_reports_unreadable(tmp_path: Path):
    home = tmp_path / "copilot"
    home.mkdir()
    database = home / "session-store.db"
    database.mkdir()
    paths = _paths(home)
    slice_ = read_database(paths, "session-a", {})
    assert slice_["records"] == []
    assert slice_["diagnostics"][0]["code"] == "copilot_database_unreadable"


def test_read_database_empty_session_returns_only_session_row(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(home, session_id="session-a")
    slice_ = read_database(_paths(home), "session-a", {})
    assert len(slice_["records"]) == 1
    assert slice_["records"][0]["payload"]["table"] == "sessions"
    assert slice_["exhausted"] is True
    assert slice_["cwd"] == "/tmp/workspace"


def test_read_database_unknown_session_is_empty_exhausted(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(home, session_id="session-a")
    slice_ = read_database(_paths(home), "missing-session", {})
    assert slice_["records"] == []
    assert slice_["exhausted"] is True
    assert slice_["cwd"] is None


# --- all three tables ---


def test_read_database_reads_sessions_turns_and_usage_events(tmp_path: Path):
    home = tmp_path / "copilot"
    usage_row = {
        "id": 13,
        "session_id": "session-a",
        "turn_index": 0,
        "agent_id": None,
        "parent_tool_call_id": None,
        "model": "gpt-5.6-luna",
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "total_nano_aiu": 1000,
        "request_multiplier": 1.0,
        "duration_ms": 100,
        "time_to_first_token_ms": 50.0,
        "output_ttft_ms": 50.0,
        "inter_token_latency_ms": None,
        "initiator": "user",
        "api_endpoint": "ws:/responses",
        "reasoning_effort": "medium",
        "finish_reason": "stop",
        "content_filter_triggered": 0,
        "token_details_json": "[]",
        "created_at": "2026-09-10T17:08:24.498Z",
    }
    _write_database(
        home,
        session_id="session-a",
        turns=[(1, "first turn"), (2, "second turn")],
        usage_rows=[usage_row],
    )
    records = _collect_all(_paths(home), "session-a")
    tables = {record["payload"]["table"] for record in records}
    assert tables == {"assistant_usage_events", "sessions", "turns"}
    assert len(_records_for_table(records, "turns")) == 2
    assert len(_records_for_table(records, "assistant_usage_events")) == 1


def test_cli_fixture_usage_rows_are_readable_from_database(tmp_path: Path):
    home = tmp_path / "copilot"
    usage_rows = _load_json(CLI_FIXTURE / "assistant-usage-events.json")
    _write_database(
        home,
        session_id=NATIVE_SESSION_ID,
        cwd="/tmp/probe",
        turns=[(0, "turn-0"), (1, "turn-1")],
        usage_rows=usage_rows,
    )
    records = _collect_all(_paths(home), NATIVE_SESSION_ID)
    usage_records = _records_for_table(records, "assistant_usage_events")
    assert len(usage_records) == 6
    assert {record["payload"]["row"]["id"] for record in usage_records} == {
        13,
        14,
        15,
        16,
        17,
        18,
    }
    assert usage_records[0]["source_kind"] == "database"
    assert usage_records[0]["ts"] == "2026-09-10T17:08:24.498Z"
    assert usage_records[0]["native_session_id"] == NATIVE_SESSION_ID


# --- source record shape ---


def test_source_record_includes_revision_and_generation(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(home, session_id="session-a", turns=[(7, "first")])
    record = read_database(_paths(home), "session-a", {})["records"][0]
    assert record["source_id"].startswith(f"copilot-db:{_paths(home)['source_key']}:session-a:")
    assert record["locator"]["table"] == "sessions"
    assert record["locator"]["content_revision"].startswith("sha256:")
    assert record["locator"]["generation"].startswith("sha256:")
    assert record["payload"]["row"]["id"] == "session-a"


# --- WAL / live changes ---


def test_reads_uncheckpointed_wal_commits(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(home, session_id="session-a", turns=[(1, "checkpointed")])
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (2, "session-a", 2, "wal-only", "2026-09-10T17:08:20.000Z"),
        )
        writer.execute(
            "INSERT INTO assistant_usage_events (id, session_id, turn_index, model, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (99, "session-a", 2, "gpt-test", "2026-09-10T17:08:21.000Z"),
        )
        writer.commit()
        wal_path = Path(f"{database}-wal")
        assert wal_path.is_file()
        assert wal_path.stat().st_size > 0

        records = _collect_all(_paths(home), "session-a")
        turn_contents = {
            record["payload"]["row"]["content"]
            for record in records
            if record["payload"]["table"] == "turns"
        }
        assert turn_contents == {"checkpointed", "wal-only"}
        assert any(
            record["payload"]["table"] == "assistant_usage_events"
            and record["payload"]["row"]["id"] == 99
            for record in records
        )
        assert wal_path.is_file()
        assert wal_path.stat().st_size > 0
    finally:
        writer.close()


def test_late_database_rows_visible_without_transcript_changes(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(home, session_id="session-a", turns=[(1, "initial")])
    first = _collect_all(_paths(home), "session-a")

    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO assistant_usage_events (id, session_id, turn_index, model, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (42, "session-a", 1, "gpt-late", "2026-09-10T17:09:00.000Z"),
    )
    connection.commit()
    connection.close()

    second = _collect_all(_paths(home), "session-a")
    assert len(second) == len(first) + 1
    assert any(
        record["payload"]["table"] == "assistant_usage_events"
        and record["payload"]["row"]["model"] == "gpt-late"
        for record in second
    )


# --- revisions and row reuse ---


def test_updated_turn_produces_new_content_revision(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(home, session_id="session-a", turns=[(7, "first")])
    paths = _paths(home)

    before = _records_for_table(_collect_all(paths, "session-a"), "turns")[0]
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE turns SET content = ?, updated_at = ? WHERE id = ?",
        ("updated", "2026-09-10T17:10:00.000Z", 7),
    )
    connection.commit()
    connection.close()

    after = _records_for_table(_collect_all(paths, "session-a"), "turns")[0]
    assert before["locator"]["primary_key"] == after["locator"]["primary_key"] == 7
    assert before["locator"]["content_revision"] != after["locator"]["content_revision"]
    assert before["source_id"] != after["source_id"]
    assert after["payload"]["row"]["content"] == "updated"


def test_unchanged_resnapshot_keeps_stable_source_id(tmp_path: Path):
    home = tmp_path / "copilot"
    paths = _paths(home)
    _write_database(home, session_id="session-a", turns=[(7, "stable")])
    first = read_database(paths, "session-a", {})["records"]
    second = read_database(paths, "session-a", {})["records"]
    turn_first = _records_for_table(first, "turns")[0]
    turn_second = _records_for_table(second, "turns")[0]
    assert turn_first["source_id"] == turn_second["source_id"]
    assert turn_first["locator"]["content_revision"] == turn_second["locator"]["content_revision"]


def test_database_replacement_changes_generation_and_resets_cursor(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(home, session_id="session-a", turns=[(1, "a"), (2, "b")])
    paths = _paths(home)

    first = read_database(paths, "session-a", {}, max_records=1)
    old_generation = first["next_cursor"]["database_generation"]
    assert first["exhausted"] is False

    replacement_home = tmp_path / "replacement-copilot"
    replacement = _write_database(
        replacement_home,
        session_id="session-a",
        turns=[(1, "replacement")],
        journal_mode="DELETE",
    )
    os.replace(replacement, database)

    after_replace = read_database(paths, "session-a", first["next_cursor"], max_records=10)
    assert after_replace["next_cursor"]["database_generation"] != old_generation
    assert after_replace["next_cursor"]["database_offset"] == len(after_replace["records"])
    assert {
        record["payload"]["row"]["content"]
        for record in _records_for_table(after_replace["records"], "turns")
    } == {"replacement"}


# --- pagination ---


def test_read_database_paginates_with_cursor(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(
        home,
        session_id="session-a",
        turns=[(index, f"turn-{index}") for index in range(1, 6)],
        journal_mode="DELETE",
    )
    paths = _paths(home)

    page_one = read_database(paths, "session-a", {}, max_records=2)
    assert len(page_one["records"]) == 2
    assert page_one["exhausted"] is False
    assert page_one["next_cursor"]["database_offset"] == 2

    page_two = read_database(paths, "session-a", page_one["next_cursor"], max_records=2)
    assert len(page_two["records"]) == 2
    assert page_two["exhausted"] is False

    remaining = read_database(paths, "session-a", page_two["next_cursor"], max_records=10)
    assert len(remaining["records"]) == 2  # two remaining turns after five total rows
    assert remaining["exhausted"] is True


def test_stale_cursor_resets_when_generation_changes(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(home, session_id="session-a", turns=[(1, "one"), (2, "two")])
    paths = _paths(home)
    first = read_database(paths, "session-a", {}, max_records=1)
    stale_cursor = {
        "database_generation": "sha256:deadbeef",
        "database_offset": 99,
    }

    os.remove(database)
    _write_database(home, session_id="session-a", turns=[(1, "fresh")])

    slice_ = read_database(paths, "session-a", stale_cursor, max_records=10)
    assert slice_["next_cursor"]["database_offset"] == len(slice_["records"])


# --- schema diagnostics ---


def test_missing_table_reports_diagnostic_but_reads_available_tables(tmp_path: Path):
    home = tmp_path / "copilot"
    home.mkdir()
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT);
        CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT);
        """
    )
    connection.execute("INSERT INTO sessions VALUES ('session-a', '/tmp')")
    connection.execute("INSERT INTO turns VALUES (1, 'session-a', 'only-turn')")
    connection.commit()
    connection.close()

    slice_ = read_database(_paths(home), "session-a", {})
    codes = {diag["code"] for diag in slice_["diagnostics"]}
    assert "copilot_database_table_missing" in codes
    tables = {record["payload"]["table"] for record in slice_["records"]}
    assert tables == {"sessions", "turns"}


def test_missing_session_column_reports_diagnostic(tmp_path: Path):
    home = tmp_path / "copilot"
    schema = """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT);
        CREATE TABLE turns (id INTEGER PRIMARY KEY, content TEXT);
        CREATE TABLE assistant_usage_events (id INTEGER PRIMARY KEY, model TEXT);
    """
    _write_database(home, session_id="session-a", schema_sql=schema, seed_session=False)
    connection = sqlite3.connect(home / "session-store.db")
    connection.execute("INSERT INTO sessions VALUES ('session-a', '/tmp')")
    connection.commit()
    connection.close()
    slice_ = read_database(_paths(home), "session-a", {})
    codes = {diag["code"] for diag in slice_["diagnostics"]}
    assert "copilot_database_missing_session_column" in codes
    assert {record["payload"]["table"] for record in slice_["records"]} == {"sessions"}


def test_missing_primary_key_reports_diagnostic(tmp_path: Path):
    home = tmp_path / "copilot"
    schema = """
        CREATE TABLE sessions (session_id TEXT, cwd TEXT);
        CREATE TABLE turns (session_id TEXT, content TEXT);
        CREATE TABLE assistant_usage_events (session_id TEXT, model TEXT);
    """
    _write_database(home, session_id="session-a", schema_sql=schema, seed_session=False)
    connection = sqlite3.connect(home / "session-store.db")
    connection.execute("INSERT INTO sessions VALUES ('session-a', '/tmp')")
    connection.commit()
    connection.close()
    slice_ = read_database(_paths(home), "session-a", {})
    codes = {diag["code"] for diag in slice_["diagnostics"]}
    assert "copilot_database_missing_primary_key" in codes
    assert slice_["records"] == []


def test_optional_columns_are_preserved_when_present(tmp_path: Path):
    home = tmp_path / "copilot"
    schema = """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, extra_flag INTEGER);
        CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT);
        CREATE TABLE assistant_usage_events (id INTEGER PRIMARY KEY, session_id TEXT, model TEXT);
    """
    _write_database(home, session_id="session-a", schema_sql=schema, seed_session=False)
    connection = sqlite3.connect(home / "session-store.db")
    connection.execute("INSERT INTO sessions (id, cwd, extra_flag) VALUES ('session-a', '/tmp', 1)")
    connection.commit()
    connection.close()

    session_record = _records_for_table(_collect_all(_paths(home), "session-a"), "sessions")[0]
    assert session_record["payload"]["row"]["extra_flag"] == 1


# --- locking ---


def test_busy_database_returns_actionable_diagnostic(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(home, session_id="session-a", turns=[(1, "locked")])
    paths = _paths(home)
    hold = threading.Event()
    release = threading.Event()
    error: list[str] = []

    def hold_exclusive_lock() -> None:
        connection = sqlite3.connect(database, timeout=30)
        try:
            connection.execute("BEGIN EXCLUSIVE")
            hold.set()
            release.wait(timeout=5)
        finally:
            connection.close()

    thread = threading.Thread(target=hold_exclusive_lock, daemon=True)
    thread.start()
    assert hold.wait(timeout=2)

    slice_ = read_database(paths, "session-a", {})
    release.set()
    thread.join(timeout=2)

    if any(diag["code"] == "copilot_database_busy" for diag in slice_["diagnostics"]):
        assert slice_["records"] == []
        assert slice_["exhausted"] is True
    else:
        # Some platforms may not block read-only URIs on EXCLUSIVE; accept successful read.
        assert slice_["records"]


# --- validation ---


def test_read_database_rejects_invalid_native_id(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(home, session_id="session-a")
    with pytest.raises(ValueError, match="native session ID"):
        read_database(_paths(home), "../escape", {})


def test_read_database_rejects_invalid_max_records(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(home, session_id="session-a")
    with pytest.raises(ValueError, match="max_records"):
        read_database(_paths(home), "session-a", {}, max_records=0)


# --- special values ---


def test_bytes_column_is_base64_encoded_in_payload(tmp_path: Path):
    home = tmp_path / "copilot"
    schema = """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT);
        CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id TEXT, blob_data BLOB);
        CREATE TABLE assistant_usage_events (id INTEGER PRIMARY KEY, session_id TEXT);
    """
    _write_database(home, session_id="session-a", schema_sql=schema, seed_session=False)
    connection = sqlite3.connect(home / "session-store.db")
    connection.execute("INSERT INTO sessions VALUES ('session-a', '/tmp')")
    connection.execute(
        "INSERT INTO turns (id, session_id, blob_data) VALUES (?, ?, ?)",
        (1, "session-a", b"\x00\xff"),
    )
    connection.commit()
    connection.close()

    turn = _records_for_table(_collect_all(_paths(home), "session-a"), "turns")[0]
    assert turn["payload"]["row"]["blob_data"] == {
        "encoding": "base64",
        "data": "AP8=",
    }


# --- review fixes: discovery, pragma PK, cursor, diagnostics, connections ---


def test_discover_database_sessions_unions_ids_from_all_allowed_tables(tmp_path: Path):
    home = tmp_path / "copilot"
    home.mkdir()
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT);
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                user_message TEXT
            );
            CREATE TABLE assistant_usage_events (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                model TEXT
            );
            """
        )
        connection.execute("INSERT INTO sessions VALUES ('in-sessions', '/tmp')")
        connection.execute("INSERT INTO turns VALUES (1, 'in-turns-only', 'hello')")
        connection.execute("INSERT INTO assistant_usage_events VALUES (1, 'in-usage-only', 'gpt')")
        connection.commit()
    finally:
        connection.close()

    assert discover_database_sessions(_paths(home)) == [
        "in-sessions",
        "in-turns-only",
        "in-usage-only",
    ]


def test_discover_database_sessions_without_sessions_table_uses_other_tables(
    tmp_path: Path,
):
    home = tmp_path / "copilot"
    home.mkdir()
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                user_message TEXT
            );
            CREATE TABLE assistant_usage_events (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                model TEXT
            );
            """
        )
        connection.execute("INSERT INTO turns VALUES (1, 'from-turns', 'hello')")
        connection.execute("INSERT INTO assistant_usage_events VALUES (1, 'from-usage', 'gpt')")
        connection.commit()
    finally:
        connection.close()

    assert discover_database_sessions(_paths(home)) == ["from-turns", "from-usage"]


def test_observed_cli_schema_preserves_session_and_turn_columns(tmp_path: Path):
    home = tmp_path / "copilot"
    schema = """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            cwd TEXT,
            repository TEXT,
            host_type TEXT,
            branch TEXT,
            summary TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            user_message TEXT,
            assistant_response TEXT,
            timestamp TEXT
        );
        CREATE TABLE assistant_usage_events (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_index INTEGER,
            model TEXT,
            created_at TEXT
        );
    """
    _write_database(home, session_id="session-a", schema_sql=schema, seed_session=False)
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "INSERT INTO sessions "
            "(id, cwd, repository, host_type, branch, summary, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "session-a",
                "/tmp/probe",
                "github.com/example/repo",
                "github",
                "main",
                "sum two files",
                "2026-09-10T17:08:00.000Z",
                "2026-09-10T17:08:50.000Z",
            ),
        )
        connection.execute(
            "INSERT INTO turns "
            "(id, session_id, turn_index, user_message, assistant_response, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                1,
                "session-a",
                0,
                "read alpha and beta",
                "42",
                "2026-09-10T17:08:10.000Z",
            ),
        )
        connection.execute(
            "INSERT INTO assistant_usage_events "
            "(id, session_id, turn_index, model, created_at) VALUES (?, ?, ?, ?, ?)",
            (13, "session-a", 0, "gpt-5.6-luna", "2026-09-10T17:08:24.498Z"),
        )
        connection.commit()
    finally:
        connection.close()

    records = _collect_all(_paths(home), "session-a")
    session = _records_for_table(records, "sessions")[0]
    turn = _records_for_table(records, "turns")[0]
    usage = _records_for_table(records, "assistant_usage_events")[0]
    assert session["payload"]["row"]["repository"] == "github.com/example/repo"
    assert session["payload"]["row"]["cwd"] == "/tmp/probe"
    assert turn["payload"]["row"]["user_message"] == "read alpha and beta"
    assert turn["payload"]["row"]["assistant_response"] == "42"
    assert turn["locator"]["primary_key"] == 1
    assert usage["payload"]["row"]["model"] == "gpt-5.6-luna"


def test_primary_key_uses_pragma_pk_when_column_is_not_named_id(tmp_path: Path):
    home = tmp_path / "copilot"
    schema = """
        CREATE TABLE sessions (session_pk TEXT PRIMARY KEY, cwd TEXT);
        CREATE TABLE turns (
            turn_pk INTEGER PRIMARY KEY,
            session_id TEXT,
            user_message TEXT
        );
        CREATE TABLE assistant_usage_events (
            usage_pk INTEGER PRIMARY KEY,
            session_id TEXT,
            model TEXT
        );
    """
    _write_database(home, session_id="session-a", schema_sql=schema, seed_session=False)
    connection = sqlite3.connect(home / "session-store.db")
    try:
        connection.execute("INSERT INTO sessions VALUES ('session-a', '/tmp')")
        connection.execute("INSERT INTO turns VALUES (9, 'session-a', 'hello')")
        connection.execute("INSERT INTO assistant_usage_events VALUES (3, 'session-a', 'gpt')")
        connection.commit()
    finally:
        connection.close()

    records = _collect_all(_paths(home), "session-a")
    session = _records_for_table(records, "sessions")[0]
    turn = _records_for_table(records, "turns")[0]
    usage = _records_for_table(records, "assistant_usage_events")[0]
    assert session["locator"]["primary_key"] == "session-a"
    assert turn["locator"]["primary_key"] == 9
    assert usage["locator"]["primary_key"] == 3
    assert "session-a" in session["source_id"]


def test_composite_primary_key_is_row_identity(tmp_path: Path):
    home = tmp_path / "copilot"
    schema = """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT);
        CREATE TABLE turns (
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            user_message TEXT,
            assistant_response TEXT,
            PRIMARY KEY (session_id, turn_index)
        );
        CREATE TABLE assistant_usage_events (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            model TEXT
        );
    """
    _write_database(home, session_id="session-a", schema_sql=schema, seed_session=False)
    connection = sqlite3.connect(home / "session-store.db")
    try:
        connection.execute("INSERT INTO sessions VALUES ('session-a', '/tmp')")
        connection.execute("INSERT INTO turns VALUES ('session-a', 0, 'first', 'reply-one')")
        connection.execute("INSERT INTO turns VALUES ('session-a', 1, 'second', 'reply-two')")
        connection.commit()
    finally:
        connection.close()

    turns = _records_for_table(_collect_all(_paths(home), "session-a"), "turns")
    assert [record["locator"]["primary_key"] for record in turns] == [
        {"session_id": "session-a", "turn_index": 0},
        {"session_id": "session-a", "turn_index": 1},
    ]
    assert turns[0]["source_id"] != turns[1]["source_id"]


def test_unrelated_wal_write_does_not_reset_pagination(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(
        home,
        session_id="session-a",
        turns=[(index, f"turn-{index}") for index in range(1, 6)],
        extra_sessions=["session-b"],
    )
    paths = _paths(home)
    page_one = read_database(paths, "session-a", {}, max_records=2)
    assert page_one["exhausted"] is False
    first_ids = [record["source_id"] for record in page_one["records"]]

    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (50, "session-b", 1, "other-session", "2026-09-10T17:11:00.000Z"),
        )
        writer.commit()
        wal_path = Path(f"{database}-wal")
        assert wal_path.is_file()
        assert wal_path.stat().st_size > 0

        page_two = read_database(paths, "session-a", page_one["next_cursor"], max_records=2)
        second_ids = [record["source_id"] for record in page_two["records"]]
        assert page_two["records"]
        assert second_ids != first_ids
        assert (
            page_two["next_cursor"]["database_generation"]
            == page_one["next_cursor"]["database_generation"]
        )
        assert page_two["next_cursor"]["database_offset"] == 4
    finally:
        writer.close()


def test_same_session_writes_do_not_starve_later_rows(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(
        home,
        session_id="session-a",
        turns=[(index, f"turn-{index}") for index in range(1, 6)],
    )
    paths = _paths(home)
    page_one = read_database(paths, "session-a", {}, max_records=2)
    assert page_one["exhausted"] is False
    first_ids = [record["source_id"] for record in page_one["records"]]

    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "UPDATE turns SET content = ?, updated_at = ? WHERE id = ?",
            ("updated-turn-1", "2026-09-10T17:13:00.000Z", 1),
        )
        writer.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (99, "session-a", 99, "late-same-session", "2026-09-10T17:13:01.000Z"),
        )
        writer.execute(
            "INSERT INTO assistant_usage_events (id, session_id, turn_index, model, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (50, "session-a", 99, "gpt-live", "2026-09-10T17:13:02.000Z"),
        )
        writer.commit()
        wal_path = Path(f"{database}-wal")
        assert wal_path.is_file()
        assert wal_path.stat().st_size > 0

        page_two = read_database(paths, "session-a", page_one["next_cursor"], max_records=2)
        second_ids = [record["source_id"] for record in page_two["records"]]
        assert page_two["records"]
        assert second_ids != first_ids
        assert (
            page_two["next_cursor"]["database_generation"]
            == page_one["next_cursor"]["database_generation"]
        )
        assert page_two["next_cursor"]["database_offset"] == 4

        collected = list(page_one["records"]) + list(page_two["records"])
        cursor = page_two["next_cursor"]
        for _ in range(20):
            slice_ = read_database(paths, "session-a", cursor, max_records=2)
            collected.extend(slice_["records"])
            cursor = slice_["next_cursor"]
            if slice_["exhausted"]:
                break
        else:
            raise AssertionError("pagination did not exhaust after same-session writes")

        turn_contents = {
            record["payload"]["row"]["content"]
            for record in collected
            if record["payload"]["table"] == "turns"
        }
        assert "turn-5" in turn_contents
        assert "late-same-session" in turn_contents
        assert any(
            record["payload"]["table"] == "assistant_usage_events"
            and record["payload"]["row"]["model"] == "gpt-live"
            for record in collected
        )
    finally:
        writer.close()


def test_repeated_same_session_inserts_still_reach_later_rows(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(
        home,
        session_id="session-a",
        turns=[(index, f"turn-{index}") for index in range(1, 9)],
    )
    paths = _paths(home)
    cursor: dict[str, Any] = {}
    collected: list[dict[str, Any]] = []
    first_page_ids: list[str] | None = None
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        for index in range(20):
            slice_ = read_database(paths, "session-a", cursor, max_records=2)
            page_ids = [record["source_id"] for record in slice_["records"]]
            if first_page_ids is None:
                first_page_ids = page_ids
            else:
                assert page_ids != first_page_ids
            collected.extend(slice_["records"])
            writer.execute(
                "INSERT INTO assistant_usage_events "
                "(id, session_id, turn_index, model, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    100 + index,
                    "session-a",
                    index,
                    f"gpt-live-{index}",
                    "2026-09-10T17:14:00.000Z",
                ),
            )
            writer.commit()
            if slice_["exhausted"]:
                break
            cursor = slice_["next_cursor"]
        else:
            raise AssertionError("continuous same-session inserts starved later rows")
    finally:
        writer.close()

    assert first_page_ids is not None
    turn_contents = {
        record["payload"]["row"]["content"]
        for record in collected
        if record["payload"]["table"] == "turns"
    }
    assert "turn-8" in turn_contents
    assert any(record["payload"]["table"] == "assistant_usage_events" for record in collected)


def test_same_row_id_different_content_is_new_revision(tmp_path: Path):
    home = tmp_path / "copilot"
    database = _write_database(home, session_id="session-a", turns=[(7, "first")])
    paths = _paths(home)
    before = _records_for_table(_collect_all(paths, "session-a"), "turns")[0]

    connection = sqlite3.connect(database)
    try:
        connection.execute("DELETE FROM turns WHERE id = 7")
        connection.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (7, "session-a", 7, "reused", "2026-09-10T17:12:00.000Z"),
        )
        connection.commit()
    finally:
        connection.close()

    after = _records_for_table(_collect_all(paths, "session-a"), "turns")[0]
    assert before["locator"]["primary_key"] == after["locator"]["primary_key"] == 7
    assert before["locator"]["content_revision"] != after["locator"]["content_revision"]
    assert before["source_id"] != after["source_id"]
    assert after["payload"]["row"]["content"] == "reused"


def test_malformed_database_is_incompatible_not_busy(tmp_path: Path):
    home = tmp_path / "copilot"
    home.mkdir()
    database = home / "session-store.db"
    database.write_bytes(b"this is not a sqlite database")
    slice_ = read_database(_paths(home), "session-a", {})
    assert slice_["records"] == []
    assert slice_["exhausted"] is True
    assert slice_["diagnostics"][0]["code"] == "copilot_database_incompatible"
    assert "busy" not in slice_["diagnostics"][0]["code"]


def test_read_and_discover_close_sqlite_connections(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_database(home, session_id="session-a", turns=[(1, "closed")])
    paths = _paths(home)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        discover_database_sessions(paths)
        read_database(paths, "session-a", {})
        gc.collect()
    leaked = [item for item in caught if "unclosed database" in str(item.message)]
    assert leaked == []
