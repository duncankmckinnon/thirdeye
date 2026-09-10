"""Read raw Copilot session evidence from its local SQLite database.

This module deliberately has no knowledge of thirdeye's archive.  It produces
lossless, revisioned source records; committing and deduplicating those records
is the archive's responsibility.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .identity import validate_native_id
from .types import SourcePaths, SourceRecord, SourceSlice

_TABLES = ("sessions", "turns", "assistant_usage_events")
_SESSION_ID_COLUMNS = {
    "sessions": ("id", "session_id"),
    "turns": ("session_id",),
    "assistant_usage_events": ("session_id",),
}
_PRIMARY_KEY_COLUMNS = ("id", "uuid", "event_id")
_TIMESTAMP_COLUMNS = ("created_at", "createdAt", "timestamp", "updated_at", "updatedAt")
_CWD_COLUMNS = ("cwd", "working_directory", "workingDirectory")


def _diagnostic(code: str, message: str, **details: Any) -> dict[str, Any]:
    """Return a content-free diagnostic suitable for CLI presentation."""

    return {"code": code, "message": message, **details}


def _observed_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    """Make SQLite values JSON-compatible without dropping a column's value."""

    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, memoryview):
        return _json_value(value.tobytes())
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return {"encoding": "repr", "data": repr(value)}
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _content_revision(row: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def _quote_identifier(name: str) -> str:
    # Names come from SQLite's schema, but quote anyway so a surprising schema
    # cannot turn an identifier into executable SQL.
    return '"' + name.replace('"', '""') + '"'


def _database_generation(database: Path) -> str:
    """Identify the database and its live WAL without opening it for writing."""

    parts: list[dict[str, Any]] = []
    for candidate in (database, Path(f"{database}-wal")):
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            parts.append({"path": candidate.name, "missing": True})
        else:
            parts.append(
                {
                    "path": candidate.name,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return "sha256:" + hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()


def _connect(database: Path) -> sqlite3.Connection:
    # ``mode=ro`` keeps SQLite's normal WAL behaviour while preventing all
    # writes.  In particular, do not use immutable=1: it ignores live WAL data.
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 100")
    connection.execute("BEGIN")
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    ]


def _available_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)", _TABLES
    )
    return {str(row[0]) for row in rows}


def _session_column(table: str, columns: Iterable[str]) -> str | None:
    known = set(columns)
    return next((name for name in _SESSION_ID_COLUMNS[table] if name in known), None)


def _primary_key_column(columns: Iterable[str]) -> str | None:
    known = set(columns)
    return next((name for name in _PRIMARY_KEY_COLUMNS if name in known), None)


def _valid_source_time(row: dict[str, Any]) -> str | None:
    for column in _TIMESTAMP_COLUMNS:
        value = row.get(column)
        if not isinstance(value, str) or not value:
            continue
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        return value
    return None


def _row_record(
    paths: SourcePaths,
    native_id: str,
    table: str,
    primary_key: Any,
    row: dict[str, Any],
    generation: str,
    observed_at: str,
) -> SourceRecord:
    revision = _content_revision(row)
    primary_identity = _canonical_json(_json_value(primary_key))
    # Keep the row identity legible for normal integer IDs, while quoting it so
    # arbitrary SQLite primary-key values never alter the source-ID structure.
    primary_component = quote(primary_identity, safe="")
    source_id = (
        f"copilot-db:{paths['source_key']}:{native_id}:{table}:{primary_component}:{revision}"
    )
    return {
        "source_id": source_id,
        "source_kind": "database",
        "native_session_id": native_id,
        "ts": _valid_source_time(row),
        "observed_at": observed_at,
        "payload": {"table": table, "row": row},
        "locator": {
            "database": paths["database"],
            "table": table,
            "primary_key": _json_value(primary_key),
            "content_revision": revision,
            "generation": generation,
        },
    }


def _empty_slice(diagnostics: list[dict[str, Any]]) -> SourceSlice:
    return {
        "records": [],
        "next_cursor": {},
        "diagnostics": diagnostics,
        "cwd": None,
        "exhausted": True,
    }


def discover_database_sessions(paths: SourcePaths) -> list[str]:
    """Return session IDs advertised by a readable Copilot sessions table.

    Discovery is intentionally conservative: a malformed or absent database is
    simply not a discoverable source.  ``read_database`` supplies the actionable
    diagnostics needed by status and sync commands.
    """

    database = Path(paths["database"])
    if not database.is_file():
        return []
    try:
        with _connect(database) as connection:
            if "sessions" not in _available_tables(connection):
                return []
            columns = _table_columns(connection, "sessions")
            session_column = _session_column("sessions", columns)
            if session_column is None:
                return []
            rows = connection.execute(
                f"SELECT {_quote_identifier(session_column)} FROM sessions "
                f"WHERE {_quote_identifier(session_column)} IS NOT NULL"
            )
            values = [str(row[0]) for row in rows if isinstance(row[0], str)]
    except sqlite3.Error:
        return []
    return sorted(set(values))


def read_database(
    paths: SourcePaths,
    native_id: str,
    cursor: dict,
    *,
    max_records: int = 1000,
) -> SourceSlice:
    """Read a bounded, transactionally consistent raw SQLite snapshot.

    Every poll re-reads the selected session, because turns and sessions are
    mutable.  A generation/offset cursor only bounds delivery of that snapshot;
    it never assumes a row ID is an immutable record identity.
    """

    validate_native_id(native_id)
    if max_records < 1:
        raise ValueError("max_records must be at least 1")

    database = Path(paths["database"])
    if not database.exists():
        return _empty_slice(
            [
                _diagnostic(
                    "copilot_database_missing",
                    "Copilot session database is not present",
                    path=str(database),
                )
            ]
        )
    if not database.is_file():
        return _empty_slice(
            [
                _diagnostic(
                    "copilot_database_unreadable",
                    "Copilot session database is not a regular file",
                    path=str(database),
                )
            ]
        )

    diagnostics: list[dict[str, Any]] = []
    observed_at = _observed_at()
    generation = _database_generation(database)
    try:
        with _connect(database) as connection:
            available = _available_tables(connection)
            rows_by_table: list[tuple[str, Any, dict[str, Any]]] = []
            cwd: str | None = None
            for table in _TABLES:
                if table not in available:
                    diagnostics.append(
                        _diagnostic(
                            "copilot_database_table_missing",
                            "Expected Copilot table is unavailable",
                            table=table,
                        )
                    )
                    continue
                columns = _table_columns(connection, table)
                session_column = _session_column(table, columns)
                primary_column = _primary_key_column(columns)
                if session_column is None:
                    diagnostics.append(
                        _diagnostic(
                            "copilot_database_missing_session_column",
                            "Copilot table cannot be scoped to a session",
                            table=table,
                            expected=list(_SESSION_ID_COLUMNS[table]),
                            columns=columns,
                        )
                    )
                    continue
                if primary_column is None:
                    diagnostics.append(
                        _diagnostic(
                            "copilot_database_missing_primary_key",
                            "Copilot table lacks a supported stable row identity",
                            table=table,
                            expected=list(_PRIMARY_KEY_COLUMNS),
                            columns=columns,
                        )
                    )
                    continue
                query = (
                    f"SELECT * FROM {_quote_identifier(table)} "
                    f"WHERE {_quote_identifier(session_column)} = ? "
                    f"ORDER BY {_quote_identifier(primary_column)}"
                )
                for sql_row in connection.execute(query, (native_id,)):
                    row = {key: _json_value(sql_row[key]) for key in sql_row.keys()}
                    if table == "sessions" and cwd is None:
                        cwd = next(
                            (row[name] for name in _CWD_COLUMNS if isinstance(row.get(name), str)),
                            None,
                        )
                    rows_by_table.append((table, row[primary_column], row))
    except sqlite3.OperationalError as error:
        return _empty_slice(
            [
                _diagnostic(
                    "copilot_database_busy",
                    "Copilot session database could not be read within 100ms; retry sync or watch",
                    path=str(database),
                    reason=str(error),
                )
            ]
        )
    except sqlite3.Error as error:
        return _empty_slice(
            [
                _diagnostic(
                    "copilot_database_incompatible",
                    "Copilot session database could not be read as SQLite evidence",
                    path=str(database),
                    reason=str(error),
                )
            ]
        )

    # The table loop above has a deterministic table order; ordering row IDs by
    # their JSON form makes heterogeneous SQLite primary key types deterministic.
    rows_by_table.sort(
        key=lambda item: (_TABLES.index(item[0]), _canonical_json(_json_value(item[1])))
    )
    incoming_generation = cursor.get("database_generation") if isinstance(cursor, dict) else None
    incoming_offset = cursor.get("database_offset", 0) if isinstance(cursor, dict) else 0
    offset = (
        incoming_offset
        if incoming_generation == generation and isinstance(incoming_offset, int)
        else 0
    )
    offset = max(0, offset)
    selected = rows_by_table[offset : offset + max_records]
    records = [
        _row_record(paths, native_id, table, primary_key, row, generation, observed_at)
        for table, primary_key, row in selected
    ]
    next_offset = offset + len(selected)
    exhausted = next_offset >= len(rows_by_table)
    next_cursor = {"database_generation": generation, "database_offset": next_offset}
    return {
        "records": records,
        "next_cursor": next_cursor,
        "diagnostics": diagnostics,
        "cwd": cwd,
        "exhausted": exhausted,
    }
