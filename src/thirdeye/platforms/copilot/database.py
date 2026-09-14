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
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
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
_TIMESTAMP_COLUMNS = ("created_at", "createdAt", "timestamp", "updated_at", "updatedAt")
_CWD_COLUMNS = ("cwd", "working_directory", "workingDirectory")
_BUSY_TOKENS = ("locked", "busy")
_INCOMPATIBLE_TOKENS = (
    "file is not a database",
    "malformed",
    "corrupt",
    "disk image is malformed",
    "not a database",
)


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
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _content_revision(row: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def _quote_identifier(name: str) -> str:
    # Names come from SQLite's schema, but quote anyway so a surprising schema
    # cannot turn an identifier into executable SQL.
    return '"' + name.replace('"', '""') + '"'


def _file_generation(database: Path) -> str:
    """Identify the database file incarnation without consulting live WAL metadata.

    WAL size and mtime change independently of this session's rows.  Device and
    inode still change when the file is replaced, which is the signal needed to
    detect row-ID reuse across a new database.
    """

    try:
        stat = database.stat()
    except FileNotFoundError:
        payload: dict[str, Any] = {"path": database.name, "missing": True}
    else:
        payload = {
            "path": database.name,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            # Windows may quickly reuse st_ino after unlink/recreate.  Creation
            # time remains stable for ordinary database writes and changes for
            # a replacement file.  It is also available as birth time on macOS.
            "birthtime_ns": getattr(stat, "st_birthtime_ns", None),
        }
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _connect(database: Path) -> sqlite3.Connection:
    # ``mode=ro`` keeps SQLite's normal WAL behaviour while preventing all
    # writes.  In particular, do not use immutable=1: it ignores live WAL data.
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 100")
        connection.execute("BEGIN")
        return connection
    except Exception:
        connection.close()
        raise


@contextmanager
def _readonly_connection(database: Path) -> Iterator[sqlite3.Connection]:
    connection = _connect(database)
    try:
        yield connection
    finally:
        connection.close()


def _table_info(connection: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return list(connection.execute(f"PRAGMA table_info({_quote_identifier(table)})"))


def _column_names(info: Iterable[sqlite3.Row]) -> list[str]:
    return [str(row["name"]) for row in info]


def _available_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)", _TABLES
    )
    return {str(row[0]) for row in rows}


def _session_column(table: str, columns: Iterable[str]) -> str | None:
    known = set(columns)
    return next((name for name in _SESSION_ID_COLUMNS[table] if name in known), None)


def _primary_key_columns(info: Iterable[sqlite3.Row]) -> list[str] | None:
    keyed = [(int(row["pk"]), str(row["name"])) for row in info if int(row["pk"]) > 0]
    if not keyed:
        return None
    keyed.sort()
    return [name for _, name in keyed]


def _session_scope_column(
    table: str, columns: Iterable[str], pk_columns: list[str] | None
) -> str | None:
    scoped = _session_column(table, columns)
    if scoped is not None:
        return scoped
    # Observed Copilot sessions use ``id``; if a compatible schema names that
    # single primary key differently, the key still *is* the native session ID.
    if table == "sessions" and pk_columns is not None and len(pk_columns) == 1:
        return pk_columns[0]
    return None


def _row_identity(row: dict[str, Any], pk_columns: Sequence[str]) -> Any:
    if len(pk_columns) == 1:
        return row[pk_columns[0]]
    return {name: row[name] for name in pk_columns}


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


def _operational_diagnostic(error: sqlite3.OperationalError, database: Path) -> dict[str, Any]:
    message = str(error).lower()
    if any(token in message for token in _BUSY_TOKENS):
        return _diagnostic(
            "copilot_database_busy",
            "Copilot session database could not be read within 100ms; retry sync or watch",
            path=str(database),
            reason=str(error),
        )
    if any(token in message for token in _INCOMPATIBLE_TOKENS):
        return _diagnostic(
            "copilot_database_incompatible",
            "Copilot session database could not be read as SQLite evidence",
            path=str(database),
            reason=str(error),
        )
    return _diagnostic(
        "copilot_database_unreadable",
        "Copilot session database could not be opened for read-only evidence",
        path=str(database),
        reason=str(error),
    )


def _session_ids_from_table(
    connection: sqlite3.Connection, table: str, session_column: str
) -> list[str]:
    rows = connection.execute(
        f"SELECT {_quote_identifier(session_column)} FROM {_quote_identifier(table)} "
        f"WHERE {_quote_identifier(session_column)} IS NOT NULL"
    )
    return [str(row[0]) for row in rows if isinstance(row[0], str)]


def discover_database_sessions(paths: SourcePaths) -> list[str]:
    """Return session IDs advertised by any readable allowlisted table.

    Discovery is intentionally conservative: a malformed or absent database is
    simply not a discoverable source.  ``read_database`` supplies the actionable
    diagnostics needed by status and sync commands.
    """

    database = Path(paths["database"])
    if not database.is_file():
        return []
    try:
        with _readonly_connection(database) as connection:
            available = _available_tables(connection)
            values: set[str] = set()
            for table in _TABLES:
                if table not in available:
                    continue
                info = _table_info(connection, table)
                columns = _column_names(info)
                session_column = _session_scope_column(table, columns, _primary_key_columns(info))
                if session_column is None:
                    continue
                values.update(_session_ids_from_table(connection, table, session_column))
    except sqlite3.Error:
        return []
    return sorted(values)


def read_database(
    paths: SourcePaths,
    native_id: str,
    cursor: dict,
    *,
    max_records: int = 1000,
) -> SourceSlice:
    """Read a bounded, transactionally consistent raw SQLite snapshot.

    Every poll re-reads the selected session, because turns and sessions are
    mutable.  Pagination is keyed to the database file incarnation so live WAL
    writes—including inserts and updates in this session—cannot starve later
    rows.  A replaced database file replays from the start so row-ID reuse
    cannot silently continue a stale offset.  Content revisions travel on each
    record; unchanged snapshots still deduplicate after a later full read.
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
    file_generation = _file_generation(database)
    rows_by_table: list[tuple[str, Any, dict[str, Any]]] = []
    cwd: str | None = None
    try:
        with _readonly_connection(database) as connection:
            available = _available_tables(connection)
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
                info = _table_info(connection, table)
                columns = _column_names(info)
                pk_columns = _primary_key_columns(info)
                session_column = _session_scope_column(table, columns, pk_columns)
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
                if pk_columns is None:
                    diagnostics.append(
                        _diagnostic(
                            "copilot_database_missing_primary_key",
                            "Copilot table lacks a SQLite PRIMARY KEY that can identify rows",
                            table=table,
                            columns=columns,
                        )
                    )
                    continue
                order = ", ".join(_quote_identifier(name) for name in pk_columns)
                query = (
                    f"SELECT * FROM {_quote_identifier(table)} "
                    f"WHERE {_quote_identifier(session_column)} = ? "
                    f"ORDER BY {order}"
                )
                for sql_row in connection.execute(query, (native_id,)):
                    row = {key: _json_value(sql_row[key]) for key in sql_row.keys()}
                    if table == "sessions" and cwd is None:
                        cwd = next(
                            (row[name] for name in _CWD_COLUMNS if isinstance(row.get(name), str)),
                            None,
                        )
                    rows_by_table.append((table, _row_identity(row, pk_columns), row))
    except sqlite3.OperationalError as error:
        return _empty_slice([_operational_diagnostic(error, database)])
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
    generation = file_generation
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
        _row_record(paths, native_id, table, primary_key, row, file_generation, observed_at)
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
