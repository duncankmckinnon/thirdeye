"""Compose Copilot's independent persisted recording sources.

This module remains above the individual readers and below the archive.  Its
only durable contract is a :class:`SourceBatch`: reader cursors stay namespaced
so a transcript replacement cannot reset SQLite pagination (and vice versa).
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from .database import discover_database_sessions, read_database
from .identity import resolve_sources, validate_native_id
from .transcript import discover_transcripts, read_transcript
from .types import SourceBatch, SourcePaths, SourceSlice

__all__ = ["discover_sessions", "read_batch", "resolve_sources"]

_DISCOVERY_PROBE_ID = "copilot-discovery-probe"
_FILE_LEVEL_DATABASE_CODES = frozenset(
    {
        "copilot_database_unreadable",
        "copilot_database_busy",
        "copilot_database_incompatible",
        "copilot_database_read_failed",
    }
)


def _discovery_diagnostic(name: str, error: BaseException) -> dict[str, Any]:
    return {
        "code": f"copilot_{name}_unreadable",
        "message": "Copilot source could not be discovered; retry sync or watch",
        "reason": type(error).__name__,
    }


def _discover_with_diagnostics(
    paths: SourcePaths,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return session IDs plus discovery diagnostics that sync must surface."""

    discovered: set[str] = set()
    diagnostics: list[dict[str, Any]] = []

    try:
        discovered.update(discover_transcripts(paths))
    except (OSError, ValueError) as error:
        diagnostics.append(_discovery_diagnostic("transcript", error))

    database_ids: list[str] | None
    try:
        database_ids = list(discover_database_sessions(paths))
        discovered.update(database_ids)
    except (OSError, ValueError) as error:
        diagnostics.append(_discovery_diagnostic("database", error))
        database_ids = None

    # discover_database_sessions treats an unreadable file as "no sessions".
    # Probe the file-level diagnostic so all-session sync is not a silent no-op.
    if database_ids == [] and Path(paths["database"]).is_file():
        probe = _read_slice("database", read_database, paths, _DISCOVERY_PROBE_ID, {})
        for diagnostic in probe["diagnostics"]:
            code = diagnostic.get("code")
            if isinstance(code, str) and code in _FILE_LEVEL_DATABASE_CODES:
                diagnostics.append(diagnostic)

    return sorted(discovered), diagnostics


def discover_sessions(paths: SourcePaths) -> list[str]:
    """Return the deterministic union of readable transcript and DB sessions."""

    sessions, _ = _discover_with_diagnostics(paths)
    return sessions


def _cursor_part(cursor: dict[str, Any], name: str) -> dict[str, Any]:
    value = cursor.get(name)
    return deepcopy(value) if isinstance(value, dict) else {}


def _failed_slice(cursor: dict[str, Any], name: str, error: BaseException) -> SourceSlice:
    return {
        "records": [],
        "next_cursor": deepcopy(cursor),
        "diagnostics": [
            {
                "code": f"copilot_{name}_read_failed",
                "message": "Copilot source could not be read; retry sync or watch",
                "reason": type(error).__name__,
            }
        ],
        "cwd": None,
        "exhausted": False,
    }


def _read_slice(
    name: str,
    reader: Callable[[SourcePaths, str, dict[str, Any]], SourceSlice],
    paths: SourcePaths,
    native_session_id: str,
    cursor: dict[str, Any],
) -> SourceSlice:
    try:
        return reader(paths, native_session_id, cursor)
    except (OSError, ValueError) as error:
        return _failed_slice(cursor, name, error)


def read_batch(paths: SourcePaths, native_session_id: str, cursor: dict) -> SourceBatch:
    """Read one bounded, lossless slice from each persisted source.

    Sources deliberately do not share a cursor.  A source failure is retained
    as a diagnostic while a healthy sibling source still contributes records.
    ``base_cursor`` is an optimistic archive marker and is stripped before the
    archive persists the next source cursor.  Exhaustion flags stay on the
    composition cursor so capture can page or report pending without confusing
    the per-source readers.
    """

    validate_native_id(native_session_id)
    source_cursor: dict[str, Any] = dict(cursor) if isinstance(cursor, dict) else {}
    transcript_cursor = _cursor_part(source_cursor, "transcript")
    database_cursor = _cursor_part(source_cursor, "database")
    transcript = _read_slice(
        "transcript", read_transcript, paths, native_session_id, transcript_cursor
    )
    database = _read_slice("database", read_database, paths, native_session_id, database_cursor)
    next_cursor: dict[str, Any] = {
        "transcript": deepcopy(transcript["next_cursor"]),
        "database": deepcopy(database["next_cursor"]),
        "base_cursor": deepcopy(source_cursor),
        "transcript_exhausted": bool(transcript["exhausted"]),
        "database_exhausted": bool(database["exhausted"]),
    }
    return {
        "source_key": paths["source_key"],
        "native_session_id": native_session_id,
        "cwd": transcript["cwd"] if transcript["cwd"] is not None else database["cwd"],
        "records": [*transcript["records"], *database["records"]],
        "next_cursor": next_cursor,
        "diagnostics": [*transcript["diagnostics"], *database["diagnostics"]],
    }
