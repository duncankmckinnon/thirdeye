"""Compose Copilot's independent persisted recording sources.

This module remains above the individual readers and below the archive.  Its
only durable contract is a :class:`SourceBatch`: reader cursors stay namespaced
so a transcript replacement cannot reset SQLite pagination (and vice versa).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .database import discover_database_sessions, read_database
from .identity import resolve_sources, validate_native_id
from .transcript import discover_transcripts, read_transcript
from .types import SourceBatch, SourcePaths, SourceSlice

__all__ = ["discover_sessions", "read_batch", "resolve_sources"]


def discover_sessions(paths: SourcePaths) -> list[str]:
    """Return the deterministic union of readable transcript and DB sessions."""

    discovered: set[str] = set()
    for discover in (discover_transcripts, discover_database_sessions):
        try:
            discovered.update(discover(paths))
        except (OSError, ValueError):
            # Discovery is advisory.  The per-session read provides the
            # actionable diagnostic while another source can still progress.
            continue
    return sorted(discovered)


def _cursor_part(cursor: dict[str, Any], name: str) -> dict[str, Any]:
    value = cursor.get(name)
    return deepcopy(value) if isinstance(value, dict) else {}


def _failed_slice(
    cursor: dict[str, Any], name: str, error: BaseException
) -> SourceSlice:
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
    archive persists the next source cursor.
    """

    validate_native_id(native_session_id)
    source_cursor: dict[str, Any] = dict(cursor) if isinstance(cursor, dict) else {}
    transcript_cursor = _cursor_part(source_cursor, "transcript")
    database_cursor = _cursor_part(source_cursor, "database")
    transcript = _read_slice(
        "transcript", read_transcript, paths, native_session_id, transcript_cursor
    )
    database = _read_slice(
        "database", read_database, paths, native_session_id, database_cursor
    )
    next_cursor: dict[str, Any] = {
        "transcript": transcript["next_cursor"],
        "database": database["next_cursor"],
        "base_cursor": deepcopy(source_cursor),
    }
    return {
        "source_key": paths["source_key"],
        "native_session_id": native_session_id,
        "cwd": transcript["cwd"] if transcript["cwd"] is not None else database["cwd"],
        "records": [*transcript["records"], *database["records"]],
        "next_cursor": next_cursor,
        "diagnostics": [*transcript["diagnostics"], *database["diagnostics"]],
    }
