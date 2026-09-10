"""Versioned, lossless source contracts shared by Copilot V1 and V2.

These TypedDicts deliberately describe raw source evidence.  They do not imply
turn reconstruction, usage accounting, or any semantic correlation.
"""

from __future__ import annotations

from typing import Any, TypedDict

from .constants import SOURCE_SCHEMA_VERSION

# The value placed beside every archived SourceRecord in a thirdeye event data
# envelope.  It is deliberately separate from a Copilot CLI version.
SCHEMA_VERSION = SOURCE_SCHEMA_VERSION


class SourcePaths(TypedDict):
    """Canonical source-home identity and its Copilot recording locations."""

    home: str
    source_key: str
    session_root: str
    database: str


class SourceRecord(TypedDict):
    """One immutable observation from a Copilot source domain."""

    source_id: str
    source_kind: str  # transcript | database | hook | metadata
    native_session_id: str
    ts: str | None  # Source time when valid; never invented.
    observed_at: str
    payload: dict[str, Any]
    locator: dict[str, Any]


class SourceBatch(TypedDict):
    """The composed capture boundary consumed by the durable archive."""

    source_key: str
    native_session_id: str
    cwd: str | None
    records: list[SourceRecord]
    next_cursor: dict[str, Any]
    diagnostics: list[dict[str, Any]]


class SyncResult(TypedDict):
    """Counts returned by a capture operation."""

    sessions: int
    records_written: int
    duplicate_records: int
    pending: int
    errors: int


class SourceSlice(TypedDict):
    """A bounded read from one source; composed into a :class:`SourceBatch`."""

    records: list[SourceRecord]
    next_cursor: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    cwd: str | None
    exhausted: bool
