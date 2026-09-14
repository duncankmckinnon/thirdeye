"""Read-only capture support for GitHub Copilot CLI recordings."""

from __future__ import annotations

from .identity import resolve_sources, stored_session_id, validate_native_id
from .types import SourceBatch, SourcePaths, SourceRecord, SourceSlice, SyncResult

__all__ = [
    "SourceBatch",
    "SourcePaths",
    "SourceRecord",
    "SourceSlice",
    "SyncResult",
    "resolve_sources",
    "stored_session_id",
    "validate_native_id",
]
