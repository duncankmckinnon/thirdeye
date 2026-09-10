"""Bounded local composition of Copilot source capture and hook spooling."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from thirdeye.config import Config

from .archive import commit_batch, iter_captured_records, load_cursor
from .hook_payload import parse_hook
from .identity import validate_native_id
from .sources import discover_sessions, read_batch
from .spool import ack_spool, enqueue_hook, read_spool
from .types import SourceBatch, SourcePaths, SourceRecord, SyncResult


def _result() -> SyncResult:
    return {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 0,
    }


def _add(total: SyncResult, result: SyncResult) -> None:
    for key in total:
        total[key] += result[key]


def _spool_sessions(config: Config, paths: SourcePaths) -> list[str]:
    """Discover hook-only sessions without reading their content directly."""

    root = Path(config.root) / "spool" / "copilot" / paths["source_key"]
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    sessions: list[str] = []
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            validate_native_id(entry.name)
        except (OSError, ValueError):
            continue
        sessions.append(entry.name)
    return sorted(sessions)


def _with_spool(batch: SourceBatch, spool_records: list[SourceRecord]) -> SourceBatch:
    """Add an invocation-time spool snapshot without changing reader cursors."""

    return {
        **batch,
        "records": [*spool_records, *batch["records"]],
        "next_cursor": deepcopy(batch["next_cursor"]),
        "diagnostics": list(batch["diagnostics"]),
    }


def _diagnostic_count(batch: SourceBatch) -> int:
    """Diagnostics are surfaced as source errors without dropping evidence."""

    return len(batch["diagnostics"])


def _diagnostic_pending(batch: SourceBatch) -> int:
    """Count sources which must be retried without treating them as complete."""

    retryable_suffixes = (
        "_unavailable",
        "_missing",
        "_busy",
        "_unreadable",
        "_read_failed",
    )
    return sum(
        1
        for diagnostic in batch["diagnostics"]
        if isinstance(diagnostic.get("code"), str)
        and diagnostic["code"].endswith(retryable_suffixes)
    )


def _stale_after_commit(
    config: Config, paths: SourcePaths, native_session_id: str, base_cursor: dict[str, Any], result: SyncResult
) -> bool:
    """Identify archive's optimistic-race result without inspecting state files."""

    return (
        result["errors"] > 0
        and result["pending"] > 0
        and load_cursor(config, paths, native_session_id) != base_cursor
    )


def capture_session(config: Config, paths: SourcePaths, native_session_id: str) -> SyncResult:
    """Commit one bounded source read plus the current durable hook spool.

    A concurrent capture may advance the archive cursor between read and
    commit.  In that case this retries exactly once from the newly committed
    cursor, preserving the original spool snapshot and leaving any second race
    pending for a later sync/watch cycle.
    """

    validate_native_id(native_session_id)
    spool_records = read_spool(config, paths, native_session_id)
    base_cursor = load_cursor(config, paths, native_session_id)
    batch = _with_spool(read_batch(paths, native_session_id, base_cursor), spool_records)
    result = commit_batch(config, paths, batch)
    diagnostics = _diagnostic_count(batch)

    if _stale_after_commit(config, paths, native_session_id, base_cursor, result):
        retry_cursor = load_cursor(config, paths, native_session_id)
        retry_batch = _with_spool(
            read_batch(paths, native_session_id, retry_cursor), spool_records
        )
        result = commit_batch(config, paths, retry_batch)
        diagnostics = _diagnostic_count(retry_batch)

    # A zero archive error means the journal checkpoint made every submitted
    # hook source ID durable (including an already-durable duplicate).
    if result["errors"] == 0 and spool_records:
        ack_spool(config, paths, [record["source_id"] for record in spool_records])

    result = dict(result)
    result["errors"] += diagnostics
    result["pending"] += _diagnostic_pending(batch)
    return result


def sync(
    config: Config, paths: SourcePaths, *, session_id: str | None = None
) -> SyncResult:
    """Capture an invocation-time snapshot of discovered and spooled sessions."""

    if session_id is not None:
        validate_native_id(session_id)
        known = set(discover_sessions(paths)) | set(_spool_sessions(config, paths))
        if session_id not in known:
            return {
                "sessions": 0,
                "records_written": 0,
                "duplicate_records": 0,
                "pending": 0,
                "errors": 1,
            }
        session_ids = [session_id]
    else:
        session_ids = sorted(set(discover_sessions(paths)) | set(_spool_sessions(config, paths)))

    total = _result()
    for native_session_id in session_ids:
        _add(total, capture_session(config, paths, native_session_id))
    return total


def _observed_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def record_hook(
    config: Config,
    paths: SourcePaths,
    event: str,
    payload: dict,
    context: dict,
) -> SyncResult:
    """Durably receive one hook observation, then make one bounded capture."""

    record = parse_hook(
        event,
        payload,
        context,
        observed_at=_observed_at(),
        observation_id=uuid4().hex,
    )
    enqueue_hook(config, paths, record)
    return capture_session(config, paths, record["native_session_id"])


__all__ = [
    "capture_session",
    "iter_captured_records",
    "record_hook",
    "sync",
]
