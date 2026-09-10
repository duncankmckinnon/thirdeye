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
from .sources import _discover_with_diagnostics, read_batch
from .spool import ack_spool, enqueue_hook, read_spool
from .types import SourceBatch, SourcePaths, SourceRecord, SyncResult

_RETRYABLE_SUFFIXES = (
    "_unavailable",
    "_missing",
    "_busy",
    "_unreadable",
    "_read_failed",
    "_incompatible",
)
_ABSENCE_CODES = frozenset({"transcript_unavailable", "copilot_database_missing"})


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


def _is_retryable_code(code: object) -> bool:
    return isinstance(code, str) and code.endswith(_RETRYABLE_SUFFIXES)


def _diagnostic_count(batch: SourceBatch) -> int:
    """Diagnostics are surfaced as source errors without dropping evidence."""

    return len(batch["diagnostics"])


def _diagnostic_pending(batch: SourceBatch) -> int:
    """Count sources which must be retried without treating them as complete."""

    return sum(
        1 for diagnostic in batch["diagnostics"] if _is_retryable_code(diagnostic.get("code"))
    )


def _source_retryable(batch: SourceBatch, name: str) -> bool:
    return any(
        isinstance(diagnostic.get("code"), str)
        and name in diagnostic["code"]
        and _is_retryable_code(diagnostic["code"])
        for diagnostic in batch["diagnostics"]
    )


def _pagination_pending(batch: SourceBatch) -> int:
    """Count sources that still have unread snapshot pages."""

    cursor = batch["next_cursor"]
    pending = 0
    for name in ("transcript", "database"):
        if cursor.get(f"{name}_exhausted", True):
            continue
        if _source_retryable(batch, name):
            continue
        pending += 1
    return pending


def _compose(archive: SyncResult, batch: SourceBatch) -> SyncResult:
    result = dict(archive)
    result["errors"] += _diagnostic_count(batch)
    result["pending"] += _diagnostic_pending(batch) + _pagination_pending(batch)
    return result  # type: ignore[return-value]


def _merge_archive(first: SyncResult, second: SyncResult) -> SyncResult:
    """Keep recovery writes from a stale attempt; take the retry's error/pending."""

    return {
        "sessions": max(first["sessions"], second["sessions"]),
        "records_written": first["records_written"] + second["records_written"],
        "duplicate_records": first["duplicate_records"] + second["duplicate_records"],
        "pending": second["pending"],
        "errors": second["errors"],
    }


def _merge_diagnostics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retryable: dict[str, dict[str, Any]] = {}
    others: list[dict[str, Any]] = []
    for item in items:
        code = item.get("code")
        if _is_retryable_code(code) and isinstance(code, str):
            retryable[code] = item
        else:
            others.append(item)
    return others + list(retryable.values())


def _stale_after_commit(
    config: Config,
    paths: SourcePaths,
    native_session_id: str,
    base_cursor: dict[str, Any],
    result: SyncResult,
) -> bool:
    """Identify archive's optimistic-race result without inspecting state files."""

    return (
        result["errors"] > 0
        and result["pending"] > 0
        and load_cursor(config, paths, native_session_id) != base_cursor
    )


def _is_absence_diagnostic(diagnostic: dict[str, Any]) -> bool:
    code = diagnostic.get("code")
    if not isinstance(code, str):
        return False
    if code in _ABSENCE_CODES:
        return True
    return code.startswith("workspace_")


def _selected_source_missing(batch: SourceBatch) -> bool:
    if batch["records"]:
        return False
    return all(_is_absence_diagnostic(diagnostic) for diagnostic in batch["diagnostics"])


def _result_from_diagnostics(diagnostics: list[dict[str, Any]]) -> SyncResult:
    return {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": sum(
            1 for diagnostic in diagnostics if _is_retryable_code(diagnostic.get("code"))
        ),
        "errors": len(diagnostics),
    }


def _capture_once(
    config: Config, paths: SourcePaths, native_session_id: str
) -> tuple[SyncResult, SourceBatch, bool]:
    """Commit one bounded source read plus the current durable hook spool.

    A concurrent capture may advance the archive cursor between read and
    commit.  In that case this retries exactly once from the newly committed
    cursor, preserving the original spool snapshot and leaving any second race
    pending for a later sync/watch cycle.
    """

    spool_records = read_spool(config, paths, native_session_id)
    base_cursor = load_cursor(config, paths, native_session_id)
    batch = _with_spool(read_batch(paths, native_session_id, base_cursor), spool_records)
    archive = commit_batch(config, paths, batch)
    unresolved_stale = False

    if _stale_after_commit(config, paths, native_session_id, base_cursor, archive):
        retry_cursor = load_cursor(config, paths, native_session_id)
        retry_batch = _with_spool(read_batch(paths, native_session_id, retry_cursor), spool_records)
        retry_archive = commit_batch(config, paths, retry_batch)
        archive = _merge_archive(archive, retry_archive)
        batch = retry_batch
        unresolved_stale = _stale_after_commit(
            config, paths, native_session_id, retry_cursor, retry_archive
        )

    # A zero archive error means the journal checkpoint made every submitted
    # hook source ID durable (including an already-durable duplicate).
    if archive["errors"] == 0 and spool_records:
        ack_spool(config, paths, [record["source_id"] for record in spool_records])

    more_pages = (not unresolved_stale) and _pagination_pending(batch) > 0
    return archive, batch, more_pages


def capture_session(config: Config, paths: SourcePaths, native_session_id: str) -> SyncResult:
    """Commit one bounded source read plus the current durable hook spool."""

    validate_native_id(native_session_id)
    archive, batch, _ = _capture_once(config, paths, native_session_id)
    return _compose(archive, batch)


def _drain_session(config: Config, paths: SourcePaths, native_session_id: str) -> SyncResult:
    """Drain one invocation-time snapshot without chasing a growing source."""

    total = _result()
    collected: list[dict[str, Any]] = []
    last_batch: SourceBatch | None = None
    prev_cursor: object = object()
    while True:
        archive, batch, more_pages = _capture_once(config, paths, native_session_id)
        last_batch = batch
        collected.extend(batch["diagnostics"])
        total["records_written"] += archive["records_written"]
        total["duplicate_records"] += archive["duplicate_records"]
        total["sessions"] = max(total["sessions"], archive["sessions"])
        total["errors"] += archive["errors"]
        total["pending"] = archive["pending"]
        current_cursor = load_cursor(config, paths, native_session_id)
        if not more_pages or current_cursor == prev_cursor:
            break
        prev_cursor = current_cursor
    assert last_batch is not None
    merged: SourceBatch = {
        **last_batch,
        "diagnostics": _merge_diagnostics(collected),
    }
    return _compose(total, merged)


def sync(config: Config, paths: SourcePaths, *, session_id: str | None = None) -> SyncResult:
    """Capture an invocation-time snapshot of discovered and spooled sessions."""

    discovered, discovery_diagnostics = _discover_with_diagnostics(paths)
    known = set(discovered) | set(_spool_sessions(config, paths))

    if session_id is not None:
        validate_native_id(session_id)
        if session_id not in known:
            probe = read_batch(paths, session_id, {})
            if _selected_source_missing(probe):
                missing: SyncResult = {
                    "sessions": 0,
                    "records_written": 0,
                    "duplicate_records": 0,
                    "pending": 0,
                    "errors": 1,
                }
                if discovery_diagnostics:
                    _add(missing, _result_from_diagnostics(discovery_diagnostics))
                return missing
        session_ids = [session_id]
    else:
        session_ids = sorted(known)

    total = _result_from_diagnostics(discovery_diagnostics)
    if not session_ids:
        return total
    for native_session_id in session_ids:
        _add(total, _drain_session(config, paths, native_session_id))
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
