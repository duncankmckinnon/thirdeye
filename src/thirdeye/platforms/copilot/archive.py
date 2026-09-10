"""Durable, local archive for raw GitHub Copilot CLI source evidence.

This module is intentionally below composition: it knows neither how records
are read nor how hooks are invoked.  Its only input is a fully composed
``SourceBatch`` and its output is the generic thirdeye event log plus a small,
recoverable Copilot checkpoint beside that log.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from thirdeye._compat.locking import LockMode, locked
from thirdeye.config import Config
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.reader import SessionReader
from thirdeye.store import Store
from thirdeye.writer import utc_iso_ms

from .constants import PLATFORM_NAME, SOURCE_SCHEMA_VERSION
from .identity import stored_session_id, validate_native_id
from .state import (
    STATE_SCHEMA_VERSION,
    clear_journal,
    journal_path,
    lock_path,
    read_json,
    state_path,
    write_journal,
    write_state,
)
from .types import SourceBatch, SourcePaths, SourceRecord, SyncResult

_EVENT_TYPES = {
    "transcript": "copilot_transcript",
    "database": "copilot_database",
    "hook": "copilot_hook",
    "metadata": "copilot_metadata",
}

# Tests and embedding applications may replace this with a deterministic
# callback that raises at a journal boundary.  It is deliberately private: no
# runtime behaviour relies on fault injection.
_fault_injector: Callable[[str], None] | None = None


def _fault(point: str) -> None:
    if _fault_injector is not None:
        _fault_injector(point)


def _result(
    *,
    sessions: int = 0,
    written: int = 0,
    duplicates: int = 0,
    pending: int = 0,
    errors: int = 0,
) -> SyncResult:
    return {
        "sessions": sessions,
        "records_written": written,
        "duplicate_records": duplicates,
        "pending": pending,
        "errors": errors,
    }


def _archive_dir(config: Config, stored_id: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored_id)


def _valid_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return value


def _event_type(record: SourceRecord) -> str:
    return _EVENT_TYPES.get(record["source_kind"], "copilot_metadata")


def _envelope(record: SourceRecord) -> dict[str, Any]:
    # ``source_record`` is named so later derived-state versions can add their
    # own fields without ever changing the raw record's shape.
    return {"schema_version": SOURCE_SCHEMA_VERSION, "source_record": record}


def _record_from_event(event: dict[str, Any]) -> SourceRecord | None:
    if event.get("t") not in _EVENT_TYPES.values():
        return None
    data = event.get("data")
    if not isinstance(data, dict) or data.get("schema_version") != SOURCE_SCHEMA_VERSION:
        return None
    record = data.get("source_record")
    if not isinstance(record, dict) or not isinstance(record.get("source_id"), str):
        return None
    return record  # type: ignore[return-value]


def _committed_records(directory: Path) -> dict[str, SourceRecord]:
    if not directory.exists():
        return {}
    committed: dict[str, SourceRecord] = {}
    for event in SessionReader(directory).iter_events(types=_EVENT_TYPES.values()):
        record = _record_from_event(event)
        if record is not None:
            committed[record["source_id"]] = record
    return committed


def _new_state(paths: SourcePaths, native_id: str) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "source_key": paths["source_key"],
        "source_home": paths["home"],
        "native_session_id": native_id,
        "cursor": {},
        "health": {"diagnostics": [], "last_successful_import": None, "capabilities": {}},
    }


def _validate_state(state: dict[str, Any], paths: SourcePaths, native_id: str) -> None:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("unsupported Copilot archive state schema")
    if state.get("source_key") != paths["source_key"]:
        raise ValueError("Copilot source-key prefix collision for stored session ID")
    if state.get("native_session_id") != native_id:
        raise ValueError("Copilot archive native session identity does not match stored session ID")


def _ensure_meta(config: Config, paths: SourcePaths, native_id: str, cwd: str | None) -> Path:
    stored_id = stored_session_id(paths, native_id)
    directory = _archive_dir(config, stored_id)
    meta = read_meta(meta_path(directory)) if directory.exists() else None
    if meta is not None:
        identity = meta.extra.get("copilot") if isinstance(meta.extra, dict) else None
        if not isinstance(identity, dict) or identity.get("source_key") != paths["source_key"]:
            raise ValueError("Copilot source-key prefix collision for stored session ID")
        if identity.get("native_session_id") != native_id:
            raise ValueError("Copilot archive native session identity does not match stored session ID")
        return directory

    # Store owns generic session metadata.  Creating it here also lets a hook
    # establish a provisional session before any transcript exists.
    writer = Store(config).open_session(
        stored_id,
        platform=PLATFORM_NAME,
        cwd=cwd or "",
        extra={
            "copilot": {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "source_key": paths["source_key"],
                "source_home": paths["home"],
                "native_session_id": native_id,
            }
        },
    )
    writer.flush_and_detach()
    return directory


def _base_cursor(batch: SourceBatch) -> dict[str, Any] | None:
    """Read an optional composition cursor without imposing a reader format.

    The public SourceBatch schema remains JSON-compatible and intentionally
    has no reader-specific field.  Composition can include either spelling in
    its cursor object when it needs optimistic contention detection.
    """
    cursor = batch["next_cursor"]
    for name in ("base_cursor", "_base_cursor"):
        value = cursor.get(name)
        if isinstance(value, dict):
            return value
    return None


def _set_health(
    state: dict[str, Any], diagnostics: list[dict[str, Any]], *, successful: bool
) -> None:
    health = state.setdefault("health", {})
    health["diagnostics"] = diagnostics
    if successful:
        health["last_successful_import"] = utc_iso_ms()


def _append_records(
    config: Config,
    directory: Path,
    stored_id: str,
    cwd: str | None,
    records: list[SourceRecord],
    committed: dict[str, SourceRecord],
) -> tuple[int, int, list[dict[str, Any]]]:
    _ = directory
    writer = Store(config).open_session(stored_id, platform=PLATFORM_NAME, cwd=cwd or "")
    written = 0
    duplicates = 0
    diagnostics: list[dict[str, Any]] = []
    try:
        for record in records:
            source_id = record.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                diagnostics.append({"kind": "invalid_source_record", "reason": "missing source_id"})
                continue
            if source_id in committed:
                duplicates += 1
                continue
            source_ts = _valid_timestamp(record.get("ts"))
            observed_at = _valid_timestamp(record.get("observed_at"))
            event_ts = source_ts or observed_at
            if source_ts is None:
                diagnostics.append({"kind": "missing_source_time", "source_id": source_id})
            writer.append(_event_type(record), _envelope(record), ts=event_ts)
            committed[source_id] = record
            written += 1
        writer.flush_and_detach()
    except BaseException:
        writer.flush_and_detach()
        raise
    return written, duplicates, diagnostics


def _recover(
    config: Config,
    paths: SourcePaths,
    native_id: str,
    directory: Path,
    state: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    journal = read_json(journal_path(directory))
    if journal is None:
        return state, 0, 0
    if journal.get("source_key") != paths["source_key"] or journal.get("native_session_id") != native_id:
        raise ValueError("Copilot archive journal belongs to another source session")
    records = journal.get("records")
    next_cursor = journal.get("next_cursor")
    if not isinstance(records, list) or not isinstance(next_cursor, dict):
        raise ValueError("invalid Copilot archive journal")
    committed = _committed_records(directory)
    written, duplicates, diagnostics = _append_records(
        config, directory, stored_session_id(paths, native_id), journal.get("cwd"), records, committed
    )
    _fault("after_recovery_append")
    state["cursor"] = next_cursor
    _set_health(state, list(journal.get("diagnostics", [])) + diagnostics, successful=True)
    write_state(directory, state)
    _fault("after_recovery_checkpoint")
    clear_journal(directory)
    return state, written, duplicates


def load_cursor(config: Config, paths: SourcePaths, native_id: str) -> dict:
    """Return a copy of the last committed cursor for one native session."""
    validate_native_id(native_id)
    directory = _archive_dir(config, stored_session_id(paths, native_id))
    if not directory.exists():
        return {}
    with locked(lock_path(directory), LockMode.EXCLUSIVE):
        state = read_json(state_path(directory))
        if state is None:
            return {}
        _validate_state(state, paths, native_id)
        # JSON copying protects the on-disk state from caller mutation.
        return json.loads(json.dumps(state.get("cursor", {})))


def commit_batch(config: Config, paths: SourcePaths, batch: SourceBatch) -> SyncResult:
    """Append one raw evidence batch and atomically advance its cursor."""
    native_id = batch["native_session_id"]
    validate_native_id(native_id)
    if batch["source_key"] != paths["source_key"]:
        raise ValueError("SourceBatch source_key does not match selected Copilot source home")
    for record in batch["records"]:
        if record.get("native_session_id") != native_id:
            raise ValueError("SourceBatch contains a record for another native session")
    stored_id = stored_session_id(paths, native_id)
    directory = _ensure_meta(config, paths, native_id, batch.get("cwd"))
    with locked(lock_path(directory), LockMode.EXCLUSIVE):
        state = read_json(state_path(directory)) or _new_state(paths, native_id)
        _validate_state(state, paths, native_id)
        state, recovered_written, recovered_duplicates = _recover(config, paths, native_id, directory, state)

        expected = _base_cursor(batch)
        current = state.get("cursor", {})
        if expected is not None and expected != current:
            diagnostic = {
                "kind": "stale_cursor",
                "reason": "archive cursor advanced while batch was being read",
                "expected_cursor": expected,
                "committed_cursor": current,
            }
            _set_health(state, list(batch["diagnostics"]) + [diagnostic], successful=False)
            write_state(directory, state)
            return _result(
                sessions=1,
                written=recovered_written,
                duplicates=recovered_duplicates,
                pending=len(batch["records"]),
                errors=1,
            )

        journal = {
            "schema_version": STATE_SCHEMA_VERSION,
            "source_key": paths["source_key"],
            "native_session_id": native_id,
            "cwd": batch.get("cwd"),
            "records": batch["records"],
            "next_cursor": batch["next_cursor"],
            "diagnostics": batch["diagnostics"],
        }
        write_journal(directory, journal)
        _fault("after_journal")
        committed = _committed_records(directory)
        written, duplicates, diagnostics = _append_records(
            config, directory, stored_id, batch.get("cwd"), batch["records"], committed
        )
        _fault("after_append")
        state["cursor"] = batch["next_cursor"]
        _set_health(state, list(batch["diagnostics"]) + diagnostics, successful=True)
        write_state(directory, state)
        _fault("after_checkpoint")
        clear_journal(directory)
        _fault("after_journal_clear")
        _apply_lifecycle(directory, batch["records"])
        return _result(
            sessions=1,
            written=recovered_written + written,
            duplicates=recovered_duplicates + duplicates,
        )


def _apply_lifecycle(directory: Path, records: list[SourceRecord]) -> None:
    """Apply only explicit top-level lifecycle evidence; child stops never close."""
    close = False
    reopen = False
    for record in records:
        if record["source_kind"] != "hook":
            continue
        payload = record.get("payload", {})
        event = payload.get("event") if isinstance(payload, dict) else None
        context = payload.get("context") if isinstance(payload, dict) else None
        is_child = isinstance(context, dict) and bool(context.get("parent_tool_call_id") or context.get("agent_id"))
        if event in {"sessionEnd", "shutdown", "session_end"} and not is_child:
            close = True
        if event in {"sessionStart", "resume", "activity", "session_start", "userPromptSubmitted"}:
            reopen = True
    if not close and not reopen:
        return
    meta_file = meta_path(directory)
    meta = read_meta(meta_file)
    if meta is None:
        return
    if reopen:
        meta.status = "open"
        meta.ended_at = None
    elif close:
        meta.status = "closed"
        meta.ended_at = utc_iso_ms()
    write_meta(meta_file, meta)


def iter_captured_records(config: Config, stored_session_id: str) -> Iterator[SourceRecord]:
    """Yield immutable raw Copilot records retained in a local stored session."""
    directory = _archive_dir(config, stored_session_id)
    if not directory.exists():
        return
    for event in SessionReader(directory).iter_events(types=_EVENT_TYPES.values()):
        record = _record_from_event(event)
        if record is not None:
            yield record
