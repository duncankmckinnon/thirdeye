"""Durable, local archive for raw GitHub Copilot CLI source evidence.

This module is intentionally below composition: it knows neither how records
are read nor how hooks are invoked.  Its only input is a fully composed
``SourceBatch`` and its output is the generic thirdeye event log plus a small,
recoverable Copilot checkpoint beside that log.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from thirdeye._compat.locking import LockMode, locked
from thirdeye.config import Config
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.reader import SessionReader
from thirdeye.store import Store
from thirdeye.writer import SessionWriter, utc_iso_ms

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
_CURSOR_CONTROL_KEYS = ("base_cursor", "_base_cursor")

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


def _source_cursor(cursor: object) -> dict[str, Any]:
    """Return the source reader cursor, without archive contention markers."""
    if not isinstance(cursor, dict):
        return {}
    return {key: value for key, value in cursor.items() if key not in _CURSOR_CONTROL_KEYS}


def _cursor_copy(cursor: object) -> dict[str, Any]:
    return json.loads(json.dumps(_source_cursor(cursor)))


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
            raise ValueError(
                "Copilot archive native session identity does not match stored session ID"
            )
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
    for name in _CURSOR_CONTROL_KEYS:
        value = cursor.get(name)
        if isinstance(value, dict):
            return value
    return None


def _observation_errors(records: list[SourceRecord]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for record in records:
        if _valid_timestamp(record.get("observed_at")) is None:
            source_id = record.get("source_id")
            errors.append(
                {
                    "kind": "invalid_observed_at",
                    "source_id": source_id if isinstance(source_id, str) else None,
                }
            )
    return errors


def _set_health(
    state: dict[str, Any], diagnostics: list[dict[str, Any]], *, successful: bool
) -> None:
    health = state.setdefault("health", {})
    health["diagnostics"] = diagnostics
    if successful:
        health["last_successful_import"] = utc_iso_ms()


def _writer_for_append(
    config: Config, directory: Path, stored_id: str, cwd: str | None
) -> SessionWriter:
    """Append into an existing archive without reopening Store session metadata."""
    meta = read_meta(meta_path(directory))
    if meta is None:
        return Store(config).open_session(stored_id, platform=PLATFORM_NAME, cwd=cwd or "")
    return SessionWriter(directory, meta)


def _append_records(
    config: Config,
    directory: Path,
    stored_id: str,
    cwd: str | None,
    records: list[SourceRecord],
    committed: dict[str, SourceRecord],
) -> tuple[int, int, list[dict[str, Any]], bool]:
    writer: SessionWriter | None = None
    written = 0
    duplicates = 0
    blocked = False
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
            observed_at = _valid_timestamp(record.get("observed_at"))
            if observed_at is None:
                diagnostics.append({"kind": "invalid_observed_at", "source_id": source_id})
                blocked = True
                continue
            source_ts = _valid_timestamp(record.get("ts"))
            event_ts = source_ts or observed_at
            if source_ts is None:
                diagnostics.append({"kind": "missing_source_time", "source_id": source_id})
            if writer is None:
                writer = _writer_for_append(config, directory, stored_id, cwd)
            writer.append(_event_type(record), _envelope(record), ts=event_ts)
            committed[source_id] = record
            written += 1
        if writer is not None:
            writer.flush_and_detach()
    except BaseException:
        if writer is not None:
            writer.flush_and_detach()
        raise
    return written, duplicates, diagnostics, blocked


def _identity_paths(
    directory: Path,
    state: dict[str, Any] | None,
    journal: dict[str, Any] | None,
) -> tuple[SourcePaths, str]:
    identity: dict[str, Any] = {}
    if isinstance(state, dict):
        identity = {
            "source_key": state.get("source_key"),
            "source_home": state.get("source_home"),
            "native_session_id": state.get("native_session_id"),
        }
    if not isinstance(identity.get("source_key"), str) or not isinstance(
        identity.get("native_session_id"), str
    ):
        meta = read_meta(meta_path(directory))
        extra = (
            meta.extra.get("copilot") if meta is not None and isinstance(meta.extra, dict) else None
        )
        if isinstance(extra, dict):
            identity = {
                "source_key": extra.get("source_key"),
                "source_home": extra.get("source_home"),
                "native_session_id": extra.get("native_session_id"),
            }
    native_id = identity.get("native_session_id")
    source_key = identity.get("source_key")
    source_home = identity.get("source_home")
    if not isinstance(native_id, str) and isinstance(journal, dict):
        native_id = journal.get("native_session_id")
    if not isinstance(source_key, str) and isinstance(journal, dict):
        source_key = journal.get("source_key")
    if (
        not isinstance(native_id, str)
        or not isinstance(source_key, str)
        or not isinstance(source_home, str)
    ):
        raise ValueError("Copilot archive identity is missing; cannot finish journal recovery")
    home = Path(source_home)
    paths: SourcePaths = {
        "home": source_home,
        "source_key": source_key,
        "session_root": str(home / "session-state"),
        "database": str(home / "session-store.db"),
    }
    return paths, native_id


def _checkpoint(
    directory: Path,
    state: dict[str, Any],
    *,
    next_cursor: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    records: list[Any],
) -> dict[str, Any]:
    _apply_lifecycle(directory, records)
    _fault("after_lifecycle")
    state["cursor"] = _source_cursor(next_cursor)
    _set_health(state, diagnostics, successful=True)
    write_state(directory, state)
    _fault("after_checkpoint")
    clear_journal(directory)
    _fault("after_journal_clear")
    return state


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
    if (
        journal.get("source_key") != paths["source_key"]
        or journal.get("native_session_id") != native_id
    ):
        raise ValueError("Copilot archive journal belongs to another source session")
    records = journal.get("records")
    next_cursor = journal.get("next_cursor")
    if not isinstance(records, list) or not isinstance(next_cursor, dict):
        raise ValueError("invalid Copilot archive journal")
    committed = _committed_records(directory)
    written, duplicates, diagnostics, blocked = _append_records(
        config, directory, directory.name, journal.get("cwd"), records, committed
    )
    _fault("after_recovery_append")
    if blocked:
        _set_health(state, list(journal.get("diagnostics", [])) + diagnostics, successful=False)
        write_state(directory, state)
        clear_journal(directory)
        return state, written, duplicates
    state = _checkpoint(
        directory,
        state,
        next_cursor=next_cursor,
        diagnostics=list(journal.get("diagnostics", [])) + diagnostics,
        records=records,
    )
    return state, written, duplicates


def _recover_directory(config: Config, directory: Path) -> None:
    journal = read_json(journal_path(directory))
    if journal is None:
        return
    state = read_json(state_path(directory))
    paths, native_id = _identity_paths(directory, state, journal)
    if state is None:
        state = _new_state(paths, native_id)
    else:
        _validate_state(state, paths, native_id)
    _recover(config, paths, native_id, directory, state)


def load_cursor(config: Config, paths: SourcePaths, native_id: str) -> dict:
    """Return a copy of the last committed cursor for one native session."""
    validate_native_id(native_id)
    directory = _archive_dir(config, stored_session_id(paths, native_id))
    if not directory.exists():
        return {}
    with locked(lock_path(directory), LockMode.EXCLUSIVE):
        journal_exists = journal_path(directory).is_file()
        state = read_json(state_path(directory))
        if state is None and not journal_exists:
            return {}
        if state is None:
            state = _new_state(paths, native_id)
        else:
            _validate_state(state, paths, native_id)
        state, _, _ = _recover(config, paths, native_id, directory, state)
        return _cursor_copy(state.get("cursor", {}))


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
        state, recovered_written, recovered_duplicates = _recover(
            config, paths, native_id, directory, state
        )

        expected = _base_cursor(batch)
        current = _source_cursor(state.get("cursor", {}))
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

        observation_errors = _observation_errors(batch["records"])
        if observation_errors:
            _set_health(state, list(batch["diagnostics"]) + observation_errors, successful=False)
            write_state(directory, state)
            return _result(
                sessions=1,
                written=recovered_written,
                duplicates=recovered_duplicates,
                pending=len(batch["records"]),
                errors=1,
            )

        next_cursor = _source_cursor(batch["next_cursor"])
        journal = {
            "schema_version": STATE_SCHEMA_VERSION,
            "source_key": paths["source_key"],
            "native_session_id": native_id,
            "cwd": batch.get("cwd"),
            "records": batch["records"],
            "next_cursor": next_cursor,
            "diagnostics": batch["diagnostics"],
        }
        write_journal(directory, journal)
        _fault("after_journal")
        committed = _committed_records(directory)
        written, duplicates, diagnostics, blocked = _append_records(
            config, directory, stored_id, batch.get("cwd"), batch["records"], committed
        )
        _fault("after_append")
        if blocked:
            _set_health(state, list(batch["diagnostics"]) + diagnostics, successful=False)
            write_state(directory, state)
            clear_journal(directory)
            return _result(
                sessions=1,
                written=recovered_written + written,
                duplicates=recovered_duplicates + duplicates,
                pending=sum(
                    1
                    for record in batch["records"]
                    if isinstance(record.get("source_id"), str)
                    and record["source_id"] not in committed
                ),
                errors=1,
            )
        _checkpoint(
            directory,
            state,
            next_cursor=next_cursor,
            diagnostics=list(batch["diagnostics"]) + diagnostics,
            records=batch["records"],
        )
        return _result(
            sessions=1,
            written=recovered_written + written,
            duplicates=recovered_duplicates + duplicates,
        )


def _is_child_hook(payload: dict[str, Any]) -> bool:
    """Child identity lives on hook_payload (and, historically, context)."""

    for mapping in (payload.get("hook_payload"), payload.get("context")):
        if not isinstance(mapping, dict):
            continue
        if mapping.get("agentId") or mapping.get("agent_id"):
            return True
        if mapping.get("parentToolCallId") or mapping.get("parent_tool_call_id"):
            return True
    return False


def _apply_lifecycle(directory: Path, records: list[Any]) -> None:
    """Apply only explicit top-level lifecycle evidence; child stops never close."""
    decision: str | None = None
    for record in records:
        if not isinstance(record, dict) or record.get("source_kind") != "hook":
            continue
        payload = record.get("payload", {})
        if not isinstance(payload, dict):
            continue
        event = payload.get("event")
        if event in {"sessionEnd", "shutdown", "session_end"} and not _is_child_hook(payload):
            decision = "close"
        elif event in {"sessionStart", "resume", "activity", "session_start", "userPromptSubmitted"}:
            decision = "reopen"
    if decision is None:
        return
    meta_file = meta_path(directory)
    meta = read_meta(meta_file)
    if meta is None:
        return
    if decision == "reopen":
        meta.status = "open"
        meta.ended_at = None
    else:
        meta.status = "closed"
        meta.ended_at = utc_iso_ms()
    write_meta(meta_file, meta)


def iter_captured_records(config: Config, stored_session_id: str) -> Iterator[SourceRecord]:
    """Yield immutable raw Copilot records retained in a local stored session."""
    directory = _archive_dir(config, stored_session_id)
    if not directory.exists():
        return
    with locked(lock_path(directory), LockMode.EXCLUSIVE):
        _recover_directory(config, directory)
    for event in SessionReader(directory).iter_events(types=_EVENT_TYPES.values()):
        record = _record_from_event(event)
        if record is not None:
            yield record
