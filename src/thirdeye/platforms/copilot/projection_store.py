"""Durable local storage for completed Copilot V2 projections.

This module is intentionally a persistence boundary.  It accepts already
constructed :class:`Projection` DTOs and archived Store events; it does not
read Copilot's live transcript/database files and does not perform semantic,
accounting, attribution, or export work.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops
from thirdeye._compat.locking import LockMode, locked
from thirdeye.config import Config
from thirdeye.meta import read_meta
from thirdeye.paths import meta_path, session_dir, usage_jsonl_path
from thirdeye.reader import SessionReader
from thirdeye.usage.index import UsageIndex
from thirdeye.usage.types import UsageRow

from .constants import PLATFORM_NAME, SOURCE_SCHEMA_VERSION
from .projection_state import (
    empty_projection_state,
    projection_lock_path,
    publish_projection_document,
    read_projection_document,
    remove_projection_state,
)
from .types import PROJECTION_SCHEMA_VERSION, Projection

_RAW_EVENT_TYPES = frozenset(
    {"copilot_transcript", "copilot_database", "copilot_hook", "copilot_metadata"}
)
_INDEX_NAMES = ("events", "turns", "usage", "attributions", "pending", "diagnostics")


class ProjectionConflictError(RuntimeError):
    """A commit observed a newer ``commit_sequence`` than it was based on.

    Raised instead of silently overwriting: a second writer already
    committed a projection built from more current builder state, so this
    (older) attempt is discarded rather than winning a last-write-races
    against the newer one.  The caller should reload projection state and
    retry.
    """


def _directory(config: Config, stored_session_id: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored_session_id)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _index_key(item: dict[str, Any], field: str, *, prefix: str) -> str:
    value = item.get(field)
    return value if isinstance(value, str) and value else f"{prefix}:{_digest(item)}"


def _require_session(directory: Path, stored_session_id: str) -> None:
    if read_meta(meta_path(directory)) is None:
        raise ValueError(f"unknown Copilot session: {stored_session_id}")


def _existing_session_dir(config: Config, stored_session_id: str) -> Path | None:
    """Return the session directory only when V1 meta is already on disk.

    ``locked()`` creates parent directories, so unknown ids must be rejected
    before acquiring the projection lock.
    """
    directory = _directory(config, stored_session_id)
    if read_meta(meta_path(directory)) is None:
        return None
    return directory


def _source_id(event: dict[str, Any]) -> str | None:
    if event.get("t") not in _RAW_EVENT_TYPES:
        return None
    data = _mapping(event.get("data"))
    if data.get("schema_version") != SOURCE_SCHEMA_VERSION:
        return None
    record = _mapping(data.get("source_record"))
    source_id = record.get("source_id")
    return source_id if isinstance(source_id, str) else None


def _read_archived_events(directory: Path) -> dict[str, dict[str, Any]]:
    if not directory.exists():
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for event in SessionReader(directory).iter_events(types=_RAW_EVENT_TYPES):
        source_id = _source_id(event)
        if source_id is not None:
            indexed[source_id] = event
    return indexed


def _add_source_ids(value: object, result: set[str]) -> None:
    if not isinstance(value, dict):
        return
    ids = value.get("source_ids")
    if isinstance(ids, list):
        result.update(source_id for source_id in ids if isinstance(source_id, str))
    references = value.get("source_references")
    if isinstance(references, list):
        for reference in references:
            source_id = _mapping(reference).get("source_id")
            if isinstance(source_id, str):
                result.add(source_id)
    attributes = value.get("attributes")
    if isinstance(attributes, dict):
        _add_source_ids(attributes, result)


def _turn_source_ids(turn: dict[str, Any], events: Iterable[dict[str, Any]]) -> set[str]:
    """Find archived evidence owned by a top-level reconstructed interaction.

    Producers can provide explicit source references on a turn/call.  The
    interaction-id fallback is evidence-preserving (not a guessed parent): it
    includes all main and nested-child events bearing that exact native
    interaction identity, which keeps child evidence inside its parent view.
    """
    source_ids: set[str] = set()
    _add_source_ids(turn, source_ids)
    for call in _items(turn.get("llm_calls")):
        _add_source_ids(_mapping(call), source_ids)
    for call in _items(turn.get("accounting_calls")):
        _add_source_ids(_mapping(call), source_ids)
    for child in _items(turn.get("subagents")):
        source_ids.update(_turn_source_ids(_mapping(child), ()))

    attributes = _mapping(turn.get("attributes"))
    interaction_id = attributes.get("interaction_id")
    turn_id = turn.get("turn_id")
    for semantic_event in events:
        event_attrs = _mapping(semantic_event.get("attributes"))
        matched = (
            isinstance(interaction_id, str) and event_attrs.get("interaction_id") == interaction_id
        ) or (isinstance(turn_id, str) and event_attrs.get("stored_turn_id") == turn_id)
        if matched:
            _add_source_ids(semantic_event, source_ids)
    return source_ids


def _is_main_turn(turn: dict[str, Any]) -> bool:
    return _mapping(turn.get("attributes")).get("agent_id") is None


def _replace_state(next_state: dict[str, Any]) -> dict[str, Any]:
    """Treat ``next_state`` as an authoritative ProjectionState snapshot."""
    state = empty_projection_state()
    archive_ids = next_state.get("archive_source_ids")
    if isinstance(archive_ids, list):
        state["archive_source_ids"] = [
            source_id for source_id in archive_ids if isinstance(source_id, str)
        ]
    semantic = _mapping(next_state.get("semantic_state"))
    state["semantic_state"] = {
        "open_interactions": dict(_mapping(semantic.get("open_interactions")))
    }
    accounting = _mapping(next_state.get("accounting_state"))
    state["accounting_state"] = {"logical_calls": dict(_mapping(accounting.get("logical_calls")))}
    revision = next_state.get("projection_revision")
    if isinstance(revision, str) and revision:
        state["projection_revision"] = revision
    state["projection_schema_version"] = PROJECTION_SCHEMA_VERSION
    return state


def _collect_identities(
    current: dict[str, Any],
    attributions: dict[str, dict[str, Any]],
    accounting_calls: dict[str, Any],
) -> dict[str, str]:
    identities = {
        key: value
        for key, value in _mapping(current).items()
        if isinstance(key, str) and isinstance(value, str)
    }
    for attribution in attributions.values():
        source_id = attribution.get("usage_source_id")
        logical_id = attribution.get("logical_call_id")
        if isinstance(source_id, str) and isinstance(logical_id, str) and logical_id:
            identities[source_id] = logical_id
            identities[logical_id] = logical_id
    for call in accounting_calls.values():
        mapped = _mapping(call)
        source_id = mapped.get("usage_source_id")
        logical_id = mapped.get("logical_call_id")
        if isinstance(logical_id, str) and logical_id:
            identities[logical_id] = logical_id
            if isinstance(source_id, str):
                identities[source_id] = logical_id
    return identities


def _usage_identity(row: UsageRow, identities: dict[str, str]) -> str:
    return identities.get(row.call_id, row.call_id)


def _drop_aliased_usage_keys(usage_index: dict[str, Any], identities: dict[str, str]) -> None:
    """Move a source-id row onto its logical id instead of deleting the call.

    An identity-only later commit may learn ``usage_source_id -> logical_call_id``
    without sending the ``UsageRow`` again.  Popping the alias without copying
    would drop the only persisted call.
    """
    for source_id, logical_id in identities.items():
        if source_id == logical_id:
            continue
        existing = usage_index.pop(source_id, None)
        if existing is None or logical_id in usage_index or not isinstance(existing, dict):
            continue
        relocated = dict(existing)
        relocated["call_id"] = logical_id
        usage_index[logical_id] = relocated


def _usage_seq(
    logical_id: str,
    row: UsageRow,
    identities: dict[str, str],
    archived_events: dict[str, dict[str, Any]],
) -> int:
    candidates = [
        source_id
        for source_id, mapped in identities.items()
        if mapped == logical_id and source_id in archived_events
    ]
    if row.call_id in archived_events:
        candidates.append(row.call_id)
    if not candidates:
        return 0
    return max(int(archived_events[source_id].get("seq", 0)) for source_id in candidates)


def _usage_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def _sidecar_rows(usage_index: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        usage_index[logical_id]
        for logical_id in sorted(usage_index)
        if isinstance(usage_index[logical_id], dict)
    ]


def _sidecar_payload(usage_index: dict[str, Any]) -> str:
    return "".join(_usage_line(row) + "\n" for row in _sidecar_rows(usage_index))


def _sidecar_matches(path: Path, usage_index: dict[str, Any]) -> bool:
    if not path.exists():
        return not usage_index
    try:
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    return lines == _sidecar_rows(usage_index)


def _write_usage_index(directory: Path, usage_index: dict[str, Any]) -> None:
    """Materialize one latest serialized row per logical call atomically.

    UsageStore is append-only and its generic reader is last-wins, which is
    insufficient when a correction must replace a row under a re-keyed
    identity.  This derived-only rewrite preserves UsageRow.to_dict JSON
    while the projection index supplies replacement semantics.
    """
    path = usage_jsonl_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_sidecar_payload(usage_index))
            stream.flush()
            os.fsync(stream.fileno())
        fsops.replace(temporary, path)
        fsops.sync_directory(path.parent)
    except BaseException:
        fsops.unlink(Path(temporary), missing_ok=True)
        raise


def _invalidate_usage_index(config: Config, stored_session_id: str, directory: Path) -> None:
    """Force UsageIndex to re-read a rewritten sidecar of unchanged size."""
    index = UsageIndex(config.root)
    connection = index.connect()
    try:
        connection.execute("DELETE FROM usage WHERE session_id = ?", (stored_session_id,))
        connection.execute("DELETE FROM usage_sync WHERE session_id = ?", (stored_session_id,))
        index.refresh_session(connection, stored_session_id, directory)
    finally:
        connection.close()


def _publish_usage(
    config: Config,
    stored_session_id: str,
    directory: Path,
    usage_index: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    path = usage_jsonl_path(directory)
    matches = _sidecar_matches(path, usage_index)
    if not matches:
        if not usage_index:
            if path.exists():
                fsops.unlink(path, missing_ok=True)
                fsops.sync_directory(directory)
        else:
            _write_usage_index(directory, usage_index)
    elif not force:
        return
    _invalidate_usage_index(config, stored_session_id, directory)


def _commit_counts(indexes: dict[str, Any]) -> dict[str, int]:
    return {name: len(_mapping(indexes.get(name))) for name in _INDEX_NAMES}


def _stored_turn(
    turn: dict[str, Any],
    source_ids: set[str],
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    turn_id = _index_key(turn, "turn_id", prefix="turn")
    merged_ids = set(source_ids)
    if prior is not None:
        merged_ids.update(
            source_id for source_id in _items(prior.get("source_ids")) if isinstance(source_id, str)
        )
    status = turn.get("status")
    return {
        "turn_id": turn_id,
        "status": status if isinstance(status, str) else "",
        "source_ids": sorted(merged_ids),
        "span": turn,
    }


def _hydrate_turn(
    stored_session_id: str,
    record: dict[str, Any],
    archived_events: dict[str, dict[str, Any]],
    cwd: str,
) -> dict[str, Any]:
    turn_id = record.get("turn_id") if isinstance(record.get("turn_id"), str) else ""
    source_ids = [item for item in _items(record.get("source_ids")) if isinstance(item, str)]
    events = [
        archived_events[source_id] for source_id in source_ids if source_id in archived_events
    ]
    events.sort(key=lambda event: int(event.get("seq", -1)))
    span = _mapping(record.get("span"))
    start_ts = span.get("start_ts") if isinstance(span.get("start_ts"), str) else None
    end_ts = span.get("end_ts") if isinstance(span.get("end_ts"), str) else None
    return {
        "id": f"{stored_session_id}:{turn_id}",
        "turn_id": turn_id,
        "session_id": stored_session_id,
        "platform": PLATFORM_NAME,
        "cwd": cwd,
        "start_seq": events[0].get("seq") if events else None,
        "end_seq": events[-1].get("seq") if events else None,
        "start_ts": events[0].get("ts") if events else start_ts,
        "end_ts": events[-1].get("ts") if events else end_ts,
        "events": events,
    }


def commit_projection(
    config: Config,
    stored_session_id: str,
    projection: Projection,
    next_state: dict[str, Any],
    *,
    replace: bool = False,
    base_commit_sequence: int | None = None,
) -> dict[str, int]:
    """Atomically merge or replace a DTO projection into replayable derived state.

    V1 events are read only from the captured Store archive to form the generic
    turn view.  No source reader is invoked, so a captured session remains
    projectable after Copilot removes its original files.

    ``replace=True`` discards prior derived indexes as part of this same
    locked operation instead of requiring a separate reset call.  Nothing is
    written to disk until every validation above this point succeeds, so a
    commit that fails validation (a malformed usage row, an unknown-schema
    document) leaves the previous projection completely untouched rather
    than losing it to a non-atomic delete-then-commit sequence.

    Durability past that validation point has two distinct failure modes.
    The projection document is the source of truth and is published with
    write-ahead journaling (see ``publish_projection_document``): once that
    call returns, the new document is durably committed, full stop.  The
    usage sidecar published immediately after is a derived, self-healing
    mirror of the document's own usage index -- kept as a separate JSONL
    file only so ``UsageIndex`` can query it without parsing the whole
    document.  If that second, mirror-only publish raises (a disk error, not
    a validation error), this function still raises so the caller learns of
    it, but the already-committed document is *not* rolled back: it reflects
    the new projection, and the next ``load_projection_state`` call
    republishes a sidecar that matches it.  A caller must not assume a raise
    from this function always means "nothing changed" -- check which phase
    failed via ``read_projection_status``/``load_projection_state`` if that
    distinction matters.

    ``base_commit_sequence``, when given, must equal the ``commit_sequence``
    a caller observed from an earlier ``load_projection_state`` call.  A
    mismatch means another writer has committed since that read -- this
    raises :class:`ProjectionConflictError` instead of overwriting the newer
    projection with one built from the stale builder state, closing the
    otherwise unlocked gap between reading prior state, building a new
    projection from it, and committing here.  This check itself happens
    before any write in this call, so a conflict never touches disk.
    """
    directory = _directory(config, stored_session_id)
    _require_session(directory, stored_session_id)
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        document = read_projection_document(directory)
        current_sequence = _mapping(document.get("state")).get("commit_sequence")
        current_sequence = current_sequence if isinstance(current_sequence, int) else 0
        if base_commit_sequence is not None and base_commit_sequence != current_sequence:
            raise ProjectionConflictError(
                f"Copilot projection for {stored_session_id!r} advanced from commit "
                f"{base_commit_sequence} to {current_sequence} since it was loaded; "
                "reload projection state and retry"
            )
        indexes = {} if replace else _mapping(document.get("indexes"))
        merged_indexes = {name: dict(_mapping(indexes.get(name))) for name in indexes}
        for name in (*_INDEX_NAMES, "usage_identities"):
            merged_indexes.setdefault(name, {})

        event_items = [
            item for item in projection.get("normalized_events", []) if isinstance(item, dict)
        ]
        for item in event_items:
            merged_indexes["events"][_index_key(item, "id", prefix="event")] = item

        attribution_items = [
            item for item in projection.get("attributions", []) if isinstance(item, dict)
        ]
        for item in attribution_items:
            merged_indexes["attributions"][
                _index_key(item, "logical_call_id", prefix="attribution")
            ] = item

        state = _replace_state(next_state)
        state["commit_sequence"] = current_sequence + 1
        identities = _collect_identities(
            merged_indexes["usage_identities"],
            merged_indexes["attributions"],
            _mapping(_mapping(state.get("accounting_state")).get("logical_calls")),
        )

        archived_events = _read_archived_events(directory)
        for item in projection.get("usage_rows", []):
            if not isinstance(item, UsageRow):
                raise TypeError("projection usage_rows must contain UsageRow instances")
            if item.session_id != stored_session_id:
                raise ValueError("usage row session_id does not match stored session")
            logical_id = _usage_identity(item, identities)
            identities[item.call_id] = logical_id
            identities[logical_id] = logical_id
            row = item.to_dict()
            row["call_id"] = logical_id
            row["seq"] = _usage_seq(logical_id, item, identities, archived_events)
            merged_indexes["usage"][logical_id] = row
        _drop_aliased_usage_keys(merged_indexes["usage"], identities)
        merged_indexes["usage_identities"] = identities

        for item in projection.get("turns", []):
            if not isinstance(item, dict) or not _is_main_turn(item):
                continue
            turn_id = _index_key(item, "turn_id", prefix="turn")
            source_ids = _turn_source_ids(item, merged_indexes["events"].values())
            merged_indexes["turns"][turn_id] = _stored_turn(
                item, source_ids, _mapping(merged_indexes["turns"].get(turn_id)) or None
            )

        merged_indexes["pending"] = {
            _index_key(item, "id", prefix="pending"): item
            for item in projection.get("pending", [])
            if isinstance(item, dict)
        }
        merged_indexes["diagnostics"] = {
            _index_key(item, "id", prefix="diagnostic"): item
            for item in projection.get("diagnostics", [])
            if isinstance(item, dict)
        }

        counts = _commit_counts(merged_indexes)
        if not state["projection_revision"]:
            state["projection_revision"] = _digest(
                {key: merged_indexes.get(key) for key in (*_INDEX_NAMES, "usage_identities")}
            )
        state["index_totals"] = counts
        next_document = {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "state": state,
            "indexes": merged_indexes,
        }
        publish_projection_document(directory, next_document)
        _publish_usage(config, stored_session_id, directory, merged_indexes["usage"])
        return counts


def load_projection_state(config: Config, stored_session_id: str) -> dict[str, Any]:
    """Return a copy of derived builder state, completing journal recovery."""
    directory = _existing_session_dir(config, stored_session_id)
    if directory is None:
        return empty_projection_state()
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        document = read_projection_document(directory)
        usage_index = _mapping(_mapping(document.get("indexes")).get("usage"))
        _publish_usage(config, stored_session_id, directory, usage_index, force=True)
        return json.loads(_canonical(_mapping(document.get("state"))))


def read_projected_turns(config: Config, stored_session_id: str) -> list[dict[str, Any]]:
    """Read completed main interaction records without semantic duplicates."""
    directory = _existing_session_dir(config, stored_session_id)
    if directory is None:
        return []
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        document = read_projection_document(directory)
        meta = read_meta(meta_path(directory))
        cwd = meta.cwd if meta is not None else ""
        archived_events = _read_archived_events(directory)
        turns = _mapping(_mapping(document.get("indexes")).get("turns"))
        values = []
        for record in turns.values():
            if not isinstance(record, dict):
                continue
            if record.get("status") != "completed":
                continue
            values.append(_hydrate_turn(stored_session_id, record, archived_events, cwd))
        values.sort(
            key=lambda turn: (str(turn.get("start_ts") or ""), str(turn.get("turn_id") or ""))
        )
        return json.loads(_canonical(values))


_STATUS_KEYS = ("events", "usage", "turns", "pending", "ambiguous", "conflicting", "errors")


def read_projection_status(config: Config, stored_session_id: str) -> dict[str, int]:
    """Return persisted projection counts without attempting a new derive.

    A caller whose current reconciliation attempt failed can use this to
    report the status of the last successfully published projection (which
    remains on disk and readable) instead of fabricating zeroed-out counts.
    """
    directory = _existing_session_dir(config, stored_session_id)
    if directory is None:
        return dict.fromkeys(_STATUS_KEYS, 0)
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        document = read_projection_document(directory)
        indexes = _mapping(document.get("indexes"))
        attributions = _mapping(indexes.get("attributions")).values()
        diagnostics = _mapping(indexes.get("diagnostics")).values()
        return {
            "events": len(_mapping(indexes.get("events"))),
            "usage": len(_mapping(indexes.get("usage"))),
            "turns": len(_mapping(indexes.get("turns"))),
            "pending": len(_mapping(indexes.get("pending"))),
            "ambiguous": sum(
                1
                for item in attributions
                if isinstance(item, dict) and item.get("status") == "ambiguous"
            ),
            "conflicting": sum(
                1
                for item in attributions
                if isinstance(item, dict) and item.get("status") == "conflicting"
            ),
            "errors": sum(
                1
                for item in diagnostics
                if isinstance(item, dict) and item.get("severity") == "error"
            ),
        }


def reset_projection_state(config: Config, stored_session_id: str) -> None:
    """Delete only reproducible projection state for a local rebuild.

    The caller must subsequently commit a full replay.  This intentionally
    does not import, reset, or otherwise interact with export-state files.
    """
    directory = _existing_session_dir(config, stored_session_id)
    if directory is None:
        return
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        remove_projection_state(directory)
        fsops.unlink(usage_jsonl_path(directory), missing_ok=True)
        _invalidate_usage_index(config, stored_session_id, directory)
        fsops.sync_directory(directory)
