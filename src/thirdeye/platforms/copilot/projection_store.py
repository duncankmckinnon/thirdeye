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


def _directory(config: Config, stored_session_id: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored_session_id)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _index_key(item: dict[str, Any], field: str, *, prefix: str) -> str:
    value = item.get(field)
    return value if isinstance(value, str) and value else f"{prefix}:{_digest(item)}"


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
    for key in ("source_ids",):
        ids = value.get(key)
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
    for call in turn.get("llm_calls", []):
        _add_source_ids(call, source_ids)
    for child in turn.get("subagents", []):
        source_ids.update(_turn_source_ids(_mapping(child), ()))

    attributes = _mapping(turn.get("attributes"))
    interaction_id = attributes.get("interaction_id")
    turn_id = turn.get("turn_id")
    for semantic_event in events:
        event_attrs = _mapping(semantic_event.get("attributes"))
        if (
            isinstance(interaction_id, str)
            and event_attrs.get("interaction_id") == interaction_id
        ) or (isinstance(turn_id, str) and event_attrs.get("stored_turn_id") == turn_id):
            ids = semantic_event.get("source_ids")
            if isinstance(ids, list):
                source_ids.update(source_id for source_id in ids if isinstance(source_id, str))
    return source_ids


def _projected_turn_record(
    stored_session_id: str,
    directory: Path,
    turn: dict[str, Any],
    semantic_events: Iterable[dict[str, Any]],
    archived_events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    turn_id = _index_key(turn, "turn_id", prefix="turn")
    source_ids = _turn_source_ids(turn, semantic_events)
    events = [archived_events[source_id] for source_id in source_ids if source_id in archived_events]
    events.sort(key=lambda event: int(event.get("seq", -1)))
    meta = read_meta(meta_path(directory)) if directory.exists() else None
    start_ts = turn.get("start_ts") if isinstance(turn.get("start_ts"), str) else None
    end_ts = turn.get("end_ts") if isinstance(turn.get("end_ts"), str) else None
    return {
        "id": f"{stored_session_id}:{turn_id}",
        "turn_id": turn_id,
        "session_id": stored_session_id,
        "platform": PLATFORM_NAME,
        "cwd": meta.cwd if meta is not None else "",
        "start_seq": events[0].get("seq") if events else None,
        "end_seq": events[-1].get("seq") if events else None,
        "start_ts": events[0].get("ts") if events else start_ts,
        "end_ts": events[-1].get("ts") if events else end_ts,
        "events": events,
    }


def _merge_state(current: dict[str, Any], next_state: dict[str, Any]) -> dict[str, Any]:
    """Merge independent incremental builders without losing another commit."""
    merged = empty_projection_state()
    merged.update({key: value for key, value in current.items() if key in merged})
    merged.update({key: value for key, value in next_state.items() if key in merged})
    current_ids = current.get("archive_source_ids")
    next_ids = next_state.get("archive_source_ids")
    current_source_ids = (
        {source_id for source_id in current_ids if isinstance(source_id, str)}
        if isinstance(current_ids, list)
        else set()
    )
    next_source_ids = (
        {source_id for source_id in next_ids if isinstance(source_id, str)}
        if isinstance(next_ids, list)
        else set()
    )
    merged["archive_source_ids"] = sorted(current_source_ids | next_source_ids)
    for section, key in (("semantic_state", "open_interactions"), ("accounting_state", "logical_calls")):
        old = _mapping(_mapping(current.get(section)).get(key))
        new = _mapping(_mapping(next_state.get(section)).get(key))
        merged[section] = {key: {**old, **new}}
    merged["projection_schema_version"] = PROJECTION_SCHEMA_VERSION
    return merged


def _usage_identity(row: UsageRow, attributions: dict[str, dict[str, Any]]) -> str:
    for logical_id, attribution in attributions.items():
        if attribution.get("usage_source_id") == row.call_id:
            return logical_id
    # Usage normalization makes call_id the durable logical ID.  The fallback
    # keeps hand-built DTOs valid without ever keying on an import sequence.
    return row.call_id


def _write_usage_index(directory: Path, usage_index: dict[str, Any]) -> None:
    """Materialize one latest serialized row per logical call atomically.

    UsageStore is append-only and its generic reader is last-wins, which is
    insufficient for correction/rebuild guarantees.  This derived-only
    materialization preserves its exact UsageRow JSON serialization while the
    projection index supplies replacement semantics.
    """
    path = usage_jsonl_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for logical_id in sorted(usage_index):
                row = usage_index[logical_id]
                if isinstance(row, dict):
                    stream.write(_canonical(row))
                    stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsops.replace(temporary, path)
        fsops.sync_directory(path.parent)
    except BaseException:
        fsops.unlink(Path(temporary), missing_ok=True)
        raise


def _commit_counts(indexes: dict[str, Any]) -> dict[str, int]:
    return {
        "events": len(_mapping(indexes.get("events"))),
        "turns": len(_mapping(indexes.get("turns"))),
        "usage": len(_mapping(indexes.get("usage"))),
        "attributions": len(_mapping(indexes.get("attributions"))),
        "pending": len(_mapping(indexes.get("pending"))),
        "diagnostics": len(_mapping(indexes.get("diagnostics"))),
    }


def commit_projection(
    config: Config,
    stored_session_id: str,
    projection: Projection,
    next_state: dict[str, Any],
) -> dict[str, int]:
    """Atomically merge a DTO projection into local, replayable derived state.

    V1 events are read only from the captured Store archive to form the generic
    turn view.  No source reader is invoked, so a captured session remains
    projectable after Copilot removes its original files.
    """
    directory = _directory(config, stored_session_id)
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        document = read_projection_document(directory)
        indexes = _mapping(document.get("indexes"))
        merged_indexes = {name: dict(_mapping(indexes.get(name))) for name in indexes}
        for name in ("events", "turns", "usage", "attributions", "pending", "diagnostics"):
            merged_indexes.setdefault(name, {})

        event_items = [item for item in projection.get("normalized_events", []) if isinstance(item, dict)]
        for item in event_items:
            merged_indexes["events"][_index_key(item, "id", prefix="event")] = item

        attribution_items = [item for item in projection.get("attributions", []) if isinstance(item, dict)]
        for item in attribution_items:
            merged_indexes["attributions"][_index_key(item, "logical_call_id", prefix="attribution")] = item

        for item in projection.get("usage_rows", []):
            if not isinstance(item, UsageRow):
                raise TypeError("projection usage_rows must contain UsageRow instances")
            row = item.to_dict()
            logical_id = _usage_identity(item, merged_indexes["attributions"])
            merged_indexes["usage"][logical_id] = row

        archived_events = _read_archived_events(directory)
        for item in projection.get("turns", []):
            if not isinstance(item, dict):
                continue
            # Child-agent spans are represented recursively by their owning
            # main interaction and never become separate generic user turns.
            if _mapping(item.get("attributes")).get("agent_id") is not None:
                continue
            turn_id = _index_key(item, "turn_id", prefix="turn")
            merged_indexes["turns"][turn_id] = _projected_turn_record(
                stored_session_id, directory, item, event_items, archived_events
            )

        for item in projection.get("pending", []):
            if isinstance(item, dict):
                merged_indexes["pending"][_index_key(item, "id", prefix="pending")] = item
        for item in projection.get("diagnostics", []):
            if isinstance(item, dict):
                merged_indexes["diagnostics"][_index_key(item, "id", prefix="diagnostic")] = item

        state = _merge_state(_mapping(document.get("state")), next_state)
        counts = _commit_counts(merged_indexes)
        state["commit_result"] = counts
        next_document = {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "state": state,
            "indexes": merged_indexes,
        }
        publish_projection_document(directory, next_document)
        # A crash after publication is repaired on the next load/commit from
        # the durable usage index; this file contains no raw V1 evidence.
        _write_usage_index(directory, merged_indexes["usage"])
        return counts


def load_projection_state(config: Config, stored_session_id: str) -> dict[str, Any]:
    """Return a copy of derived builder state, completing journal recovery."""
    directory = _directory(config, stored_session_id)
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        document = read_projection_document(directory)
        # Recover a sidecar lost after durable state publication without
        # changing V1 evidence or export bookkeeping.
        usage_index = _mapping(_mapping(document.get("indexes")).get("usage"))
        if usage_index:
            _write_usage_index(directory, usage_index)
        return json.loads(_canonical(_mapping(document.get("state"))))


def read_projected_turns(config: Config, stored_session_id: str) -> list[dict[str, Any]]:
    """Read completed main interaction records without semantic duplicates."""
    directory = _directory(config, stored_session_id)
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        document = read_projection_document(directory)
        turns = _mapping(_mapping(document.get("indexes")).get("turns"))
        values = [value for value in turns.values() if isinstance(value, dict)]
        values.sort(key=lambda turn: (str(turn.get("start_ts") or ""), str(turn.get("turn_id") or "")))
        return json.loads(_canonical(values))


def reset_projection_state(config: Config, stored_session_id: str) -> None:
    """Delete only reproducible projection state for a local rebuild.

    The caller must subsequently commit a full replay.  This intentionally
    does not import, reset, or otherwise interact with export-state files.
    """
    directory = _directory(config, stored_session_id)
    with locked(projection_lock_path(directory), LockMode.EXCLUSIVE):
        remove_projection_state(directory)
        fsops.unlink(usage_jsonl_path(directory), missing_ok=True)
        fsops.sync_directory(directory)
