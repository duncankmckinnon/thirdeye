"""Durable Copilot export eligibility and accounting-placement state.

This state intentionally has no relationship to projection state.  Projection
state is disposable and rebuilt from the V1 archive; export placement is an
accounting decision and survives rebuilds.  The generic OTel worker cannot
atomically acknowledge a remote collector and this file, so an absent job is
never treated as proof of delivery.  A crash after a remote flush can still
lead to a deterministic retry and therefore a duplicate remote span.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops
from thirdeye._compat.locking import LockMode, locked
from thirdeye.config import Config
from thirdeye.paths import session_dir

from .constants import PLATFORM_NAME
from .jsonio import atomic_write_json, read_json_object

EXPORT_STATE_FILENAME = "copilot.export.ledger.json"
EXPORT_LOCK_FILENAME = "copilot.export.lock"
EXPORT_STATE_SCHEMA_VERSION = 1


def export_state_path(directory: Path) -> Path:
    return directory / EXPORT_STATE_FILENAME


def export_lock_path(directory: Path) -> Path:
    return directory / EXPORT_LOCK_FILENAME


def _directory(config: Config, stored_session_id: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored_session_id)


def empty_export_state() -> dict[str, Any]:
    """Return the versioned state used before the first V2 export activation."""
    return {
        "schema_version": EXPORT_STATE_SCHEMA_VERSION,
        "activated": False,
        "excluded_turn_ids": [],
        "excluded_accounting_ids": [],
        "placements": {},
        "conflicts": {},
    }


def _ids(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({value for value in values if isinstance(value, str) and value})


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _state(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None or value.get("schema_version") != EXPORT_STATE_SCHEMA_VERSION:
        return empty_export_state()
    placements = _mapping(value.get("placements"))
    conflicts = _mapping(value.get("conflicts"))
    return {
        "schema_version": EXPORT_STATE_SCHEMA_VERSION,
        "activated": bool(value.get("activated", False)),
        "excluded_turn_ids": _ids(value.get("excluded_turn_ids")),
        "excluded_accounting_ids": _ids(value.get("excluded_accounting_ids")),
        "placements": {key: item for key, item in placements.items() if isinstance(key, str) and isinstance(item, dict)},
        "conflicts": {key: item for key, item in conflicts.items() if isinstance(key, str) and isinstance(item, dict)},
    }


def _read(directory: Path) -> dict[str, Any]:
    try:
        return _state(
            read_json_object(
                export_state_path(directory), invalid_message="invalid Copilot export ledger"
            )
        )
    except ValueError:
        # Export accounting is not disposable.  Do not overwrite a corrupt
        # ledger and risk emitting tokens at another location.
        raise ValueError("invalid Copilot export ledger") from None


def _write(directory: Path, state: dict[str, Any]) -> None:
    atomic_write_json(export_state_path(directory), state)


def load_export_state(config: Config, stored_session_id: str) -> dict[str, Any]:
    """Read a snapshot of the independent export ledger.

    Callers that mutate the result must use :func:`update_export_state`; this
    helper intentionally returns a detached JSON-safe dictionary.
    """
    directory = _directory(config, stored_session_id)
    with locked(export_lock_path(directory), LockMode.SHARED):
        return json.loads(json.dumps(_read(directory)))


def update_export_state(
    config: Config, stored_session_id: str, update: Any
) -> dict[str, Any]:
    """Atomically apply ``update(state)`` and return the published state."""
    directory = _directory(config, stored_session_id)
    with locked(export_lock_path(directory), LockMode.EXCLUSIVE):
        state = _read(directory)
        updated = update(state)
        if not isinstance(updated, dict):
            raise TypeError("Copilot export state update must return a dictionary")
        state = _state(updated)
        _write(directory, state)
        return json.loads(json.dumps(state))


def usage_digest(usage: dict[str, Any]) -> str:
    """Stable correction detector for a serialized ``UsageRow``."""
    encoded = json.dumps(usage, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_turn_eligible(state: dict[str, Any], turn_id: str) -> bool:
    return turn_id not in set(_ids(state.get("excluded_turn_ids")))


def is_accounting_eligible(state: dict[str, Any], accounting_id: str) -> bool:
    return accounting_id not in set(_ids(state.get("excluded_accounting_ids")))


def initialize_eligibility(
    state: dict[str, Any],
    *,
    terminal_turn_ids: list[str],
    accounting_ids: list[str],
    include_history: bool,
) -> dict[str, Any]:
    """Set the first-activation boundary without changing any placement.

    Normal activation excludes terminal history.  An explicit historical
    export opts the supplied completed turns and accounting identities in by
    removing them from that boundary.  New evidence is absent from both lists
    and is therefore eligible after restart.
    """
    result = _state(state)
    turn_ids = set(_ids(terminal_turn_ids))
    usage_ids = set(_ids(accounting_ids))
    if not result["activated"]:
        result["activated"] = True
        if not include_history:
            result["excluded_turn_ids"] = sorted(turn_ids)
            result["excluded_accounting_ids"] = sorted(usage_ids)
        return result
    if include_history:
        result["excluded_turn_ids"] = sorted(set(result["excluded_turn_ids"]) - turn_ids)
        result["excluded_accounting_ids"] = sorted(
            set(result["excluded_accounting_ids"]) - usage_ids
        )
    return result


def record_placement(
    state: dict[str, Any],
    *,
    accounting_id: str,
    destination: str,
    span_id: str,
    usage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Persist the first token location for an accounting identity.

    Returns ``(state, entry, accepted)``.  A source correction or a changed
    destination after placement is a durable conflict: silently replacing a
    queued job could produce two different token totals at the same span.
    """
    result = _state(state)
    digest = usage_digest(usage)
    placements = result["placements"]
    existing = _mapping(placements.get(accounting_id))
    if existing:
        same = (
            existing.get("destination") == destination
            and existing.get("span_id") == span_id
            and existing.get("usage_digest") == digest
        )
        if same:
            return result, existing, True
        result["conflicts"][accounting_id] = {
            "accounting_id": accounting_id,
            "reason": "accounting placement or usage changed after queueing",
            "existing": existing,
            "candidate": {
                "destination": destination,
                "span_id": span_id,
                "usage_digest": digest,
            },
        }
        return result, existing, False
    entry = {
        "accounting_id": accounting_id,
        "destination": destination,
        "span_id": span_id,
        "usage_digest": digest,
        "emitted": False,
        "last_error": None,
    }
    placements[accounting_id] = entry
    return result, entry, True


def mark_placement_error(state: dict[str, Any], accounting_id: str, error: str) -> dict[str, Any]:
    result = _state(state)
    entry = _mapping(result["placements"].get(accounting_id))
    if entry:
        entry["last_error"] = error
        result["placements"][accounting_id] = entry
    return result


def mark_placement_delivered(state: dict[str, Any], accounting_id: str) -> dict[str, Any]:
    """Record an externally confirmed delivery, never inferred from a job file.

    The current detached worker has no transactional callback into this
    ledger.  This hook is deliberately available for a future confirmed
    transport, while ordinary queueing leaves ``emitted`` false.
    """
    result = _state(state)
    entry = _mapping(result["placements"].get(accounting_id))
    if entry:
        entry["emitted"] = True
        entry["last_error"] = None
        result["placements"][accounting_id] = entry
    return result


def remove_export_state(config: Config, stored_session_id: str) -> None:
    """Remove export state only for an explicit export-ledger reset.

    Projection rebuilds must not call this function.
    """
    directory = _directory(config, stored_session_id)
    with locked(export_lock_path(directory), LockMode.EXCLUSIVE):
        fsops.unlink(export_state_path(directory), missing_ok=True)
        fsops.sync_directory(directory)
