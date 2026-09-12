"""Durable Copilot export eligibility and accounting-placement state.

This state intentionally has no relationship to projection state.  Projection
state is disposable and rebuilt from the V1 archive; export placement is an
accounting decision and survives rebuilds.  The accounting worker cannot
atomically acknowledge a remote collector and this file, so an absent job is
never treated as proof of delivery.  A crash after a remote flush can still
lead to a deterministic retry and therefore a duplicate remote span.

``record_placement`` takes an explicit ``delivered`` flag from the caller
(who reads the generic transport's independent, durable
``otel_export.accounting_export_sent`` claim before calling in).  That flag,
not merely a changed destination, is what turns a correction into a
quarantined conflict: a correction to a job that never left this machine is
always safe to replace, while a correction after confirmed delivery can no
longer relocate tokens the remote collector already has.
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
        "turn_errors": {},
    }


def _ids(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({value for value in values if isinstance(value, str) and value})


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize(value: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults on a dict already known to carry the current schema.

    Only safe for state this module itself produced (loaded and version
    checked by :func:`_read`, or built fresh by another function here) —
    never call this directly on unvalidated bytes off disk.
    """
    placements = _mapping(value.get("placements"))
    conflicts = _mapping(value.get("conflicts"))
    turn_errors = _mapping(value.get("turn_errors"))
    return {
        "schema_version": EXPORT_STATE_SCHEMA_VERSION,
        "activated": bool(value.get("activated", False)),
        "excluded_turn_ids": _ids(value.get("excluded_turn_ids")),
        "excluded_accounting_ids": _ids(value.get("excluded_accounting_ids")),
        "placements": {
            key: item for key, item in placements.items() if isinstance(key, str) and isinstance(item, dict)
        },
        "conflicts": {
            key: item for key, item in conflicts.items() if isinstance(key, str) and isinstance(item, dict)
        },
        "turn_errors": {
            key: value2 for key, value2 in turn_errors.items() if isinstance(key, str) and isinstance(value2, str)
        },
    }


def _read(directory: Path) -> dict[str, Any]:
    """Load the ledger, failing closed on anything but a genuinely absent file.

    A missing file is the only case that legitimately means "no export has
    ever run for this session" and may start from empty state. Malformed
    JSON, a non-object document, or an unrecognized ``schema_version`` are
    all corruption or a future/unknown format from this module's point of
    view: silently treating them as empty would forget every placement this
    ledger recorded and risk emitting tokens at a second location.
    """
    raw = read_json_object(export_state_path(directory), invalid_message="invalid Copilot export ledger")
    if raw is None:
        return empty_export_state()
    version = raw.get("schema_version")
    if version != EXPORT_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported Copilot export ledger schema_version: {version!r}")
    return _normalize(raw)


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


def update_export_state(config: Config, stored_session_id: str, update: Any) -> dict[str, Any]:
    """Atomically apply ``update(state)`` and return the published state."""
    directory = _directory(config, stored_session_id)
    with locked(export_lock_path(directory), LockMode.EXCLUSIVE):
        state = _read(directory)
        updated = update(state)
        if not isinstance(updated, dict):
            raise TypeError("Copilot export state update must return a dictionary")
        state = _normalize(updated)
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

    Callers are responsible for passing only turn ids and accounting ids that
    are *actually* already historical (a terminal main interaction, or
    accounting owned by one) — an interaction still open at first activation
    must not appear here, or it would stay excluded forever even once it
    later completes.
    """
    result = _normalize(state)
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
        result["excluded_accounting_ids"] = sorted(set(result["excluded_accounting_ids"]) - usage_ids)
    return result


def record_placement(
    state: dict[str, Any],
    *,
    accounting_id: str,
    destination: str,
    span_id: str,
    usage: dict[str, Any],
    delivered: bool = False,
    old_job_state: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Persist the first token location for an accounting identity.

    ``delivered`` is the caller's fresh read of the generic transport's
    durable delivery claim for this identity (see
    ``otel_export.accounting_export_sent``), not this ledger's own possibly
    stale ``emitted`` flag — the local job file backing that flag is deleted
    by the worker on success, so this ledger cannot detect delivery on its
    own and must be told.

    ``old_job_state`` is the Copilot transport's result after atomically
    attempting to cancel the *existing* placement's own job. ``"cancelled"``
    means relocation is safe. ``"claimed"`` means a worker holds it right now
    and could deliver at any moment; ``"emitted"`` means it already completed.
    Both states quarantine the correction.

    Returns ``(state, entry, accepted)``.  A correction is only a durable
    conflict when the existing placement was (or is now known to have been)
    delivered, or when its job might still be in flight: rebinding a
    destination or usage value the remote collector already has, or might
    still receive, risks two different token totals for the same logical
    call. A correction to a placement proven to have never been queued, or
    proven to have permanently failed, is always safe to replace/requeue.
    """
    result = _normalize(state)
    digest = usage_digest(usage)
    placements = result["placements"]
    existing = _mapping(placements.get(accounting_id))
    if existing:
        already_delivered = bool(existing.get("emitted")) or delivered
        same = (
            existing.get("destination") == destination
            and existing.get("span_id") == span_id
            and existing.get("usage_digest") == digest
        )
        if same:
            if delivered and not existing.get("emitted"):
                existing = {**existing, "emitted": True, "last_error": None}
                placements[accounting_id] = existing
            return result, existing, True
        if already_delivered or old_job_state in {"claimed", "emitted"}:
            if delivered and not existing.get("emitted"):
                # The candidate is rejected, but the fresh delivery read is
                # still new information about the *existing* placement —
                # record it so the ledger stops looking like it was only
                # ever queued.
                existing = {**existing, "emitted": True}
                placements[accounting_id] = existing
            reason = (
                "accounting placement or usage changed after confirmed delivery"
                if already_delivered
                else "accounting placement or usage changed while the prior job was in flight"
            )
            result["conflicts"][accounting_id] = {
                "accounting_id": accounting_id,
                "reason": reason,
                "existing": existing,
                "candidate": {
                    "destination": destination,
                    "span_id": span_id,
                    "usage_digest": digest,
                },
            }
            return result, existing, False
        # Nothing was ever confirmed delivered, and no job is provably in
        # flight, for this identity: replace the queued/failed placement
        # outright rather than quarantining it.
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
    entry = {
        "accounting_id": accounting_id,
        "destination": destination,
        "span_id": span_id,
        "usage_digest": digest,
        "emitted": bool(delivered),
        "last_error": None,
    }
    placements[accounting_id] = entry
    return result, entry, True


def mark_placement_error(state: dict[str, Any], accounting_id: str, error: str) -> dict[str, Any]:
    result = _normalize(state)
    entry = _mapping(result["placements"].get(accounting_id))
    if entry:
        entry["last_error"] = error
        result["placements"][accounting_id] = entry
    return result


def clear_placement_error(state: dict[str, Any], accounting_id: str) -> dict[str, Any]:
    """Drop a stale ``last_error`` once a retried job successfully queues."""
    result = _normalize(state)
    entry = _mapping(result["placements"].get(accounting_id))
    if entry and entry.get("last_error") is not None:
        entry["last_error"] = None
        result["placements"][accounting_id] = entry
    return result


def mark_placement_job_status(
    state: dict[str, Any], accounting_id: str, status: dict[str, Any]
) -> dict[str, Any]:
    """Persist the worker lifecycle state and its latest delivery error."""
    result = _normalize(state)
    entry = _mapping(result["placements"].get(accounting_id))
    if not entry:
        return result
    entry["job_state"] = status.get("state")
    entry["job_attempt"] = status.get("attempt")
    worker_error = status.get("last_error")
    if isinstance(worker_error, str) and worker_error:
        entry["last_error"] = worker_error
    elif status.get("state") in {"queued", "claimed"}:
        entry["last_error"] = None
    if status.get("state") == "emitted":
        entry["emitted"] = True
        entry["last_error"] = None
    result["placements"][accounting_id] = entry
    return result


def mark_placement_delivered(state: dict[str, Any], accounting_id: str) -> dict[str, Any]:
    """Record an externally confirmed delivery, never inferred from a job file.

    Ordinary reconciliation no longer needs to call this directly —
    ``record_placement``'s ``delivered`` flag self-heals ``emitted`` from the
    durable transport claim on every pass — but it remains available for a
    caller with its own confirmed-delivery source.
    """
    result = _normalize(state)
    entry = _mapping(result["placements"].get(accounting_id))
    if entry:
        entry["emitted"] = True
        entry["last_error"] = None
        result["placements"][accounting_id] = entry
    return result


def mark_turn_error(state: dict[str, Any], turn_id: str, error: str) -> dict[str, Any]:
    """Record that a whole-turn export job failed to queue locally.

    Kept separate from ``placements`` (keyed by accounting id, not turn id)
    so a turn-job failure is visible without inventing a fake accounting
    record for it.
    """
    result = _normalize(state)
    result["turn_errors"][turn_id] = error
    return result


def clear_turn_error(state: dict[str, Any], turn_id: str) -> dict[str, Any]:
    result = _normalize(state)
    result["turn_errors"].pop(turn_id, None)
    return result


def remove_export_state(config: Config, stored_session_id: str) -> None:
    """Remove export state only for an explicit export-ledger reset.

    Projection rebuilds must not call this function.
    """
    directory = _directory(config, stored_session_id)
    with locked(export_lock_path(directory), LockMode.EXCLUSIVE):
        fsops.unlink(export_state_path(directory), missing_ok=True)
        fsops.sync_directory(directory)
