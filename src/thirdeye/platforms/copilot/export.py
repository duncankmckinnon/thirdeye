"""Assemble eligible Copilot projections into generic detached OTel jobs.

No network operation happens here.  This module only writes/dispatches the
generic transport's local jobs; the detached worker performs remote delivery.
The split cannot provide transactional exactly-once delivery: a crash after a
remote flush and before acknowledgement can retry a deterministic span. The
transport's own durable claims (`otel_export.turn_export_sent` /
`accounting_export_sent`) are what let this module tell "already confirmed
delivered" apart from "merely queued" across restarts, since the worker
deletes its own job file on success.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from thirdeye import otel_export
from thirdeye.config import Config
from thirdeye.meta import read_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.span_ids import chat_span_id

from .constants import PLATFORM_NAME
from .export_state import (
    clear_placement_error,
    clear_turn_error,
    initialize_eligibility,
    is_accounting_eligible,
    is_turn_eligible,
    mark_placement_error,
    mark_turn_error,
    record_placement,
    update_export_state,
)
from .types import Projection

_TERMINAL = frozenset({"completed", "interrupted", "errored"})
_EXPORTABLE_ATTRIBUTIONS = frozenset({"matched", "ambiguous"})


def _directory(config: Config, stored_session_id: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored_session_id)


def _terminal(turn: dict[str, Any] | None) -> bool:
    return bool(turn) and turn.get("status") in _TERMINAL


def _walk_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for turn in turns:
        found.append(turn)
        children = turn.get("subagents")
        if isinstance(children, list):
            found.extend(_walk_turns([child for child in children if isinstance(child, dict)]))
    return found


def _main_terminal_turns(projection: Projection) -> list[dict[str, Any]]:
    return [
        turn
        for turn in projection["turns"]
        if isinstance(turn, dict)
        and _terminal(turn)
        and not isinstance((turn.get("attributes") or {}).get("agent_id"), str)
    ]


def _root_owner_index(projection: Projection) -> dict[str, dict[str, Any]]:
    """Map every turn id, main or nested, to its owning main ``TurnSpanDict``.

    Eligibility and "is this history" are properties of a whole interaction
    (a main turn and everything nested under it, exported together as one
    job), never of a nested child's own id — the child is never exported as
    an independent top-level job, so its id alone is never a member of the
    boundary lists built from ``_main_terminal_turns``.
    """
    index: dict[str, dict[str, Any]] = {}

    def walk(turn: dict[str, Any], root: dict[str, Any]) -> None:
        turn_id = turn.get("turn_id")
        if isinstance(turn_id, str):
            index[turn_id] = root
        for child in turn.get("subagents") or []:
            if isinstance(child, dict):
                walk(child, root)

    for turn in projection["turns"]:
        if isinstance(turn, dict):
            walk(turn, turn)
    return index


def _root_for(root_index: dict[str, dict[str, Any]], turn_id: object) -> dict[str, Any] | None:
    if isinstance(turn_id, str):
        return root_index.get(turn_id)
    return None


def _accounting_calls(projection: Projection) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Map an identity to its owner turn and serialized generic accounting."""
    found: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for turn in _walk_turns([turn for turn in projection["turns"] if isinstance(turn, dict)]):
        for item in turn.get("accounting_calls") or []:
            if not isinstance(item, dict):
                continue
            accounting_id = item.get("accounting_id")
            usage = item.get("usage")
            if isinstance(accounting_id, str) and isinstance(usage, dict):
                found[accounting_id] = (turn, item)
    return found


def _historical_accounting_ids(
    projection: Projection, root_index: dict[str, dict[str, Any]]
) -> list[str]:
    """Accounting identities that are already part of terminal history.

    An identity owned by a still-open interaction must be excluded from this
    set (and therefore stay eligible for later export once that interaction
    completes). An identity with no interaction to gate on at all (an
    ownerless usage row, or a ``stored_turn_id`` this projection cannot
    resolve) has no way to become "no longer historical" later, so it keeps
    the conservative default of being treated as already-seen history.
    """
    owners: dict[str, dict[str, Any] | None] = {}
    for accounting_id, (owner, _item) in _accounting_calls(projection).items():
        owners[accounting_id] = _root_for(root_index, owner.get("turn_id"))
    for attribution in projection["attributions"]:
        accounting_id = attribution["logical_call_id"]
        if accounting_id not in owners:
            owners[accounting_id] = _root_for(root_index, attribution["stored_turn_id"])
    for row in projection["usage_rows"]:
        owners.setdefault(row.call_id, None)
    return sorted(
        accounting_id for accounting_id, root in owners.items() if root is None or _terminal(root)
    )


def _span_id(session_id: str, turn_id: str | None, accounting_id: str) -> str:
    if turn_id is None:
        return f"accounting:{session_id}:{accounting_id}"
    return f"accounting:{session_id}:{turn_id}:{accounting_id}"


def _turn_span_id(stored_session_id: str, turn_id: str) -> str:
    """Return the OTel-safe deterministic span id for a source-derived turn."""
    return str(chat_span_id(PLATFORM_NAME, stored_session_id, f"turn:{turn_id}"))


def _configured(config: Config) -> bool:
    return config.logfire.enabled and bool(config.logfire.token)


def _error_state(config: Config, stored_session_id: str, accounting_id: str, message: str) -> None:
    def update(state: dict[str, Any]) -> dict[str, Any]:
        return mark_placement_error(state, accounting_id, message)

    update_export_state(config, stored_session_id, update)


def _turn_error_state(config: Config, stored_session_id: str, turn_id: str, message: str) -> None:
    def update(state: dict[str, Any]) -> dict[str, Any]:
        return mark_turn_error(state, turn_id, message)

    update_export_state(config, stored_session_id, update)


def _clear_error_state(config: Config, stored_session_id: str, accounting_id: str) -> None:
    def update(state: dict[str, Any]) -> dict[str, Any]:
        return clear_placement_error(state, accounting_id)

    update_export_state(config, stored_session_id, update)


def _clear_turn_error_state(config: Config, stored_session_id: str, turn_id: str) -> None:
    def update(state: dict[str, Any]) -> dict[str, Any]:
        return clear_turn_error(state, turn_id)

    update_export_state(config, stored_session_id, update)


def _eligible_state(
    config: Config,
    stored_session_id: str,
    projection: Projection,
    root_index: dict[str, dict[str, Any]],
    *,
    include_history: bool,
) -> dict[str, Any]:
    terminal_ids = [str(turn["turn_id"]) for turn in _main_terminal_turns(projection)]

    def update(state: dict[str, Any]) -> dict[str, Any]:
        return initialize_eligibility(
            state,
            terminal_turn_ids=terminal_ids,
            accounting_ids=_historical_accounting_ids(projection, root_index),
            include_history=include_history,
        )

    return update_export_state(config, stored_session_id, update)


def _place(
    config: Config,
    stored_session_id: str,
    *,
    accounting_id: str,
    destination: str,
    span_id: str,
    usage: dict[str, Any],
    delivered: bool,
) -> tuple[dict[str, Any] | None, bool]:
    captured: dict[str, Any] = {}

    def update(state: dict[str, Any]) -> dict[str, Any]:
        next_state, entry, accepted = record_placement(
            state,
            accounting_id=accounting_id,
            destination=destination,
            span_id=span_id,
            usage=usage,
            delivered=delivered,
        )
        captured["entry"] = entry
        captured["accepted"] = accepted
        return next_state

    update_export_state(config, stored_session_id, update)
    return captured.get("entry"), bool(captured.get("accepted"))


def _turn_with_placed_accounting(
    turn: dict[str, Any], placements: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Return a recursive turn copy containing only its chat-placed tokens."""
    result = deepcopy(turn)
    calls = []
    for accounting in result.get("accounting_calls") or []:
        entry = placements.get(str(accounting.get("accounting_id")))
        if entry and entry.get("destination") == "chat-span":
            calls.append(accounting)
    result["accounting_calls"] = calls
    result["subagents"] = [
        _turn_with_placed_accounting(child, placements)
        for child in result.get("subagents") or []
        if isinstance(child, dict)
    ]
    return result


def _with_deterministic_turn_ids(turn: dict[str, Any], stored_session_id: str) -> dict[str, Any]:
    """Finalize a complete turn tree with ids stable across archive replays.

    Copilot's stable turn identity is a source-derived string rather than the
    integer sequence used by the older live adapters.  ``chat_span_id`` is a
    domain-separated deterministic 64-bit id function, and the ``turn:``
    prefix keeps this turn namespace distinct from model-call ids.
    """
    result = deepcopy(turn)
    turn_id = result.get("turn_id")
    if isinstance(turn_id, str) and turn_id:
        result["turn_span_id"] = _turn_span_id(stored_session_id, turn_id)
    result["subagents"] = [
        _with_deterministic_turn_ids(child, stored_session_id)
        for child in result.get("subagents") or []
        if isinstance(child, dict)
    ]
    return result


def queue_exports(
    config: Config,
    stored_session_id: str,
    projection: Projection,
    *,
    include_history: bool = False,
) -> int:
    """Queue eligible terminal projection evidence without reading live Copilot data.

    ``include_history`` is the explicit ``--export`` opt-in supplied by
    runtime composition.  The default first activation records all currently
    terminal evidence as local-only.  Subsequent terminal interactions and
    later accounting rows are eligible.  The returned count is local jobs
    accepted for dispatch, not confirmed remote delivery.
    """
    directory = _directory(config, stored_session_id)
    meta = read_meta(meta_path(directory))
    if meta is None:
        raise ValueError(f"unknown Copilot session: {stored_session_id}")
    root_index = _root_owner_index(projection)
    state = _eligible_state(
        config, stored_session_id, projection, root_index, include_history=include_history
    )
    if not _configured(config):
        return 0

    accounting = _accounting_calls(projection)
    placements: dict[str, dict[str, Any]] = {}
    queued = 0

    # Place terminal owned accounting first.  This decision happens before a
    # whole-turn job is assembled, so a future local match cannot move a
    # fallback charge onto a chat span.
    for accounting_id, (owner, item) in accounting.items():
        owner_id = owner.get("turn_id")
        root = _root_for(root_index, owner_id)
        usage = item["usage"]
        if (
            not isinstance(owner_id, str)
            or root is None
            or not _terminal(root)
            or item.get("attribution_status") not in _EXPORTABLE_ATTRIBUTIONS
        ):
            continue
        root_id = str(root["turn_id"])
        if not is_turn_eligible(state, root_id) or not is_accounting_eligible(state, accounting_id):
            continue
        call_id = item.get("call_id")
        # A chat span already flushed to Logfire is immutable history: usage
        # that resolves to "matched" only *after* that flush can no longer
        # land on it and must use the turn-owned fallback span instead.
        chat_available = not otel_export.turn_export_sent(directory, root_id)
        if (
            chat_available
            and item.get("attribution_status") == "matched"
            and isinstance(call_id, str)
            and call_id
        ):
            destination = "chat-span"
            span_id = str(call_id)
        else:
            destination = "turn-accounting-span"
            span_id = _span_id(stored_session_id, owner_id, accounting_id)
        delivered = otel_export.accounting_export_sent(directory, accounting_id)
        entry, accepted = _place(
            config,
            stored_session_id,
            accounting_id=accounting_id,
            destination=destination,
            span_id=span_id,
            usage=usage,
            delivered=delivered,
        )
        if not accepted:
            continue
        if entry is not None:
            placements[accounting_id] = entry
        if entry is not None and entry.get("emitted"):
            # Confirmed delivered, whether just now or on an earlier pass.
            # Never recreate an accounting job after that point.
            continue
        if destination != "turn-accounting-span" or entry is None:
            continue
        sent = otel_export.export_turn_accounting(
            config,
            directory,
            stored_session_id,
            PLATFORM_NAME,
            meta.cwd,
            owner_id,
            item,
            turn_span_id=_turn_span_id(stored_session_id, owner_id),
        )
        if sent:
            queued += 1
            if entry.get("last_error") is not None:
                _clear_error_state(config, stored_session_id, accounting_id)
        else:
            _error_state(config, stored_session_id, accounting_id, "accounting job was not queued")

    # Usage with no known user-turn owner is intentionally a session accounting
    # job.  It never manufactures a prompt/turn merely to satisfy tracing.
    rows = {row.call_id: row.to_dict() for row in projection["usage_rows"]}
    attached_ids = set(accounting)
    for attribution in projection["attributions"]:
        accounting_id = attribution["logical_call_id"]
        if attribution["status"] not in _EXPORTABLE_ATTRIBUTIONS:
            # Pending attribution is intentionally local-only until later
            # archive evidence resolves it.  A conflicting join is likewise
            # quarantined rather than guessed into an accounting span.
            continue
        if accounting_id in attached_ids or not is_accounting_eligible(state, accounting_id):
            continue
        usage = rows.get(accounting_id)
        if usage is None:
            continue
        span_id = _span_id(stored_session_id, None, accounting_id)
        delivered = otel_export.accounting_export_sent(directory, accounting_id)
        entry, accepted = _place(
            config,
            stored_session_id,
            accounting_id=accounting_id,
            destination="session-accounting-span",
            span_id=span_id,
            usage=usage,
            delivered=delivered,
        )
        if not accepted or entry is None:
            continue
        if entry.get("emitted"):
            continue
        item = {
            "accounting_id": accounting_id,
            "usage": usage,
            "attribution_status": attribution["status"],
            "agent_id": attribution["agent_id"],
            "call_id": None,
            "attributes": {
                "logical_call_id": accounting_id,
                "usage_source_id": attribution["usage_source_id"],
                "evidence": list(attribution["evidence"]),
            },
        }
        sent = otel_export.export_session_accounting(
            config, directory, stored_session_id, PLATFORM_NAME, meta.cwd, item
        )
        if sent:
            queued += 1
            if entry.get("last_error") is not None:
                _clear_error_state(config, stored_session_id, accounting_id)
        else:
            _error_state(config, stored_session_id, accounting_id, "accounting job was not queued")

    # Generic transport provides a persistent completed-turn claim.  Sending
    # a duplicate local job is harmless there; no network I/O occurs here.
    for turn in _main_terminal_turns(projection):
        turn_id = str(turn["turn_id"])
        if not is_turn_eligible(state, turn_id):
            continue
        assembled = _with_deterministic_turn_ids(
            _turn_with_placed_accounting(turn, placements), stored_session_id
        )
        sent = otel_export.export_turn(
            config, directory, stored_session_id, PLATFORM_NAME, meta.cwd, assembled
        )
        if sent:
            queued += 1
            if turn_id in (state.get("turn_errors") or {}):
                _clear_turn_error_state(config, stored_session_id, turn_id)
        else:
            _turn_error_state(config, stored_session_id, turn_id, "turn export job was not queued")
    return queued
