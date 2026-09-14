"""Assemble eligible Copilot projections into generic detached OTel jobs.

No network operation happens here.  This module only writes/dispatches the
generic transport's local jobs; the detached worker performs remote delivery.
The split cannot provide transactional exactly-once delivery: a crash after a
remote flush and before acknowledgement can retry a deterministic span. The
transport's durable turn claims and Copilot's accounting delivery claims are
what let this module tell "already confirmed delivered" apart from "merely
queued" across restarts, since workers delete their job files on success.
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

from . import export_transport
from .constants import PLATFORM_NAME
from .export_state import (
    clear_turn_error,
    initialize_eligibility,
    is_accounting_eligible,
    is_turn_eligible,
    mark_turn_error,
    record_placement,
    set_pending_unowned_accounting,
    update_export_state,
    usage_digest,
)
from .types import Projection

_TERMINAL = frozenset({"completed", "interrupted", "errored"})


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
    projection: Projection,
    root_index: dict[str, dict[str, Any]],
    *,
    session_closed: bool,
) -> list[str]:
    """Accounting identities that are already part of terminal history.

    An identity owned by a still-open interaction stays eligible for later
    export once that interaction completes. Missing ownership in an open
    session is likewise provisional and cannot become immutable history merely
    because transcript reconstruction currently lags the usage row. Once the
    session is explicitly closed, a non-pending row with no resolved root is
    settled session history. A pending join remains unresolved in either case.
    """
    owners: dict[str, dict[str, Any] | None] = {}
    for accounting_id, (owner, _item) in _accounting_calls(projection).items():
        owners[accounting_id] = _root_for(root_index, owner.get("turn_id"))
    unresolved: set[str] = set()
    for attribution in projection["attributions"]:
        accounting_id = attribution["logical_call_id"]
        if accounting_id in owners:
            continue
        if attribution["status"] == "matched":
            unresolved.add(accounting_id)
            continue
        if attribution["status"] == "pending":
            unresolved.add(accounting_id)
            continue
        root = _root_for(root_index, attribution["stored_turn_id"])
        if root is None and not session_closed:
            unresolved.add(accounting_id)
            continue
        owners[accounting_id] = root
    for row in projection["usage_rows"]:
        if row.call_id in unresolved:
            continue
        if row.call_id not in owners and not session_closed:
            continue
        owners.setdefault(row.call_id, None)
    return sorted(
        accounting_id for accounting_id, root in owners.items() if root is None or _terminal(root)
    )


def _turn_span_id(stored_session_id: str, turn_id: str) -> str:
    """Return the OTel-safe deterministic span id for a source-derived turn."""
    return str(chat_span_id(PLATFORM_NAME, stored_session_id, f"turn:{turn_id}"))


def _configured(config: Config) -> bool:
    return config.logfire.enabled and bool(config.logfire.token)


def _turn_error_state(config: Config, stored_session_id: str, turn_id: str, message: str) -> None:
    def update(state: dict[str, Any]) -> dict[str, Any]:
        return mark_turn_error(state, turn_id, message)

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
    session_closed: bool,
) -> dict[str, Any]:
    terminal_ids = [str(turn["turn_id"]) for turn in _main_terminal_turns(projection)]

    def update(state: dict[str, Any]) -> dict[str, Any]:
        return initialize_eligibility(
            state,
            terminal_turn_ids=terminal_ids,
            accounting_ids=_historical_accounting_ids(
                projection, root_index, session_closed=session_closed
            ),
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
        existing = state.get("placements", {}).get(accounting_id) or {}
        old_span_id = existing.get("span_id")
        old_job_state: str | None = None
        digest = usage_digest(usage)
        usage_changed = existing.get("usage_digest") != digest
        if isinstance(old_span_id, str) and old_span_id and old_span_id != span_id:
            # Relocating to a different span: find out whether the job under
            # the *old* span id is provably inert before deciding it is safe
            # to replace, and if it is, cancel it so an already-dispatched
            # worker can never deliver tokens this ledger no longer points
            # at once it decides on the new destination below.
            status = export_transport.cancel(config.root, old_span_id)
            old_job_state = status.get("state")
        elif (
            isinstance(old_span_id, str)
            and old_span_id == span_id
            and usage_changed
            and not delivered
        ):
            # Same span, new usage, still undelivered: rewrite the queued
            # payload. A claimed in-flight job cannot be overwritten.
            current = export_transport.status(config.root, old_span_id)
            current_state = (current or {}).get("state")
            if current_state == "claimed":
                old_job_state = "claimed"
            elif current_state in {"queued", "retrying", "failed"}:
                status = export_transport.cancel(config.root, old_span_id)
                old_job_state = status.get("state")
        next_state, entry, accepted = record_placement(
            state,
            accounting_id=accounting_id,
            destination=destination,
            span_id=span_id,
            usage=usage,
            delivered=delivered,
            old_job_state=old_job_state,
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
    session_closed = meta.status == "closed"
    root_index = _root_owner_index(projection)
    state = _eligible_state(
        config,
        stored_session_id,
        projection,
        root_index,
        include_history=include_history,
        session_closed=session_closed,
    )
    if not _configured(config):
        return 0

    accounting = _accounting_calls(projection)
    placements: dict[str, dict[str, Any]] = {}
    deferred_ids: set[str] = set()
    blocked_root_ids: set[str] = set()
    queued = 0

    # Place terminal accounting only when its owning chat call is present.
    # Copilot can publish a matched usage row before transcript reconstruction
    # exposes that call, including briefly after an interaction's sessionEnd.
    for accounting_id, (owner, item) in accounting.items():
        owner_id = owner.get("turn_id")
        root = _root_for(root_index, owner_id)
        usage = item["usage"]
        if (
            not isinstance(owner_id, str)
            or root is None
            or not _terminal(root)
            or item.get("attribution_status") != "matched"
        ):
            continue
        root_id = str(root["turn_id"])
        if not is_turn_eligible(state, root_id) or not is_accounting_eligible(state, accounting_id):
            continue
        call_id = item.get("call_id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not any(call.get("call_id") == call_id for call in owner.get("llm_calls") or [])
        ):
            deferred_ids.add(accounting_id)
            blocked_root_ids.add(root_id)
            continue
        existing_entry = (state.get("placements") or {}).get(accounting_id) or {}
        turn_sent = otel_export.turn_export_sent(directory, root_id)
        delivered = export_transport.delivery_sent(directory, accounting_id) or (
            turn_sent and existing_entry.get("destination") == "chat-span"
        )
        # A chat span already flushed to Logfire is immutable. Usage matched
        # only after that flush is omitted because Copilot does not emit
        # synthetic accounting spans. An existing chat placement remains
        # valid because the turn delivery is what confirmed it.
        chat_available = not turn_sent or (
            delivered and existing_entry.get("destination") == "chat-span"
        )
        if not chat_available:
            continue
        destination = "chat-span"
        span_id = str(call_id)
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

    # A matched attribution can precede reconstruction of its owning call.
    # Keep it local until it can ride inside the normal turn export. Ambiguous
    # accounting is intentionally omitted rather than emitted on a synthetic
    # fallback span.
    attached_ids = set(accounting)
    for attribution in projection["attributions"]:
        accounting_id = attribution["logical_call_id"]
        if attribution["status"] != "matched":
            continue
        if accounting_id in attached_ids or not is_accounting_eligible(state, accounting_id):
            continue
        call_id = attribution.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        owner_id = attribution.get("stored_turn_id")
        root = _root_for(root_index, owner_id)
        deferred_ids.add(accounting_id)
        if root is not None:
            blocked_root_ids.add(str(root["turn_id"]))

    state = update_export_state(
        config,
        stored_session_id,
        lambda current: set_pending_unowned_accounting(current, sorted(deferred_ids)),
    )

    # Generic transport provides a persistent completed-turn claim.  Sending
    # a duplicate local job is harmless there; no network I/O occurs here.
    for turn in _main_terminal_turns(projection):
        turn_id = str(turn["turn_id"])
        if turn_id in blocked_root_ids or not is_turn_eligible(state, turn_id):
            continue
        assembled = _with_deterministic_turn_ids(
            _turn_with_placed_accounting(turn, placements), stored_session_id
        )
        sent = export_transport.queue_turn(
            config, directory, stored_session_id, meta.cwd, assembled
        )
        if sent:
            queued += 1
            if turn_id in (state.get("turn_errors") or {}):
                _clear_turn_error_state(config, stored_session_id, turn_id)
        else:
            _turn_error_state(config, stored_session_id, turn_id, "turn export job was not queued")
    return queued
