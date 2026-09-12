"""Archive-only composition for Copilot's local V2 projection.

Reconciliation deliberately reads the immutable V1 archive rather than the
live Copilot home.  That makes it safe to run after a session has ended (or
after Copilot has removed its transient transcript files), and keeps capture
failures independent from derived-state failures.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from thirdeye.config import Config

from .archive import iter_captured_records
from .projection import build_projection
from .projection_store import (
    commit_projection,
    load_projection_state,
    reset_projection_state,
)
from .types import Projection

_COUNT_KEYS = ("events", "usage", "turns", "exports", "pending", "ambiguous", "conflicting")


def _empty_result(*, errors: int = 0) -> dict[str, int]:
    result = {key: 0 for key in _COUNT_KEYS}
    result["errors"] = errors
    return result


def _stored_counts(state: dict[str, Any]) -> dict[str, int]:
    """Return safe status counts when a projection attempt cannot proceed."""

    result = _empty_result()
    totals = state.get("index_totals")
    if not isinstance(totals, dict):
        return result
    for source, target in (
        ("events", "events"),
        ("usage", "usage"),
        ("turns", "turns"),
        ("pending", "pending"),
    ):
        value = totals.get(source)
        if isinstance(value, int) and value >= 0:
            result[target] = value
    return result


def _attribution_counts(projection: Projection) -> tuple[int, int]:
    ambiguous = 0
    conflicting = 0
    for attribution in projection["attributions"]:
        status = attribution.get("status")
        if status == "ambiguous":
            ambiguous += 1
        elif status == "conflicting":
            conflicting += 1
    return ambiguous, conflicting


def _diagnostic_errors(projection: Projection) -> int:
    return sum(
        1
        for diagnostic in projection["diagnostics"]
        if diagnostic.get("severity") == "error"
    )


def queue_exports(config: Config, stored_session_id: str, projection: Projection) -> int:
    """Queue export work only when an explicit caller requests it.

    Export assembly is intentionally not an import-time dependency of local
    reconciliation.  Runtime integration can call this boundary after the
    export implementation is installed; archive-only callers never load it.
    """

    module = import_module(".export", package=__package__)
    enqueue = getattr(module, "queue_exports")
    if not callable(enqueue):
        raise TypeError("Copilot export assembly does not provide queue_exports")
    exporter: Callable[[Config, str, Projection], int] = enqueue
    return exporter(config, stored_session_id, projection)


def reconcile_archive(
    config: Config,
    stored_session_id: str,
    *,
    rebuild: bool = False,
    export: bool = False,
) -> dict[str, int]:
    """Rebuild local Copilot projections from the immutable V1 archive.

    A normal reconciliation is a complete archive replay.  Stable derived
    identities make committing that replay idempotent, while a full replay
    keeps late usage rows eligible to join semantic evidence captured in an
    earlier pass.  ``rebuild`` deletes only reproducible projection state;
    raw archive records and the separate export ledger are untouched.

    Errors are reported as counts instead of escaping so source capture can
    remain operational when a derived projection is malformed.  Export is a
    second, opt-in phase performed only after the local commit succeeds.
    """

    try:
        # Build a replacement before deleting a previous projection.  This is
        # especially important for an operator-triggered rebuild: malformed
        # archived evidence must not turn a recoverable projection error into
        # a loss of the last readable local view.  Rebuild input is empty so
        # no unfinished incremental state can leak into the full replay.
        prior_state = {} if rebuild else load_projection_state(config, stored_session_id)
        records = list(iter_captured_records(config, stored_session_id))
        projection, next_state = build_projection(records, prior_state)
        next_state["archive_source_ids"] = [record["source_id"] for record in records]
        if rebuild:
            reset_projection_state(config, stored_session_id)
        counts = commit_projection(config, stored_session_id, projection, next_state)
    except Exception:
        # Do not reset or mutate V1 capture state when derivation fails.
        try:
            result = _stored_counts(load_projection_state(config, stored_session_id))
        except Exception:
            result = _empty_result()
        result["errors"] += 1
        return result

    ambiguous, conflicting = _attribution_counts(projection)
    result = {
        "events": counts["events"],
        "usage": counts["usage"],
        "turns": counts["turns"],
        "exports": 0,
        "pending": counts["pending"],
        "ambiguous": ambiguous,
        "conflicting": conflicting,
        "errors": _diagnostic_errors(projection),
    }
    if not export:
        return result

    try:
        result["exports"] = queue_exports(config, stored_session_id, projection)
    except Exception:
        # Export delivery must never roll back a successful local projection.
        result["errors"] += 1
    return result
