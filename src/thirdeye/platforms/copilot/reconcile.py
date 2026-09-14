"""Archive-only composition for Copilot's local V2 projection.

Reconciliation deliberately reads the immutable V1 archive rather than the
live Copilot home.  That makes it safe to run after a session has ended (or
after Copilot has removed its transient transcript files), and keeps capture
failures independent from derived-state failures.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from thirdeye.config import Config

from .archive import iter_captured_records
from .projection import build_projection
from .projection_store import commit_projection, load_projection_state, read_projection_status
from .types import Projection

_COUNT_KEYS = ("events", "usage", "turns", "exports", "pending", "ambiguous", "conflicting")


def _empty_result(*, errors: int = 0) -> dict[str, int]:
    result = {key: 0 for key in _COUNT_KEYS}
    result["errors"] = errors
    return result


def _failure_result(config: Config, stored_session_id: str) -> dict[str, int]:
    """Report the last successfully published projection's status on failure.

    The local projection on disk (if any) is untouched by a failed attempt,
    so its counts -- not zeros -- describe what a reader will actually see.
    """
    try:
        status = read_projection_status(config, stored_session_id)
    except Exception:
        return _empty_result(errors=1)
    result = {**_empty_result(), **status}
    result["errors"] = status.get("errors", 0) + 1
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
        1 for diagnostic in projection["diagnostics"] if diagnostic.get("severity") == "error"
    )


def queue_exports(
    config: Config,
    stored_session_id: str,
    projection: Projection,
    *,
    include_history: bool = False,
) -> int:
    """Queue export work only when an explicit caller requests it.

    Export assembly is intentionally not an import-time dependency of local
    reconciliation.  Runtime integration can call this boundary after the
    export implementation is installed; archive-only callers never load it.
    """

    module = import_module(".export", package=__package__)
    enqueue = module.queue_exports
    if not callable(enqueue):
        raise TypeError("Copilot export assembly does not provide queue_exports")
    exporter: Callable[..., int] = enqueue
    return exporter(config, stored_session_id, projection, include_history=include_history)


def reconcile_archive(
    config: Config,
    stored_session_id: str,
    *,
    rebuild: bool = False,
    export: bool = False,
    include_history: bool = False,
) -> dict[str, int]:
    """Rebuild local Copilot projections from the immutable V1 archive.

    A normal reconciliation is a complete archive replay.  Stable derived
    identities make committing that replay idempotent.  ``rebuild`` discards
    reproducible projection state as part of the same locked commit used for
    an incremental merge (see ``commit_projection(..., replace=True)``): raw
    archive records and the separate export ledger are untouched, and nothing
    is written to disk until the replacement projection passes validation, so
    a rebuild that fails validation (a malformed usage row, a stale schema)
    cannot erase the last readable local view.  A failure *after* that point
    -- the projection document itself durably publishes, but the derived
    usage-sidecar mirror then hits an I/O error -- still reports an error
    here, but the counts below reflect the document that was, in fact,
    committed; see ``commit_projection`` for why that distinction is not a
    bug.  Either way the next reconciliation call self-heals the sidecar.

    Loading prior state, building the new projection from it, and committing
    are not one locked operation -- ``build_projection`` runs unlocked so a
    slow archive replay does not hold the projection lock.  To close the gap
    that leaves, the ``commit_sequence`` observed here is passed through to
    ``commit_projection`` as ``base_commit_sequence``: if another writer
    committed in the meantime, the commit is refused (``ProjectionConflictError``,
    reported below as an error) instead of overwriting newer derived state
    with one built from what is now stale builder state.  This is checked
    even for ``rebuild``, which otherwise discards ``prior_state`` entirely.

    Errors are reported as counts instead of escaping so source capture can
    remain operational when a derived projection is malformed.  Export is a
    second, opt-in phase performed only after the local commit succeeds.
    """

    try:
        loaded_state = load_projection_state(config, stored_session_id)
        base_commit_sequence = loaded_state.get("commit_sequence")
        prior_state = {} if rebuild else loaded_state
        records = list(iter_captured_records(config, stored_session_id))
        projection, next_state = build_projection(records, prior_state)
        next_state["archive_source_ids"] = [record["source_id"] for record in records]
        counts = commit_projection(
            config,
            stored_session_id,
            projection,
            next_state,
            replace=rebuild,
            base_commit_sequence=base_commit_sequence,
        )
    except Exception:
        return _failure_result(config, stored_session_id)

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
        result["exports"] = queue_exports(
            config, stored_session_id, projection, include_history=include_history
        )
    except Exception:
        # Export delivery must never roll back a successful local projection.
        result["errors"] += 1
    return result
