"""Recoverable, versioned storage for Copilot's derived V2 projection.

This state deliberately lives beside, rather than inside, the V1 capture
checkpoint.  The capture archive is immutable evidence; deleting this file is
therefore a safe way to request a local projection rebuild.  It must never
touch the export ledger (which has its own lock and lifecycle).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops

from .types import PROJECTION_SCHEMA_VERSION

PROJECTION_STATE_FILENAME = "copilot.projection.state.json"
PROJECTION_JOURNAL_FILENAME = "copilot.projection.journal.json"
PROJECTION_LOCK_FILENAME = "copilot.projection.lock"
PROJECTION_JOURNAL_SCHEMA_VERSION = 1

# Kept private, just like the V1 archive fault hook.  It gives durability tests
# precise crash boundaries without making fault injection a runtime feature.
_fault_injector: Callable[[str], None] | None = None


def _fault(point: str) -> None:
    if _fault_injector is not None:
        _fault_injector(point)


def projection_state_path(session_dir: Path) -> Path:
    return session_dir / PROJECTION_STATE_FILENAME


def projection_journal_path(session_dir: Path) -> Path:
    return session_dir / PROJECTION_JOURNAL_FILENAME


def projection_lock_path(session_dir: Path) -> Path:
    return session_dir / PROJECTION_LOCK_FILENAME


def empty_projection_state() -> dict[str, Any]:
    """Return the public state shape used as pure-builder input."""
    return {
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "archive_source_ids": [],
        "semantic_state": {"open_interactions": {}},
        "accounting_state": {"logical_calls": {}},
        "projection_revision": "",
    }


def empty_projection_document() -> dict[str, Any]:
    """Return the private envelope around public state and replay indexes."""
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "state": empty_projection_state(),
        "indexes": {
            "events": {},
            "turns": {},
            "usage": {},
            "attributions": {},
            "pending": {},
            "diagnostics": {},
        },
    }


def _atomic_json(path: Path, value: dict[str, Any], *, fault_point: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsops.replace(temp_name, path)
        fsops.sync_directory(path.parent)
        _fault(fault_point)
    except BaseException:
        fsops.unlink(Path(temp_name), missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(fsops.read_text(path, encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"invalid Copilot projection state: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"invalid Copilot projection state: {path}")
    return value


def _validate_document(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != PROJECTION_SCHEMA_VERSION:
        raise ValueError("unsupported Copilot projection state schema")
    state = document.get("state")
    indexes = document.get("indexes")
    if not isinstance(state, dict) or not isinstance(indexes, dict):
        raise ValueError("invalid Copilot projection state")
    if state.get("projection_schema_version") != PROJECTION_SCHEMA_VERSION:
        raise ValueError("unsupported Copilot projection schema")
    for name in ("events", "turns", "usage", "attributions", "pending", "diagnostics"):
        if not isinstance(indexes.get(name), dict):
            raise ValueError("invalid Copilot projection indexes")
    return document


def read_projection_document(session_dir: Path) -> dict[str, Any]:
    """Read the durable document, returning an empty one before first use.

    Callers holding :func:`projection_lock_path` may rely on this to finish a
    previously published journal before reading.  Keeping recovery here makes
    it impossible for a later commit to merge against an incomplete snapshot.
    """
    journal = _read_json(projection_journal_path(session_dir))
    if journal is not None:
        if journal.get("schema_version") != PROJECTION_JOURNAL_SCHEMA_VERSION:
            raise ValueError("unsupported Copilot projection journal schema")
        document = journal.get("document")
        if not isinstance(document, dict):
            raise ValueError("invalid Copilot projection journal")
        _validate_document(document)
        _atomic_json(
            projection_state_path(session_dir), document, fault_point="after_projection_recovery"
        )
        fsops.unlink(projection_journal_path(session_dir), missing_ok=True)
        fsops.sync_directory(session_dir)
        _fault("after_projection_recovery_clear")
        return document

    document = _read_json(projection_state_path(session_dir))
    return empty_projection_document() if document is None else _validate_document(document)


def publish_projection_document(session_dir: Path, document: dict[str, Any]) -> None:
    """Publish ``document`` with write-ahead journaling.

    The caller owns the projection lock.  If interrupted after either replace,
    the next reader replays the complete immutable document from the journal.
    """
    _validate_document(document)
    journal = {"schema_version": PROJECTION_JOURNAL_SCHEMA_VERSION, "document": document}
    _atomic_json(
        projection_journal_path(session_dir), journal, fault_point="after_projection_journal"
    )
    _atomic_json(
        projection_state_path(session_dir), document, fault_point="after_projection_state"
    )
    fsops.unlink(projection_journal_path(session_dir), missing_ok=True)
    fsops.sync_directory(session_dir)
    _fault("after_projection_journal_clear")


def remove_projection_state(session_dir: Path) -> None:
    """Remove only rebuildable local projection files, never V1 or export data."""
    fsops.unlink(projection_state_path(session_dir), missing_ok=True)
    fsops.unlink(projection_journal_path(session_dir), missing_ok=True)
    fsops.sync_directory(session_dir)
