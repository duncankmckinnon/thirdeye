"""Private, atomically-published state for the Copilot evidence archive."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops

from .jsonio import atomic_write_json, read_json_object

STATE_SCHEMA_VERSION = 1
STATE_FILENAME = "copilot.state.json"
JOURNAL_FILENAME = "copilot.journal.json"
LOCK_FILENAME = "copilot.archive.lock"

# Tests may replace this with a deterministic callback that raises at a
# publication boundary.  Runtime behaviour never depends on fault injection.
_fault_injector: Callable[[str], None] | None = None


def _fault(point: str) -> None:
    if _fault_injector is not None:
        _fault_injector(point)


def state_path(session_dir: Path) -> Path:
    return session_dir / STATE_FILENAME


def journal_path(session_dir: Path) -> Path:
    return session_dir / JOURNAL_FILENAME


def lock_path(session_dir: Path) -> Path:
    return session_dir / LOCK_FILENAME


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    after_replace = None
    if path.name == STATE_FILENAME:
        after_replace = "after_state_replace"
    elif path.name == JOURNAL_FILENAME:
        after_replace = "after_journal_replace"
    atomic_write_json(
        path,
        value,
        on_replaced=(lambda: _fault(after_replace)) if after_replace else None,
        on_synced=lambda: _fault("after_dirsync"),
    )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return read_json_object(path, invalid_message=f"invalid Copilot archive state: {path}")
    except OSError as exc:
        # A malformed or unreadable state file is never silently used as a
        # fresh cursor.  The archive caller turns this into a diagnostic and
        # leaves the evidence log itself readable.
        raise ValueError(f"invalid Copilot archive state: {path}") from exc


def write_state(session_dir: Path, value: dict[str, Any]) -> None:
    _atomic_json(state_path(session_dir), value)


def write_journal(session_dir: Path, value: dict[str, Any]) -> None:
    _atomic_json(journal_path(session_dir), value)


def clear_journal(session_dir: Path) -> None:
    fsops.unlink(journal_path(session_dir), missing_ok=True)
    fsops.sync_directory(session_dir)
