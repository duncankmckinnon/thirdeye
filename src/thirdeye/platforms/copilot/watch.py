"""Foreground polling for local Copilot CLI evidence.

The watcher intentionally watches filesystem *identity*, not Copilot's
process.  This makes it useful after an editor-terminal session has ended and
also means it never needs to start, inspect, or authenticate a Copilot CLI.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from thirdeye.config import Config

from .capture import sync
from .database import discover_database_sessions
from .identity import validate_native_id
from .sources import discover_sessions
from .types import SourcePaths

_EVENTS_FILENAME = "events.jsonl"
_SLEEP: Callable[[float], None] = time.sleep


def _file_stamp(path: Path) -> tuple[int, int, int, int] | None:
    """Return a cheap replacement/append-sensitive stamp for one file."""

    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _transcript_stamps(paths: SourcePaths, session_ids: set[str]) -> dict[str, object]:
    root = Path(paths["session_root"])
    stamps: dict[str, object] = {}
    for native_id in session_ids:
        try:
            validate_native_id(native_id)
        except ValueError:
            continue
        stamps[native_id] = _file_stamp(root / native_id / _EVENTS_FILENAME)
    return stamps


def _database_stamp(paths: SourcePaths) -> tuple[object, object, object]:
    """Include WAL and SHM changes: live SQLite commits need no DB mtime."""

    database = Path(paths["database"])
    return (
        _file_stamp(database),
        _file_stamp(database.with_name(f"{database.name}-wal")),
        _file_stamp(database.with_name(f"{database.name}-shm")),
    )


def _spool_stamps(config: Config, paths: SourcePaths) -> dict[str, tuple[tuple[str, object], ...]]:
    """Return per-session spool stamps without loading prompt-bearing records."""

    root = Path(config.root) / "spool" / "copilot" / paths["source_key"]
    try:
        entries = list(root.iterdir())
    except OSError:
        return {}
    result: dict[str, tuple[tuple[str, object], ...]] = {}
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            validate_native_id(entry.name)
            files = tuple((item.name, _file_stamp(item)) for item in sorted(entry.glob("*.json")))
        except (OSError, ValueError):
            continue
        result[entry.name] = files
    return result


def _source_snapshot(config: Config, paths: SourcePaths) -> dict[str, Any]:
    """Discover source IDs and cheap change positions for the next poll."""

    # ``discover_sessions`` is deliberately the public composition discovery
    # entrypoint.  Database-specific discovery is additionally retained so a
    # WAL-only change need not re-read transcript-only sessions.
    all_sessions = set(discover_sessions(paths))
    database_sessions = set(discover_database_sessions(paths))
    all_sessions.update(database_sessions)
    spool = _spool_stamps(config, paths)
    all_sessions.update(spool)
    return {
        "sessions": all_sessions,
        "database_sessions": database_sessions,
        "transcripts": _transcript_stamps(paths, all_sessions),
        "database": _database_stamp(paths),
        "spool": spool,
    }


def _changed_sessions(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Choose only source positions that can have produced new evidence."""

    before_sessions = before["sessions"]
    after_sessions = after["sessions"]
    changed = set(after_sessions - before_sessions)

    for native_id in after_sessions:
        if before["transcripts"].get(native_id) != after["transcripts"].get(native_id):
            changed.add(native_id)
        if before["spool"].get(native_id) != after["spool"].get(native_id):
            changed.add(native_id)

    if before["database"] != after["database"]:
        # Retain IDs observed before the change as well: a transaction may
        # delete/update rows and source disappearance is never session close.
        changed.update(before["database_sessions"])
        changed.update(after["database_sessions"])
    return changed


def _result_needs_retry(result: dict[str, int], *, present: bool) -> bool:
    """Retry incomplete or failed work only while the source is still discoverable.

    A missing selected ID is a one-shot error for that poll: source removal is
    never session completion, but it also must not become a permanent retry
    loop.  Recreated files change stamps and are selected again.  Pending work
    (busy/unreadable/incomplete pages) retries even if discovery briefly drops
    the ID, because those codes are not absence.
    """

    if result.get("pending", 0) > 0:
        return True
    return bool(present and result.get("errors", 0) > 0)


def watch(config: Config, paths: SourcePaths, *, interval: float = 1.0) -> None:
    """Poll local recordings until interrupted.

    The initial sync drains the bounded source snapshot.  Later cycles invoke
    per-session sync only after a transcript, database/WAL, or spool position
    changes (or after a retryable result), avoiding repeated parsing of quiet
    completed sessions.  All capture remains local; this function never
    exports data and never starts a background service.
    """

    if not isinstance(interval, (int, float)) or isinstance(interval, bool):
        raise ValueError("interval must be a finite number of seconds, at least 0.1")
    if not math.isfinite(interval) or interval < 0.1:
        raise ValueError("interval must be a finite number of seconds, at least 0.1")

    try:
        # Take the baseline before the initial drain.  A source can append
        # while that bounded drain is running; comparing its post-drain stamp
        # on the first poll ensures that append receives a later capture.
        previous = _source_snapshot(config, paths)
        initial = sync(config, paths)
        retry = set(previous["sessions"]) if _result_needs_retry(initial, present=True) else set()
        while True:
            _SLEEP(float(interval))
            current = _source_snapshot(config, paths)
            selected = _changed_sessions(previous, current) | retry
            retry = set()
            for native_id in sorted(selected):
                # KeyboardInterrupt is intentionally checked between sessions;
                # a current archive commit remains crash-recoverable.
                result = sync(config, paths, session_id=native_id)
                if _result_needs_retry(result, present=native_id in current["sessions"]):
                    retry.add(native_id)
            previous = current
    except KeyboardInterrupt:
        # A foreground CLI command treats Ctrl-C as ordinary termination.
        return


__all__ = ["watch"]
