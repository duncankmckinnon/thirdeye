"""Private, atomically-published state for the Copilot evidence archive."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops

STATE_SCHEMA_VERSION = 1
STATE_FILENAME = "copilot.state.json"
JOURNAL_FILENAME = "copilot.journal.json"
LOCK_FILENAME = "copilot.archive.lock"


def state_path(session_dir: Path) -> Path:
    return session_dir / STATE_FILENAME


def journal_path(session_dir: Path) -> Path:
    return session_dir / JOURNAL_FILENAME


def lock_path(session_dir: Path) -> Path:
    return session_dir / LOCK_FILENAME


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsops.replace(temp_name, path)
    except BaseException:
        fsops.unlink(Path(temp_name), missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(fsops.read_text(path, encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        # A malformed state file is never silently used as a fresh cursor.  The
        # archive caller turns this into a diagnostic and leaves the evidence
        # log itself readable.
        raise ValueError(f"invalid Copilot archive state: {path}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"invalid Copilot archive state: {path}")
    return raw


def write_state(session_dir: Path, value: dict[str, Any]) -> None:
    _atomic_json(state_path(session_dir), value)


def write_journal(session_dir: Path, value: dict[str, Any]) -> None:
    _atomic_json(journal_path(session_dir), value)


def clear_journal(session_dir: Path) -> None:
    fsops.unlink(journal_path(session_dir), missing_ok=True)
