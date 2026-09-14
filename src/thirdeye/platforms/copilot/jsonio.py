"""Atomic JSON publication shared by Copilot archive and projection state."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    on_replaced: Callable[[], None] | None = None,
    on_synced: Callable[[], None] | None = None,
) -> None:
    """Replace ``path`` with canonical JSON, fsyncing the file and directory.

    ``on_replaced`` runs after the durable rename and before the directory
    sync so callers can inject crash boundaries at the same points as before.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsops.replace(temp_name, path)
        if on_replaced is not None:
            on_replaced()
        fsops.sync_directory(path.parent)
        if on_synced is not None:
            on_synced()
    except BaseException:
        fsops.unlink(Path(temp_name), missing_ok=True)
        raise


def read_json_object(path: Path, *, invalid_message: str) -> dict[str, Any] | None:
    """Read a JSON object, distinguishing absence, corruption, and I/O errors.

    ``FileNotFoundError`` returns ``None``.  Malformed JSON or a non-object
    becomes ``ValueError``.  Transient ``OSError`` (sharing violations, EIO)
    propagates unchanged so callers do not treat a busy file as corruption.
    """
    try:
        raw = json.loads(fsops.read_text(path, encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        raise ValueError(invalid_message) from None
    if not isinstance(raw, dict):
        raise ValueError(invalid_message)
    return raw
