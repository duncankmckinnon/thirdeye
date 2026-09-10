"""Durable per-record spool for Copilot hook observations.

Enqueue is independent of archive capture: each observation is written
atomically to its own file without taking the session archive lock.  Read
returns complete records; acknowledge deletes only the named committed
source IDs.  Malformed neighbors stay on disk and do not drop valid records.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops
from thirdeye.config import Config
from thirdeye.platforms.copilot.identity import validate_native_id
from thirdeye.platforms.copilot.types import SourcePaths, SourceRecord

_MAX_DIAGNOSTIC_REASON = 200


def _spool_root(config: Config, paths: SourcePaths) -> Path:
    return Path(config.root) / "spool" / "copilot" / paths["source_key"]


def _spool_dir(config: Config, paths: SourcePaths, native_id: str) -> Path:
    validate_native_id(native_id)
    return _spool_root(config, paths) / native_id


def _write_diagnostic(path: Path, reason: str) -> None:
    """Record a bounded location/reason diagnostic; never prompt bodies."""

    diag_path = path.with_name(f"{path.name}.diag")
    clipped = reason[:_MAX_DIAGNOSTIC_REASON]
    payload = json.dumps({"path": path.name, "reason": clipped}, ensure_ascii=True)
    try:
        diag_path.write_text(payload + "\n", encoding="utf-8")
    except OSError:
        return


def _record_error(data: Any, *, expected_native_id: str) -> str | None:
    """Return a diagnostic reason when *data* is not a complete hook SourceRecord."""

    if not isinstance(data, dict):
        return "spool file is not a SourceRecord object"
    for key in ("source_id", "source_kind", "native_session_id", "observed_at"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            return "spool file is not a complete SourceRecord object"
    if "ts" not in data or (data["ts"] is not None and not isinstance(data["ts"], str)):
        return "spool file is not a complete SourceRecord object"
    if not isinstance(data.get("payload"), dict) or not isinstance(data.get("locator"), dict):
        return "spool file is not a complete SourceRecord object"
    if data["source_kind"] != "hook":
        return "spool file source_kind is not hook"
    if data["native_session_id"] != expected_native_id:
        return "native_session_id does not match spool directory"
    return None


def _load_record(path: Path, *, expected_native_id: str) -> SourceRecord | None:
    try:
        raw = fsops.read_text(path, encoding="utf-8")
        data: Any = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        _write_diagnostic(path, f"{type(exc).__name__}: {exc}")
        return None
    reason = _record_error(data, expected_native_id=expected_native_id)
    if reason is not None:
        _write_diagnostic(path, reason)
        return None
    return data  # type: ignore[return-value]


def enqueue_hook(config: Config, paths: SourcePaths, record: SourceRecord) -> str:
    """Atomically persist one hook observation and return its spool path."""

    native_id = record["native_session_id"]
    directory = _spool_dir(config, paths, native_id)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{uuid.uuid4().hex}.json"
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        fsops.replace(tmp_name, dest)
    except BaseException:
        fsops.unlink(Path(tmp_name), missing_ok=True)
        raise
    return str(dest)


def read_spool(config: Config, paths: SourcePaths, native_id: str) -> list[SourceRecord]:
    """Return complete spool records for *native_id*, skipping malformed files."""

    directory = _spool_dir(config, paths, native_id)
    if not directory.is_dir():
        return []
    records: list[SourceRecord] = []
    for path in sorted(directory.glob("*.json")):
        record = _load_record(path, expected_native_id=native_id)
        if record is not None:
            records.append(record)
    return records


def ack_spool(config: Config, paths: SourcePaths, source_ids: list[str]) -> None:
    """Delete spool files whose committed source IDs are listed.

    Unknown IDs are no-ops.  Malformed files are left in place.
    """

    wanted = {item for item in source_ids if item}
    if not wanted:
        return
    root = _spool_root(config, paths)
    if not root.is_dir():
        return
    for path in root.glob("*/*.json"):
        record = _load_record(path, expected_native_id=path.parent.name)
        if record is None:
            continue
        if record["source_id"] in wanted:
            fsops.unlink(path, missing_ok=True)
