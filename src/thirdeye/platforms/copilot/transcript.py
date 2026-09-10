"""Lossless, bounded reads of Copilot CLI transcript recordings.

This module deliberately knows nothing about thirdeye's archive.  It turns the
``events.jsonl`` and the small, adjacent ``workspace.yaml`` document into raw
source evidence that a later composition layer can commit durably.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .constants import SOURCE_SCHEMA_VERSION
from .identity import validate_native_id
from .types import SourcePaths, SourceRecord, SourceSlice

_EVENTS_FILENAME = "events.jsonl"
_WORKSPACE_FILENAME = "workspace.yaml"
_CONTINUITY_WINDOW = 4096


def _observed_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _diagnostic(code: str, message: str, **locator: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "locator": locator}


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _session_directory(paths: SourcePaths, native_id: str) -> Path:
    """Return a checked session directory without allowing path traversal."""

    validate_native_id(native_id)
    root = Path(paths["session_root"]).expanduser().resolve(strict=False)
    session = (root / native_id).resolve(strict=False)
    if not _within(session, root):
        raise ValueError("native session ID escapes the Copilot session root")
    return session


def _generation(path: Path) -> str:
    """Identify the current file object, while remaining stable for appends."""

    stat = path.stat()
    # st_dev/st_ino distinguishes atomic replacement on the platforms we
    # support and, unlike mtime/ctime, does not change on an ordinary append.
    # A subsequent size decrease still detects truncation on filesystems where
    # inode data is unavailable.
    return f"{stat.st_dev:x}-{stat.st_ino:x}"


def _valid_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _continuity_anchor(path: Path, offset: int) -> tuple[int, str]:
    """Fingerprint source bytes already consumed without penalizing appends."""

    start = max(0, offset - _CONTINUITY_WINDOW)
    with path.open("rb") as stream:
        stream.seek(start)
        digest = hashlib.sha256(stream.read(offset - start)).hexdigest()
    return start, digest


def _json_value(value: Any) -> Any:
    """Reject YAML-only values so metadata remains a JSON-compatible contract."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(f"workspace metadata contains unsupported value {type(value).__name__}")


def _source_id(paths: SourcePaths, native_id: str, event: dict[str, Any], generation: str, offset: int, raw: bytes) -> tuple[str, dict[str, Any]]:
    event_id = event.get("id")
    if isinstance(event_id, (str, int)) and str(event_id):
        value = str(event_id)
        return (
            f"{paths['source_key']}/{native_id}/{value}",
            {"native_event_id": value},
        )

    digest = hashlib.sha256(raw).hexdigest()
    return (
        f"{paths['source_key']}/{native_id}/{generation}/{offset}/{digest}",
        {"content_digest": digest, "identity_confidence": "lower"},
    )


def _record_for_line(
    paths: SourcePaths,
    native_id: str,
    generation: str,
    offset: int,
    line: bytes,
    observed_at: str,
) -> tuple[SourceRecord, dict[str, Any] | None]:
    """Create evidence for one complete newline-terminated physical line."""

    raw = line[:-1] if line.endswith(b"\n") else line
    locator: dict[str, Any] = {
        "file": _EVENTS_FILENAME,
        "file_generation": generation,
        "byte_offset": offset,
        "byte_length": len(line),
    }
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        digest = hashlib.sha256(raw).hexdigest()
        locator["content_digest"] = digest
        record: SourceRecord = {
            "source_id": f"{paths['source_key']}/{native_id}/{generation}/{offset}/{digest}",
            "source_kind": "transcript",
            "native_session_id": native_id,
            "ts": None,
            "observed_at": observed_at,
            "payload": {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "malformed": "invalid_utf8",
                "raw_bytes_base64": base64.b64encode(raw).decode("ascii"),
            },
            "locator": locator,
        }
        return record, _diagnostic("transcript_invalid_utf8", "complete transcript line is not UTF-8", **locator)

    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        digest = hashlib.sha256(raw).hexdigest()
        locator["content_digest"] = digest
        record = {
            "source_id": f"{paths['source_key']}/{native_id}/{generation}/{offset}/{digest}",
            "source_kind": "transcript",
            "native_session_id": native_id,
            "ts": None,
            "observed_at": observed_at,
            "payload": {"schema_version": SOURCE_SCHEMA_VERSION, "malformed": "invalid_json", "raw_line": text},
            "locator": locator,
        }
        return record, _diagnostic("transcript_invalid_json", "complete transcript line is not JSON", **locator)

    if not isinstance(value, dict):
        digest = hashlib.sha256(raw).hexdigest()
        locator["content_digest"] = digest
        record = {
            "source_id": f"{paths['source_key']}/{native_id}/{generation}/{offset}/{digest}",
            "source_kind": "transcript",
            "native_session_id": native_id,
            "ts": None,
            "observed_at": observed_at,
            "payload": {"schema_version": SOURCE_SCHEMA_VERSION, "malformed": "non_object_json", "raw_value": value},
            "locator": locator,
        }
        return record, _diagnostic("transcript_non_object", "complete transcript line is not a JSON object", **locator)

    source_id, event_locator = _source_id(paths, native_id, value, generation, offset, raw)
    locator.update(event_locator)
    timestamp = _valid_timestamp(value.get("timestamp"))
    diagnostic = None
    if value.get("timestamp") is not None and timestamp is None:
        diagnostic = _diagnostic("transcript_invalid_timestamp", "event timestamp is not ISO-8601", **locator)
    # Keep every top-level field, including unknown future fields.  The schema
    # version comes last so an untrusted event cannot alter our envelope.
    payload = {**value, "schema_version": SOURCE_SCHEMA_VERSION}
    record = {
        "source_id": source_id,
        "source_kind": "transcript",
        "native_session_id": native_id,
        "ts": timestamp,
        "observed_at": observed_at,
        "payload": payload,
        "locator": locator,
    }
    return record, diagnostic


def _workspace_record(
    paths: SourcePaths, native_id: str, directory: Path, observed_at: str
) -> tuple[SourceRecord | None, str | None, list[dict[str, Any]]]:
    """Read only safe workspace metadata and return it as independent evidence."""

    path = directory / _WORKSPACE_FILENAME
    if not path.is_file():
        return None, None, []
    try:
        raw = path.read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
        data = _json_value(data)
    except (OSError, TypeError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, None, [_diagnostic("workspace_metadata_invalid", "workspace.yaml could not be safely parsed", file=str(path), reason=type(exc).__name__)]
    if not isinstance(data, dict):
        return None, None, [_diagnostic("workspace_metadata_invalid", "workspace.yaml must contain a mapping", file=str(path))]

    digest = hashlib.sha256(raw).hexdigest()
    try:
        generation = _generation(path)
    except OSError:
        generation = f"content-{digest}"
    cwd = data.get("cwd")
    if not isinstance(cwd, str):
        cwd = data.get("workspacePath") if isinstance(data.get("workspacePath"), str) else None
    record: SourceRecord = {
        "source_id": f"{paths['source_key']}/{native_id}/workspace/{digest}",
        "source_kind": "metadata",
        "native_session_id": native_id,
        "ts": None,
        "observed_at": observed_at,
        "payload": {"schema_version": SOURCE_SCHEMA_VERSION, "file": _WORKSPACE_FILENAME, "data": data},
        "locator": {"file": _WORKSPACE_FILENAME, "file_generation": generation, "content_digest": digest},
    }
    return record, cwd, []


def discover_transcripts(paths: SourcePaths) -> list[str]:
    """Return native session IDs with an immediately readable events file."""

    root = Path(paths["session_root"]).expanduser().resolve(strict=False)
    try:
        candidates = list(root.iterdir())
    except OSError:
        return []
    result: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
            if not _within(resolved, root) or not resolved.is_dir() or not (resolved / _EVENTS_FILENAME).is_file():
                continue
            validate_native_id(candidate.name)
        except (OSError, ValueError):
            continue
        result.append(candidate.name)
    return sorted(result)


def read_transcript(
    paths: SourcePaths,
    native_id: str,
    cursor: dict,
    *,
    max_records: int = 1000,
    max_bytes: int = 4_194_304,
) -> SourceSlice:
    """Read a stable bounded snapshot of one transcript.

    A newline is the commit boundary: a trailing partial JSON or UTF-8 line is
    left untouched for the next read.  Cursors retain a snapshot endpoint so a
    caller can drain the state observed at the start even while Copilot appends.
    """

    if max_records < 0 or max_bytes < 0:
        raise ValueError("max_records and max_bytes must be non-negative")
    directory = _session_directory(paths, native_id)
    event_path = directory / _EVENTS_FILENAME
    diagnostics: list[dict[str, Any]] = []
    observed_at = _observed_at()
    workspace, cwd, workspace_diagnostics = _workspace_record(paths, native_id, directory, observed_at)
    diagnostics.extend(workspace_diagnostics)
    records: list[SourceRecord] = []

    try:
        stat = event_path.stat()
        if not event_path.is_file():
            raise FileNotFoundError(event_path)
        generation = _generation(event_path)
    except OSError:
        diagnostics.append(_diagnostic("transcript_unavailable", "events.jsonl is unavailable; it is not considered complete", file=str(event_path)))
        return {"records": records, "next_cursor": dict(cursor), "diagnostics": diagnostics, "cwd": cwd, "exhausted": False}

    size = stat.st_size
    prior_generation = cursor.get("file_generation")
    prior_offset = cursor.get("byte_offset", 0)
    if not isinstance(prior_offset, int) or prior_offset < 0:
        prior_offset = 0
        diagnostics.append(_diagnostic("transcript_cursor_invalid", "invalid byte offset; replaying transcript", file=str(event_path)))
    reset = prior_generation is not None and prior_generation != generation
    if prior_offset > size:
        reset = True
    anchor_start = cursor.get("continuity_start")
    anchor_digest = cursor.get("continuity_digest")
    if not reset and prior_offset and isinstance(anchor_start, int) and isinstance(anchor_digest, str):
        try:
            current_start, current_digest = _continuity_anchor(event_path, prior_offset)
        except OSError:
            current_start, current_digest = -1, ""
        if (current_start, current_digest) != (anchor_start, anchor_digest):
            reset = True
    if reset:
        diagnostics.append(_diagnostic("transcript_replaced", "transcript was replaced or truncated; replaying from byte zero", file=str(event_path), previous_generation=prior_generation, file_generation=generation))
        prior_offset = 0

    prior_end = cursor.get("snapshot_end")
    if reset or not isinstance(prior_end, int) or prior_end < prior_offset:
        snapshot_end = size
    elif prior_offset < prior_end:
        snapshot_end = min(prior_end, size)
    else:
        snapshot_end = size

    # Metadata is a snapshot too.  It is included when changed, but it does
    # not prevent an events-only caller from making progress at a tiny bound.
    workspace_digest = workspace["locator"]["content_digest"] if workspace else None
    workspace_emitted = False
    if workspace is not None and cursor.get("workspace_digest") != workspace_digest and max_records > 0:
        records.append(workspace)
        workspace_emitted = True

    offset = prior_offset
    consumed = 0
    limit_hit = False
    try:
        with event_path.open("rb") as stream:
            stream.seek(offset)
            while offset < snapshot_end:
                remaining = snapshot_end - offset
                line = stream.readline(remaining)
                if not line or not line.endswith(b"\n"):
                    # A line extending beyond the observed snapshot is not yet
                    # a source record, even if its prefix happens to decode.
                    break
                if len(records) >= max_records:
                    limit_hit = True
                    break
                line_size = len(line)
                if consumed and consumed + line_size > max_bytes:
                    limit_hit = True
                    break
                if not consumed and line_size > max_bytes:
                    diagnostics.append(_diagnostic("transcript_record_oversize", "one complete transcript record exceeds max_bytes and was accepted for progress", file=str(event_path), byte_offset=offset, byte_length=line_size, max_bytes=max_bytes))
                record, diagnostic = _record_for_line(paths, native_id, generation, offset, line, observed_at)
                records.append(record)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                offset += line_size
                consumed += line_size
    except OSError:
        diagnostics.append(_diagnostic("transcript_read_failed", "events.jsonl could not be read; it is not considered complete", file=str(event_path)))
        return {"records": records, "next_cursor": dict(cursor), "diagnostics": diagnostics, "cwd": cwd, "exhausted": False}

    next_cursor: dict[str, Any] = {
        "byte_offset": offset,
        "file_generation": generation,
        "snapshot_end": snapshot_end,
    }
    try:
        anchor_start, anchor_digest = _continuity_anchor(event_path, offset)
        next_cursor["continuity_start"] = anchor_start
        next_cursor["continuity_digest"] = anchor_digest
    except OSError:
        # The read above remains useful.  A later invocation will report the
        # unavailable source instead of pretending that it reached completion.
        pass
    if workspace_digest is not None and (workspace_emitted or cursor.get("workspace_digest") == workspace_digest):
        next_cursor["workspace_digest"] = workspace_digest
    exhausted = offset >= snapshot_end and not limit_hit
    return {"records": records, "next_cursor": next_cursor, "diagnostics": diagnostics, "cwd": cwd, "exhausted": exhausted}
