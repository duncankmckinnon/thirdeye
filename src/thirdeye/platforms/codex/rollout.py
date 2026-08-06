from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"

# Anchored, whitelist-only: a leading alphanumeric followed by up to 127 more of
# [alnum _ -]. This is the injection guard for resolve_rollout — an id that does
# not fully match never reaches a glob, so a `*`, `[`, `/`, or `..` cannot escape
# into pattern territory or a parent directory.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def resolve_rollout(session_id: str, sessions_root: Path | None = None) -> Path | None:
    """Locate the rollout JSONL for session_id, or None.

    Hardened against the ambiguity and injection risks of interpolating the id
    into a glob: the id is validated against SESSION_ID_RE, only the literal
    pattern ``rollout-*.jsonl`` is globbed, symlink escapes from the root are
    rejected, and (when present) the file's own ``session_meta`` frame must name
    the expected session.
    """
    if not SESSION_ID_RE.fullmatch(session_id):
        return None

    root = sessions_root if sessions_root is not None else CODEX_SESSIONS_ROOT
    if not root.is_dir():
        return None

    try:
        resolved_root = root.resolve()
    except OSError:
        return None

    suffix = f"-{session_id}.jsonl"
    for candidate in root.rglob("rollout-*.jsonl"):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        # Reject symlink escapes: the resolved root must be an ancestor of the
        # resolved candidate.
        if resolved_root not in resolved.parents:
            continue
        if not resolved.name.endswith(suffix):
            continue
        if not _session_meta_matches(resolved, session_id):
            continue
        return resolved
    return None


def _session_meta_matches(path: Path, session_id: str) -> bool:
    """Whether path's first session_meta frame names session_id.

    Reads only until the first ``session_meta`` frame. If none exists (older
    rollouts), accept on the filename alone by returning True.
    """
    try:
        for _, frame in iter_frames(path, 0):
            if frame.get("type") != "session_meta":
                continue
            payload = frame.get("payload")
            payload_id = payload.get("id") if isinstance(payload, dict) else None
            return payload_id == session_id
    except OSError:
        return False
    return True


def iter_frames(path: Path, offset: int) -> Iterator[tuple[int, dict]]:
    """Yield (line_start_offset, frame) for each parseable JSON line from offset.

    Skips blank and malformed lines. The offset is the byte position of the
    line's first byte, usable as a stable per-frame identity. Tolerates a
    truncated final line (a rollout being written concurrently): only lines
    terminated by a newline are yielded.
    """
    with path.open("rb") as f:
        f.seek(offset)
        pos = offset
        for raw in f:
            line_offset = pos
            pos += len(raw)
            if not raw.endswith(b"\n"):
                # Incomplete final line — do not yield a partial record.
                break
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                frame = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(frame, dict):
                yield line_offset, frame


def end_offset(path: Path, offset: int) -> int:
    """Byte offset after the last complete line read from offset.

    A complete line ends in a newline; a truncated trailing line is excluded so
    the bookmark never lands mid-record.
    """
    with path.open("rb") as f:
        f.seek(offset)
        pos = offset
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            pos += len(raw)
    return pos
