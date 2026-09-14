"""Best-effort recovery of Cursor's local CLI transcript metadata.

Cursor CLI persists the rendered user/assistant exchange separately from the
hook payloads. These helpers are deliberately additive: unreadable, absent, or
incomplete local files simply yield empty fields for the hook-derived turn to
keep authoritative.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CursorLocalTurn:
    input_text: str = ""
    output_text: str = ""
    request_id: str = ""


def _safe_session_id(session_id: str) -> bool:
    return bool(session_id) and session_id not in {".", ".."} and not any(
        separator and separator in session_id for separator in (os.sep, os.altsep)
    )


def transcript_roots() -> list[Path]:
    override = os.environ.get("THIRDEYE_CURSOR_TRANSCRIPT_ROOTS", "")
    if override:
        return [Path(part) for part in override.split(os.pathsep) if part]
    projects = Path.home() / ".cursor" / "projects"
    if not projects.is_dir():
        return []
    try:
        return [path / "agent-transcripts" for path in projects.iterdir() if path.is_dir()]
    except OSError:
        return []


def chat_roots() -> list[Path]:
    override = os.environ.get("THIRDEYE_CURSOR_CHAT_ROOTS", "")
    if override:
        return [Path(part) for part in override.split(os.pathsep) if part]
    root = Path.home() / ".cursor" / "chats"
    return [root] if root.is_dir() else []


def _message_text(record: dict[str, Any]) -> str:
    message = record.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"].strip()
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    return "\n".join(parts)


def _transcript_turn(session_id: str) -> tuple[str, str, int]:
    for root in transcript_roots():
        path = root / session_id / f"{session_id}.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        messages: list[tuple[str, str]] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("role") not in {"user", "assistant"}:
                continue
            if text := _message_text(record):
                messages.append((str(record["role"]), text))
        for index in range(len(messages) - 1, -1, -1):
            role, input_text = messages[index]
            if role != "user":
                continue
            output_text = next(
                (text for next_role, text in reversed(messages[index + 1 :]) if next_role == "assistant"),
                "",
            )
            return input_text, output_text, sum(1 for message_role, _ in messages if message_role == "user")
    return "", "", 0


_REQUEST_ID = re.compile(r'"requestId"\s*:\s*"([^"\\]+)"')


def _request_ids(path: Path) -> list[str]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = connection.execute("SELECT data FROM blobs ORDER BY rowid").fetchall()
    except (OSError, sqlite3.Error):
        return []
    finally:
        if "connection" in locals():
            connection.close()
    ids: list[str] = []
    for (blob,) in rows:
        if not isinstance(blob, (bytes, bytearray, str)):
            continue
        text = blob.decode("utf-8", errors="ignore") if isinstance(blob, (bytes, bytearray)) else blob
        user_at = text.find('"role":"user"')
        if user_at < 0:
            user_at = text.find('"role": "user"')
        if user_at < 0:
            continue
        match = _REQUEST_ID.search(text, user_at)
        if match:
            ids.append(match.group(1))
    return ids


def _request_id(session_id: str, *, expected_count: int) -> str:
    if expected_count <= 0:
        return ""
    for root in chat_roots():
        try:
            paths = sorted(root.glob(f"*/{session_id}/store.db"))
        except OSError:
            continue
        for path in paths:
            if (ids := _request_ids(path)) and len(ids) == expected_count:
                return ids[-1]
    return ""


def local_turn(session_id: str) -> CursorLocalTurn:
    """Return the newest local transcript turn and its locally stored request ID."""
    if not _safe_session_id(session_id):
        return CursorLocalTurn()
    input_text, output_text, user_count = _transcript_turn(session_id)
    return CursorLocalTurn(input_text, output_text, _request_id(session_id, expected_count=user_count))
