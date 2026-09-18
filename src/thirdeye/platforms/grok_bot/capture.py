"""Grok Bot store.db capture → TurnSpanDict → shared otel_export.

Wave 2: parse message / send-message / event; poll transcript_entries fail-open;
tool-call body mapping deferred.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.platforms.grok_bot.constants import PLATFORM_NAME
from thirdeye.tracing.model import TurnSpanDict

_MAPPED_KINDS = frozenset({"message", "send-message", "event"})
_DEFERRED_KINDS = frozenset({"tool-call"})


def _ms_to_iso(ms: Any) -> str:
    try:
        value = int(ms)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value % 1000:03d}Z"


def parse_entry(raw: dict[str, Any] | Any) -> dict[str, Any]:
    """Normalize one transcript entry; ``kind`` is the discriminant."""
    if not isinstance(raw, dict):
        return {"kind": None, "skipped": True}
    kind = raw.get("kind")
    out = dict(raw)
    out["kind"] = kind
    if kind in _DEFERRED_KINDS:
        out["deferred"] = True
        out["skipped"] = True
    elif kind not in _MAPPED_KINDS:
        out["skipped"] = True
    return out


def normalize_entry(raw: dict[str, Any] | Any) -> dict[str, Any]:
    return parse_entry(raw)


def _message_text(entry: dict[str, Any]) -> str:
    content = entry.get("content")
    if isinstance(content, str):
        return content
    message = entry.get("message")
    if isinstance(message, dict):
        inner = message.get("content")
        if isinstance(inner, str):
            return inner
    return ""


def _group_key(entry: dict[str, Any]) -> str:
    rid = entry.get("requestId")
    if isinstance(rid, str) and rid:
        return rid
    eid = entry.get("id")
    if isinstance(eid, str) and eid:
        return eid
    return f"anon-{id(entry)}"


def build_turns_from_entries(
    entries: list[dict[str, Any]],
    *,
    conversation_id: str,
    agent_id: str,
    agent_name: str = "",
) -> list[TurnSpanDict]:
    """Correlate entries by ``requestId`` into TurnSpanDict list."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for raw in entries:
        parsed = parse_entry(raw)
        kind = parsed.get("kind")
        if kind in _DEFERRED_KINDS:
            continue
        if kind not in _MAPPED_KINDS:
            continue
        key = _group_key(parsed)
        if key not in buckets:
            order.append(key)
        buckets[key].append(parsed)

    turns: list[TurnSpanDict] = []
    for key in order:
        group = buckets[key]
        user_texts: list[str] = []
        assistant_texts: list[str] = []
        timestamps: list[Any] = []
        for entry in group:
            kind = entry.get("kind")
            timestamps.append(entry.get("timestampMs"))
            if kind == "message":
                role = entry.get("role")
                text = _message_text(entry)
                if role == "user":
                    user_texts.append(text)
                elif role == "assistant":
                    assistant_texts.append(text)

        if not user_texts and not assistant_texts:
            continue

        start_ms = next((t for t in timestamps if t is not None), 0)
        end_ms = next((t for t in reversed(timestamps) if t is not None), start_ms)
        attrs: dict[str, Any] = {
            "thirdeye.platform": PLATFORM_NAME,
            "platform": PLATFORM_NAME,
            "gen_ai.conversation.id": conversation_id,
            "thirdeye.grok_bot.agent_id": agent_id,
            "request_id": key,
            "thirdeye.grok_bot.request_id": key,
        }
        if agent_name:
            attrs["gen_ai.agent.name"] = agent_name

        turn: TurnSpanDict = {
            "turn_id": key,
            "start_ts": _ms_to_iso(start_ms),
            "end_ts": _ms_to_iso(end_ms),
            "input_message": "\n".join(user_texts),
            "output_message": "\n".join(assistant_texts),
            "status": "completed",
            "llm_calls": [],
            "permission_requests": [],
            "subagents": [],
            "attributes": attrs,
        }
        turns.append(turn)
    return turns


def entries_to_turns(
    entries: list[dict[str, Any]],
    *,
    conversation_id: str,
    agent_id: str,
    agent_name: str = "",
) -> list[TurnSpanDict]:
    return build_turns_from_entries(
        entries,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
    )


def _read_entries(db_path: Path) -> list[dict[str, Any]] | None:
    """Return entries, [] if empty, or None on fail-open errors."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=0.1)
    except sqlite3.Error:
        return None
    try:
        try:
            rows = conn.execute(
                "SELECT seq, id, entry FROM transcript_entries ORDER BY seq"
            ).fetchall()
        except sqlite3.Error:
            return None
        out: list[dict[str, Any]] = []
        for _seq, _eid, body in rows:
            try:
                parsed = json.loads(body)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def poll_and_export(
    db_path: Path | str,
    *,
    conversation_id: str,
    agent_id: str,
    agent_name: str = "",
    cwd: str = "",
) -> dict[str, Any]:
    """Poll one agent store.db and export completed turns via otel_export.

    Fail-open: empty DB, corrupt file, or SQLite BUSY → no raise, no export.
    Never calls ``export_turn`` with an empty ``{}`` turn.
    """
    from thirdeye import otel_export

    path = Path(db_path)
    result: dict[str, Any] = {"exported": 0, "skipped": True}
    try:
        if not path.exists():
            return result
    except OSError:
        return result

    try:
        entries = _read_entries(path)
    except sqlite3.Error:
        return result
    if entries is None:
        return result
    if not entries:
        return {"exported": 0, "skipped": True}

    turns = build_turns_from_entries(
        entries,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
    )
    if not turns:
        return {"exported": 0, "skipped": True}

    try:
        config = Config.load()
    except Exception:
        config = None  # type: ignore[assignment]

    session_dir = Path(cwd) if cwd else path.parent
    exported = 0
    for turn in turns:
        if not turn:
            continue
        if not (turn.get("input_message") or turn.get("output_message")):
            continue
        try:
            otel_export.export_turn(
                config,  # type: ignore[arg-type]
                session_dir,
                conversation_id,
                PLATFORM_NAME,
                cwd or str(session_dir),
                turn,
            )
            exported += 1
        except Exception:
            continue
    return {"exported": exported, "skipped": exported == 0}


def sync_store(
    db_path: Path | str,
    *,
    conversation_id: str,
    agent_id: str,
    agent_name: str = "",
    cwd: str = "",
) -> dict[str, Any]:
    return poll_and_export(
        db_path,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
    )


def poll_store(
    db_path: Path | str,
    *,
    conversation_id: str,
    agent_id: str,
    agent_name: str = "",
    cwd: str = "",
) -> dict[str, Any]:
    return poll_and_export(
        db_path,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
    )
