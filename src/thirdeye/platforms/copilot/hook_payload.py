"""Normalize Copilot hook observations into SourceRecord envelopes.

This module is input-only: it must not open files, spawn processes, or talk
to SQLite.  Callers own durable spooling and later capture.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from thirdeye.platforms.copilot.constants import CLI_HOOK_EVENT_ALIASES, SOURCE_SCHEMA_VERSION
from thirdeye.platforms.copilot.identity import validate_native_id
from thirdeye.platforms.copilot.types import SourceRecord

# Claude-style PascalCase names accepted as input aliases.  Canonical stored
# event names remain Copilot CLI camelCase from CLI_HOOK_EVENT_ALIASES.
_PASCAL_CASE_ALIASES: dict[str, str] = {
    "SessionStart": "sessionStart",
    "UserPromptSubmit": "userPromptSubmitted",
    "PreToolUse": "preToolUse",
    "PostToolUse": "postToolUse",
    "Stop": "agentStop",
    "SubagentStart": "subagentStart",
    "SubagentStop": "subagentStop",
    "SessionEnd": "sessionEnd",
}

# Allowlisted hook-time context.  ``env`` is the captured environment mapping;
# the remaining keys are optional explicit trace identifiers.
_CONTEXT_KEYS = ("env", "trace_id", "span_id", "parent_span_id", "trace_context", "traceparent")

# Epoch milliseconds are >= 1e12 for dates after 2001-09-09.
_MILLISECOND_THRESHOLD = 1_000_000_000_000


def _canonical_event(event: str) -> str:
    if event in CLI_HOOK_EVENT_ALIASES:
        return event
    aliased = _PASCAL_CASE_ALIASES.get(event)
    if aliased is not None:
        return aliased
    return event


def _session_id(payload: dict[str, Any]) -> str:
    session_id = payload.get("sessionId")
    if not isinstance(session_id, str):
        raise ValueError("hook payload is missing a string sessionId")
    validate_native_id(session_id)
    return session_id


def _valid_iso_datetime(value: str) -> bool:
    """Return True when *value* is a parseable ISO-8601 datetime."""

    iso = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(iso)
    except ValueError:
        return False
    return True


def _source_ts(payload: dict[str, Any]) -> str | None:
    """Return the original source timestamp when it is valid; never invent one."""

    if "timestamp" not in payload:
        return None
    value: Any = payload["timestamp"]
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if _valid_iso_datetime(stripped):
            return stripped
        try:
            value = float(stripped)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    seconds = value / 1000.0 if abs(value) >= _MILLISECOND_THRESHOLD else float(value)
    try:
        dt = datetime.fromtimestamp(seconds, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _allowlisted_context(context: dict[str, Any]) -> dict[str, Any]:
    stored: dict[str, Any] = {}
    for key in _CONTEXT_KEYS:
        if key not in context:
            continue
        value = context[key]
        stored[key] = deepcopy(value)
    return stored


def parse_hook(
    event: str,
    payload: dict[str, Any],
    context: dict[str, Any],
    *,
    observed_at: str,
    observation_id: str,
) -> SourceRecord:
    """Build one hook SourceRecord from a normalized observation.

    The raw *payload* is retained unmodified.  Session routing uses Copilot's
    ``sessionId`` and rejects invalid native IDs.  Child hooks that share a
    parent session ID are recorded as-is; this function does not guess
    ownership.
    """

    native_session_id = _session_id(payload)
    canonical_event = _canonical_event(event)
    retained_payload = deepcopy(payload)
    return {
        "source_id": f"hook/{native_session_id}/{observation_id}",
        "source_kind": "hook",
        "native_session_id": native_session_id,
        "ts": _source_ts(payload),
        "observed_at": observed_at,
        "payload": {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "event": canonical_event,
            "hook_payload": retained_payload,
            "context": _allowlisted_context(context),
        },
        "locator": {
            "observation_id": observation_id,
            "event": canonical_event,
        },
    }
