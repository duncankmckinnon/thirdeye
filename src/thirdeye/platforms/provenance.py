"""Pure helpers for classifying hook payload provenance."""

from __future__ import annotations

from typing import Any

_KNOWN_PLATFORMS = frozenset({"claude", "codex", "cursor"})
_CURSOR_MARKERS = ("cursor_version", "composer_mode")


def foreign_payload_reason(payload: dict[str, Any], expected: str) -> str | None:
    """Return positive evidence of a foreign origin, otherwise ``None``.

    The classifier intentionally fails open: fields that do not provide an
    unambiguous platform signal are ignored.
    """
    if expected not in _KNOWN_PLATFORMS:
        return None

    if expected != "cursor":
        for marker in _CURSOR_MARKERS:
            if marker in payload:
                return f"Cursor marker {marker} present"

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str) or not event_name:
        return None

    first_character = event_name[0]
    if first_character.islower() and expected != "cursor":
        return f"Cursor event {event_name} received for {expected}"
    if first_character.isupper() and expected == "cursor":
        return f"PascalCase event {event_name} received for cursor"

    return None
