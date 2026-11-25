"""Span names for the hook payload fields Claude Code and Codex share.

Both harnesses send hook payloads with the same field names. Copied onto a
span as-is they read as unexplained raw keys beside the ``gen_ai.*`` and
``thirdeye.*`` vocabulary, and several repeat what the exporter already sets.
Adapters pass a payload through :func:`hook_span_attributes` before putting
it on a tool call, permission request, or subagent turn.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Fields worth keeping, under the names a span query should use. The tool
# fields take their OTel GenAI names directly; the exporter keeps an existing
# `gen_ai.tool.call.*` value rather than projecting its own.
_RENAMED = {
    "tool_name": "gen_ai.tool.name",
    "tool_use_id": "gen_ai.tool.call.id",
    "tool_input": "gen_ai.tool.call.arguments",
    "tool_response": "gen_ai.tool.call.result",
    "agent_id": "thirdeye.subagent.id",
    "agent_type": "thirdeye.subagent.type",
    "permission_mode": "thirdeye.permission_mode",
    "effort": "thirdeye.effort",
    "duration_ms": "thirdeye.tool.duration_ms",
    "is_interrupt": "thirdeye.tool.interrupted",
    "mcp_server": "thirdeye.tool.mcp_server",
    "reason": "thirdeye.permission.reason",
}

# Plumbing, or a value a span already carries another way.
_DROPPED = frozenset(
    {
        "hook_event_name",  # the span's name says which event it is
        "prompt_id",  # thirdeye.turn.id identifies the turn
        "turn_id",
        "scratchpad_dir",
        "stop_hook_active",
        "error",  # the tool span's ERROR status description
    }
)

_UNKNOWN_PREFIX = "thirdeye.hook."


def hook_span_attributes(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``data`` with shared hook fields renamed or dropped.

    A field this module doesn't know is kept under ``thirdeye.hook.`` rather
    than dropped, so a harness adding a field never silently loses it.
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in _DROPPED:
            continue
        out[_RENAMED.get(key, f"{_UNKNOWN_PREFIX}{key}")] = value
    return out
