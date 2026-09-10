"""Stable Copilot platform identifiers and CLI hook vocabulary.

Paths are intentionally absent from this module.  Source paths are resolved at
invocation time by :func:`thirdeye.platforms.copilot.identity.resolve_sources`.
"""

from __future__ import annotations

PLATFORM_NAME = "copilot"
DISPLAY_NAME = "GitHub Copilot CLI"
SOURCE_SCHEMA_VERSION = 1
# Compatibility-friendly names for consumers that refer to the record/event
# envelope rather than the source format.  All three are the same V1 contract.
SCHEMA_VERSION = SOURCE_SCHEMA_VERSION
SOURCE_RECORD_SCHEMA_VERSION = SOURCE_SCHEMA_VERSION
EVENT_ENVELOPE_VERSION = SOURCE_SCHEMA_VERSION

COPILOT_HOME_ENV = "COPILOT_HOME"
HOOKS_DIRECTORY_NAME = "hooks"
OWNED_HOOK_FILENAME = "thirdeye.json"

# Copilot CLI 1.0.83's documented hook names are camelCase.  The aliases are
# intentionally data-only: installation and hook runtime decide which events
# they support without importing source-resolution code.
CLI_HOOK_EVENT_ALIASES: dict[str, str] = {
    "sessionStart": "session_start",
    "userPromptSubmitted": "user_prompt_submitted",
    "preToolUse": "pre_tool_use",
    "postToolUse": "post_tool_use",
    "agentStop": "agent_stop",
    "subagentStart": "subagent_start",
    "subagentStop": "subagent_stop",
    "sessionEnd": "session_end",
    "errorOccurred": "error_occurred",
}

CLI_HOOK_EVENTS = tuple(CLI_HOOK_EVENT_ALIASES)
HOOK_EVENT_ALIASES = CLI_HOOK_EVENT_ALIASES
HOOK_EVENTS = CLI_HOOK_EVENTS
