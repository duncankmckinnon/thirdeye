from __future__ import annotations

from pathlib import Path

PLATFORM_NAME = "cursor"
DISPLAY_NAME = "Cursor"
HOOKS_FILE = Path.home() / ".cursor" / "hooks.json"
HOOK_BIN_NAME = "thirdeye-cursor-hook"
HOOK_TIMEOUT_S = 30

# Cursor IDE emits the full set. Cursor CLI currently emits a subset, but it
# is safe to register every event in one shared hooks file.
TRACED_EVENTS: tuple[str, ...] = (
    "sessionStart",
    "sessionEnd",
    "beforeSubmitPrompt",
    "afterAgentResponse",
    "afterAgentThought",
    "beforeShellExecution",
    "afterShellExecution",
    "beforeMCPExecution",
    "afterMCPExecution",
    "beforeReadFile",
    "afterFileEdit",
    "beforeTabFileRead",
    "afterTabFileEdit",
    "postToolUse",
    "subagentStop",
    "stop",
)

READ_TOOL_NAMES = frozenset(
    {
        "read_file",
        "read",
        "view_file",
        "view",
    }
)

# Tools in this set already produce a complete record from their dedicated
# Cursor callbacks, so their generic postToolUse notification would duplicate it.
# Most have a real after-callback that emits the paired `tool_result`.
# `tab_file_read` is the exception: Cursor registers `beforeTabFileRead` with no
# matching after-event, and that before-callback emits a self-contained instant
# event, so its postToolUse is dropped rather than paired.
DEDICATED_AFTER_TOOL_NAMES = frozenset(
    {
        "shell",
        "terminal",
        "bash",
        "run_command",
        "run_shell",
        "edit_file",
        "edit",
        "write_file",
        "write",
        "create_file",
        "delete_file",
        "tab_file_read",
        "tab_file_edit",
        "mcp",
        "mcp_execution",
    }
)

# Routing/PII fields are represented by the session envelope or are not useful
# diagnostic payload. generation_id remains: it is Cursor's turn correlation key.
STRIP_KEYS = frozenset(
    {
        "conversation_id",
        "conversationId",
        "cwd",
        "workspace_roots",
        "transcript_path",
        "user_email",
        "hook_event_name",
        "hookEventName",
    }
)
