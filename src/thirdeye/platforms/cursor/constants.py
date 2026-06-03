from __future__ import annotations

from pathlib import Path

PLATFORM_NAME = "cursor"
DISPLAY_NAME = "Cursor"

HOOKS_FILE = Path.home() / ".cursor" / "hooks.json"

HOOK_BIN_NAME = "thirdeye-cursor-hook"

HOOK_TIMEOUT_S = 30

TRACED_EVENTS: tuple[str, ...] = (
    "sessionStart",
    "sessionEnd",
    "beforeSubmitPrompt",
    "afterAgentResponse",
    "beforeShellExecution",
    "afterShellExecution",
    "beforeMCPExecution",
    "afterMCPExecution",
    "beforeReadFile",
    "afterFileEdit",
    "beforeTabFileRead",
    "afterTabFileEdit",
    "postToolUse",
    "stop",
)

DEDICATED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "shell",
        "terminal",
        "bash",
        "run_command",
        "run_shell",
        "read_file",
        "read",
        "view_file",
        "view",
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

STRIP_KEYS: frozenset[str] = frozenset(
    {
        "conversation_id",
        "conversationId",
        "workspace_roots",
        "transcript_path",
        "user_email",
        "hook_event_name",
        "hookEventName",
    }
)
