from __future__ import annotations

from pathlib import Path

PLATFORM_NAME = "claude"
DISPLAY_NAME = "Claude Code"
SETTINGS_FILE = Path.home() / ".claude" / "settings.json"

HOOK_EVENTS: dict[str, str] = {
    "SessionStart": "thirdeye-claude-session-start",
    "UserPromptSubmit": "thirdeye-claude-user-prompt-submit",
    "PreToolUse": "thirdeye-claude-pre-tool-use",
    "PostToolUse": "thirdeye-claude-post-tool-use",
    "Stop": "thirdeye-claude-stop",
    "SubagentStop": "thirdeye-claude-subagent-stop",
    "StopFailure": "thirdeye-claude-stop-failure",
    "Notification": "thirdeye-claude-notification",
    "PermissionRequest": "thirdeye-claude-permission-request",
    "SessionEnd": "thirdeye-claude-session-end",
    "PostToolUseFailure": "thirdeye-claude-post-tool-use-failure",
    "SubagentStart": "thirdeye-claude-subagent-start",
    "UserPromptExpansion": "thirdeye-claude-user-prompt-expansion",
    "PreCompact": "thirdeye-claude-pre-compact",
    "PostCompact": "thirdeye-claude-post-compact",
    "PermissionDenied": "thirdeye-claude-permission-denied",
}
