from __future__ import annotations

from pathlib import Path

PLATFORM_NAME = "codex"
DISPLAY_NAME = "Codex CLI"
CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_CONFIG_FILE = CODEX_CONFIG_DIR / "config.toml"
NOTIFY_BIN_NAME = "thirdeye-codex-notify"

# Codex's separate, newer hooks.json mechanism: same event names and JSON
# payload shape as Claude Code's own hooks, unlike notify's argv/thread-id
# convention. See platforms/codex/hooks_json.py for the handlers.
CODEX_HOOKS_FILE = CODEX_CONFIG_DIR / "hooks.json"

# Event -> the thirdeye-codex-* binary that owns it. Every entry here is
# additive in hooks.json (thirdeye's command is appended alongside whatever
# else is already registered for that event), never exclusive ownership like
# notify's single argv slot. Limited to the event names Codex's own
# hooks.json actually recognizes (visible as [hooks.state] trust entries in
# config.toml once approved) — notably no "Notification", unlike Claude Code.
HOOKS_JSON_BIN_NAMES: dict[str, str] = {
    "SessionStart": "thirdeye-codex-session-start",
    "SessionEnd": "thirdeye-codex-session-end",
    "UserPromptSubmit": "thirdeye-codex-user-prompt-submit",
    "SubagentStart": "thirdeye-codex-subagent-start",
    "SubagentStop": "thirdeye-codex-subagent-stop",
    "PermissionRequest": "thirdeye-codex-permission-request",
    "PreCompact": "thirdeye-codex-pre-compact",
    "PostCompact": "thirdeye-codex-post-compact",
}

# Events deliberately never wired to a thirdeye-codex-* binary: notify's
# rollout reconstruction already gives these a complete, richer picture (real
# token usage, reconstructed message content) than a live per-event hook
# could, so wiring them too would capture the same tool calls and turns
# twice. install() actively strips any thirdeye-claude-* entry it finds under
# these — that combination is always a misconfiguration (Codex's hooks
# pointed at Claude's own handlers), never a legitimate integration.
HOOKS_JSON_UNSUPPORTED_EVENTS = ("PreToolUse", "PostToolUse", "Stop")
