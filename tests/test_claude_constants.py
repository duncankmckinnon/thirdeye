from __future__ import annotations

import tomllib
from pathlib import Path

from thirdeye.platforms.claude.constants import (
    DISPLAY_NAME,
    HOOK_EVENTS,
    PLATFORM_NAME,
    SETTINGS_FILE,
)

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_platform_name():
    assert PLATFORM_NAME == "claude"


def test_display_name():
    assert DISPLAY_NAME == "Claude Code"


def test_settings_file_under_claude_home():
    parts = SETTINGS_FILE.parts
    assert ".claude" in parts
    assert parts[-1] == "settings.json"


def test_hook_events_covers_known_lifecycle():
    expected = {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SubagentStop",
        "StopFailure",
        "Notification",
        "PermissionRequest",
        "SessionEnd",
        "PostToolUseFailure",
        "SubagentStart",
        "UserPromptExpansion",
        "PreCompact",
        "PostCompact",
        "PermissionDenied",
    }
    assert set(HOOK_EVENTS.keys()) == expected


def test_hook_events_has_sixteen_entries():
    assert len(HOOK_EVENTS) == 16


def test_new_hook_events_have_expected_script_names():
    assert HOOK_EVENTS["PostToolUseFailure"] == "thirdeye-claude-post-tool-use-failure"
    assert HOOK_EVENTS["SubagentStart"] == "thirdeye-claude-subagent-start"
    assert HOOK_EVENTS["UserPromptExpansion"] == "thirdeye-claude-user-prompt-expansion"
    assert HOOK_EVENTS["PreCompact"] == "thirdeye-claude-pre-compact"
    assert HOOK_EVENTS["PostCompact"] == "thirdeye-claude-post-compact"
    assert HOOK_EVENTS["PermissionDenied"] == "thirdeye-claude-permission-denied"


def test_hook_event_scripts_unique():
    assert len(set(HOOK_EVENTS.values())) == len(HOOK_EVENTS)


def test_hook_event_scripts_have_thirdeye_prefix():
    for script in HOOK_EVENTS.values():
        assert script.startswith("thirdeye-claude-")


def test_every_hook_event_has_project_scripts_entry():
    # A hook wired without a matching [project.scripts] entry point fails
    # silently at runtime; parse pyproject.toml to catch it in CI instead.
    with _PYPROJECT.open("rb") as fh:
        pyproject = tomllib.load(fh)
    scripts = pyproject["project"]["scripts"]
    for script in HOOK_EVENTS.values():
        assert script in scripts, f"missing [project.scripts] entry for {script}"
