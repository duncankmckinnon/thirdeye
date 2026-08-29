from thirdeye.platforms.cursor.constants import (
    DEDICATED_AFTER_TOOL_NAMES,
    READ_TOOL_NAMES,
    TRACED_EVENTS,
)


def test_traced_events_include_pre_tool_and_both_subagent_edges_once():
    for event_name in ("preToolUse", "subagentStart", "subagentStop"):
        assert event_name in TRACED_EVENTS
        assert TRACED_EVENTS.count(event_name) == 1


def test_read_aliases_are_not_skipped_by_the_dedicated_after_set():
    """`_post_tool_use` checks the dedicated set first, so an overlap would
    silently drop read results again."""
    assert READ_TOOL_NAMES.isdisjoint(DEDICATED_AFTER_TOOL_NAMES)


def test_read_tool_names_covers_known_read_aliases():
    assert READ_TOOL_NAMES == {"read_file", "read", "view_file", "view"}


def test_dedicated_after_set_keeps_shell_mcp_and_edit_write_aliases():
    assert {
        "shell",
        "terminal",
        "bash",
        "run_command",
        "run_shell",
        "mcp",
        "mcp_execution",
        "edit_file",
        "edit",
        "write_file",
        "write",
        "create_file",
        "delete_file",
    } <= DEDICATED_AFTER_TOOL_NAMES


def test_tool_name_sets_are_lowercase():
    """Membership is tested against `name.lower()`, so uppercase entries would
    be unreachable."""
    for name in READ_TOOL_NAMES | DEDICATED_AFTER_TOOL_NAMES:
        assert name == name.lower()
