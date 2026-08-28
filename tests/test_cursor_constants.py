from thirdeye.platforms.cursor.constants import (
    DEDICATED_AFTER_TOOL_NAMES,
    READ_TOOL_NAMES,
    TRACED_EVENTS,
)


def test_traced_events_contains_subagent_stop_once():
    assert TRACED_EVENTS.count("subagentStop") == 1


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
