from __future__ import annotations

from thirdeye.platforms.cursor import constants


class TestNames:
    def test_platform_name(self):
        assert constants.PLATFORM_NAME == "cursor"

    def test_display_name(self):
        assert constants.DISPLAY_NAME == "Cursor"

    def test_hook_bin_name(self):
        assert constants.HOOK_BIN_NAME == "thirdeye-cursor-hook"


class TestHooksFile:
    def test_under_cursor_dir(self):
        assert constants.HOOKS_FILE.name == "hooks.json"
        assert constants.HOOKS_FILE.parent.name == ".cursor"


class TestTracedEvents:
    def test_contains_all_documented_events(self):
        expected = {
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
        }
        assert set(constants.TRACED_EVENTS) == expected

    def test_is_tuple_not_list(self):
        assert isinstance(constants.TRACED_EVENTS, tuple)

    def test_no_duplicates(self):
        assert len(constants.TRACED_EVENTS) == len(set(constants.TRACED_EVENTS))


class TestDedicatedToolNames:
    def test_includes_shell_variants(self):
        for name in ("shell", "bash", "terminal", "run_command"):
            assert name in constants.DEDICATED_TOOL_NAMES

    def test_includes_file_variants(self):
        for name in ("read_file", "edit_file", "write_file"):
            assert name in constants.DEDICATED_TOOL_NAMES

    def test_includes_mcp(self):
        assert "mcp" in constants.DEDICATED_TOOL_NAMES


class TestStripKeys:
    def test_strips_routing_keys(self):
        for key in ("conversation_id", "conversationId", "hook_event_name", "hookEventName"):
            assert key in constants.STRIP_KEYS

    def test_strips_pii(self):
        assert "user_email" in constants.STRIP_KEYS

    def test_strips_noisy_paths(self):
        assert "transcript_path" in constants.STRIP_KEYS
        assert "workspace_roots" in constants.STRIP_KEYS
