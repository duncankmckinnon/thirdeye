from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.agent.harness import (
    _FIX_ARGS,
    _REVIEW_ARGS,
    _STREAM_FIX_ARGS,
    _STREAM_REVIEW_ARGS,
    AgentHarness,
)
from thirdeye.eval.agents.base import AgentConfig, ConfigAdapter
from thirdeye.eval.agents.claude import ClaudeAdapter
from thirdeye.eval.agents.codex import CodexAdapter
from thirdeye.eval.agents.gemini import GeminiAdapter

# --- construction ---


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        AgentHarness(ClaudeAdapter(), "invalid")


def test_mode_property():
    h = AgentHarness(ClaudeAdapter(), "review")
    assert h.mode == "review"


def test_adapter_name_property():
    h = AgentHarness(ClaudeAdapter(), "review")
    assert h.adapter_name == "claude"


def test_command_property_matches_adapter():
    h = AgentHarness(ClaudeAdapter(), "review")
    assert h.command == "claude"


# --- ClaudeAdapter overrides ---


def test_claude_review_uses_text_output():
    h = AgentHarness(ClaudeAdapter(), "review")
    cmd = h.build_command("my task", Path("/tmp"))
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "text"


def test_claude_review_has_allowed_tools():
    h = AgentHarness(ClaudeAdapter(), "review")
    cmd = h.build_command("x", Path("/"))
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    tools = cmd[idx + 1]
    assert "Bash(thirdeye *)" in tools
    assert "Read" in tools
    assert "Grep" in tools


def test_claude_review_does_not_allow_edit_or_write():
    h = AgentHarness(ClaudeAdapter(), "review")
    cmd = h.build_command("x", Path("/"))
    idx = cmd.index("--allowedTools")
    tools = cmd[idx + 1]
    assert "Edit" not in tools
    assert "Write" not in tools


def test_claude_fix_uses_text_output():
    h = AgentHarness(ClaudeAdapter(), "fix")
    cmd = h.build_command("x", Path("/"))
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "text"


def test_claude_fix_has_no_allowed_tools_restriction():
    h = AgentHarness(ClaudeAdapter(), "fix")
    cmd = h.build_command("x", Path("/"))
    assert "--allowedTools" not in cmd


def test_claude_fix_does_not_have_json_output():
    h = AgentHarness(ClaudeAdapter(), "fix")
    cmd = h.build_command("x", Path("/"))
    assert "json" not in cmd


# --- GeminiAdapter overrides ---


def test_gemini_review_uses_plan_approval_mode():
    h = AgentHarness(GeminiAdapter(), "review")
    cmd = h.build_command("x", Path("/"))
    assert "--approval-mode" in cmd
    idx = cmd.index("--approval-mode")
    assert cmd[idx + 1] == "plan"


def test_gemini_review_no_output_format_json():
    h = AgentHarness(GeminiAdapter(), "review")
    cmd = h.build_command("x", Path("/"))
    assert "--output-format" not in cmd


def test_gemini_fix_has_no_approval_mode():
    h = AgentHarness(GeminiAdapter(), "fix")
    cmd = h.build_command("x", Path("/"))
    assert "--approval-mode" not in cmd


# --- CodexAdapter overrides ---


def test_codex_review_uses_sandbox_read_only():
    h = AgentHarness(CodexAdapter(), "review")
    cmd = h.build_command("x", Path("/"))
    assert "--sandbox" in cmd
    idx = cmd.index("--sandbox")
    assert cmd[idx + 1] == "read-only"


def test_codex_review_no_json_flag():
    h = AgentHarness(CodexAdapter(), "review")
    cmd = h.build_command("x", Path("/"))
    assert "--json" not in cmd


def test_codex_fix_no_sandbox():
    h = AgentHarness(CodexAdapter(), "fix")
    cmd = h.build_command("x", Path("/"))
    assert "--sandbox" not in cmd


# --- prompt substitution ---


def test_prompt_is_substituted_in_command():
    h = AgentHarness(ClaudeAdapter(), "review")
    cmd = h.build_command("evaluate this session", Path("/"))
    assert "evaluate this session" in cmd
    assert "{prompt}" not in cmd


def test_prompt_with_spaces_stays_single_argv_element():
    h = AgentHarness(ClaudeAdapter(), "review")
    cmd = h.build_command("a task with spaces", Path("/"))
    assert "a task with spaces" in cmd
    # Must appear as a single element, not split on spaces
    assert cmd.count("a task with spaces") == 1


# --- ConfigAdapter fallback ---


def test_unknown_adapter_uses_its_own_args():
    custom = ConfigAdapter(
        name="my-agent",
        config=AgentConfig(command="my-agent", args=["-p", "{prompt}", "--flag"]),
    )
    h = AgentHarness(custom, "review")
    cmd = h.build_command("hello", Path("/"))
    assert cmd == ["my-agent", "-p", "hello", "--flag"]


def test_unknown_adapter_fix_mode_also_uses_own_args():
    custom = ConfigAdapter(
        name="my-agent",
        config=AgentConfig(command="my-agent", args=["-p", "{prompt}"]),
    )
    h = AgentHarness(custom, "fix")
    cmd = h.build_command("hello", Path("/"))
    assert cmd == ["my-agent", "-p", "hello"]


# --- override table invariants ---


def test_review_args_table_covers_all_builtin_names():
    for name in ("claude", "gemini", "codex"):
        assert name in _REVIEW_ARGS


def test_fix_args_table_covers_all_builtin_names():
    for name in ("claude", "gemini", "codex"):
        assert name in _FIX_ARGS


def test_all_review_args_contain_prompt_placeholder():
    for name, args in _REVIEW_ARGS.items():
        assert "{prompt}" in args, f"{name} review args missing {{prompt}}"


def test_all_fix_args_contain_prompt_placeholder():
    for name, args in _FIX_ARGS.items():
        assert "{prompt}" in args, f"{name} fix args missing {{prompt}}"


# --- streaming mode ---


def test_streaming_defaults_to_false():
    h = AgentHarness(ClaudeAdapter(), "review")
    assert h.streaming is False


def test_streaming_property_reflects_init():
    h = AgentHarness(ClaudeAdapter(), "review", streaming=True)
    assert h.streaming is True


def test_streaming_review_uses_stream_json_for_claude():
    h = AgentHarness(ClaudeAdapter(), "review", streaming=True)
    cmd = h.build_command("task", Path("/tmp"))
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "stream-json"


def test_streaming_fix_uses_stream_json_for_claude():
    h = AgentHarness(ClaudeAdapter(), "fix", streaming=True)
    cmd = h.build_command("task", Path("/tmp"))
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "stream-json"


def test_non_streaming_review_still_uses_text_for_claude():
    h = AgentHarness(ClaudeAdapter(), "review", streaming=False)
    cmd = h.build_command("task", Path("/tmp"))
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "text"


def test_stream_review_args_table_covers_all_builtin_names():
    for name in ("claude", "gemini", "codex"):
        assert name in _STREAM_REVIEW_ARGS


def test_stream_fix_args_table_covers_all_builtin_names():
    for name in ("claude", "gemini", "codex"):
        assert name in _STREAM_FIX_ARGS


def test_stream_review_args_contain_prompt_placeholder():
    for name, args in _STREAM_REVIEW_ARGS.items():
        assert "{prompt}" in args, f"{name} stream-review args missing {{prompt}}"


def test_stream_fix_args_contain_prompt_placeholder():
    for name, args in _STREAM_FIX_ARGS.items():
        assert "{prompt}" in args, f"{name} stream-fix args missing {{prompt}}"


def test_streaming_review_includes_verbose_for_claude():
    h = AgentHarness(ClaudeAdapter(), "review", streaming=True)
    cmd = h.build_command("task", Path("/tmp"))
    assert "--verbose" in cmd


def test_streaming_fix_includes_verbose_for_claude():
    h = AgentHarness(ClaudeAdapter(), "fix", streaming=True)
    cmd = h.build_command("task", Path("/tmp"))
    assert "--verbose" in cmd
