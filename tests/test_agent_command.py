from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from thirdeye.cli import main

# Patch targets for the two heavy dependencies.
_PATCH_STREAM = "thirdeye.commands.agent.run_agent_streaming"
_PATCH_PROMPT = "thirdeye.commands.agent.build_agent_prompt"


def _invoke(*args, **kwargs):
    """Invoke the agent command via the top-level CLI (catches sys.exit)."""
    return CliRunner().invoke(main, ["agent", *args], catch_exceptions=False, **kwargs)


# --- happy path ---


def test_help_text_mentions_agent():
    result = CliRunner().invoke(main, ["agent", "--help"])
    assert result.exit_code == 0
    assert "task" in result.output.lower() or "agent" in result.output.lower()


def test_help_mentions_fix_flag():
    result = CliRunner().invoke(main, ["agent", "--help"])
    assert "--fix" in result.output


def test_help_mentions_skill_flags():
    result = CliRunner().invoke(main, ["agent", "--help"])
    assert "--skill" in result.output
    assert "--skills" in result.output


def test_successful_run_exits_zero():
    with (
        patch(_PATCH_STREAM, return_value=(0, 1000)) as mock_stream,
        patch(_PATCH_PROMPT, return_value="composed prompt"),
    ):
        result = CliRunner().invoke(
            main, ["agent", "review sessions from today"], catch_exceptions=False
        )
    assert result.exit_code == 0


def test_default_agent_is_claude():
    captured_harness = []

    def _fake_stream(harness, prompt, cwd, **kwargs):
        captured_harness.append(harness)
        return (0, 500)

    with patch(_PATCH_STREAM, side_effect=_fake_stream), patch(_PATCH_PROMPT, return_value="p"):
        CliRunner().invoke(main, ["agent", "x"], catch_exceptions=False)

    assert len(captured_harness) == 1
    assert captured_harness[0].adapter_name == "claude"


def test_fix_flag_sets_fix_mode():
    captured_harness = []

    def _fake_stream(harness, prompt, cwd, **kwargs):
        captured_harness.append(harness)
        return (0, 500)

    with patch(_PATCH_STREAM, side_effect=_fake_stream), patch(_PATCH_PROMPT, return_value="p"):
        CliRunner().invoke(main, ["agent", "x", "--fix"], catch_exceptions=False)

    assert captured_harness[0].mode == "fix"


def test_without_fix_flag_mode_is_review():
    captured_harness = []

    def _fake_stream(harness, prompt, cwd, **kwargs):
        captured_harness.append(harness)
        return (0, 500)

    with patch(_PATCH_STREAM, side_effect=_fake_stream), patch(_PATCH_PROMPT, return_value="p"):
        CliRunner().invoke(main, ["agent", "x"], catch_exceptions=False)

    assert captured_harness[0].mode == "review"


def test_skills_flag_lists_builtins_and_exits():
    result = CliRunner().invoke(main, ["agent", "--skills"], catch_exceptions=False)
    assert result.exit_code == 0
    from thirdeye.agent.prompt import VALID_SKILLS

    for name in VALID_SKILLS:
        assert name in result.output


def test_skills_flag_does_not_require_task():
    result = CliRunner().invoke(main, ["agent", "--skills"], catch_exceptions=False)
    assert result.exit_code == 0


def test_skills_flag_does_not_run_the_agent():
    with patch(_PATCH_STREAM) as mock_stream, patch(_PATCH_PROMPT) as mock_prompt:
        CliRunner().invoke(main, ["agent", "--skills"], catch_exceptions=False)
    mock_stream.assert_not_called()
    mock_prompt.assert_not_called()


def test_skill_path_is_loaded_and_injected(tmp_path):
    skill_file = tmp_path / "my-skill.md"
    skill_file.write_text("# My Custom Skill\nDo something useful.\n")

    captured_kwargs: list[dict] = []

    def _fake_prompt(task, **kwargs):
        captured_kwargs.append(kwargs)
        return "composed"

    with (
        patch(_PATCH_STREAM, return_value=(0, 500)),
        patch(_PATCH_PROMPT, side_effect=_fake_prompt),
    ):
        result = CliRunner().invoke(
            main,
            ["agent", "do stuff", "--skill", str(skill_file)],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "skill_bodies" in captured_kwargs[0]
    bodies = captured_kwargs[0]["skill_bodies"]
    # custom skill present
    assert any("My Custom Skill" in b for b in bodies)
    # built-in defaults also present
    from thirdeye.agent.prompt import DEFAULT_SKILLS

    assert len(bodies) > len(DEFAULT_SKILLS)


def test_multiple_skill_paths_are_all_injected(tmp_path):
    skill_a = tmp_path / "skill-a.md"
    skill_b = tmp_path / "skill-b.md"
    skill_a.write_text("# Skill A\n")
    skill_b.write_text("# Skill B\n")

    captured_kwargs: list[dict] = []

    def _fake_prompt(task, **kwargs):
        captured_kwargs.append(kwargs)
        return "composed"

    with (
        patch(_PATCH_STREAM, return_value=(0, 500)),
        patch(_PATCH_PROMPT, side_effect=_fake_prompt),
    ):
        CliRunner().invoke(
            main,
            ["agent", "do stuff", "--skill", str(skill_a), "--skill", str(skill_b)],
            catch_exceptions=False,
        )

    from thirdeye.agent.prompt import DEFAULT_SKILLS

    bodies = captured_kwargs[0]["skill_bodies"]
    assert len(bodies) == len(DEFAULT_SKILLS) + 2


def test_no_skill_path_uses_defaults():
    captured_kwargs: list[dict] = []

    def _fake_prompt(task, **kwargs):
        captured_kwargs.append(kwargs)
        return "composed"

    with (
        patch(_PATCH_STREAM, return_value=(0, 500)),
        patch(_PATCH_PROMPT, side_effect=_fake_prompt),
    ):
        CliRunner().invoke(main, ["agent", "do stuff"], catch_exceptions=False)

    assert "skill_bodies" not in captured_kwargs[0]


def test_missing_task_without_skills_flag_errors():
    result = CliRunner().invoke(main, ["agent"])
    assert result.exit_code != 0


# --- error paths ---


def test_unknown_agent_produces_click_exception():
    result = CliRunner().invoke(main, ["agent", "x", "--agent", "no-such-agent"])
    assert result.exit_code != 0
    assert "unknown agent" in result.output.lower() or "error" in result.output.lower()


def test_file_not_found_from_exec_produces_click_exception():
    with (
        patch(_PATCH_STREAM, side_effect=FileNotFoundError("claude not found")),
        patch(_PATCH_PROMPT, return_value="p"),
    ):
        result = CliRunner().invoke(main, ["agent", "x"])
    assert result.exit_code != 0
    assert "claude" in result.output.lower() or "error" in result.output.lower()


def test_agent_nonzero_exit_propagates():
    with patch(_PATCH_STREAM, return_value=(2, 500)), patch(_PATCH_PROMPT, return_value="p"):
        result = CliRunner().invoke(main, ["agent", "x"])
    assert result.exit_code == 2


def test_thirdeye_home_is_passed_to_run_agent_streaming():
    """commands/agent.py must forward config.root as thirdeye_home so new sessions get tagged."""
    captured_kwargs: list[dict] = []

    def _fake_stream(harness, prompt, cwd, **kwargs):
        captured_kwargs.append(kwargs)
        return (0, 500)

    with patch(_PATCH_STREAM, side_effect=_fake_stream), patch(_PATCH_PROMPT, return_value="p"):
        CliRunner().invoke(main, ["agent", "x"], catch_exceptions=False)

    assert len(captured_kwargs) == 1
    assert "thirdeye_home" in captured_kwargs[0]
    assert captured_kwargs[0]["thirdeye_home"] is not None


# --- CLI registration ---


def test_agent_command_is_registered_in_main_cli():
    result = CliRunner().invoke(main, ["--help"])
    assert "agent" in result.output


def test_cwd_option_passed_down(tmp_path):
    captured_prompt_cwd = []
    captured_stream_cwd = []

    def _fake_prompt(task, *, cwd=None, **kwargs):
        captured_prompt_cwd.append(cwd)
        return "composed prompt"

    def _fake_stream(harness, prompt, cwd, **kwargs):
        captured_stream_cwd.append(cwd)
        return (0, 500)

    # Let's create a custom directory to pass to --cwd
    custom_dir = tmp_path / "custom-cwd"
    custom_dir.mkdir()

    with (
        patch(_PATCH_STREAM, side_effect=_fake_stream),
        patch(_PATCH_PROMPT, side_effect=_fake_prompt),
    ):
        result = CliRunner().invoke(
            main,
            ["agent", "do something", "--cwd", str(custom_dir)],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert len(captured_prompt_cwd) == 1
    assert captured_prompt_cwd[0] == custom_dir
    assert len(captured_stream_cwd) == 1
    assert captured_stream_cwd[0] == custom_dir


# --- --stream flag ---


def test_help_mentions_stream_flag():
    result = CliRunner().invoke(main, ["agent", "--help"])
    assert "--stream" in result.output


def test_stream_flag_creates_streaming_harness():
    """--stream creates an AgentHarness with streaming=True."""
    captured_harnesses: list = []

    def _fake_stream(harness, prompt, cwd, **kwargs):
        captured_harnesses.append(harness)
        return (0, 500)

    with (
        patch(_PATCH_STREAM, side_effect=_fake_stream),
        patch(_PATCH_PROMPT, return_value="p"),
    ):
        CliRunner().invoke(main, ["agent", "x", "--stream"], catch_exceptions=False)

    assert captured_harnesses[0].streaming is True


def test_no_stream_flag_creates_non_streaming_harness():
    """Without --stream, harness.streaming is False."""
    captured_harnesses: list = []

    def _fake_stream(harness, prompt, cwd, **kwargs):
        captured_harnesses.append(harness)
        return (0, 500)

    with (
        patch(_PATCH_STREAM, side_effect=_fake_stream),
        patch(_PATCH_PROMPT, return_value="p"),
    ):
        CliRunner().invoke(main, ["agent", "x"], catch_exceptions=False)

    assert captured_harnesses[0].streaming is False
