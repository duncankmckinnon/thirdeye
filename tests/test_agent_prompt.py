from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.agent.prompt import (
    DEFAULT_SKILLS,
    VALID_SKILLS,
    build_agent_prompt,
    load_builtin_skills,
    load_skill_file,
)


def test_default_returns_string():
    result = build_agent_prompt("review sessions from today")
    assert isinstance(result, str)


def test_task_appears_at_end():
    result = build_agent_prompt("my task here")
    assert result.endswith("TASK:\nmy task here")


def test_default_skills_includes_all_valid_skills():
    assert set(DEFAULT_SKILLS) == VALID_SKILLS


def test_default_prompt_contains_all_skills():
    result = build_agent_prompt("x")
    assert "## Overview" in result  # use-thirdeye
    assert "Reviewing thirdeye" in result  # thirdeye-review
    assert "Evaluating thirdeye" in result  # thirdeye-evals


def test_single_skill_only_loads_that_skill():
    result = build_agent_prompt("x", skills=["use-thirdeye"])
    assert "use-thirdeye" in result or "## Overview" in result
    # thirdeye-review heading must NOT appear
    assert "Reviewing thirdeye traces" not in result


def test_unknown_skill_raises_value_error():
    with pytest.raises(ValueError, match="unknown skill"):
        build_agent_prompt("x", skills=["nonexistent-skill"])


def test_context_block_contains_date():
    from datetime import date

    result = build_agent_prompt("x")
    assert f"date: {date.today().isoformat()}" in result


def test_context_block_contains_cwd_when_provided():
    result = build_agent_prompt("x", cwd=Path("/my/project"))
    assert "cwd: /my/project" in result


def test_context_block_contains_thirdeye_home_when_provided():
    result = build_agent_prompt("x", thirdeye_home=Path("/home/user/.thirdeye"))
    assert "thirdeye_home: /home/user/.thirdeye" in result


def test_context_block_omits_cwd_when_not_provided():
    result = build_agent_prompt("x")
    assert "cwd:" not in result


def test_empty_skills_list_produces_context_and_task_only():
    result = build_agent_prompt("my task", skills=[])
    assert "TASK:\nmy task" in result
    assert "Context:" in result
    # No skill content — only one separator between context and task
    assert result.count("---") == 1


def test_separator_appears_between_sections():
    result = build_agent_prompt("x", skills=["use-thirdeye", "thirdeye-review"])
    assert "\n\n---\n\n" in result


def test_valid_skills_set_contains_expected_names():
    assert "use-thirdeye" in VALID_SKILLS
    assert "thirdeye-review" in VALID_SKILLS
    assert "thirdeye-evals" in VALID_SKILLS
    assert "thirdeye-filter" in VALID_SKILLS


def test_all_valid_skills_load_without_error():
    for name in VALID_SKILLS:
        result = build_agent_prompt("x", skills=[name])
        assert len(result) > 100  # each skill has substantial content


# --- load_skill_file ---


def test_load_skill_file_reads_content(tmp_path):
    skill = tmp_path / "my-skill.md"
    skill.write_text("# My Skill\nDo useful things.\n")
    result = load_skill_file(skill)
    assert "My Skill" in result
    assert "Do useful things." in result


def test_load_skill_file_strips_frontmatter(tmp_path):
    skill = tmp_path / "my-skill.md"
    skill.write_text("---\nname: my-skill\ndescription: test\n---\n# Body\nContent here.\n")
    result = load_skill_file(skill)
    assert "---" not in result
    assert "name: my-skill" not in result
    assert "Body" in result


# --- skill_bodies ---


def test_skill_bodies_overrides_builtin_skills(tmp_path):
    skill = tmp_path / "custom.md"
    skill.write_text("# Custom Skill\nCustom content.\n")
    body = load_skill_file(skill)
    result = build_agent_prompt("x", skill_bodies=[body])
    assert "Custom content." in result
    assert "Reviewing thirdeye" not in result


def test_skill_bodies_content_appears_in_prompt(tmp_path):
    skill = tmp_path / "custom.md"
    skill.write_text("# My Skill\nDo something special.\n")
    body = load_skill_file(skill)
    result = build_agent_prompt("my task", skill_bodies=[body])
    assert "Do something special." in result
    assert "TASK:\nmy task" in result


def test_load_builtin_skills_returns_all_defaults():
    bodies = load_builtin_skills()
    assert len(bodies) == len(DEFAULT_SKILLS)
    assert all(isinstance(b, str) and len(b) > 0 for b in bodies)


def test_skill_bodies_empty_list_produces_context_and_task_only():
    result = build_agent_prompt("my task", skill_bodies=[])
    assert "TASK:\nmy task" in result
    assert result.count("---") == 1
