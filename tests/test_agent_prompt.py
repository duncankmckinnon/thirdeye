from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.agent.prompt import (
    DEFAULT_SKILLS,
    VALID_SKILLS,
    build_agent_prompt,
)


def test_default_returns_string():
    result = build_agent_prompt("review sessions from today")
    assert isinstance(result, str)


def test_task_appears_at_end():
    result = build_agent_prompt("my task here")
    assert result.endswith("TASK:\nmy task here")


def test_default_skills_are_use_thirdeye_and_review():
    assert DEFAULT_SKILLS == ["use-thirdeye", "thirdeye-review"]


def test_default_prompt_contains_both_default_skills():
    result = build_agent_prompt("x")
    # Each skill body is non-empty — check that at least one distinctive phrase
    # from each shipped skill appears. use-thirdeye opens with "## Overview",
    # thirdeye-review opens with "# Reviewing thirdeye traces".
    assert "## Overview" in result
    assert "thirdeye-review" in result or "Reviewing thirdeye" in result


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
    # No skill content — the prompt is just context + task
    assert "---" not in result  # no separator without skills


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
