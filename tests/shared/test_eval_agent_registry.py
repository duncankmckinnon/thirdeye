from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from thirdeye.eval.agents import (
    BUILTIN_ADAPTERS,
    get_adapter,
    list_agent_names,
)
from thirdeye.eval.agents.claude import ClaudeAdapter
from thirdeye.paths import eval_agents_config_path


def test_builtin_names_present():
    assert set(BUILTIN_ADAPTERS.keys()) == {"claude", "codex"}


def test_get_builtin_adapter():
    a = get_adapter("claude")
    assert isinstance(a, ClaudeAdapter)
    assert a.name == "claude"


def test_get_unknown_agent_raises():
    with pytest.raises(ValueError, match="unknown agent"):
        get_adapter("nonexistent")


def test_get_with_no_home_skips_overrides():
    a = get_adapter("claude", thirdeye_home=None)
    assert isinstance(a, ClaudeAdapter)


def test_user_override_replaces_builtin(tmp_path: Path):
    cfg = eval_agents_config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "claude": {
                    "command": "my-claude",
                    "args": ["--custom", "{prompt}"],
                    "output_format": "text",
                }
            }
        )
    )
    a = get_adapter("claude", thirdeye_home=tmp_path)
    assert a.config.command == "my-claude"
    assert "--custom" in a.config.args


def test_user_can_define_new_agent(tmp_path: Path):
    cfg = eval_agents_config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "myagent": {
                    "command": "myagent",
                    "args": ["-p", "{prompt}"],
                }
            }
        )
    )
    a = get_adapter("myagent", thirdeye_home=tmp_path)
    assert a.name == "myagent"
    assert a.config.command == "myagent"


def test_list_agent_names_builtins_only():
    names = list_agent_names()
    assert names == ["claude", "codex"]


def test_list_agent_names_includes_overrides(tmp_path: Path):
    cfg = eval_agents_config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "myagent": {"command": "myagent", "args": ["{prompt}"]},
            }
        )
    )
    assert list_agent_names(tmp_path) == ["claude", "codex", "myagent"]


def test_malformed_overrides_yaml_falls_back_to_builtins(tmp_path: Path):
    cfg = eval_agents_config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{not valid yaml")
    a = get_adapter("claude", thirdeye_home=tmp_path)
    assert isinstance(a, ClaudeAdapter)


def test_non_dict_overrides_ignored(tmp_path: Path):
    cfg = eval_agents_config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "claude": "not a dict — should be skipped",
            }
        )
    )
    a = get_adapter("claude", thirdeye_home=tmp_path)
    assert isinstance(a, ClaudeAdapter)
