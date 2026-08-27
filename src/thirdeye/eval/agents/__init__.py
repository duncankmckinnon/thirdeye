"""Built-in adapters registry plus user YAML override loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from thirdeye.eval.agents.base import (
    AgentAdapter,
    AgentConfig,
    ConfigAdapter,
    OutputFormat,
)
from thirdeye.eval.agents.claude import ClaudeAdapter
from thirdeye.eval.agents.codex import CodexAdapter
from thirdeye.paths import eval_agents_config_path

BUILTIN_ADAPTERS: dict[str, type[AgentAdapter]] = {
    "claude": ClaudeAdapter,
    "codex": CodexAdapter,
}


def _load_user_overrides(thirdeye_home: Path) -> dict[str, dict[str, Any]]:
    path = eval_agents_config_path(thirdeye_home)
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def get_adapter(name: str, *, thirdeye_home: Path | None = None) -> AgentAdapter:
    """Return an adapter for `name`. User overrides win over builtins.

    Resolution order:
    1. User override at <thirdeye_home>/eval-agents.yaml (if home given)
    2. Built-in adapter for the name
    3. ValueError if neither matches
    """
    if thirdeye_home is not None:
        overrides = _load_user_overrides(thirdeye_home)
        if name in overrides:
            return ConfigAdapter.from_config(name, overrides[name])

    builtin = BUILTIN_ADAPTERS.get(name)
    if builtin is not None:
        return builtin()

    raise ValueError(f"unknown agent: {name!r}")


def list_agent_names(thirdeye_home: Path | None = None) -> list[str]:
    """Return sorted list of available agent names (builtins ∪ overrides)."""
    names: set[str] = set(BUILTIN_ADAPTERS.keys())
    if thirdeye_home is not None:
        names.update(_load_user_overrides(thirdeye_home).keys())
    return sorted(names)


__all__ = [
    "AgentAdapter",
    "AgentConfig",
    "ConfigAdapter",
    "OutputFormat",
    "ClaudeAdapter",
    "CodexAdapter",
    "BUILTIN_ADAPTERS",
    "get_adapter",
    "list_agent_names",
]
