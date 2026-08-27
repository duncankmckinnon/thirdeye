from __future__ import annotations

import json
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


@dataclass
class AgentConfig:
    """YAML-serializable configuration for an agent adapter."""

    command: str
    args: list[str] = field(default_factory=lambda: ["{prompt}"])
    output_format: OutputFormat = OutputFormat.TEXT
    json_result_key: str = "result"
    json_cost_key: str = "cost_usd"

    def __post_init__(self) -> None:
        if isinstance(self.output_format, str):
            self.output_format = OutputFormat(self.output_format)
        if not self.command:
            raise ValueError("command must not be empty")
        if not self.args:
            raise ValueError("args must not be empty")
        if "{prompt}" not in self.args:
            raise ValueError("args must contain '{prompt}' placeholder")

    def to_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "command": self.command,
            "args": list(self.args),
            "output_format": self.output_format.value,
        }
        if self.output_format == OutputFormat.JSON:
            entry["json_result_key"] = self.json_result_key
            entry["json_cost_key"] = self.json_cost_key
        return entry

    @classmethod
    def from_dict(cls, entry: dict[str, Any], default_command: str = "") -> AgentConfig:
        return cls(
            command=entry.get("command", default_command) or default_command,
            args=list(entry.get("args", ["{prompt}"])),
            output_format=entry.get("output_format", "text"),
            json_result_key=entry.get("json_result_key", "result"),
            json_cost_key=entry.get("json_cost_key", "cost_usd"),
        )


class AgentAdapter(ABC):
    """Abstraction for a CLI agent platform (Claude, Codex, etc.).

    Built-in adapters set ``self.config`` in ``__init__``. Custom adapters
    are constructed from YAML via ``ConfigAdapter.from_config(name, entry)``.
    """

    name: str
    config: AgentConfig

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        """Build the subprocess argv, substituting ``{prompt}`` in args."""
        resolved = [a.replace("{prompt}", prompt) for a in self.config.args]
        return [self.config.command, *resolved]

    def parse_output(self, raw: str) -> tuple[str, dict[str, Any]]:
        """Return ``(agent_text, cost_dict)``. Default handles text + JSON."""
        if self.config.output_format == OutputFormat.JSON:
            try:
                data = json.loads(raw)
                result = data.get(self.config.json_result_key, raw)
                cost = data.get(self.config.json_cost_key, {})
                if not isinstance(cost, dict):
                    cost = {}
                return (str(result), cost)
            except (json.JSONDecodeError, TypeError):
                return (raw, {})
        return (raw.strip(), {})

    def to_config(self) -> dict[str, Any]:
        return self.config.to_dict()


@dataclass
class ConfigAdapter(AgentAdapter):
    """Adapter driven by a YAML config entry (no hard-coded args)."""

    name: str
    config: AgentConfig

    @classmethod
    def from_config(cls, name: str, entry: dict[str, Any]) -> ConfigAdapter:
        return cls(name=name, config=AgentConfig.from_dict(entry, default_command=name))
