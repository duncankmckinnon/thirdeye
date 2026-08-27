from __future__ import annotations

from pathlib import Path
from typing import Literal

from thirdeye.eval.agents.base import AgentAdapter

# Per-adapter arg overrides for review mode (read-only + thirdeye CLI).
_REVIEW_ARGS: dict[str, list[str]] = {
    "claude": [
        "-p",
        "{prompt}",
        "--output-format",
        "text",
        "--allowedTools",
        "Bash(thirdeye *) Bash(jq *) Read Glob Grep",
    ],
    "codex": ["exec", "--sandbox", "read-only", "{prompt}"],
}

# Per-adapter arg overrides for fix mode (full tool access).
_FIX_ARGS: dict[str, list[str]] = {
    "claude": ["-p", "{prompt}", "--output-format", "text"],
    "codex": ["exec", "{prompt}"],
}

# Streaming variants: emit each event as newline-delimited JSON for real-time display.
_STREAM_REVIEW_ARGS: dict[str, list[str]] = {
    "claude": [
        "-p",
        "{prompt}",
        "--output-format",
        "stream-json",
        "--verbose",
        "--allowedTools",
        "Bash(thirdeye *) Bash(jq *) Read Glob Grep",
    ],
    "codex": ["exec", "--sandbox", "read-only", "{prompt}"],
}

_STREAM_FIX_ARGS: dict[str, list[str]] = {
    "claude": ["-p", "{prompt}", "--output-format", "stream-json", "--verbose"],
    "codex": ["exec", "{prompt}"],
}


class AgentHarness:
    """Wraps an AgentAdapter with mode-specific arg overrides for agent mode.

    In review mode (default): restricts tools to thirdeye CLI + read-only ops.
    In fix mode (--fix flag): unrestricted tool access so the agent can edit files.
    In streaming mode (--stream flag): uses stream-json output format so intermediate
    tool calls and results are emitted in real time rather than collected until the end.
    For adapters not in the override tables (e.g. custom ConfigAdapter entries),
    the adapter's own config.args are used as-is.
    """

    def __init__(
        self,
        adapter: AgentAdapter,
        mode: Literal["review", "fix"],
        *,
        streaming: bool = False,
    ) -> None:
        if mode not in ("review", "fix"):
            raise ValueError(f"mode must be 'review' or 'fix', got {mode!r}")
        self._adapter = adapter
        self._mode = mode
        self._streaming = streaming
        if streaming:
            table = _STREAM_REVIEW_ARGS if mode == "review" else _STREAM_FIX_ARGS
        else:
            table = _REVIEW_ARGS if mode == "review" else _FIX_ARGS
        self._args = table.get(adapter.name, list(adapter.config.args))

    @property
    def command(self) -> str:
        return self._adapter.config.command

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def streaming(self) -> bool:
        return self._streaming

    @property
    def adapter_name(self) -> str:
        return self._adapter.name

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        """Return the full subprocess argv with {prompt} substituted."""
        resolved = [a.replace("{prompt}", prompt) for a in self._args]
        return [self.command, *resolved]
