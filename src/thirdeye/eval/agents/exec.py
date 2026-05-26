from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from thirdeye.eval.agents.base import AgentAdapter


@dataclass
class AgentInvocation:
    """Captures the raw subprocess outcome for one agent run."""

    stdout: str
    stderr: str
    returncode: int
    duration_ms: int


def invoke_agent(adapter: AgentAdapter, prompt: str, cwd: Path) -> AgentInvocation:
    """Run the agent subprocess and capture stdout/stderr/returncode."""
    if shutil.which(adapter.config.command) is None:
        raise FileNotFoundError(f"`{adapter.config.command}` not found on PATH — install it first")
    cmd = adapter.build_command(prompt, cwd)
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = int((time.monotonic() - t0) * 1000)
    return AgentInvocation(
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
        duration_ms=elapsed,
    )
