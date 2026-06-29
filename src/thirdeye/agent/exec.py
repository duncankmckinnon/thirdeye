from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import click

from thirdeye.agent.harness import AgentHarness


def _open_pty() -> tuple[int, int]:
    import pty  # deferred: not available on Windows

    return pty.openpty()


def _read_fd(fd: int, n: int) -> bytes:
    return os.read(fd, n)


def _list_sessions(thirdeye_home: Path, platform: str) -> set[str]:
    """Return the set of session IDs currently recorded for platform. Best-effort."""
    try:
        from thirdeye.config import Config
        from thirdeye.store import Store

        config = Config(root=thirdeye_home)
        return {s.session_id for s in Store(config).list_sessions(platform=platform)}
    except Exception:
        return set()


def _tag_sessions(thirdeye_home: Path, platform: str, session_ids: set[str]) -> None:
    """Write the 'thirdeye-agent' tag onto seq 0 of each new session. Best-effort."""
    if not session_ids:
        return
    try:
        from thirdeye.paths import session_dir
        from thirdeye.tags import TagStore

        for sid in session_ids:
            sd = session_dir(thirdeye_home, platform, sid)
            if sd.exists():
                TagStore(sd).add(0, "thirdeye-agent", source="auto")
    except Exception:
        pass  # tagging is observability; never block the command on it


def run_agent_streaming(
    harness: AgentHarness,
    prompt: str,
    cwd: Path,
    *,
    output: Callable[[str], None] | None = None,
    thirdeye_home: Path | None = None,
    use_pty: bool = False,
) -> tuple[int, int]:
    """Run the agent subprocess, streaming stdout to the terminal.

    Args:
        harness:        AgentHarness wrapping the desired adapter + mode.
        prompt:         The composed prompt string to pass to the agent.
        cwd:            Working directory for the subprocess.
        output:         Callable that receives each stdout chunk/line. Defaults to
                        click.echo(line, nl=False) so callers can inject a
                        mock for testing.
        thirdeye_home:  If provided, any new sessions the agent spawns on its
                        platform are tagged 'thirdeye-agent' after the run.
        use_pty:        If True, allocate a pseudo-terminal for stdout so the
                        subprocess sees a TTY and flushes output in real time.
                        If False (default), use a pipe.

    Returns:
        (returncode, duration_ms)

    Raises:
        FileNotFoundError: if the agent binary is not on PATH.
    """
    if shutil.which(harness.command) is None:
        raise FileNotFoundError(f"`{harness.command}` not found on PATH — install it first")

    if output is None:

        def output(line: str) -> None:
            click.echo(line, nl=False)

    _use_pty = use_pty

    # Snapshot existing sessions before the run so we can detect new ones.
    pre_ids: set[str] = set()
    if thirdeye_home is not None:
        pre_ids = _list_sessions(thirdeye_home, harness.adapter_name)

    cmd = harness.build_command(prompt, cwd)
    t0 = time.monotonic()

    stderr_lines: list[str] = []

    if _use_pty:
        master_fd, slave_fd = _open_pty()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.close(slave_fd)
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    if _use_pty:
        buf = b""
        try:
            while True:
                try:
                    chunk = _read_fd(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    output(line_bytes.decode("utf-8", errors="replace") + "\n")
        finally:
            os.close(master_fd)
        if buf:
            output(buf.decode("utf-8", errors="replace"))
    else:
        assert proc.stdout is not None
        for line in proc.stdout:
            output(line)

    proc.wait()
    stderr_thread.join(timeout=5)
    duration_ms = int((time.monotonic() - t0) * 1000)

    if proc.returncode != 0:
        for line in stderr_lines[-5:]:
            click.echo(line, nl=False, err=True)

    # Tag any new sessions the agent opened during the run.
    if thirdeye_home is not None:
        post_ids = _list_sessions(thirdeye_home, harness.adapter_name)
        _tag_sessions(thirdeye_home, harness.adapter_name, post_ids - pre_ids)

    return proc.returncode, duration_ms
