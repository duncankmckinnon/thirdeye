"""Tests for cross-platform detached-process helpers."""

import subprocess
import sys
import time
from pathlib import Path

from thirdeye._compat.proc import pid_alive, spawn_detached


def test_spawn_detached_child_completes(tmp_path: Path) -> None:
    marker = tmp_path / "completed"
    process = spawn_detached(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
            str(marker),
        ]
    )

    assert process.wait(timeout=2) == 0
    assert marker.is_file()


def test_pid_alive_true_then_false() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    try:
        assert pid_alive(process.pid) is True
        assert process.wait(timeout=2) == 0
        assert pid_alive(process.pid) is False
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def test_pid_alive_does_not_terminate_target() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        assert pid_alive(process.pid) is True
        assert pid_alive(process.pid) is True
        time.sleep(0.05)
        assert process.poll() is None

        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
