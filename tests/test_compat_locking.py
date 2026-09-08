"""Behavioral tests for the cross-platform lock shim."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from thirdeye._compat.locking import LockMode, LockTimeout, locked, locked_fd


def _lock_holder(path: Path) -> subprocess.Popen[str]:
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ | {"PYTHONPATH": str(source_root)}
    script = """
from pathlib import Path
import sys
from thirdeye._compat.locking import LockMode, locked

with locked(Path(sys.argv[1]), LockMode.EXCLUSIVE):
    print("locked", flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"
    return process


def _release(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.close()
    process.wait(timeout=2)


def test_exclusive_excludes_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "locks" / "index.lock"
    holder = _lock_holder(path)
    try:
        with pytest.raises(LockTimeout):
            with locked(path, LockMode.EXCLUSIVE, timeout=0.05):
                pass
        _release(holder)
        with locked(path, LockMode.EXCLUSIVE, timeout=0.05):
            pass
    finally:
        if holder.poll() is None:
            holder.terminate()
            holder.wait(timeout=2)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows shared locks are exclusive")
def test_shared_does_not_exclude_shared(tmp_path: Path) -> None:
    path = tmp_path / "index.lock"
    with locked(path, LockMode.SHARED):
        with locked(path, LockMode.SHARED, timeout=0.05):
            pass


def test_bounded_timeout_raises_lock_timeout(tmp_path: Path) -> None:
    holder = _lock_holder(tmp_path / "index.lock")
    started = time.monotonic()
    try:
        with pytest.raises(LockTimeout):
            with locked(tmp_path / "index.lock", LockMode.EXCLUSIVE, timeout=0.05):
                pass
        assert time.monotonic() - started < 0.2
    finally:
        _release(holder)


def test_lock_timeout_is_an_oserror() -> None:
    with pytest.raises(OSError):
        raise LockTimeout()


def test_locked_creates_missing_file_and_parent(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "parents" / "index.lock"
    with locked(path, LockMode.EXCLUSIVE):
        assert path.is_file()


def test_locked_fd_restores_file_position(tmp_path: Path) -> None:
    path = tmp_path / "index.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        os.write(fd, b"abcdef")
        os.lseek(fd, 4, os.SEEK_SET)
        with locked_fd(fd, LockMode.EXCLUSIVE):
            pass
        assert os.lseek(fd, 0, os.SEEK_CUR) == 4
    finally:
        os.close(fd)
