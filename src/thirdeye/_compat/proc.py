"""Cross-platform detached-process spawning and liveness probing.

On POSIX ``spawn_detached`` uses ``start_new_session=True`` and ``pid_alive``
uses ``os.kill(pid, 0)`` -- the idioms the call sites use today. On Windows
``start_new_session`` is silently ignored by CPython, so a detached child needs
``creationflags``; and ``os.kill(pid, 0)`` calls ``TerminateProcess`` and would
kill the process being probed, so ``pid_alive`` probes via ``OpenProcess`` /
``WaitForSingleObject`` instead.
"""

import os
import subprocess
from pathlib import Path
from typing import Any

from thirdeye._compat import IS_WINDOWS

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _PROCESS_SYNCHRONIZE = 0x00100000
    _WAIT_TIMEOUT = 0x00000102

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


def spawn_detached(
    argv: list[str],
    *,
    cwd: Path | str | None = None,
    stdin: Any = subprocess.DEVNULL,
    stdout: Any = subprocess.DEVNULL,
    stderr: Any = subprocess.DEVNULL,
) -> subprocess.Popen:
    """Spawn ``argv`` fully detached from this process and return the Popen."""
    if IS_WINDOWS:
        # ``start_new_session`` is dropped by CPython's Windows _execute_child.
        # DETACHED_PROCESS gives the child no console; CREATE_NEW_PROCESS_GROUP
        # detaches it from this process's Ctrl-C group. CREATE_NO_WINDOW is
        # invalid alongside DETACHED_PROCESS and would be redundant anyway.
        extra: dict[str, Any] = {
            "creationflags": (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        }
    else:
        extra = {"start_new_session": True, "close_fds": True}
    return subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        **extra,
    )


def _pid_alive_windows(pid: int) -> bool:
    handle = _kernel32.OpenProcess(_PROCESS_SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        # WAIT_TIMEOUT means the process object is not signalled: still running.
        # Prefer this over GetExitCodeProcess, whose STILL_ACTIVE (259) is
        # ambiguous with a worker legitimately exiting 259.
        return _kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
    finally:
        _kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """Return whether ``pid`` names a live process, without disturbing it."""
    if IS_WINDOWS:
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
