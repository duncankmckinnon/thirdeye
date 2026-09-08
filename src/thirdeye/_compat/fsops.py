"""Filesystem operations that tolerate Windows sharing semantics.

Windows refuses to rename over or delete a file another process holds open;
POSIX allows both. On POSIX these helpers are a direct pass-through with no
retry, no sleep, and no added latency -- a ``PermissionError`` there propagates
on the first try, unchanged. On Windows they retry ``PermissionError`` (and only
``PermissionError``) with bounded exponential backoff.
"""

import os
import time
from collections.abc import Callable
from pathlib import Path

from thirdeye._compat import IS_WINDOWS

_RETRY_INITIAL_DELAY_S = 0.005
_RETRY_MAX_DELAY_S = 0.05
_RETRY_BUDGET_S = 1.0


def _with_windows_retry(operation: Callable[[], None]) -> None:
    deadline = time.monotonic() + _RETRY_BUDGET_S
    delay = _RETRY_INITIAL_DELAY_S
    while True:
        try:
            operation()
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _RETRY_MAX_DELAY_S)


def replace(src: Path | str, dst: Path | str) -> None:
    """``os.replace`` with bounded retry on Windows ``PermissionError``."""
    if IS_WINDOWS:
        _with_windows_retry(lambda: os.replace(src, dst))
    else:
        os.replace(src, dst)


def unlink(path: Path, *, missing_ok: bool = False) -> None:
    """``Path.unlink`` with bounded retry on Windows ``PermissionError``."""
    path = Path(path)
    if IS_WINDOWS:
        _with_windows_retry(lambda: path.unlink(missing_ok=missing_ok))
    else:
        path.unlink(missing_ok=missing_ok)
