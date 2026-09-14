"""Filesystem operations that tolerate Windows sharing semantics.

Windows refuses to rename over or delete a file another process holds open, and
symmetrically refuses to *open* a file another process is atomically replacing
(ERROR_SHARING_VIOLATION, surfaced as ``PermissionError``). POSIX allows all
three. On POSIX these helpers are a direct pass-through with no retry, no sleep,
and no added latency -- a ``PermissionError`` there propagates on the first try,
unchanged. On Windows they retry ``PermissionError`` (and only
``PermissionError``) with bounded exponential backoff.

The read side matters as much as the write side: a file published by the
tmp-file-plus-``replace`` pattern is readable before and after the swap but can
be briefly unopenable during it, so a concurrent reader fails on a file that is
present and intact a millisecond later.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from thirdeye._compat import IS_WINDOWS

_T = TypeVar("_T")

_RETRY_INITIAL_DELAY_S = 0.005
_RETRY_MAX_DELAY_S = 0.05
_RETRY_BUDGET_S = 1.0


def _with_windows_retry(operation: Callable[[], _T]) -> _T:
    deadline = time.monotonic() + _RETRY_BUDGET_S
    delay = _RETRY_INITIAL_DELAY_S
    while True:
        try:
            return operation()
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
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


def sync_directory(path: Path | str) -> None:
    """Best-effort ``fsync`` of a directory after ``replace`` or ``unlink``.

    Publishing a file with tmp-plus-replace (or removing a journal) is not
    durable until the directory entry itself is synced. Some platforms,
    notably Windows, reject directory ``fsync``; those errors are ignored so
    callers stay portable.
    """
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)


def read_text(path: Path | str, *, encoding: str = "utf-8") -> str:
    """``Path.read_text`` with bounded retry on Windows ``PermissionError``.

    For files published by the tmp-file-plus-:func:`replace` pattern, which a
    concurrent writer can make momentarily unopenable on Windows. Only
    ``PermissionError`` is retried: a genuinely missing file still raises
    ``FileNotFoundError`` on the first try, on every platform.
    """
    path = Path(path)
    if IS_WINDOWS:
        return _with_windows_retry(lambda: path.read_text(encoding=encoding))
    return path.read_text(encoding=encoding)
