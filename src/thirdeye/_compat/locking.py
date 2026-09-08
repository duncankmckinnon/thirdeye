"""Cross-platform advisory file locking.

``locked`` and ``locked_fd`` give the rest of the codebase a single lock
primitive that behaves the same way it does today on POSIX and also works on
Windows.

POSIX uses ``fcntl.flock``: a ``timeout=None`` acquisition is a plain blocking
``flock`` call, and a bounded acquisition polls ``LOCK_NB`` inside an exponential
backoff loop. Windows uses ``msvcrt.locking`` on a one-byte region at offset 0,
always via ``LK_NBLCK`` inside the same backoff loop (looping forever when
``timeout`` is ``None``).

**This shim is deliberately not reentrant.** Acquiring the same path twice from a
single process deadlocks on POSIX. Callers that need reentrancy (for example
``thirdeye.platforms.claude.hooks``) own that bookkeeping themselves; do not add
depth counting or an "already held" check here.
"""

import contextlib
import os
import time
from collections.abc import Callable, Iterator
from enum import Enum, auto
from pathlib import Path

from thirdeye._compat import IS_WINDOWS

if IS_WINDOWS:
    import msvcrt
else:
    import fcntl

_RETRY_INITIAL_DELAY_S = 0.005
_RETRY_MAX_DELAY_S = 0.025


class LockMode(Enum):
    SHARED = auto()
    EXCLUSIVE = auto()


class LockTimeout(TimeoutError):
    """Raised when a bounded lock acquisition exceeds its budget.

    Subclasses :class:`TimeoutError` (and therefore :class:`OSError`) because
    every hook call site already wraps its lock use in ``except OSError`` /
    ``except Exception``.
    """


def _acquire_with_backoff(
    try_once: Callable[[], None],
    *,
    contention_errors: tuple[type[BaseException], ...],
    timeout: float | None,
) -> None:
    """Call ``try_once`` until it stops raising a contention error.

    ``try_once`` performs a single non-blocking acquisition: it returns on
    success and raises one of ``contention_errors`` when the lock is held
    elsewhere. ``timeout=None`` retries forever; a float bounds the wait with
    exponential backoff (initial 0.005s, doubling, capped at 0.025s per sleep,
    never sleeping past the deadline) and raises :class:`LockTimeout` on expiry.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    delay = _RETRY_INITIAL_DELAY_S
    while True:
        try:
            try_once()
            return
        except contention_errors:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LockTimeout(f"timed out after {timeout}s waiting for lock") from None
                sleep_for = min(delay, remaining)
            else:
                sleep_for = delay
            time.sleep(max(0.0, sleep_for))
            delay = min(delay * 2, _RETRY_MAX_DELAY_S)


def _locked_fd_posix(fd: int, mode: LockMode, timeout: float | None) -> Iterator[None]:
    operation = fcntl.LOCK_SH if mode is LockMode.SHARED else fcntl.LOCK_EX
    if timeout is None:
        fcntl.flock(fd, operation)
    else:
        _acquire_with_backoff(
            lambda: fcntl.flock(fd, operation | fcntl.LOCK_NB),
            contention_errors=(BlockingIOError,),
            timeout=timeout,
        )
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _locked_fd_windows(fd: int, mode: LockMode, timeout: float | None) -> Iterator[None]:
    # ``msvcrt.locking`` operates at the current file position and must be
    # unlocked at the same offset, so seek to 0 around each call and restore the
    # caller's position afterwards. Both LockMode values lock exclusively here.
    original_position = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        _acquire_with_backoff(
            lambda: msvcrt.locking(fd, msvcrt.LK_NBLCK, 1),
            contention_errors=(OSError,),
            timeout=timeout,
        )
    finally:
        os.lseek(fd, original_position, os.SEEK_SET)
    try:
        yield
    finally:
        release_position = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.lseek(fd, release_position, os.SEEK_SET)


@contextlib.contextmanager
def locked_fd(fd: int, mode: LockMode, *, timeout: float | None = None) -> Iterator[None]:
    """Lock an already-open descriptor the caller owns, yield, and release.

    ``timeout=None`` blocks indefinitely; a float bounds the wait and raises
    :class:`LockTimeout` on expiry.
    """
    if IS_WINDOWS:
        yield from _locked_fd_windows(fd, mode, timeout)
    else:
        yield from _locked_fd_posix(fd, mode, timeout)


@contextlib.contextmanager
def locked(path: Path, mode: LockMode, *, timeout: float | None = None) -> Iterator[None]:
    """Lock ``path`` (creating it if needed), yield, and release.

    Creates the parent directory, opens the lock file with ``path.open("a+")``
    -- this exact mode, so permissions match what existing POSIX users already
    have on disk -- and closes it on exit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        with locked_fd(handle.fileno(), mode, timeout=timeout):
            yield
