from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import PurePosixPath, PureWindowsPath

import thirdeye._compat as _compat


def command_basename(command: str) -> str:
    """Basename of a configured hook command, in the active platform's path syntax.

    On Windows a resolved command is a backslash path ending in ``.exe``; on
    POSIX it is a forward-slash path whose extension is meaningful. Only strip
    ``.exe``, and only on Windows -- a blanket :meth:`~pathlib.PurePath.stem`
    would also eat a POSIX user's ``thirdeye-claude-session-start.sh`` wrapper.
    """
    pure = PureWindowsPath if _compat.IS_WINDOWS else PurePosixPath
    name = pure(command).name
    if _compat.IS_WINDOWS and name[-4:].lower() == ".exe":
        name = name[:-4]
    return name


def command_matches(command: object, bin_name: str) -> bool:
    """Whether a configured hook command refers to our ``bin_name`` binary."""
    if not isinstance(command, str) or not command:
        return False
    return command_basename(command) == bin_name


def resolve_command(bin_name: str) -> str:
    """Absolute path to ``bin_name``, or the bare name when that is safer.

    Returns ``shutil.which(bin_name) or bin_name``, except that on Windows a
    resolved path containing a space is dropped in favour of the bare name:
    ``which`` found it, so it is on PATH, and letting a shell re-resolve it
    sidesteps quoting bugs around paths like ``C:\\Users\\First Last\\...``.
    """
    resolved = shutil.which(bin_name) or bin_name
    if _compat.IS_WINDOWS and resolved != bin_name and " " in resolved:
        return bin_name
    return resolved


class Platform(ABC):
    name: str
    display_name: str

    @abstractmethod
    def install(self) -> None: ...

    @abstractmethod
    def uninstall(self) -> None: ...

    def is_installed(self) -> bool:
        """Return whether thirdeye's complete integration is configured."""
        return False
