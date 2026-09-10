"""Installation of thirdeye's passive GitHub Copilot CLI hooks.

Copilot reads version-1 hook documents from ``$COPILOT_HOME/hooks``.  This
module owns only ``thirdeye.json`` and only command entries whose executable
is thirdeye's dispatcher; it never changes Copilot-wide settings or hooks
owned by another integration.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path, PureWindowsPath
from typing import Any

import click

import thirdeye._compat as _compat
from thirdeye.platforms.base import Platform, command_basename, resolve_command

from .constants import (
    CLI_HOOK_EVENTS,
    DISPLAY_NAME,
    HOOK_BIN_NAME,
    HOOK_CONFIG_VERSION,
    HOOK_TIMEOUT_S,
    HOOKS_DIRECTORY_NAME,
    OWNED_HOOK_FILENAME,
    PLATFORM_NAME,
)
from .identity import resolve_sources


def _quote_bash(value: str) -> str:
    """Quote one literal argument for Copilot's POSIX shell hook runner."""

    return shlex.quote(value)


def _quote_powershell(value: str) -> str:
    """Quote one literal argument for PowerShell's single-quote syntax."""

    return "'" + value.replace("'", "''") + "'"


def _bash_command(entrypoint: str, event: str) -> str:
    return f"{_quote_bash(entrypoint)} {_quote_bash(event)}"


def _powershell_command(entrypoint: str, event: str) -> str:
    # ``&`` is required for a quoted executable path to be invoked rather
    # than treated as a string expression.
    return f"& {_quote_powershell(entrypoint)} {_quote_powershell(event)}"


def _command_executable(command: object, shell: str) -> str | None:
    """Return the executable in one of our generated commands, if parseable."""

    if not isinstance(command, str) or not command.strip():
        return None
    if shell == "bash":
        try:
            parts = shlex.split(command, posix=True)
        except ValueError:
            return None
        return parts[0] if parts else None

    # We generate ``& 'path' 'event'``.  Accept both quote styles for a
    # previous installation, but intentionally do not try to interpret an
    # arbitrary PowerShell program: configuration ownership must be narrow.
    match = re.match(r"^\s*&\s+(?:'((?:[^']|'')*)'|\"([^\"]*)\")", command)
    if not match:
        return None
    quoted = match.group(1) if match.group(1) is not None else match.group(2)
    return quoted.replace("''", "'")


def _is_our_command(command: object, shell: str) -> bool:
    executable = _command_executable(command, shell)
    if executable is None:
        return False
    # command_basename follows the host path rules.  Also recognize a Windows
    # executable when inspecting a fixture on a non-Windows host.
    if command_basename(executable) == HOOK_BIN_NAME:
        return True
    name = PureWindowsPath(executable).name
    return name.lower() in {HOOK_BIN_NAME.lower(), f"{HOOK_BIN_NAME}.exe".lower()}


def _entry_is_ours(entry: object) -> bool:
    return isinstance(entry, dict) and any(
        _is_our_command(entry.get(shell), shell) for shell in ("bash", "powershell")
    )


def _load_document(path: Path) -> dict[str, Any]:
    """Load a valid version-1 Copilot hook document without normalizing it.

    Invalid documents are configuration errors, not empty documents: replacing
    them would destroy an operator's hooks and conceal a broken setup.
    """

    if not path.exists():
        return {"version": HOOK_CONFIG_VERSION, "hooks": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(
            f"Cannot update Copilot hooks at {path}: the file is not valid JSON. "
            "Fix or move it, then run the command again; it was left unchanged."
        ) from exc
    if not isinstance(data, dict):
        raise click.ClickException(
            f"Cannot update Copilot hooks at {path}: the document must be a JSON object. "
            "It was left unchanged."
        )
    if data.get("version") != HOOK_CONFIG_VERSION:
        raise click.ClickException(
            f"Cannot update Copilot hooks at {path}: expected version "
            f"{HOOK_CONFIG_VERSION}, found {data.get('version')!r}. It was left unchanged."
        )
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        raise click.ClickException(
            f"Cannot update Copilot hooks at {path}: 'hooks' must be an object. "
            "It was left unchanged."
        )
    for event in CLI_HOOK_EVENTS:
        if event in hooks and not isinstance(hooks[event], list):
            raise click.ClickException(
                f"Cannot update Copilot hooks at {path}: hooks.{event} must be a list. "
                "It was left unchanged."
            )
    return data


def _save_document(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


class CopilotPlatform(Platform):
    """Passive Copilot CLI hook installation scoped to one source home."""

    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(
        self,
        source_home: Path | None = None,
        hooks_file: Path | None = None,
        entrypoint: str | Path | None = None,
        *,
        windows: bool | None = None,
    ) -> None:
        self._source_home = source_home
        self._hooks_file = hooks_file
        self._entrypoint = str(entrypoint) if entrypoint is not None else None
        self._windows = _compat.IS_WINDOWS if windows is None else windows

    @property
    def hooks_file(self) -> Path:
        if self._hooks_file is not None:
            return self._hooks_file
        paths = resolve_sources(self._source_home)
        return Path(paths["home"]) / HOOKS_DIRECTORY_NAME / OWNED_HOOK_FILENAME

    @property
    def entrypoint(self) -> str:
        return self._entrypoint or resolve_command(HOOK_BIN_NAME)

    def _command(self, event: str) -> tuple[str, str]:
        if self._windows:
            return "powershell", _powershell_command(self.entrypoint, event)
        return "bash", _bash_command(self.entrypoint, event)

    def install(self) -> None:
        path = self.hooks_file
        data = _load_document(path)
        hooks = data["hooks"]
        changed = False
        for event in CLI_HOOK_EVENTS:
            entries = hooks.get(event, [])
            # The event type has already been validated by _load_document.
            if not isinstance(entries, list):  # Defensive for typed JSON input.
                raise AssertionError(f"validated hook list changed shape for {event}")
            retained = [entry for entry in entries if not _entry_is_ours(entry)]
            shell, command = self._command(event)
            desired = {
                "type": "command",
                shell: command,
                "timeoutSec": HOOK_TIMEOUT_S,
            }
            if retained != entries or desired not in retained:
                hooks[event] = [*retained, desired]
                changed = True
        if changed or not path.exists():
            _save_document(path, data)

    def is_installed(self) -> bool:
        path = self.hooks_file
        try:
            data = _load_document(path)
        except click.ClickException:
            return False
        hooks = data["hooks"]
        for event in CLI_HOOK_EVENTS:
            entries = hooks.get(event)
            if not isinstance(entries, list):
                return False
            shell, command = self._command(event)
            if not any(
                isinstance(entry, dict)
                and entry.get("type") == "command"
                and entry.get(shell) == command
                and entry.get("timeoutSec") == HOOK_TIMEOUT_S
                for entry in entries
            ):
                return False
        return True

    def uninstall(self) -> None:
        path = self.hooks_file
        if not path.exists():
            return
        data = _load_document(path)
        hooks = data["hooks"]
        changed = False
        for event in list(hooks):
            entries = hooks[event]
            if not isinstance(entries, list):
                # Events outside our supported set are unrelated; leave their
                # invalid shape intact rather than risking destructive repair.
                continue
            retained = [entry for entry in entries if not _entry_is_ours(entry)]
            if retained == entries:
                continue
            changed = True
            if retained:
                hooks[event] = retained
            else:
                del hooks[event]
        # Version and hooks are the only fields we create.  An otherwise empty
        # owned document can disappear; extra fields always remain untouched.
        if not hooks and set(data) == {"version", "hooks"}:
            path.unlink()
        elif changed:
            _save_document(path, data)
