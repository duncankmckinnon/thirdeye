from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import click

from thirdeye.platforms.base import Platform
from thirdeye.platforms.codex.constants import (
    CODEX_CONFIG_FILE,
    CODEX_HOOKS_FILE,
    DISPLAY_NAME,
    HOOKS_JSON_BIN_NAMES,
    HOOKS_JSON_UNSUPPORTED_EVENTS,
    NOTIFY_BIN_NAME,
    PLATFORM_NAME,
)

_NOTIFY_LINE_RE = re.compile(r"^notify\s*=\s*(\[.*?\])\s*$", re.MULTILINE | re.DOTALL)
_STALE_CLAUDE_PREFIX = "thirdeye-claude-"
_OWN_PREFIX = "thirdeye-codex-"


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _command_name(command: object) -> str:
    return Path(str(command)).name if isinstance(command, str) and command else ""


def _filter_event_commands(hooks: dict[str, Any], event: str, drop_prefix: str) -> bool:
    """Remove hook entries under `event` whose command basename starts with
    `drop_prefix`, pruning now-empty groups and the event key itself.
    Returns whether anything changed.
    """
    groups = hooks.get(event)
    if not isinstance(groups, list):
        return False
    changed = False
    kept_groups: list[Any] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
            kept_groups.append(group)
            continue
        entries = group["hooks"]
        kept_entries = [
            e
            for e in entries
            if not (isinstance(e, dict) and _command_name(e.get("command")).startswith(drop_prefix))
        ]
        if len(kept_entries) != len(entries):
            changed = True
        if kept_entries:
            kept_groups.append({**group, "hooks": kept_entries})
    if changed:
        if kept_groups:
            hooks[event] = kept_groups
        else:
            hooks.pop(event, None)
    return changed


def _ensure_event_command(hooks: dict[str, Any], event: str, cmd: str) -> bool:
    """Add `cmd` as a hook for `event` unless already present. Returns
    whether anything changed.
    """
    groups = hooks.get(event)
    if not isinstance(groups, list):
        groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        for entry in group.get("hooks") or []:
            if isinstance(entry, dict) and entry.get("command") == cmd:
                return False
    groups.append({"hooks": [{"type": "command", "command": cmd}]})
    hooks[event] = groups
    return True


def _parse_notify_array(value: str) -> list[str]:
    """Parse a TOML inline array like ['a', "b"] -> ['a', 'b']."""
    items: list[str] = []
    for m in re.finditer(r'"([^"]*)"|\'([^\']*)\'', value):
        items.append(m.group(1) if m.group(1) is not None else m.group(2))
    return items


def _format_notify_array(items: list[str]) -> str:
    quoted = ", ".join("'" + item.replace("'", "\\'") + "'" for item in items)
    return f"notify = [{quoted}]"


class CodexPlatform(Platform):
    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(
        self,
        config_file: Path | None = None,
        force: bool = False,
        hooks_file: Path | None = None,
    ) -> None:
        self._config_file = config_file or CODEX_CONFIG_FILE
        self._hooks_file = hooks_file or CODEX_HOOKS_FILE
        self._force = force

    def notify_conflict(self) -> str | None:
        """Return the program that currently owns Codex's notify slot, if foreign."""
        cmd = shutil.which(NOTIFY_BIN_NAME) or NOTIFY_BIN_NAME
        match = _NOTIFY_LINE_RE.search(_read_text(self._config_file))
        if not match:
            return None
        existing = _parse_notify_array(match.group(1))
        if not existing or existing == [cmd]:
            return None
        return existing[0]

    def is_installed(self) -> bool:
        """Return whether both Codex integration mechanisms are configured."""
        text = _read_text(self._config_file)
        match = _NOTIFY_LINE_RE.search(text)
        if not match:
            return False
        notify = _parse_notify_array(match.group(1))
        if not notify or Path(notify[0]).name != NOTIFY_BIN_NAME:
            return False

        hooks = _read_json(self._hooks_file).get("hooks")
        if not isinstance(hooks, dict):
            return False
        for event, bin_name in HOOKS_JSON_BIN_NAMES.items():
            groups = hooks.get(event)
            if not isinstance(groups, list):
                return False
            found = False
            for group in groups:
                if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                    continue
                if any(
                    isinstance(entry, dict) and _command_name(entry.get("command")) == bin_name
                    for entry in group["hooks"]
                ):
                    found = True
                    break
            if not found:
                return False
        return True

    def install(self) -> None:
        """Install thirdeye's notify handler and hooks.json bindings.

        Codex's ``notify`` is a single program's argv, not a list of callbacks.
        thirdeye therefore owns the whole value or nothing:

        * absent, or ``[]`` -> ``notify = [our_cmd]``
        * exactly ``[our_cmd]`` -> no-op (idempotent)
        * anything else -> conflict; raise without touching the file, unless
          ``force=True``, in which case take over the slot.

        hooks.json is a separate, newer, per-event mechanism (see
        HOOKS_JSON_BIN_NAMES) and unlike notify is additive: thirdeye's
        command is appended to whatever else is already registered for an
        event, never taking exclusive ownership of it, so it always runs
        unconditionally regardless of ``force``. It also strips any
        thirdeye-claude-* entry it finds under any event, including the
        three thirdeye deliberately never wires (PreToolUse/PostToolUse/Stop
        — see hooks_json.py's docstring for why) — that combination is
        always a misconfiguration (Codex's hooks pointed at Claude's own
        handlers, mislabeling every captured session as platform=claude),
        never a legitimate integration.
        """
        self._install_hooks_json()
        cmd = shutil.which(NOTIFY_BIN_NAME) or NOTIFY_BIN_NAME
        text = _read_text(self._config_file)
        match = _NOTIFY_LINE_RE.search(text)
        if match:
            existing = _parse_notify_array(match.group(1))
            if existing == [cmd]:
                # Already ours; nothing to do.
                return
            if existing and not self._force:
                # A foreign notify program owns the slot. Validate before
                # writing anything so the file is left byte-identical.
                incumbent = existing[0]
                raise click.ClickException(
                    f"Codex 'notify' is already set to {incumbent!r} in "
                    f"{self._config_file}. Codex's notify is a single program's argv, "
                    "not a list of callbacks, so thirdeye-codex-notify cannot be added "
                    "without breaking it. Either remove the existing notify value, or "
                    "install a dispatcher program that invokes both it and "
                    "thirdeye-codex-notify (e.g. via a --previous-notify argument). "
                    "Pass --force (`thirdeye add --codex --force`) to have thirdeye "
                    "take over the notify slot instead."
                )
            # Empty array, or force=True: take over the slot.
            new_line = _format_notify_array([cmd])
            new_text = text[: match.start()] + new_line + text[match.end() :]
        else:
            notify_line = _format_notify_array([cmd]) + "\n"
            # Insert before first section header to keep it top-level
            section_match = re.search(r"^\[", text, re.MULTILINE)
            if text and section_match:
                insert_pos = section_match.start()
                new_text = text[:insert_pos] + notify_line + "\n" + text[insert_pos:]
            else:
                prefix = text + ("\n" if text and not text.endswith("\n") else "")
                new_text = prefix + notify_line
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        self._config_file.write_text(new_text)

    def uninstall(self) -> None:
        """Remove the notify handler, but only when thirdeye owns slot 0.

        Under argv semantics only the first element is the program. If a
        foreign dispatcher owns slot 0 we must not touch it, even if our
        command name appears later as a mere argument.
        """
        text = _read_text(self._config_file)
        if not text:
            return
        match = _NOTIFY_LINE_RE.search(text)
        if not match:
            return
        existing = _parse_notify_array(match.group(1))
        if not existing or Path(existing[0]).name != NOTIFY_BIN_NAME:
            # We don't own the program slot; leave everything as-is.
            return
        # Remove the entire notify line (and one trailing newline).
        start = match.start()
        end = match.end()
        if end < len(text) and text[end] == "\n":
            end += 1
        new_text = text[:start] + text[end:]
        if not new_text.strip():
            if self._config_file.exists():
                self._config_file.unlink()
        else:
            self._config_file.write_text(new_text)
        self._uninstall_hooks_json()

    def _install_hooks_json(self) -> None:
        data = _read_json(self._hooks_file)
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}
        changed = False

        # Every event Codex's hooks.json recognizes, whether or not thirdeye
        # wires it: a stale thirdeye-claude-* entry under any of them,
        # including the three thirdeye deliberately skips, is always wrong.
        for event in set(hooks) | set(HOOKS_JSON_BIN_NAMES) | set(HOOKS_JSON_UNSUPPORTED_EVENTS):
            if _filter_event_commands(hooks, event, _STALE_CLAUDE_PREFIX):
                changed = True

        for event, bin_name in HOOKS_JSON_BIN_NAMES.items():
            cmd = shutil.which(bin_name) or bin_name
            if _ensure_event_command(hooks, event, cmd):
                changed = True

        if not changed:
            return
        data["hooks"] = hooks
        self._hooks_file.parent.mkdir(parents=True, exist_ok=True)
        self._hooks_file.write_text(json.dumps(data, indent=2) + "\n")

    def _uninstall_hooks_json(self) -> None:
        data = _read_json(self._hooks_file)
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            return
        changed = False
        for event in list(hooks):
            if _filter_event_commands(hooks, event, _OWN_PREFIX):
                changed = True
        if not changed:
            return
        data["hooks"] = hooks
        if hooks or set(data) != {"hooks"}:
            self._hooks_file.write_text(json.dumps(data, indent=2) + "\n")
        elif self._hooks_file.exists():
            self._hooks_file.unlink()
