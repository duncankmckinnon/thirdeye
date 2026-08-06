from __future__ import annotations

import re
import shutil
from pathlib import Path

import click

from thirdeye.platforms.base import Platform
from thirdeye.platforms.codex.constants import (
    CODEX_CONFIG_FILE,
    DISPLAY_NAME,
    NOTIFY_BIN_NAME,
    PLATFORM_NAME,
)

_NOTIFY_LINE_RE = re.compile(r"^notify\s*=\s*(\[.*?\])\s*$", re.MULTILINE | re.DOTALL)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


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

    def __init__(self, config_file: Path | None = None, force: bool = False) -> None:
        self._config_file = config_file or CODEX_CONFIG_FILE
        self._force = force

    def install(self) -> None:
        """Install thirdeye's notify handler.

        Codex's ``notify`` is a single program's argv, not a list of callbacks.
        thirdeye therefore owns the whole value or nothing:

        * absent, or ``[]`` -> ``notify = [our_cmd]``
        * exactly ``[our_cmd]`` -> no-op (idempotent)
        * anything else -> conflict; raise without touching the file, unless
          ``force=True``, in which case take over the slot.
        """
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
                    "Pass force=True (thirdeye add --force) to have thirdeye take over "
                    "the notify slot instead."
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
            return
        self._config_file.write_text(new_text)
