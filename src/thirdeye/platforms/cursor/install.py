from __future__ import annotations

import json
import shutil
from pathlib import Path

from thirdeye.platforms.base import Platform
from thirdeye.platforms.cursor.constants import (
    DISPLAY_NAME,
    HOOK_BIN_NAME,
    HOOK_TIMEOUT_S,
    HOOKS_FILE,
    PLATFORM_NAME,
    TRACED_EVENTS,
)


def _load(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "hooks": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "hooks": {}}
    if not isinstance(data, dict):
        return {"version": 1, "hooks": {}}
    data.setdefault("version", 1)
    if not isinstance(data.get("hooks"), dict):
        data["hooks"] = {}
    return data


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _is_ours(entry: object) -> bool:
    return isinstance(entry, dict) and Path(str(entry.get("command") or "")).name == HOOK_BIN_NAME


class CursorPlatform(Platform):
    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(self, hooks_file: Path | None = None) -> None:
        self._hooks_file = hooks_file or HOOKS_FILE

    def install(self) -> None:
        data = _load(self._hooks_file)
        hooks = data["hooks"]
        command = shutil.which(HOOK_BIN_NAME) or HOOK_BIN_NAME
        for event in TRACED_EVENTS:
            entries = hooks.setdefault(event, [])
            if not isinstance(entries, list):
                entries = hooks[event] = []
            if not any(_is_ours(entry) for entry in entries):
                entries.append({"type": "command", "command": command, "timeout": HOOK_TIMEOUT_S})
        _save(self._hooks_file, data)

    def is_installed(self) -> bool:
        hooks = _load(self._hooks_file).get("hooks")
        if not isinstance(hooks, dict):
            return False
        return all(
            isinstance(hooks.get(event), list)
            and any(_is_ours(entry) for entry in hooks[event])
            for event in TRACED_EVENTS
        )

    def uninstall(self) -> None:
        if not self._hooks_file.exists():
            return
        data = _load(self._hooks_file)
        hooks = data["hooks"]
        for event in list(hooks):
            entries = hooks[event]
            if not isinstance(entries, list):
                continue
            remaining = [entry for entry in entries if not _is_ours(entry)]
            if remaining:
                hooks[event] = remaining
            else:
                del hooks[event]
        _save(self._hooks_file, data)
