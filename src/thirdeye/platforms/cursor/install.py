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
        if not isinstance(data, dict):
            return {"version": 1, "hooks": {}}
        data.setdefault("version", 1)
        data.setdefault("hooks", {})
        return data
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "hooks": {}}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _resolve_command() -> str:
    return shutil.which(HOOK_BIN_NAME) or HOOK_BIN_NAME


def _is_our_entry(entry: dict) -> bool:
    cmd = entry.get("command", "")
    return Path(cmd).name == HOOK_BIN_NAME


class CursorPlatform(Platform):
    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(self, hooks_file: Path | None = None) -> None:
        self._hooks_file = hooks_file or HOOKS_FILE

    def install(self) -> None:
        data = _load(self._hooks_file)
        hooks = data["hooks"]
        cmd = _resolve_command()
        for event in TRACED_EVENTS:
            entries = hooks.setdefault(event, [])
            if not any(_is_our_entry(e) for e in entries):
                entries.append(
                    {"type": "command", "command": cmd, "timeout": HOOK_TIMEOUT_S}
                )
        _save(self._hooks_file, data)

    def uninstall(self) -> None:
        if not self._hooks_file.exists():
            return
        data = _load(self._hooks_file)
        hooks = data.get("hooks", {})
        for event in list(hooks.keys()):
            entries = [e for e in hooks[event] if not _is_our_entry(e)]
            if entries:
                hooks[event] = entries
            else:
                del hooks[event]
        _save(self._hooks_file, data)
