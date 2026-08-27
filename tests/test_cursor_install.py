from __future__ import annotations

import json
from pathlib import Path

from thirdeye.platforms.cursor.constants import HOOK_BIN_NAME, TRACED_EVENTS
from thirdeye.platforms.cursor.install import CursorPlatform


def test_install_registers_every_cursor_event_and_is_idempotent(tmp_path: Path):
    path = tmp_path / "hooks.json"
    platform = CursorPlatform(hooks_file=path)
    platform.install()
    platform.install()
    data = json.loads(path.read_text())
    assert set(data["hooks"]) == set(TRACED_EVENTS)
    for entries in data["hooks"].values():
        assert len(entries) == 1
        assert Path(entries[0]["command"]).name == HOOK_BIN_NAME


def test_install_and_uninstall_preserve_foreign_hooks(tmp_path: Path):
    path = tmp_path / "hooks.json"
    foreign = {"type": "command", "command": "/opt/foreign-hook", "timeout": 10}
    path.write_text(json.dumps({"version": 1, "theme": "dark", "hooks": {"stop": [foreign]}}))
    platform = CursorPlatform(hooks_file=path)
    platform.install()
    platform.uninstall()
    data = json.loads(path.read_text())
    assert data["theme"] == "dark"
    assert data["hooks"] == {"stop": [foreign]}
