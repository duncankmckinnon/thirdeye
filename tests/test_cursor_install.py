from __future__ import annotations

import json
from pathlib import Path

from thirdeye.platforms.cursor.constants import HOOK_BIN_NAME, HOOK_TIMEOUT_S, TRACED_EVENTS
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


def test_install_registers_subagent_stop(tmp_path: Path, monkeypatch):
    path = tmp_path / "hooks.json"
    monkeypatch.setattr("thirdeye.platforms.cursor.install.shutil.which", lambda _name: None)

    CursorPlatform(hooks_file=path).install()

    data = json.loads(path.read_text())
    assert data["hooks"]["subagentStop"] == [
        {
            "type": "command",
            "command": HOOK_BIN_NAME,
            "timeout": HOOK_TIMEOUT_S,
        }
    ]


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
