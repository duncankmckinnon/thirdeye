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


def test_install_registers_subagent_lifecycle_and_pre_tool(tmp_path: Path, monkeypatch):
    path = tmp_path / "hooks.json"
    monkeypatch.setattr("thirdeye.platforms.cursor.install.shutil.which", lambda _name: None)

    CursorPlatform(hooks_file=path).install()

    data = json.loads(path.read_text())
    expected = {
        "type": "command",
        "command": HOOK_BIN_NAME,
        "timeout": HOOK_TIMEOUT_S,
    }
    for event_name in ("preToolUse", "subagentStart", "subagentStop"):
        assert data["hooks"][event_name] == [expected]


def test_install_upgrades_old_thirdeye_cursor_hooks(tmp_path: Path, monkeypatch):
    path = tmp_path / "hooks.json"
    monkeypatch.setattr("thirdeye.platforms.cursor.install.shutil.which", lambda _name: None)
    old_events = [event for event in TRACED_EVENTS if event not in {"preToolUse", "subagentStart"}]
    ours = {"type": "command", "command": HOOK_BIN_NAME, "timeout": HOOK_TIMEOUT_S}
    foreign = {
        "type": "command",
        "command": "/opt/user-hooks/cursor-hook",
        "timeout": 17,
        "metadata": {"owner": "user"},
    }
    hooks = {event: [ours.copy()] for event in old_events}
    hooks["stop"].append(foreign)
    path.write_text(json.dumps({"version": 1, "hooks": hooks}))

    CursorPlatform(hooks_file=path).install()

    data = json.loads(path.read_text())
    for event_name in TRACED_EVENTS:
        assert (
            sum(Path(entry["command"]).name == HOOK_BIN_NAME for entry in data["hooks"][event_name])
            == 1
        )
    assert data["hooks"]["preToolUse"] == [ours]
    assert data["hooks"]["subagentStart"] == [ours]
    assert data["hooks"]["stop"][1] == foreign


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
