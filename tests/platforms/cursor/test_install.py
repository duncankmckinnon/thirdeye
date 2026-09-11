from __future__ import annotations

import json
from pathlib import Path

from thirdeye.platforms.base import command_basename
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
        # Not Path(...).name: on Windows `which` resolves to a ".EXE" that
        # command_basename (what install matching itself uses) strips.
        assert command_basename(entries[0]["command"]) == HOOK_BIN_NAME


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


class TestInstallerIdentity:
    def test_install_then_is_installed(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("thirdeye.platforms.cursor.install.shutil.which", lambda _: None)
        platform = CursorPlatform(hooks_file=tmp_path / "hooks.json")

        platform.install()

        assert platform.is_installed()

    def test_install_twice_appends_no_duplicate(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("thirdeye._compat.IS_WINDOWS", True)
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda name: rf"C:\Users\First Last\Scripts\{name}.exe",
        )
        path = tmp_path / "hooks.json"
        platform = CursorPlatform(hooks_file=path)
        platform.install()
        first = json.loads(path.read_text())
        platform.install()
        second = json.loads(path.read_text())

        for event in TRACED_EVENTS:
            assert len(second["hooks"][event]) == len(first["hooks"][event])

    def test_absolute_then_bare_appends_no_duplicate(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "hooks.json"
        platform = CursorPlatform(hooks_file=path)
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which", lambda name: f"/opt/bin/{name}"
        )
        platform.install()
        monkeypatch.setattr("thirdeye.platforms.cursor.install.shutil.which", lambda _: None)
        platform.install()
        data = json.loads(path.read_text())

        assert all(len(data["hooks"][event]) == 1 for event in TRACED_EVENTS)

    def test_uninstall_removes_ours_keeps_foreign(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("thirdeye._compat.IS_WINDOWS", True)
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda name: rf"C:\Tools\Scripts\{name}.exe",
        )
        path = tmp_path / "hooks.json"
        platform = CursorPlatform(hooks_file=path)
        platform.install()
        data = json.loads(path.read_text())
        foreign = {"type": "command", "command": "/opt/foreign-hook", "timeout": 10}
        data["hooks"]["stop"].append(foreign)
        path.write_text(json.dumps(data))

        platform.uninstall()

        assert json.loads(path.read_text())["hooks"] == {"stop": [foreign]}

    def test_windows_exe_resolution_round_trips(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("thirdeye._compat.IS_WINDOWS", True)
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda name: rf"C:\Tools\Scripts\{name}.exe",
        )
        path = tmp_path / "hooks.json"
        platform = CursorPlatform(hooks_file=path)

        platform.install()
        assert platform.is_installed()
        platform.uninstall()

        assert json.loads(path.read_text())["hooks"] == {}
