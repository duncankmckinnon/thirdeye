from __future__ import annotations

import json
from pathlib import Path

from thirdeye.platforms.cursor.constants import (
    HOOK_BIN_NAME,
    HOOK_TIMEOUT_S,
    TRACED_EVENTS,
)
from thirdeye.platforms.cursor.install import CursorPlatform


class TestCursorPlatformAttributes:
    def test_name_is_cursor(self):
        p = CursorPlatform(hooks_file=Path("/fake/hooks.json"))
        assert p.name == "cursor"

    def test_display_name(self):
        p = CursorPlatform(hooks_file=Path("/fake/hooks.json"))
        assert p.display_name == "Cursor"

    def test_is_platform_subclass(self):
        from thirdeye.platforms.base import Platform

        assert issubclass(CursorPlatform, Platform)


class TestInstallFreshFile:
    def test_writes_all_traced_events(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        assert set(data["hooks"].keys()) == set(TRACED_EVENTS)

    def test_writes_version_1(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        assert data["version"] == 1

    def test_creates_parent_dir(self, tmp_path: Path):
        hooks_file = tmp_path / "nested" / "deeper" / "hooks.json"
        CursorPlatform(hooks_file=hooks_file).install()
        assert hooks_file.exists()

    def test_each_event_has_one_entry(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        for event, entries in data["hooks"].items():
            assert len(entries) == 1, f"expected 1 entry for {event}"

    def test_command_basename_matches_hook_bin_name(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        for event, entries in data["hooks"].items():
            for entry in entries:
                assert Path(entry["command"]).name == HOOK_BIN_NAME

    def test_each_entry_has_command_type_and_timeout(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        for event, entries in data["hooks"].items():
            for entry in entries:
                assert entry["type"] == "command"
                assert entry["timeout"] == HOOK_TIMEOUT_S
                assert isinstance(entry["command"], str)


class TestInstallIdempotent:
    def test_running_install_twice_does_not_duplicate_entries(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        p = CursorPlatform(hooks_file=hooks_file)
        p.install()
        p.install()
        data = json.loads(hooks_file.read_text())
        for event, entries in data["hooks"].items():
            ours = [e for e in entries if Path(e.get("command", "")).name == HOOK_BIN_NAME]
            assert len(ours) == 1, f"expected 1 thirdeye entry for {event}"

    def test_running_install_after_path_change_dedups_by_basename(
        self, tmp_path: Path, monkeypatch
    ):
        hooks_file = tmp_path / "hooks.json"
        hooks_file.parent.mkdir(parents=True, exist_ok=True)
        hooks_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {
                        "sessionStart": [
                            {
                                "type": "command",
                                "command": f"/old/path/{HOOK_BIN_NAME}",
                                "timeout": HOOK_TIMEOUT_S,
                            }
                        ]
                    },
                }
            )
        )
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda _: f"/new/path/{HOOK_BIN_NAME}",
        )
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        ours = [
            e
            for e in data["hooks"]["sessionStart"]
            if Path(e.get("command", "")).name == HOOK_BIN_NAME
        ]
        assert len(ours) == 1


class TestInstallPreservesExistingHooks:
    def test_preserves_unrelated_hook_under_same_event(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {
                        "sessionStart": [
                            {
                                "type": "command",
                                "command": "/path/to/arize-hook-cursor",
                                "timeout": 30,
                            }
                        ]
                    },
                }
            )
        )
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        cmds = [e["command"] for e in data["hooks"]["sessionStart"]]
        assert "/path/to/arize-hook-cursor" in cmds
        assert any(Path(c).name == HOOK_BIN_NAME for c in cmds)

    def test_preserves_unknown_top_level_keys(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({"version": 1, "hooks": {}, "custom": "x"}))
        CursorPlatform(hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        assert data["custom"] == "x"


class TestUninstallFreshState:
    def test_uninstall_missing_file_is_noop(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        CursorPlatform(hooks_file=hooks_file).uninstall()
        assert not hooks_file.exists()

    def test_uninstall_empty_hooks_dict_is_noop(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({"version": 1, "hooks": {}}))
        CursorPlatform(hooks_file=hooks_file).uninstall()
        data = json.loads(hooks_file.read_text())
        assert data == {"version": 1, "hooks": {}}


class TestUninstallRemovesOnlyOurEntries:
    def test_removes_only_thirdeye_entries(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {
                        "sessionStart": [
                            {
                                "type": "command",
                                "command": f"/usr/local/bin/{HOOK_BIN_NAME}",
                                "timeout": HOOK_TIMEOUT_S,
                            },
                            {
                                "type": "command",
                                "command": "/path/to/arize-hook-cursor",
                                "timeout": 30,
                            },
                        ]
                    },
                }
            )
        )
        CursorPlatform(hooks_file=hooks_file).uninstall()
        data = json.loads(hooks_file.read_text())
        cmds = [e["command"] for e in data["hooks"]["sessionStart"]]
        assert "/path/to/arize-hook-cursor" in cmds
        assert not any(Path(c).name == HOOK_BIN_NAME for c in cmds)

    def test_collapses_empty_event_arrays(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        p = CursorPlatform(hooks_file=hooks_file)
        p.install()
        p.uninstall()
        data = json.loads(hooks_file.read_text())
        assert data["hooks"] == {}

    def test_preserves_version_field(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        p = CursorPlatform(hooks_file=hooks_file)
        p.install()
        p.uninstall()
        data = json.loads(hooks_file.read_text())
        assert data["version"] == 1

    def test_does_not_delete_file(self, tmp_path: Path):
        hooks_file = tmp_path / "hooks.json"
        p = CursorPlatform(hooks_file=hooks_file)
        p.install()
        p.uninstall()
        assert hooks_file.exists()
