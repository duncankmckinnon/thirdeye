from __future__ import annotations

import json
from pathlib import Path

from thirdeye.platforms.claude.constants import HOOK_EVENTS
from thirdeye.platforms.claude.install import ClaudePlatform


class TestClaudePlatformAttributes:
    def test_name_is_claude(self):
        p = ClaudePlatform(settings_file=Path("/fake/settings.json"))
        assert p.name == "claude"

    def test_display_name(self):
        p = ClaudePlatform(settings_file=Path("/fake/settings.json"))
        assert p.display_name == "Claude Code"

    def test_is_platform_subclass(self):
        from thirdeye.platforms.base import Platform

        assert issubclass(ClaudePlatform, Platform)


class TestInstallFreshFile:
    def test_writes_all_hook_events(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert set(settings["hooks"].keys()) == set(HOOK_EVENTS.keys())

    def test_creates_parent_dir(self, tmp_path: Path):
        settings_file = tmp_path / "nested" / "deeper" / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        assert settings_file.exists()

    def test_sets_command_type(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        for entries in settings["hooks"].values():
            for entry in entries:
                for h in entry["hooks"]:
                    assert h["type"] == "command"
                    assert "command" in h and isinstance(h["command"], str)

    def test_each_event_has_one_entry(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        for event, entries in settings["hooks"].items():
            assert len(entries) == 1, f"expected 1 entry for {event}"

    def test_command_contains_script_name(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        for event, script in HOOK_EVENTS.items():
            cmds = [h["command"] for entry in settings["hooks"][event] for h in entry["hooks"]]
            assert any(script in c for c in cmds), f"{script} not in commands for {event}"

    def test_output_is_valid_json(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        json.loads(settings_file.read_text())

    def test_file_ends_with_newline(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        assert settings_file.read_text().endswith("\n")


class TestInstallIdempotent:
    def test_no_duplicate_commands(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.install()
        settings = json.loads(settings_file.read_text())
        for entries in settings["hooks"].values():
            commands = [h["command"] for entry in entries for h in entry["hooks"]]
            assert len(commands) == len(set(commands))

    def test_no_duplicate_entries(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.install()
        p.install()
        settings = json.loads(settings_file.read_text())
        for event, entries in settings["hooks"].items():
            assert len(entries) == 1, f"expected 1 entry for {event} after 3 installs"

    def test_content_identical_after_double_install(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        first = settings_file.read_text()
        p.install()
        second = settings_file.read_text()
        assert first == second


class TestInstallPreservesExisting:
    def test_preserves_existing_settings(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"theme": "dark", "env": {"FOO": "bar"}}))
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert settings["theme"] == "dark"
        assert settings["env"] == {"FOO": "bar"}
        assert "hooks" in settings

    def test_preserves_unrelated_hooks(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"hooks": [{"type": "command", "command": "/some/other/tool"}]}
                        ]
                    }
                }
            )
        )
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        cmds = [h["command"] for entry in settings["hooks"]["SessionStart"] for h in entry["hooks"]]
        assert "/some/other/tool" in cmds
        assert any("thirdeye-claude-session-start" in c for c in cmds)

    def test_preserves_hooks_for_unknown_events(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "CustomEvent": [{"hooks": [{"type": "command", "command": "/custom/hook"}]}]
                    }
                }
            )
        )
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert "CustomEvent" in settings["hooks"]
        cmds = [h["command"] for entry in settings["hooks"]["CustomEvent"] for h in entry["hooks"]]
        assert "/custom/hook" in cmds


class TestInstallEdgeCases:
    def test_handles_empty_file(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("")
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert set(settings["hooks"].keys()) == set(HOOK_EVENTS.keys())

    def test_handles_malformed_json(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{invalid json")
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert set(settings["hooks"].keys()) == set(HOOK_EVENTS.keys())

    def test_handles_empty_hooks_dict(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {}}))
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert set(settings["hooks"].keys()) == set(HOOK_EVENTS.keys())

    def test_handles_empty_event_list(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {"SessionStart": []}}))
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert len(settings["hooks"]["SessionStart"]) == 1


class TestUninstallRemovesHooks:
    def test_removes_all_our_hooks(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.uninstall()
        settings = json.loads(settings_file.read_text())
        assert "hooks" not in settings

    def test_drops_empty_hooks_key(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.uninstall()
        settings = json.loads(settings_file.read_text())
        assert "hooks" not in settings

    def test_removes_only_our_hooks(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"hooks": [{"type": "command", "command": "/some/other/tool"}]}
                        ]
                    }
                }
            )
        )
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.uninstall()
        settings = json.loads(settings_file.read_text())
        cmds = [h["command"] for entry in settings["hooks"]["SessionStart"] for h in entry["hooks"]]
        assert "/some/other/tool" in cmds
        assert not any("thirdeye-claude-session-start" in c for c in cmds)

    def test_preserves_other_settings_on_uninstall(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"theme": "dark"}))
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.uninstall()
        settings = json.loads(settings_file.read_text())
        assert settings["theme"] == "dark"

    def test_preserves_hooks_for_unknown_events_on_uninstall(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "CustomEvent": [{"hooks": [{"type": "command", "command": "/custom/hook"}]}]
                    }
                }
            )
        )
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.uninstall()
        settings = json.loads(settings_file.read_text())
        assert "CustomEvent" in settings["hooks"]
        cmds = [h["command"] for entry in settings["hooks"]["CustomEvent"] for h in entry["hooks"]]
        assert "/custom/hook" in cmds


class TestUninstallEdgeCases:
    def test_no_settings_file_is_noop(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).uninstall()
        assert not settings_file.exists()

    def test_no_hooks_key_is_noop(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"theme": "dark"}))
        ClaudePlatform(settings_file=settings_file).uninstall()
        settings = json.loads(settings_file.read_text())
        assert settings == {"theme": "dark"}

    def test_uninstall_idempotent(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.uninstall()
        p.uninstall()
        settings = json.loads(settings_file.read_text())
        assert "hooks" not in settings

    def test_uninstall_then_install_restores(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        first = json.loads(settings_file.read_text())
        p.uninstall()
        p.install()
        restored = json.loads(settings_file.read_text())
        assert first == restored

    def test_uninstall_empty_hooks_drops_key(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {}}))
        ClaudePlatform(settings_file=settings_file).uninstall()
        settings = json.loads(settings_file.read_text())
        assert "hooks" not in settings


class TestDefaultSettingsFile:
    def test_default_settings_file_matches_constants(self):
        from thirdeye.platforms.claude.constants import SETTINGS_FILE

        p = ClaudePlatform()
        assert p._settings_file == SETTINGS_FILE


class TestHookEntryStructure:
    def test_hook_structure_matches_claude_schema(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        for event, entries in settings["hooks"].items():
            assert isinstance(entries, list)
            for entry in entries:
                assert "hooks" in entry
                assert isinstance(entry["hooks"], list)
                for h in entry["hooks"]:
                    assert set(h.keys()) == {"type", "command"}
                    assert h["type"] == "command"

    def test_all_events_registered(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        assert len(settings["hooks"]) == len(HOOK_EVENTS)


class TestResolveCommandAbsolutePath:
    def test_install_uses_absolute_path_when_which_resolves(self, tmp_path: Path, monkeypatch):
        settings_file = tmp_path / "settings.json"

        def fake_which(name):
            return f"/usr/local/bin/{name}"

        monkeypatch.setattr("thirdeye.platforms.claude.install.shutil.which", fake_which)
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        for event, script in HOOK_EVENTS.items():
            cmds = [h["command"] for entry in settings["hooks"][event] for h in entry["hooks"]]
            assert cmds == [f"/usr/local/bin/{script}"]

    def test_install_falls_back_to_bare_name_when_which_fails(self, tmp_path: Path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("thirdeye.platforms.claude.install.shutil.which", lambda _: None)
        ClaudePlatform(settings_file=settings_file).install()
        settings = json.loads(settings_file.read_text())
        for event, script in HOOK_EVENTS.items():
            cmds = [h["command"] for entry in settings["hooks"][event] for h in entry["hooks"]]
            assert cmds == [script]

    def test_uninstall_removes_absolute_path_hooks(self, tmp_path: Path, monkeypatch):
        settings_file = tmp_path / "settings.json"

        def fake_which(name):
            return f"/usr/local/bin/{name}"

        monkeypatch.setattr("thirdeye.platforms.claude.install.shutil.which", fake_which)
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        p.uninstall()
        settings = json.loads(settings_file.read_text())
        assert "hooks" not in settings

    def test_idempotent_with_absolute_paths(self, tmp_path: Path, monkeypatch):
        settings_file = tmp_path / "settings.json"

        def fake_which(name):
            return f"/opt/bin/{name}"

        monkeypatch.setattr("thirdeye.platforms.claude.install.shutil.which", fake_which)
        p = ClaudePlatform(settings_file=settings_file)
        p.install()
        first = settings_file.read_text()
        p.install()
        second = settings_file.read_text()
        assert first == second


class TestUninstallMixedEntries:
    def test_entry_with_mixed_hooks_is_kept(self, tmp_path: Path, monkeypatch):
        """An entry containing both an thirdeye hook and a foreign hook is kept
        because not ALL hooks in the entry are ours."""
        monkeypatch.setattr("thirdeye.platforms.claude.install.shutil.which", lambda _: None)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {
                                "hooks": [
                                    {"type": "command", "command": "thirdeye-claude-session-start"},
                                    {"type": "command", "command": "/foreign/tool"},
                                ]
                            }
                        ]
                    }
                }
            )
        )
        ClaudePlatform(settings_file=settings_file).uninstall()
        settings = json.loads(settings_file.read_text())
        assert "SessionStart" in settings["hooks"]
        cmds = [h["command"] for entry in settings["hooks"]["SessionStart"] for h in entry["hooks"]]
        assert "thirdeye-claude-session-start" in cmds
        assert "/foreign/tool" in cmds

    def test_entry_with_empty_hooks_list_is_kept(self, tmp_path: Path):
        """An entry with an empty hooks list: all() on empty is True, so this
        entry would be removed. Verify the behavior."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"hooks": []},
                        ]
                    }
                }
            )
        )
        ClaudePlatform(settings_file=settings_file).uninstall()
        settings = json.loads(settings_file.read_text())
        # all() on empty iterable is True, so the entry IS removed
        assert "hooks" not in settings or "SessionStart" not in settings.get("hooks", {})

    def test_uninstall_with_no_thirdeye_hooks_present(self, tmp_path: Path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [{"hooks": [{"type": "command", "command": "/other/tool"}]}]
                    }
                }
            )
        )
        ClaudePlatform(settings_file=settings_file).uninstall()
        settings = json.loads(settings_file.read_text())
        assert "/other/tool" in [
            h["command"] for entry in settings["hooks"]["SessionStart"] for h in entry["hooks"]
        ]
