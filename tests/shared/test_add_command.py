from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.commands.add import PLATFORMS, find_orphaned_hooks
from thirdeye.platforms.claude.install import ClaudePlatform
from thirdeye.platforms.codex.install import CodexPlatform
from thirdeye.platforms.cursor.install import CursorPlatform

# -- command registration ------------------------------------------------------


def test_add_appears_in_help():
    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "add" in r.output


def test_remove_appears_in_help():
    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "remove" in r.output


def test_add_help_mentions_claude():
    r = CliRunner().invoke(main, ["add", "--help"])
    assert r.exit_code == 0
    assert "--claude" in r.output


def test_remove_help_mentions_claude():
    r = CliRunner().invoke(main, ["remove", "--help"])
    assert r.exit_code == 0
    assert "--claude" in r.output


def test_add_help_mentions_codex():
    r = CliRunner().invoke(main, ["add", "--help"])
    assert r.exit_code == 0
    assert "--codex" in r.output


def test_remove_help_mentions_codex():
    r = CliRunner().invoke(main, ["remove", "--help"])
    assert r.exit_code == 0
    assert "--codex" in r.output


# -- platform flag required ----------------------------------------------------


def test_add_requires_platform():
    r = CliRunner().invoke(main, ["add"])
    assert r.exit_code != 0
    assert "platform" in r.output.lower()


def test_remove_requires_platform():
    r = CliRunner().invoke(main, ["remove"])
    assert r.exit_code != 0
    assert "platform" in r.output.lower()


# -- install (add --claude) ----------------------------------------------------


def test_add_claude_writes_settings(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    r = CliRunner().invoke(main, ["add", "--claude"])
    assert r.exit_code == 0, r.output
    assert "Installed" in r.output
    assert "Claude Code" in r.output
    data = json.loads(settings.read_text())
    assert "hooks" in data and len(data["hooks"]) >= 1


def test_add_claude_creates_settings_file(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    assert not settings.exists()
    CliRunner().invoke(main, ["add", "--claude"])
    assert settings.exists()


def test_add_claude_registers_all_hook_events(tmp_path: Path, monkeypatch):
    from thirdeye.platforms.claude.constants import HOOK_EVENTS

    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    CliRunner().invoke(main, ["add", "--claude"])
    data = json.loads(settings.read_text())
    assert set(data["hooks"].keys()) == set(HOOK_EVENTS.keys())


def test_add_claude_idempotent(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    runner = CliRunner()
    runner.invoke(main, ["add", "--claude"])
    first = settings.read_text()
    runner.invoke(main, ["add", "--claude"])
    second = settings.read_text()
    assert first == second


def test_add_claude_preserves_existing_settings(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"theme": "dark"}))
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    CliRunner().invoke(main, ["add", "--claude"])
    data = json.loads(settings.read_text())
    assert data["theme"] == "dark"
    assert "hooks" in data


# -- uninstall (remove --claude) -----------------------------------------------


def test_remove_claude(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    runner = CliRunner()
    runner.invoke(main, ["add", "--claude"])
    r = runner.invoke(main, ["remove", "--claude"])
    assert r.exit_code == 0, r.output
    assert "Removed" in r.output
    assert "Claude Code" in r.output
    data = json.loads(settings.read_text())
    assert "hooks" not in data


def test_remove_claude_noop_when_not_installed(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    r = CliRunner().invoke(main, ["remove", "--claude"])
    assert r.exit_code == 0, r.output
    assert "Removed" in r.output


def test_remove_claude_preserves_other_settings(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"theme": "dark"}))
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    runner = CliRunner()
    runner.invoke(main, ["add", "--claude"])
    runner.invoke(main, ["remove", "--claude"])
    data = json.loads(settings.read_text())
    assert data["theme"] == "dark"
    assert "hooks" not in data


def test_reinstall_after_remove(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    runner = CliRunner()
    runner.invoke(main, ["add", "--claude"])
    first = json.loads(settings.read_text())
    runner.invoke(main, ["remove", "--claude"])
    runner.invoke(main, ["add", "--claude"])
    restored = json.loads(settings.read_text())
    assert first == restored


# -- output messages -----------------------------------------------------------


def test_add_install_output_contains_platform_name(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    r = CliRunner().invoke(main, ["add", "--claude"])
    assert r.exit_code == 0
    assert "Claude Code" in r.output


def test_remove_output_contains_platform_name(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    runner = CliRunner()
    runner.invoke(main, ["add", "--claude"])
    r = runner.invoke(main, ["remove", "--claude"])
    assert r.exit_code == 0
    assert "Claude Code" in r.output


# -- existing commands unaffected ----------------------------------------------


def test_existing_commands_still_registered():
    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    for cmd in ["ingest", "list", "show", "events", "tail", "event", "search", "stats"]:
        assert cmd in r.output, f"command {cmd!r} missing from --help"


def test_ingest_still_works(tmp_path: Path):
    runner = CliRunner()
    env = {"THIRDEYE_HOME": str(tmp_path)}
    payload = json.dumps({"t": "msg", "data": "hi"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "REGR1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0


# -- PLATFORMS dict correctness ------------------------------------------------


def test_platforms_dict_is_exactly_supported_platforms():
    assert set(PLATFORMS) == {"claude", "codex", "cursor"}


def test_platforms_dict_has_claude():
    assert "claude" in PLATFORMS
    assert PLATFORMS["claude"] is ClaudePlatform


def test_platforms_dict_has_codex():
    assert "codex" in PLATFORMS
    assert PLATFORMS["codex"] is CodexPlatform


def test_platforms_dict_has_cursor():
    assert "cursor" in PLATFORMS
    assert PLATFORMS["cursor"] is CursorPlatform


def test_platform_flag_value_maps_to_platforms_key():
    for key, cls in PLATFORMS.items():
        instance = cls()
        assert instance.name == key


# -- removed platforms rejected ------------------------------------------------


def test_add_gemini_fails():
    r = CliRunner().invoke(main, ["add", "gemini"])
    assert r.exit_code != 0


def test_add_gemini_flag_fails():
    r = CliRunner().invoke(main, ["add", "--gemini"])
    assert r.exit_code != 0


# -- implementation uses PLATFORMS dict ----------------------------------------


def test_add_uses_platforms_dict(monkeypatch):
    """The add command should dispatch via PLATFORMS[platform_flag](), not hardcode ClaudePlatform()."""
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Mock Platform"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "claude", mock_cls)
    r = CliRunner().invoke(main, ["add", "--claude"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.install.assert_called_once()


def test_remove_uses_platforms_dict(monkeypatch):
    """The remove command should dispatch via PLATFORMS[platform_flag](), not hardcode ClaudePlatform()."""
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Mock Platform"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "claude", mock_cls)
    r = CliRunner().invoke(main, ["remove", "--claude"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.uninstall.assert_called_once()


# -- install (add --codex) -----------------------------------------------------


def test_add_codex_calls_install(monkeypatch):
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Codex CLI"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "codex", mock_cls)
    r = CliRunner().invoke(main, ["add", "--codex"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.install.assert_called_once()


def test_add_codex_writes_config(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.toml"
    monkeypatch.setitem(PLATFORMS, "codex", lambda: CodexPlatform(config_file=config))
    r = CliRunner().invoke(main, ["add", "--codex"])
    assert r.exit_code == 0, r.output
    assert "Installed" in r.output
    assert "Codex CLI" in r.output
    text = config.read_text()
    assert "notify" in text
    assert "thirdeye" in text


def test_remove_codex_calls_uninstall(monkeypatch):
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Codex CLI"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "codex", mock_cls)
    r = CliRunner().invoke(main, ["remove", "--codex"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.uninstall.assert_called_once()


def test_remove_codex_removes_notify(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.toml"
    monkeypatch.setitem(PLATFORMS, "codex", lambda: CodexPlatform(config_file=config))
    runner = CliRunner()
    runner.invoke(main, ["add", "--codex"])
    r = runner.invoke(main, ["remove", "--codex"])
    assert r.exit_code == 0, r.output
    assert "Removed" in r.output
    assert "Codex CLI" in r.output


def test_codex_platform_name_matches_key():
    instance = CodexPlatform(config_file=Path("/fake"))
    assert instance.name == "codex"


def test_add_codex_foreign_notify_errors_without_force(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("notify = ['/usr/local/bin/other-notify']\n")
    monkeypatch.setitem(PLATFORMS, "codex", lambda **kw: CodexPlatform(config_file=config, **kw))
    r = CliRunner().invoke(main, ["add", "--codex"])
    assert r.exit_code != 0
    assert "--force" in r.output


def test_add_codex_force_takes_over_foreign_notify(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("notify = ['/usr/local/bin/other-notify']\n")
    monkeypatch.setitem(PLATFORMS, "codex", lambda **kw: CodexPlatform(config_file=config, **kw))
    r = CliRunner().invoke(main, ["add", "--codex", "--force"])
    assert r.exit_code == 0, r.output
    text = config.read_text()
    assert "thirdeye-codex-notify" in text
    assert "other-notify" not in text


# -- add --list ----------------------------------------------------------------


def test_list_shows_supported_platforms(monkeypatch):
    # Avoid depending on the real ~/.gemini or ~/.cursor config on this machine.
    monkeypatch.setattr("thirdeye.commands.add.ORPHAN_CONFIG_PATHS", ())
    r = CliRunner().invoke(main, ["add", "--list"])
    assert r.exit_code == 0, r.output
    assert "claude" in r.output
    assert "codex" in r.output
    assert "gemini" not in r.output
    assert "cursor" in r.output


# -- find_orphaned_hooks -------------------------------------------------------


def test_find_orphaned_hooks_detects_gemini_session_start(tmp_path: Path):
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/usr/local/bin/thirdeye-gemini-session-start",
                                    "timeout": 30000,
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )
    result = find_orphaned_hooks([config])
    assert result == [(config, "/usr/local/bin/thirdeye-gemini-session-start")]


def test_find_orphaned_hooks_ignores_unrelated_commands(tmp_path: Path):
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "/usr/bin/echo hi"}]}
                    ]
                }
            }
        )
    )
    assert find_orphaned_hooks([config]) == []


def test_find_orphaned_hooks_missing_file(tmp_path: Path):
    assert find_orphaned_hooks([tmp_path / "does-not-exist.json"]) == []


def test_find_orphaned_hooks_malformed_json(tmp_path: Path):
    config = tmp_path / "settings.json"
    config.write_text("{ this is not valid json ]")
    assert find_orphaned_hooks([config]) == []


def test_find_orphaned_hooks_ignores_supported_cursor_hook(tmp_path: Path):
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "afterFileEdit": [{"command": "/opt/tools/thirdeye-cursor-hook", "timeout": 30}]
                },
            }
        )
    )
    result = find_orphaned_hooks([config])
    assert result == []


def test_find_orphaned_hooks_does_not_match_claude_stop(tmp_path: Path):
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"command": "thirdeye-claude-stop"}]}]}})
    )
    assert find_orphaned_hooks([config]) == []
