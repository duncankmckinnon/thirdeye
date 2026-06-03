from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.commands.add import PLATFORMS
from thirdeye.platforms.claude.install import ClaudePlatform
from thirdeye.platforms.codex.install import CodexPlatform
from thirdeye.platforms.cursor.install import CursorPlatform
from thirdeye.platforms.gemini.install import GeminiPlatform

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


def test_platforms_dict_has_claude():
    assert "claude" in PLATFORMS
    assert PLATFORMS["claude"] is ClaudePlatform


def test_platform_flag_value_maps_to_platforms_key():
    for key, cls in PLATFORMS.items():
        instance = cls()
        assert instance.name == key


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


# -- hook command structure ----------------------------------------------------


def test_add_claude_hook_entries_have_command_type(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    CliRunner().invoke(main, ["add", "--claude"])
    data = json.loads(settings.read_text())
    for event_name, entries in data["hooks"].items():
        for entry in entries:
            for hook in entry["hooks"]:
                assert hook["type"] == "command", f"{event_name} hook missing type=command"
                assert "command" in hook, f"{event_name} hook missing command key"


def test_add_claude_hook_commands_contain_thirdeye(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    CliRunner().invoke(main, ["add", "--claude"])
    data = json.loads(settings.read_text())
    for event_name, entries in data["hooks"].items():
        for entry in entries:
            for hook in entry["hooks"]:
                cmd = hook["command"]
                assert "thirdeye" in cmd, f"{event_name} command {cmd!r} missing 'thirdeye'"


# -- PLATFORMS dict: gemini and codex ------------------------------------------


def test_platforms_dict_has_gemini():
    assert "gemini" in PLATFORMS
    assert PLATFORMS["gemini"] is GeminiPlatform


def test_platforms_dict_has_codex():
    assert "codex" in PLATFORMS
    assert PLATFORMS["codex"] is CodexPlatform


# -- help text lists all platform flags ----------------------------------------


def test_add_help_mentions_gemini():
    r = CliRunner().invoke(main, ["add", "--help"])
    assert r.exit_code == 0
    assert "--gemini" in r.output


def test_add_help_mentions_codex():
    r = CliRunner().invoke(main, ["add", "--help"])
    assert r.exit_code == 0
    assert "--codex" in r.output


def test_remove_help_mentions_gemini():
    r = CliRunner().invoke(main, ["remove", "--help"])
    assert r.exit_code == 0
    assert "--gemini" in r.output


def test_remove_help_mentions_codex():
    r = CliRunner().invoke(main, ["remove", "--help"])
    assert r.exit_code == 0
    assert "--codex" in r.output


# -- error message lists all three flags ---------------------------------------


def test_add_no_flag_error_lists_all_platforms():
    r = CliRunner().invoke(main, ["add"])
    assert r.exit_code != 0
    assert "--claude" in r.output
    assert "--gemini" in r.output
    assert "--codex" in r.output


def test_remove_no_flag_error_lists_all_platforms():
    r = CliRunner().invoke(main, ["remove"])
    assert r.exit_code != 0
    assert "--claude" in r.output
    assert "--gemini" in r.output
    assert "--codex" in r.output


# -- install (add --gemini) ----------------------------------------------------


def test_add_gemini_calls_install(monkeypatch):
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Gemini CLI"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "gemini", mock_cls)
    r = CliRunner().invoke(main, ["add", "--gemini"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.install.assert_called_once()


def test_add_gemini_writes_settings(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "gemini", lambda: GeminiPlatform(settings_file=settings))
    r = CliRunner().invoke(main, ["add", "--gemini"])
    assert r.exit_code == 0, r.output
    assert "Installed" in r.output
    assert "Gemini CLI" in r.output
    data = json.loads(settings.read_text())
    assert "hooks" in data


def test_add_gemini_registers_all_hook_events(tmp_path: Path, monkeypatch):
    from thirdeye.platforms.gemini.constants import HOOK_EVENTS

    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "gemini", lambda: GeminiPlatform(settings_file=settings))
    CliRunner().invoke(main, ["add", "--gemini"])
    data = json.loads(settings.read_text())
    assert set(data["hooks"].keys()) == set(HOOK_EVENTS.keys())


# -- uninstall (remove --gemini) -----------------------------------------------


def test_remove_gemini_calls_uninstall(monkeypatch):
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Gemini CLI"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "gemini", mock_cls)
    r = CliRunner().invoke(main, ["remove", "--gemini"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.uninstall.assert_called_once()


def test_remove_gemini_removes_hooks(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "gemini", lambda: GeminiPlatform(settings_file=settings))
    runner = CliRunner()
    runner.invoke(main, ["add", "--gemini"])
    r = runner.invoke(main, ["remove", "--gemini"])
    assert r.exit_code == 0, r.output
    assert "Removed" in r.output
    assert "Gemini CLI" in r.output


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


# -- uninstall (remove --codex) ------------------------------------------------


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


# -- claude regression ---------------------------------------------------------


# -- cursor: help, dispatch, PLATFORMS dict ------------------------------------


def test_add_help_mentions_cursor():
    r = CliRunner().invoke(main, ["add", "--help"])
    assert r.exit_code == 0
    assert "--cursor" in r.output


def test_remove_help_mentions_cursor():
    r = CliRunner().invoke(main, ["remove", "--help"])
    assert r.exit_code == 0
    assert "--cursor" in r.output


def test_platforms_dict_has_cursor():
    assert "cursor" in PLATFORMS
    assert PLATFORMS["cursor"] is CursorPlatform


def test_add_cursor_dispatches_to_cursor_platform(monkeypatch):
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Cursor"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "cursor", mock_cls)
    r = CliRunner().invoke(main, ["add", "--cursor"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.install.assert_called_once()


def test_remove_cursor_dispatches_to_cursor_platform(monkeypatch):
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Cursor"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "cursor", mock_cls)
    r = CliRunner().invoke(main, ["remove", "--cursor"])
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.uninstall.assert_called_once()


def test_add_claude_still_works(tmp_path: Path, monkeypatch):
    """Regression: --claude must still work after adding new platforms."""
    settings = tmp_path / "settings.json"
    monkeypatch.setitem(PLATFORMS, "claude", lambda: ClaudePlatform(settings_file=settings))
    r = CliRunner().invoke(main, ["add", "--claude"])
    assert r.exit_code == 0, r.output
    assert "Installed" in r.output
    assert "Claude Code" in r.output


# -- platform flag_value maps to PLATFORMS key ---------------------------------


def test_all_platform_flag_values_map_to_platforms_keys():
    """Every key in PLATFORMS should have a corresponding --flag on the CLI."""
    for key in PLATFORMS:
        r = CliRunner().invoke(main, ["add", f"--{key}", "--help"])
        # If the flag doesn't exist, Click will error before showing help
        # We just need it not to fail with "no such option"
        assert "no such option" not in r.output.lower(), f"--{key} flag not registered"


def test_gemini_platform_name_matches_key():
    instance = GeminiPlatform(settings_file=Path("/fake"))
    assert instance.name == "gemini"


def test_codex_platform_name_matches_key():
    instance = CodexPlatform(config_file=Path("/fake"))
    assert instance.name == "codex"


# -- mutual exclusivity: last flag wins ----------------------------------------


def test_add_multiple_platform_flags_last_wins(monkeypatch):
    """Passing two platform flags: last flag wins (Click flag_value behavior)."""
    from unittest.mock import MagicMock

    mock_platform = MagicMock()
    mock_platform.display_name = "Gemini CLI"
    mock_cls = MagicMock(return_value=mock_platform)

    monkeypatch.setitem(PLATFORMS, "gemini", mock_cls)
    r = CliRunner().invoke(main, ["add", "--claude", "--gemini"])
    # Click's flag_value makes the last flag win, so gemini is resolved
    assert r.exit_code == 0, r.output
    mock_cls.assert_called_once()
    mock_platform.install.assert_called_once()
