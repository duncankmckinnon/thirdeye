from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.config import Config, LogfireSettings
from thirdeye.logfire_auth import LogfireAuthError


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_platform(name: str, installed: bool = False) -> MagicMock:
    platform = MagicMock()
    platform.display_name = name
    platform.is_installed.return_value = installed
    platform.notify_conflict.return_value = None
    return platform


def _fake_resolver(
    monkeypatch: pytest.MonkeyPatch,
    platforms: dict[str, MagicMock],
) -> None:
    monkeypatch.setattr(
        "thirdeye.commands.setup.add_commands._resolve_platform",
        lambda name, **_: platforms[name],
    )


def test_setup_appears_in_help() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "setup" in result.output


def test_setup_multiselects_agents_skills_and_logfire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = {name: _fake_platform(name) for name in ("claude", "codex", "cursor")}
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: True)
    monkeypatch.setattr(
        "thirdeye.commands.setup.logfire_cmd.mint_write_token", lambda: "pylf_v1_us_test"
    )

    result = CliRunner().invoke(
        main,
        ["setup"],
        input="1,3\nall\ny\n",
    )

    assert result.exit_code == 0, result.output
    platforms["claude"].install.assert_called_once()
    platforms["codex"].install.assert_not_called()
    platforms["cursor"].install.assert_called_once()
    assert Path(".claude/skills/use-thirdeye").is_symlink()
    assert Path(".agents/skills/use-thirdeye").is_symlink()
    config = Config.load()
    assert config.logfire.enabled is True
    assert config.logfire.token == "pylf_v1_us_test"
    assert "pylf_v1_us_test" not in result.output


def test_setup_can_skip_agent_and_skill_multiselects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = {name: _fake_platform(name) for name in ("claude", "codex", "cursor")}
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: False)

    result = CliRunner().invoke(main, ["setup"], input="none\nnone\n")

    assert result.exit_code == 0, result.output
    assert all(not platform.install.called for platform in platforms.values())
    assert "Tracing: no platforms configured" in result.output
    assert "Skills: skipped" in result.output
    assert "Logfire: extension not installed" in result.output


def test_setup_skips_entire_logfire_step_when_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = {name: _fake_platform(name) for name in ("claude", "codex", "cursor")}
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: False)

    result = CliRunner().invoke(main, ["setup"], input="none\nnone\n")

    assert result.exit_code == 0, result.output
    assert "Pydantic Logfire" not in result.output
    assert "Enable live Logfire" not in result.output


def test_setup_does_not_ask_about_already_configured_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = {
        name: _fake_platform(name, installed=True) for name in ("claude", "codex", "cursor")
    }
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup._install_new_skills", lambda _: "up to date")
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: False)

    result = CliRunner().invoke(main, ["setup"])

    assert result.exit_code == 0, result.output
    assert "All supported agents are already configured" in result.output
    assert "Configure tracing for [" not in result.output
    assert all(not platform.install.called for platform in platforms.values())


def test_setup_installs_only_new_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    from thirdeye.commands import skill

    platforms = {
        "claude": _fake_platform("claude", installed=True),
        "codex": _fake_platform("codex"),
        "cursor": _fake_platform("cursor"),
    }
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: False)
    existing = skill._install_one("use-thirdeye", Path(".claude/skills/use-thirdeye"), force=False)

    result = CliRunner().invoke(main, ["setup"], input="none\nall\n")

    assert result.exit_code == 0, result.output
    assert "already installed" not in result.output
    assert "Skills: installed 4 new" in result.output
    assert "Installed" in existing


def test_setup_offers_to_replace_a_foreign_codex_notifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thirdeye.platforms.codex.install import CodexPlatform

    config_file = tmp_path / "config.toml"
    hooks_file = tmp_path / "hooks.json"
    config_file.write_text("notify = ['/usr/local/bin/existing-notifier']\n")
    codex = CodexPlatform(config_file=config_file, hooks_file=hooks_file)
    platforms = {
        "claude": _fake_platform("claude"),
        "codex": codex,
        "cursor": _fake_platform("cursor"),
    }

    def resolve(name: str, **kwargs: object) -> object:
        if name == "codex" and kwargs.get("force"):
            return CodexPlatform(config_file=config_file, hooks_file=hooks_file, force=True)
        return platforms[name]

    monkeypatch.setattr("thirdeye.commands.setup.add_commands._resolve_platform", resolve)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: False)

    result = CliRunner().invoke(main, ["setup"], input="2\ny\nnone\n")

    assert result.exit_code == 0, result.output
    assert "existing-notifier" in result.output
    assert "may disable features" in result.output
    assert "Installed tracing for Codex CLI" in result.output
    assert "thirdeye-codex-notify" in config_file.read_text()


def test_setup_preserves_foreign_codex_notifier_when_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thirdeye.platforms.codex.install import CodexPlatform

    config_file = tmp_path / "config.toml"
    hooks_file = tmp_path / "hooks.json"
    original = "notify = ['/usr/local/bin/existing-notifier']\n"
    config_file.write_text(original)
    platforms = {
        "claude": _fake_platform("claude"),
        "codex": CodexPlatform(config_file=config_file, hooks_file=hooks_file),
        "cursor": _fake_platform("cursor"),
    }
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: False)

    result = CliRunner().invoke(main, ["setup"], input="2\nn\nnone\n")

    assert result.exit_code == 0, result.output
    assert "Skipped Codex tracing" in result.output
    assert config_file.read_text() == original
    assert not hooks_file.exists()


def test_setup_keeps_existing_logfire_token_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = {
        name: _fake_platform(name, installed=True) for name in ("claude", "codex", "cursor")
    }
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup._install_new_skills", lambda _: "up to date")
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: True)
    Config.load().write_logfire_settings(LogfireSettings(enabled=True, token="old-token"))

    result = CliRunner().invoke(main, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "already configured" in result.output
    assert Config.load().logfire.token == "old-token"
    assert "old-token" not in result.output


def test_setup_can_replace_existing_logfire_token(monkeypatch: pytest.MonkeyPatch) -> None:
    platforms = {
        name: _fake_platform(name, installed=True) for name in ("claude", "codex", "cursor")
    }
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup._install_new_skills", lambda _: "up to date")
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: True)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.mint_write_token", lambda: "new-token")
    Config.load().write_logfire_settings(LogfireSettings(enabled=True, token="old-token"))

    result = CliRunner().invoke(main, ["setup"], input="y\n")

    assert result.exit_code == 0, result.output
    assert Config.load().logfire.token == "new-token"
    assert "old-token" not in result.output
    assert "new-token" not in result.output


def test_setup_surfaces_logfire_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    platforms = {name: _fake_platform(name) for name in ("claude", "codex", "cursor")}
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: True)
    monkeypatch.setattr(
        "thirdeye.commands.setup.logfire_cmd.mint_write_token",
        MagicMock(side_effect=LogfireAuthError("login cancelled")),
    )

    result = CliRunner().invoke(main, ["setup"], input="none\nnone\ny\n")

    assert result.exit_code != 0
    assert "login cancelled" in result.output
    assert Config.load().logfire.token is None


def test_setup_can_enable_an_existing_disabled_logfire_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = {
        name: _fake_platform(name, installed=True) for name in ("claude", "codex", "cursor")
    }
    _fake_resolver(monkeypatch, platforms)
    monkeypatch.setattr("thirdeye.commands.setup._install_new_skills", lambda _: "up to date")
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: True)
    Config.load().write_logfire_settings(LogfireSettings(enabled=False, token="saved-token"))

    result = CliRunner().invoke(main, ["setup"], input="n\ny\n")

    assert result.exit_code == 0, result.output
    assert Config.load().logfire.enabled is True
    assert Config.load().logfire.token == "saved-token"
    assert "saved-token" not in result.output


@pytest.mark.parametrize(
    ("platforms", "targets"),
    [
        (["claude"], [Path(".claude/skills")]),
        (["codex"], [Path(".codex/skills")]),
        (["cursor"], [Path(".agents/skills")]),
        ([], [Path(".agents/skills")]),
        (
            ["claude", "codex", "cursor"],
            [Path(".claude/skills"), Path(".codex/skills"), Path(".agents/skills")],
        ),
    ],
)
def test_skill_targets_follow_configured_agents(platforms: list[str], targets: list[Path]) -> None:
    from thirdeye.commands.setup import _skill_targets

    assert _skill_targets(platforms) == targets
