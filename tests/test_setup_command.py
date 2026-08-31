from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.config import Config


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_setup_appears_in_help() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "setup" in result.output


def test_setup_configures_selected_platforms_skills_and_logfire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[str] = []

    def fake_platform(name: str) -> MagicMock:
        platform = MagicMock()
        platform.display_name = name
        platform.notify_conflict.return_value = None
        platform.install.side_effect = lambda: installed.append(name)
        return platform

    monkeypatch.setattr(
        "thirdeye.commands.setup.add_commands._resolve_platform",
        lambda name: fake_platform(name),
    )
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: True)

    result = CliRunner().invoke(
        main,
        ["setup"],
        input="y\nn\ny\ny\ny\npylf_v1_us_test\n",
    )

    assert result.exit_code == 0, result.output
    assert installed == ["claude", "cursor"]
    assert (Path(".claude/skills/use-thirdeye")).is_symlink()
    assert (Path(".agents/skills/use-thirdeye")).is_symlink()
    config = Config.load()
    assert config.logfire.enabled is True
    assert config.logfire.token == "pylf_v1_us_test"
    assert "pylf_v1_us_test" not in result.output
    assert "Setup complete." in result.output


def test_setup_can_skip_every_optional_step(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = MagicMock()
    monkeypatch.setattr("thirdeye.commands.setup.add_commands._resolve_platform", resolve)

    result = CliRunner().invoke(main, ["setup"], input="n\nn\nn\nn\nn\n")

    assert result.exit_code == 0, result.output
    resolve.assert_not_called()
    assert "Tracing: no platforms configured" in result.output
    assert "Skills: skipped" in result.output
    assert "Logfire: skipped" in result.output


def test_setup_explains_missing_logfire_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thirdeye.commands.setup.logfire_cmd.is_available", lambda: False)

    result = CliRunner().invoke(main, ["setup"], input="n\nn\nn\nn\ny\n")

    assert result.exit_code == 0, result.output
    assert "thrdi[logfire]" in result.output
    assert Config.load().logfire.enabled is False


def test_setup_offers_to_replace_a_foreign_codex_notifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thirdeye.commands.setup import add_commands
    from thirdeye.platforms.codex.install import CodexPlatform

    config_file = tmp_path / "config.toml"
    hooks_file = tmp_path / "hooks.json"
    config_file.write_text("notify = ['/usr/local/bin/existing-notifier']\n")
    monkeypatch.setitem(
        add_commands.PLATFORMS,
        "codex",
        lambda **kwargs: CodexPlatform(
            config_file=config_file,
            hooks_file=hooks_file,
            **kwargs,
        ),
    )

    result = CliRunner().invoke(main, ["setup"], input="n\ny\ny\nn\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert "existing-notifier" in result.output
    assert "may disable features" in result.output
    assert "Installed tracing for Codex CLI" in result.output
    assert "thirdeye-codex-notify" in config_file.read_text()
    assert "existing-notifier" not in config_file.read_text()


def test_setup_preserves_a_foreign_codex_notifier_when_force_is_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thirdeye.commands.setup import add_commands
    from thirdeye.platforms.codex.install import CodexPlatform

    config_file = tmp_path / "config.toml"
    hooks_file = tmp_path / "hooks.json"
    original = "notify = ['/usr/local/bin/existing-notifier']\n"
    config_file.write_text(original)
    monkeypatch.setitem(
        add_commands.PLATFORMS,
        "codex",
        lambda **kwargs: CodexPlatform(
            config_file=config_file,
            hooks_file=hooks_file,
            **kwargs,
        ),
    )

    result = CliRunner().invoke(main, ["setup"], input="n\ny\nn\nn\nn\nn\n")

    assert result.exit_code == 0, result.output
    assert "Skipped Codex tracing" in result.output
    assert "Tracing: no platforms configured" in result.output
    assert config_file.read_text() == original
    assert not hooks_file.exists()


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
def test_skill_targets_follow_selected_agents(platforms: list[str], targets: list[Path]) -> None:
    from thirdeye.commands.setup import _skill_targets

    assert _skill_targets(platforms) == targets
