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
