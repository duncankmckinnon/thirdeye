from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from thirdeye.commands.logfire_cmd import logfire_group
from thirdeye.config import Config, LogfireSettings
from thirdeye.logfire_auth import LogfireAuthError


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    # `enable` refuses to run without the logfire package; stub availability
    # so these tests exercise the config-writing logic regardless of whether
    # the optional extra happens to be installed in the dev environment.
    monkeypatch.setattr("thirdeye.commands.logfire_cmd.is_available", lambda: True)
    return tmp_path


def test_enable_without_package_installed_fails_clearly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("thirdeye.commands.logfire_cmd.is_available", lambda: False)
    result = CliRunner().invoke(logfire_group, ["enable"])
    assert result.exit_code != 0
    assert "logfire" in result.output.lower()


def test_status_disabled_by_default():
    result = CliRunner().invoke(logfire_group, ["status"])
    assert result.exit_code == 0
    assert "enabled           : False" in result.output


def test_enable_reuses_saved_token_when_confirmed(monkeypatch: pytest.MonkeyPatch):
    Config.load().write_logfire_settings(LogfireSettings(enabled=False, token="saved-token"))
    mint = MagicMock(return_value="minted-token")
    monkeypatch.setattr("thirdeye.commands.logfire_cmd.mint_write_token", mint)

    result = CliRunner().invoke(logfire_group, ["enable"], input="y\n")

    assert result.exit_code == 0, result.output
    config = Config.load()
    assert config.logfire.enabled is True
    assert config.logfire.token == "saved-token"
    mint.assert_not_called()


def test_enable_mints_token_when_none_saved(monkeypatch: pytest.MonkeyPatch):
    mint = MagicMock(return_value="pylf_v1_us_abcd1234")
    monkeypatch.setattr("thirdeye.commands.logfire_cmd.mint_write_token", mint)

    result = CliRunner().invoke(logfire_group, ["enable"])

    assert result.exit_code == 0, result.output
    config = Config.load()
    assert config.logfire.enabled is True
    assert config.logfire.token == "pylf_v1_us_abcd1234"
    mint.assert_called_once()


def test_enable_mints_token_when_saved_token_declined(monkeypatch: pytest.MonkeyPatch):
    Config.load().write_logfire_settings(LogfireSettings(enabled=True, token="old-token"))
    mint = MagicMock(return_value="new-minted")
    monkeypatch.setattr("thirdeye.commands.logfire_cmd.mint_write_token", mint)

    result = CliRunner().invoke(logfire_group, ["enable"], input="n\n")

    assert result.exit_code == 0, result.output
    assert Config.load().logfire.token == "new-minted"
    mint.assert_called_once()


def test_enable_surfaces_auth_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "thirdeye.commands.logfire_cmd.mint_write_token",
        MagicMock(side_effect=LogfireAuthError("login cancelled")),
    )

    result = CliRunner().invoke(logfire_group, ["enable"])

    assert result.exit_code != 0
    assert "login cancelled" in result.output
    assert Config.load().logfire.token is None


def test_enable_persists_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "thirdeye.commands.logfire_cmd.mint_write_token", lambda: "pylf_v1_us_abcd1234"
    )
    result = CliRunner().invoke(logfire_group, ["enable"])
    assert result.exit_code == 0, result.output
    config = Config.load()
    assert config.logfire.enabled is True
    assert config.logfire.token == "pylf_v1_us_abcd1234"


def test_enable_masks_token_in_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "thirdeye.commands.logfire_cmd.mint_write_token", lambda: "pylf_v1_us_abcd1234"
    )
    result = CliRunner().invoke(logfire_group, ["enable"])
    assert "abcd1234" not in result.output
    assert "1234" not in result.output
    assert "********" in result.output


def test_enable_rejects_token_command_line_option():
    result = CliRunner().invoke(logfire_group, ["enable", "--token", "secret"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_disable_keeps_token_but_flips_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("thirdeye.commands.logfire_cmd.mint_write_token", lambda: "tok")
    CliRunner().invoke(logfire_group, ["enable"])
    result = CliRunner().invoke(logfire_group, ["disable"])
    assert result.exit_code == 0
    config = Config.load()
    assert config.logfire.enabled is False
    assert config.logfire.token == "tok"


def test_status_reflects_enabled_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("thirdeye.commands.logfire_cmd.mint_write_token", lambda: "tok")
    CliRunner().invoke(logfire_group, ["enable"])
    result = CliRunner().invoke(logfire_group, ["status"])
    assert "enabled           : True" in result.output


def test_enable_rejects_removed_project_option():
    result = CliRunner().invoke(logfire_group, ["enable", "--project", "p"])
    assert result.exit_code != 0
    assert "No such option" in result.output
