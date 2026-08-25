from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from thirdeye.commands.logfire_cmd import logfire_group
from thirdeye.config import Config


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


def test_enable_persists_token_and_project():
    result = CliRunner().invoke(
        logfire_group, ["enable", "--project", "myproj"], input="pylf_v1_us_abcd1234\n"
    )
    assert result.exit_code == 0, result.output
    config = Config.load()
    assert config.logfire.enabled is True
    assert config.logfire.token == "pylf_v1_us_abcd1234"
    assert config.logfire.project == "myproj"


def test_enable_masks_token_in_output():
    result = CliRunner().invoke(logfire_group, ["enable"], input="pylf_v1_us_abcd1234\n")
    assert "abcd1234" not in result.output
    assert "1234" not in result.output
    assert "********" in result.output


def test_enable_rejects_token_command_line_option():
    result = CliRunner().invoke(logfire_group, ["enable", "--token", "secret"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_disable_keeps_token_but_flips_flag():
    CliRunner().invoke(logfire_group, ["enable", "--project", "p"], input="tok\n")
    result = CliRunner().invoke(logfire_group, ["disable"])
    assert result.exit_code == 0
    config = Config.load()
    assert config.logfire.enabled is False
    assert config.logfire.token == "tok"
    assert config.logfire.project == "p"


def test_status_reflects_enabled_state():
    CliRunner().invoke(logfire_group, ["enable", "--project", "p"], input="tok\n")
    result = CliRunner().invoke(logfire_group, ["status"])
    assert "enabled           : True" in result.output
    assert "project           : p" in result.output
