from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from thirdeye.commands.capture_env import capture_env_group
from thirdeye.config import Config


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    monkeypatch.delenv("THIRDEYE_CAPTURE_ENV", raising=False)
    return tmp_path


def test_show_when_unconfigured_reports_the_default():
    result = CliRunner().invoke(capture_env_group, ["show"])
    assert result.exit_code == 0
    assert "patterns : WB_*" in result.output
    assert "built-in default" in result.output


def test_set_persists_and_show_reports_config_source():
    r = CliRunner().invoke(capture_env_group, ["set", "WB_*"])
    assert r.exit_code == 0
    assert Config.load().capture_env_patterns == ("WB_*",)

    shown = CliRunner().invoke(capture_env_group, ["show"])
    assert "patterns : WB_*" in shown.output
    assert "config.yaml" in shown.output


def test_set_accepts_multiple_and_comma_forms():
    CliRunner().invoke(capture_env_group, ["set", "WB_*", "BUILD_LABEL"])
    assert Config.load().capture_env_patterns == ("WB_*", "BUILD_LABEL")
    CliRunner().invoke(capture_env_group, ["set", "A_*,B_*"])
    assert Config.load().capture_env_patterns == ("A_*", "B_*")


def test_clear_disables_capture_rather_than_reverting_to_the_default():
    CliRunner().invoke(capture_env_group, ["set", "WB_*"])
    r = CliRunner().invoke(capture_env_group, ["clear"])
    assert r.exit_code == 0
    assert Config.load().capture_env_patterns == ()

    shown = CliRunner().invoke(capture_env_group, ["show"])
    assert "patterns : (none)" in shown.output
    assert "config.yaml" in shown.output


def test_env_var_is_reported_as_overriding(monkeypatch: pytest.MonkeyPatch):
    CliRunner().invoke(capture_env_group, ["set", "WB_*"])
    monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "OTHER_*")
    shown = CliRunner().invoke(capture_env_group, ["show"])
    assert "patterns : OTHER_*" in shown.output
    assert "THIRDEYE_CAPTURE_ENV" in shown.output
