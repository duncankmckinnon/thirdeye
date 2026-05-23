from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.config import Config


def test_load_without_capture_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THIRDEYE_CAPTURE_ENV", raising=False)
    config = Config.load()
    assert config.capture_env_patterns == ()


def test_load_with_capture_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*,OTHER")
    config = Config.load()
    assert config.capture_env_patterns == ("WB_*", "OTHER")


def test_load_tolerates_whitespace_and_empties(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "  WB_* , , OTHER ,")
    config = Config.load()
    assert config.capture_env_patterns == ("WB_*", "OTHER")


def test_config_positional_root_default_patterns() -> None:
    config = Config(root=Path("/tmp"))
    assert config.capture_env_patterns == ()


def test_config_is_hashable_with_patterns() -> None:
    config = Config(root=Path("/tmp"), capture_env_patterns=("X",))
    assert {config} == {config}
