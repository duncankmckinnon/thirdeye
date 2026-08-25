from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.config import Config, LogfireSettings


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


def test_logfire_settings_default_disabled() -> None:
    assert Config(root=Path("/tmp")).logfire == LogfireSettings()
    assert Config(root=Path("/tmp")).logfire.enabled is False


class TestLogfireSettingsPersistence:
    def test_load_reads_persisted_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
        Config.load().write_logfire_settings(LogfireSettings(enabled=True, token="tok"))
        reloaded = Config.load()
        assert reloaded.logfire == LogfireSettings(enabled=True, token="tok")

    def test_load_ignores_legacy_project_and_next_write_removes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.safe_dump({"logfire": {"enabled": True, "token": "tok", "project": "legacy"}})
        )

        config = Config.load()
        assert config.logfire == LogfireSettings(enabled=True, token="tok")

        config.write_logfire_settings(config.logfire)
        assert "project" not in yaml.safe_load(config_file.read_text())["logfire"]

    def test_write_returns_updated_copy(self, tmp_path: Path) -> None:
        config = Config(root=tmp_path)
        updated = config.write_logfire_settings(LogfireSettings(enabled=True, token="t"))
        assert updated.logfire.enabled is True
        assert config.logfire.enabled is False  # original untouched (frozen)

    def test_write_preserves_other_config_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
        config = Config.load()
        config.config_file.parent.mkdir(parents=True, exist_ok=True)
        config.config_file.write_text(yaml.safe_dump({"unrelated": "value"}))
        config.write_logfire_settings(LogfireSettings(enabled=True, token="t"))
        on_disk = yaml.safe_load(config.config_file.read_text())
        assert on_disk["unrelated"] == "value"
        assert on_disk["logfire"]["enabled"] is True

    def test_missing_config_file_yields_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
        assert Config.load().logfire == LogfireSettings()

    def test_malformed_config_file_yields_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("not: valid: yaml: [")
        assert Config.load().logfire == LogfireSettings()
