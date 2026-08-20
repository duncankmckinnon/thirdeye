from __future__ import annotations

import json

# Use tomllib (3.11+) or tomli (3.10) for verifying written TOML
import tomllib as _toml_read
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TOML_WITH_SECTIONS = """\
[otel.exporter.otlp-http]
endpoint = 'https://example.com/v1/traces'

[marketplaces.x]
url = "https://marketplace.example.com"

[plugins."my-plugin"]
enabled = true
version = "1.2.3"
"""

SAMPLE_TOML_WITH_NOTIFY = """\
notify = ['/some/other/tool']

[otel.exporter.otlp-http]
endpoint = 'https://example.com/v1/traces'
"""


# ---------------------------------------------------------------------------
# TestCodexPlatformAttributes
# ---------------------------------------------------------------------------


class TestCodexPlatformAttributes:
    def test_name_is_codex(self):
        from thirdeye.platforms.codex.install import CodexPlatform

        p = CodexPlatform(config_file=Path("/fake/config.toml"))
        assert p.name == "codex"

    def test_display_name(self):
        from thirdeye.platforms.codex.install import CodexPlatform

        p = CodexPlatform(config_file=Path("/fake/config.toml"))
        assert p.display_name == "Codex CLI"

    def test_is_platform_subclass(self):
        from thirdeye.platforms.base import Platform
        from thirdeye.platforms.codex.install import CodexPlatform

        assert issubclass(CodexPlatform, Platform)

    def test_default_config_file_matches_constants(self):
        from thirdeye.platforms.codex import install
        from thirdeye.platforms.codex.install import CodexPlatform

        p = CodexPlatform()
        # Compare against install's own imported binding, not constants.py's
        # directly: the autouse _never_touch_real_platform_configs fixture
        # patches the former (what __init__'s default actually resolves)
        # for every test in the suite, not the latter.
        assert p._config_file == install.CODEX_CONFIG_FILE


# ---------------------------------------------------------------------------
# TestInstallFreshFile
# ---------------------------------------------------------------------------


class TestInstallFreshFile:
    def test_writes_notify_array(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        CodexPlatform(config_file=config_file).install()
        assert config_file.exists()
        data = _toml_read.loads(config_file.read_text())
        assert "notify" in data
        assert isinstance(data["notify"], list)
        assert len(data["notify"]) == 1
        assert "thirdeye-codex-notify" in data["notify"][0]

    def test_creates_parent_dir(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "nested" / "deeper" / "config.toml"
        CodexPlatform(config_file=config_file).install()
        assert config_file.exists()

    def test_file_ends_with_newline(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        CodexPlatform(config_file=config_file).install()
        assert config_file.read_text().endswith("\n")

    def test_written_file_is_valid_toml(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        CodexPlatform(config_file=config_file).install()
        # Should not raise
        _toml_read.loads(config_file.read_text())


# ---------------------------------------------------------------------------
# TestInstallExistingNoNotify
# ---------------------------------------------------------------------------


class TestInstallExistingNoNotify:
    def test_adds_notify_line(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert "notify" in data
        assert "thirdeye-codex-notify" in data["notify"][0]

    def test_preserves_otel_section(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert "otel" in data
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"

    def test_preserves_marketplaces_section(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert "marketplaces" in data
        assert data["marketplaces"]["x"]["url"] == "https://marketplace.example.com"

    def test_preserves_plugins_section(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert "plugins" in data
        assert data["plugins"]["my-plugin"]["enabled"] is True


# ---------------------------------------------------------------------------
# TestInstallExistingNotify
# ---------------------------------------------------------------------------


class TestInstallExistingNotify:
    def test_foreign_notify_raises(self, tmp_path: Path, monkeypatch):
        """A foreign notify program means conflict: never append, never overwrite."""
        import click

        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        with pytest.raises(click.ClickException):
            CodexPlatform(config_file=config_file).install()

    def test_foreign_notify_leaves_file_byte_identical(self, tmp_path: Path, monkeypatch):
        import click

        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        before = config_file.read_bytes()
        with pytest.raises(click.ClickException):
            CodexPlatform(config_file=config_file).install()
        assert config_file.read_bytes() == before

    def test_conflict_error_names_incumbent(self, tmp_path: Path, monkeypatch):
        import click

        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        with pytest.raises(click.ClickException) as exc_info:
            CodexPlatform(config_file=config_file).install()
        assert "/some/other/tool" in str(exc_info.value)

    def test_force_overwrites_foreign_notify(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        CodexPlatform(config_file=config_file, force=True).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == ["thirdeye-codex-notify"]

    def test_force_preserves_other_content(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        CodexPlatform(config_file=config_file, force=True).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"


# ---------------------------------------------------------------------------
# TestInstallIdempotent
# ---------------------------------------------------------------------------


class TestInstallIdempotent:
    def test_no_duplicate_on_second_install(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.install()
        data = _toml_read.loads(config_file.read_text())
        notify_entries = [x for x in data["notify"] if "thirdeye-codex-notify" in x]
        assert len(notify_entries) == 1

    def test_byte_identical_after_double_install(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        first = config_file.read_bytes()
        p.install()
        second = config_file.read_bytes()
        assert first == second

    def test_idempotent_after_force_over_existing_notify(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        p = CodexPlatform(config_file=config_file, force=True)
        p.install()
        first = config_file.read_bytes()
        p.install()
        second = config_file.read_bytes()
        assert first == second

    def test_triple_install_no_duplicates(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.install()
        p.install()
        data = _toml_read.loads(config_file.read_text())
        assert len(data["notify"]) == 1


# ---------------------------------------------------------------------------
# TestUninstallRemovesNotify
# ---------------------------------------------------------------------------


class TestUninstallRemovesNotify:
    def test_removes_our_entry(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.uninstall()
        text = config_file.read_text() if config_file.exists() else ""
        if text.strip():
            data = _toml_read.loads(text)
            assert "notify" not in data or "thirdeye-codex-notify" not in str(
                data.get("notify", [])
            )
        # If file is empty or gone, that's fine too

    def test_leaves_foreign_notify_alone(self, tmp_path: Path, monkeypatch):
        """uninstall must not touch a notify slot owned by a foreign program."""
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        CodexPlatform(config_file=config_file).uninstall()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == ["/some/other/tool"]

    def test_drops_notify_line_entirely_when_empty(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.uninstall()
        text = config_file.read_text() if config_file.exists() else ""
        assert "notify" not in text

    def test_deletes_file_if_becomes_empty(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.uninstall()
        # File should be deleted or contain only whitespace
        if config_file.exists():
            assert config_file.read_text().strip() == ""


# ---------------------------------------------------------------------------
# TestUninstallEdgeCases
# ---------------------------------------------------------------------------


class TestUninstallEdgeCases:
    def test_file_does_not_exist_is_noop(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        # Don't create the file
        CodexPlatform(config_file=config_file).uninstall()
        assert not config_file.exists()

    def test_file_empty_is_noop(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        CodexPlatform(config_file=config_file).uninstall()
        # Should not raise, file stays as-is or is removed

    def test_file_has_only_foreign_notify_entries_is_noop(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text("notify = ['/some/other/tool', '/another/tool']\n")
        CodexPlatform(config_file=config_file).uninstall()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == ["/some/other/tool", "/another/tool"]

    def test_uninstall_idempotent(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.uninstall()
        p.uninstall()  # Second uninstall should not error

    def test_uninstall_no_notify_line_is_noop(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        original = config_file.read_text()
        CodexPlatform(config_file=config_file).uninstall()
        assert config_file.read_text() == original


# ---------------------------------------------------------------------------
# TestResolveCommandAbsolutePath
# ---------------------------------------------------------------------------


class TestResolveCommandAbsolutePath:
    def test_install_uses_absolute_path_when_which_resolves(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr(
            "thirdeye.platforms.codex.install.shutil.which",
            lambda name: f"/usr/local/bin/{name}",
        )
        config_file = tmp_path / "config.toml"
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == ["/usr/local/bin/thirdeye-codex-notify"]

    def test_install_falls_back_to_bare_name_when_which_fails(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == ["thirdeye-codex-notify"]

    def test_uninstall_removes_absolute_path_variant(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr(
            "thirdeye.platforms.codex.install.shutil.which",
            lambda name: f"/usr/local/bin/{name}",
        )
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.uninstall()
        text = config_file.read_text() if config_file.exists() else ""
        assert "/usr/local/bin/thirdeye-codex-notify" not in text

    def test_uninstall_removes_bare_name_variant(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        # Install with bare name
        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        # Now uninstall with which resolving (should still remove bare-name entry)
        monkeypatch.setattr(
            "thirdeye.platforms.codex.install.shutil.which",
            lambda name: f"/usr/local/bin/{name}",
        )
        p.uninstall()
        text = config_file.read_text() if config_file.exists() else ""
        assert "thirdeye-codex-notify" not in text

    def test_uninstall_removes_whole_line_when_we_own_slot_0(self, tmp_path: Path, monkeypatch):
        """When thirdeye owns slot 0, the whole notify value is ours to remove."""
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "notify = ['thirdeye-codex-notify', '/usr/local/bin/thirdeye-codex-notify',"
            " '/some/other/tool']\n"
        )
        monkeypatch.setattr(
            "thirdeye.platforms.codex.install.shutil.which",
            lambda name: f"/usr/local/bin/{name}",
        )
        CodexPlatform(config_file=config_file).uninstall()
        text = config_file.read_text() if config_file.exists() else ""
        assert "notify" not in text

    def test_idempotent_with_absolute_paths(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr(
            "thirdeye.platforms.codex.install.shutil.which",
            lambda name: f"/opt/bin/{name}",
        )
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        first = config_file.read_bytes()
        p.install()
        second = config_file.read_bytes()
        assert first == second


# ---------------------------------------------------------------------------
# TestPreservesNonNotifyContent
# ---------------------------------------------------------------------------


class TestPreservesNonNotifyContent:
    def test_install_preserves_all_sections(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        CodexPlatform(config_file=config_file).install()
        text = config_file.read_text()
        data = _toml_read.loads(text)
        # All original sections preserved
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"
        assert data["marketplaces"]["x"]["url"] == "https://marketplace.example.com"
        assert data["plugins"]["my-plugin"]["enabled"] is True
        assert data["plugins"]["my-plugin"]["version"] == "1.2.3"

    def test_uninstall_preserves_all_sections(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.uninstall()
        text = config_file.read_text()
        data = _toml_read.loads(text)
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"
        assert data["marketplaces"]["x"]["url"] == "https://marketplace.example.com"
        assert data["plugins"]["my-plugin"]["enabled"] is True
        assert data["plugins"]["my-plugin"]["version"] == "1.2.3"

    def test_install_uninstall_roundtrip_no_section_changes(self, tmp_path: Path, monkeypatch):
        """After install+uninstall, no section content changed except the notify line."""
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_SECTIONS)
        original_data = _toml_read.loads(config_file.read_text())
        p = CodexPlatform(config_file=config_file)
        p.install()
        p.uninstall()
        final_data = _toml_read.loads(config_file.read_text())
        # Remove notify key from comparison if present
        original_data.pop("notify", None)
        final_data.pop("notify", None)
        assert original_data == final_data

    def test_force_install_uninstall_preserves_content(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_NOTIFY)
        CodexPlatform(config_file=config_file, force=True).install()
        CodexPlatform(config_file=config_file).uninstall()
        data = _toml_read.loads(config_file.read_text())
        # We took over the slot, so on uninstall notify is removed entirely.
        assert "notify" not in data
        # OTel section still there
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"


# ---------------------------------------------------------------------------
# TestHelpers - _parse_notify_array and _format_notify_array
# ---------------------------------------------------------------------------


class TestParseNotifyArray:
    def test_parses_single_quoted_strings(self):
        from thirdeye.platforms.codex.install import _parse_notify_array

        result = _parse_notify_array("['hello', 'world']")
        assert result == ["hello", "world"]

    def test_parses_double_quoted_strings(self):
        from thirdeye.platforms.codex.install import _parse_notify_array

        result = _parse_notify_array('["hello", "world"]')
        assert result == ["hello", "world"]

    def test_parses_mixed_quotes(self):
        from thirdeye.platforms.codex.install import _parse_notify_array

        result = _parse_notify_array("""['single', "double"]""")
        assert result == ["single", "double"]

    def test_parses_empty_array(self):
        from thirdeye.platforms.codex.install import _parse_notify_array

        result = _parse_notify_array("[]")
        assert result == []

    def test_parses_paths_with_slashes(self):
        from thirdeye.platforms.codex.install import _parse_notify_array

        result = _parse_notify_array("['/usr/local/bin/tool', '/opt/bin/other']")
        assert result == ["/usr/local/bin/tool", "/opt/bin/other"]

    def test_parses_no_space_between_items(self):
        from thirdeye.platforms.codex.install import _parse_notify_array

        result = _parse_notify_array("['a','b','c']")
        assert result == ["a", "b", "c"]


class TestFormatNotifyArray:
    def test_formats_single_item(self):
        from thirdeye.platforms.codex.install import _format_notify_array

        result = _format_notify_array(["thirdeye-codex-notify"])
        assert result == "notify = ['thirdeye-codex-notify']"

    def test_formats_multiple_items(self):
        from thirdeye.platforms.codex.install import _format_notify_array

        result = _format_notify_array(["/some/tool", "thirdeye-codex-notify"])
        assert result == "notify = ['/some/tool', 'thirdeye-codex-notify']"

    def test_formats_empty_list(self):
        from thirdeye.platforms.codex.install import _format_notify_array

        result = _format_notify_array([])
        assert result == "notify = []"

    def test_output_is_valid_toml(self):
        from thirdeye.platforms.codex.install import _format_notify_array

        result = _format_notify_array(["/usr/local/bin/thirdeye-codex-notify"])
        # Add newline for valid TOML doc
        data = _toml_read.loads(result + "\n")
        assert data["notify"] == ["/usr/local/bin/thirdeye-codex-notify"]

    def test_escapes_single_quotes_in_values(self):
        from thirdeye.platforms.codex.install import _format_notify_array

        result = _format_notify_array(["it's a test"])
        assert "\\'" in result or "it's a test" not in result


# ---------------------------------------------------------------------------
# Single-argv semantics (fixed command via a shared fixture)
# ---------------------------------------------------------------------------

FAKE_CMD = "/fake/bin/thirdeye-codex-notify"

# A foreign multi-element notify — the real dispatcher shape on the dev machine.
SAMPLE_TOML_FOREIGN_DISPATCHER = """\
notify = ['/opt/other/dispatcher', 'turn-ended']

[otel.exporter.otlp-http]
endpoint = 'https://example.com/v1/traces'
"""

# A notify array written across multiple lines.
SAMPLE_TOML_MULTILINE_FOREIGN = """\
notify = [
  "/opt/other/dispatcher",
  "turn-ended"
]

[otel.exporter.otlp-http]
endpoint = 'https://example.com/v1/traces'
"""

SAMPLE_TOML_MULTILINE_OURS = """\
notify = [
  "/fake/bin/thirdeye-codex-notify"
]

[otel.exporter.otlp-http]
endpoint = 'https://example.com/v1/traces'
"""

# Content with comments and a top-level model key, to guard preservation.
SAMPLE_TOML_WITH_COMMENTS = """\
# thirdeye config
model = "gpt-5.6-sol"  # inline comment

[otel.exporter.otlp-http]
endpoint = 'https://example.com/v1/traces'
"""


@pytest.fixture
def fixed_cmd(monkeypatch):
    """Resolve the notify command deterministically to FAKE_CMD."""
    monkeypatch.setattr(
        "thirdeye.platforms.codex.install.shutil.which",
        lambda _: FAKE_CMD,
    )
    return FAKE_CMD


class TestSingleArgvSemantics:
    def test_absent_notify_sets_our_cmd(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == [FAKE_CMD]

    def test_empty_array_sets_our_cmd(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text("notify = []\n")
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == [FAKE_CMD]

    def test_already_our_cmd_is_byte_identical(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(f"notify = ['{FAKE_CMD}']\n")
        before = config_file.read_bytes()
        CodexPlatform(config_file=config_file).install()
        assert config_file.read_bytes() == before

    def test_bare_name_when_which_returns_none(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        CodexPlatform(config_file=config_file).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == ["thirdeye-codex-notify"]

    def test_foreign_multi_element_raises(self, tmp_path: Path, fixed_cmd):
        import click

        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_FOREIGN_DISPATCHER)
        before = config_file.read_bytes()
        with pytest.raises(click.ClickException):
            CodexPlatform(config_file=config_file).install()
        assert config_file.read_bytes() == before

    def test_force_takes_over_foreign_multi_element(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_FOREIGN_DISPATCHER)
        CodexPlatform(config_file=config_file, force=True).install()
        data = _toml_read.loads(config_file.read_text())
        assert data["notify"] == [FAKE_CMD]
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"


class TestMultilineNotify:
    def test_multiline_foreign_raises_byte_identical(self, tmp_path: Path, fixed_cmd):
        import click

        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_MULTILINE_FOREIGN)
        before = config_file.read_bytes()
        with pytest.raises(click.ClickException):
            CodexPlatform(config_file=config_file).install()
        assert config_file.read_bytes() == before

    def test_multiline_ours_install_is_noop(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_MULTILINE_OURS)
        before = config_file.read_bytes()
        CodexPlatform(config_file=config_file).install()
        assert config_file.read_bytes() == before

    def test_multiline_ours_uninstall_removes(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_MULTILINE_OURS)
        CodexPlatform(config_file=config_file).uninstall()
        data = _toml_read.loads(config_file.read_text())
        assert "notify" not in data
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"


class TestUninstallArgvOwnership:
    def test_foreign_slot0_with_our_cmd_as_arg_is_noop(self, tmp_path: Path, monkeypatch):
        """The corrupt state the old installer produced: a foreign program owns
        slot 0 and our command trails as a mere argument. Uninstall must not
        touch it — nothing changed."""
        from thirdeye.platforms.codex.install import CodexPlatform

        monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "notify = ['/opt/other/dispatcher', 'turn-ended', 'thirdeye-codex-notify']\n"
        )
        before = config_file.read_bytes()
        CodexPlatform(config_file=config_file).uninstall()
        assert config_file.read_bytes() == before


class TestPreservesCommentsAndModel:
    def test_install_preserves_comments_and_model(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_COMMENTS)
        CodexPlatform(config_file=config_file).install()
        text = config_file.read_text()
        assert "# thirdeye config" in text
        assert "# inline comment" in text
        data = _toml_read.loads(text)
        assert data["model"] == "gpt-5.6-sol"
        assert data["notify"] == [FAKE_CMD]

    def test_force_install_preserves_comments_and_model(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        # Start with a foreign notify plus comments and model.
        config_file.write_text(
            "# thirdeye config\n"
            'model = "gpt-5.6-sol"  # inline comment\n'
            "notify = ['/opt/other/dispatcher', 'turn-ended']\n"
            "\n"
            "[otel.exporter.otlp-http]\n"
            "endpoint = 'https://example.com/v1/traces'\n"
        )
        CodexPlatform(config_file=config_file, force=True).install()
        text = config_file.read_text()
        assert "# thirdeye config" in text
        assert "# inline comment" in text
        data = _toml_read.loads(text)
        assert data["model"] == "gpt-5.6-sol"
        assert data["notify"] == [FAKE_CMD]
        assert data["otel"]["exporter"]["otlp-http"]["endpoint"] == "https://example.com/v1/traces"

    def test_uninstall_preserves_comments_and_model(self, tmp_path: Path, fixed_cmd):
        from thirdeye.platforms.codex.install import CodexPlatform

        config_file = tmp_path / "config.toml"
        config_file.write_text(SAMPLE_TOML_WITH_COMMENTS)
        CodexPlatform(config_file=config_file).install()
        CodexPlatform(config_file=config_file).uninstall()
        text = config_file.read_text()
        assert "# thirdeye config" in text
        assert "# inline comment" in text
        data = _toml_read.loads(text)
        assert data["model"] == "gpt-5.6-sol"
        assert "notify" not in data


# ---------------------------------------------------------------------------
# TestHooksJson — Codex's separate, newer per-event hooks.json mechanism
# ---------------------------------------------------------------------------

SUPPORTED_HOOKS_JSON_EVENTS = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "SubagentStart",
    "SubagentStop",
    "PermissionRequest",
    "PreCompact",
    "PostCompact",
)

# The three events thirdeye deliberately never wires (see hooks_json.py):
# notify's rollout reconstruction already covers them more richly.
UNSUPPORTED_HOOKS_JSON_EVENTS = ("PreToolUse", "PostToolUse", "Stop")


def _no_which(monkeypatch) -> None:
    """So a resolved command is just the bare binary name, deterministically."""
    monkeypatch.setattr("thirdeye.platforms.codex.install.shutil.which", lambda _: None)


def _commands_for(hooks_data: dict, event: str) -> list[str]:
    out: list[str] = []
    for group in hooks_data.get("hooks", {}).get(event, []):
        for entry in group.get("hooks", []):
            out.append(entry["command"])
    return out


def _hooks_json_with(events: dict[str, str]) -> dict:
    return {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": cmd}]}]
            for event, cmd in events.items()
        }
    }


class TestHooksJsonInstall:
    def test_fresh_file_gets_every_supported_event(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        for event in SUPPORTED_HOOKS_JSON_EVENTS:
            commands = _commands_for(data, event)
            assert len(commands) == 1
            assert Path(commands[0]).name.startswith("thirdeye-codex-")

    def test_unsupported_events_are_never_added(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        for event in UNSUPPORTED_HOOKS_JSON_EVENTS:
            assert event not in data.get("hooks", {})

    def test_idempotent_on_repeated_install(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        p = CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file)
        p.install()
        p.install()
        data = json.loads(hooks_file.read_text())
        for event in SUPPORTED_HOOKS_JSON_EVENTS:
            assert len(_commands_for(data, event)) == 1

    def test_foreign_tool_entries_are_preserved_alongside_ours(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps(_hooks_json_with({"SessionStart": "/some/other/tool"})))
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        commands = _commands_for(data, "SessionStart")
        assert "/some/other/tool" in commands
        assert any(Path(c).name == "thirdeye-codex-session-start" for c in commands)

    def test_stale_claude_entry_on_supported_event_is_replaced(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(
            json.dumps(_hooks_json_with({"SessionStart": "/opt/homebrew/bin/thirdeye-claude-session-start"}))
        )
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        commands = _commands_for(data, "SessionStart")
        assert not any("thirdeye-claude-" in c for c in commands)
        assert any(Path(c).name == "thirdeye-codex-session-start" for c in commands)

    def test_stale_claude_entry_on_unsupported_event_is_stripped_not_replaced(
        self, tmp_path: Path, monkeypatch
    ):
        """The exact misconfiguration this feature exists to fix: Codex's
        PreToolUse/PostToolUse/Stop pointed at Claude's own handlers, which
        mislabels every captured Codex session as platform=claude.
        """
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(
            json.dumps(
                _hooks_json_with(
                    {
                        "PreToolUse": "/opt/homebrew/bin/thirdeye-claude-pre-tool-use",
                        "PostToolUse": "/opt/homebrew/bin/thirdeye-claude-post-tool-use",
                        "Stop": "/opt/homebrew/bin/thirdeye-claude-stop",
                    }
                )
            )
        )
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        for event in UNSUPPORTED_HOOKS_JSON_EVENTS:
            assert event not in data.get("hooks", {})

    def test_preserves_a_foreign_entry_on_an_unsupported_event(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps(_hooks_json_with({"Stop": "/some/other/tool"})))
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).install()
        data = json.loads(hooks_file.read_text())
        assert _commands_for(data, "Stop") == ["/some/other/tool"]

    def test_creates_parent_dir(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "nested" / "deeper" / "hooks.json"
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).install()
        assert hooks_file.exists()

    def test_no_op_when_already_correctly_configured(self, tmp_path: Path, monkeypatch):
        """install() must not rewrite (and so not touch the mtime of) a
        hooks.json that's already exactly right.
        """
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        p = CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file)
        p.install()
        before = hooks_file.read_text()
        p.install()
        assert hooks_file.read_text() == before


class TestHooksJsonUninstall:
    def test_removes_only_our_entries(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        p = CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file)
        p.install()
        p.uninstall()
        assert not hooks_file.exists()

    def test_preserves_foreign_entries(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps(_hooks_json_with({"SessionStart": "/some/other/tool"})))
        p = CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file)
        p.install()
        p.uninstall()
        data = json.loads(hooks_file.read_text())
        assert _commands_for(data, "SessionStart") == ["/some/other/tool"]

    def test_missing_file_is_noop(self, tmp_path: Path, monkeypatch):
        from thirdeye.platforms.codex.install import CodexPlatform

        _no_which(monkeypatch)
        hooks_file = tmp_path / "hooks.json"
        CodexPlatform(config_file=tmp_path / "config.toml", hooks_file=hooks_file).uninstall()
        assert not hooks_file.exists()
