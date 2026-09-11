from __future__ import annotations

from thirdeye.platforms.codex.constants import (
    CODEX_CONFIG_DIR,
    CODEX_CONFIG_FILE,
    DISPLAY_NAME,
    NOTIFY_BIN_NAME,
    PLATFORM_NAME,
)


def test_platform_name():
    assert PLATFORM_NAME == "codex"


def test_display_name():
    assert DISPLAY_NAME == "Codex CLI"


def test_config_file_under_codex_dir():
    parts = CODEX_CONFIG_FILE.parts
    assert ".codex" in parts
    assert parts[-1] == "config.toml"


def test_config_dir_is_parent_of_config_file():
    assert CODEX_CONFIG_FILE.parent == CODEX_CONFIG_DIR


def test_notify_bin_name():
    assert NOTIFY_BIN_NAME == "thirdeye-codex-notify"
