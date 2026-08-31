from __future__ import annotations

import json
from pathlib import Path

from thirdeye.platforms.claude.constants import HOOK_EVENTS
from thirdeye.platforms.claude.install import ClaudePlatform
from thirdeye.platforms.codex.constants import HOOKS_JSON_BIN_NAMES
from thirdeye.platforms.codex.install import CodexPlatform
from thirdeye.platforms.cursor.constants import TRACED_EVENTS
from thirdeye.platforms.cursor.install import CursorPlatform


def test_claude_requires_every_hook_to_be_installed(tmp_path: Path) -> None:
    settings_file = tmp_path / "claude.json"
    platform = ClaudePlatform(settings_file=settings_file)
    assert platform.is_installed() is False

    platform.install()
    assert platform.is_installed() is True

    data = json.loads(settings_file.read_text())
    data["hooks"].pop(next(iter(HOOK_EVENTS)))
    settings_file.write_text(json.dumps(data))
    assert platform.is_installed() is False


def test_cursor_requires_every_hook_to_be_installed(tmp_path: Path) -> None:
    hooks_file = tmp_path / "cursor.json"
    platform = CursorPlatform(hooks_file=hooks_file)
    assert platform.is_installed() is False

    platform.install()
    assert platform.is_installed() is True

    data = json.loads(hooks_file.read_text())
    data["hooks"].pop(TRACED_EVENTS[0])
    hooks_file.write_text(json.dumps(data))
    assert platform.is_installed() is False


def test_codex_requires_notify_and_every_json_hook(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    hooks_file = tmp_path / "codex-hooks.json"
    platform = CodexPlatform(config_file=config_file, hooks_file=hooks_file)
    assert platform.is_installed() is False

    platform.install()
    assert platform.is_installed() is True

    data = json.loads(hooks_file.read_text())
    data["hooks"].pop(next(iter(HOOKS_JSON_BIN_NAMES)))
    hooks_file.write_text(json.dumps(data))
    assert platform.is_installed() is False


def test_installed_checks_tolerate_malformed_hook_groups(tmp_path: Path) -> None:
    claude_file = tmp_path / "claude.json"
    claude_file.write_text(
        json.dumps({"hooks": {event: [{"hooks": None}] for event in HOOK_EVENTS}})
    )
    assert ClaudePlatform(settings_file=claude_file).is_installed() is False

    codex_config = tmp_path / "config.toml"
    codex_config.write_text("notify = ['thirdeye-codex-notify']\n")
    codex_hooks = tmp_path / "codex.json"
    codex_hooks.write_text(
        json.dumps({"hooks": {event: [{"hooks": None}] for event in HOOKS_JSON_BIN_NAMES}})
    )
    assert CodexPlatform(config_file=codex_config, hooks_file=codex_hooks).is_installed() is False
