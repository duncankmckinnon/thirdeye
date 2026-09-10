from __future__ import annotations

import json
import shlex
from pathlib import Path

import click
import pytest

from thirdeye.platforms.base import Platform
from thirdeye.platforms.copilot.constants import (
    CLI_HOOK_EVENTS,
    DISPLAY_NAME,
    HOOK_BIN_NAME,
    HOOK_CONFIG_VERSION,
    HOOK_TIMEOUT_S,
    HOOKS_DIRECTORY_NAME,
    OWNED_HOOK_FILENAME,
    PLATFORM_NAME,
)
from thirdeye.platforms.copilot.install import CopilotPlatform


def _platform(
    tmp_path: Path,
    *,
    hooks_file: Path | None = None,
    source_home: Path | None = None,
    entrypoint: str | None = None,
    windows: bool | None = None,
) -> CopilotPlatform:
    return CopilotPlatform(
        source_home=source_home,
        hooks_file=hooks_file,
        entrypoint=entrypoint or "/opt/thirdeye/bin/thirdeye-copilot-hook",
        windows=windows,
    )


def _expected_entry(
    platform: CopilotPlatform,
    event: str,
) -> dict[str, object]:
    shell, command = platform._command(event)
    return {
        "type": "command",
        shell: command,
        "timeoutSec": HOOK_TIMEOUT_S,
    }


class TestCopilotPlatformAttributes:
    def test_name_and_display_name(self, tmp_path: Path):
        platform = _platform(tmp_path)
        assert platform.name == PLATFORM_NAME
        assert platform.display_name == DISPLAY_NAME

    def test_is_platform_subclass(self):
        assert issubclass(CopilotPlatform, Platform)


class TestInstallFreshFile:
    def test_registers_every_supported_event(self, tmp_path: Path):
        path = tmp_path / "hooks" / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path)
        platform.install()
        data = json.loads(path.read_text())
        assert data["version"] == HOOK_CONFIG_VERSION
        assert set(data["hooks"]) == set(CLI_HOOK_EVENTS)
        for event in CLI_HOOK_EVENTS:
            assert data["hooks"][event] == [_expected_entry(platform, event)]

    def test_creates_parent_directory(self, tmp_path: Path):
        path = tmp_path / "nested" / "hooks" / OWNED_HOOK_FILENAME
        _platform(tmp_path, hooks_file=path).install()
        assert path.exists()

    def test_output_is_valid_json_with_trailing_newline(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        _platform(tmp_path, hooks_file=path).install()
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        json.loads(text)


class TestInstallIdempotent:
    def test_double_install_does_not_duplicate_entries(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path)
        platform.install()
        first = json.loads(path.read_text())
        platform.install()
        second = json.loads(path.read_text())
        assert first == second
        for event in CLI_HOOK_EVENTS:
            assert len(second["hooks"][event]) == 1

    def test_install_then_is_installed(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path)
        assert platform.is_installed() is False
        platform.install()
        assert platform.is_installed() is True


class TestInstallMergeAndUpgrade:
    def test_preserves_foreign_hooks_and_extra_fields(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        foreign = {
            "type": "command",
            "bash": "/opt/foreign-hook sessionStart",
            "timeoutSec": 17,
        }
        path.write_text(
            json.dumps(
                {
                    "version": HOOK_CONFIG_VERSION,
                    "theme": "dark",
                    "hooks": {"sessionStart": [foreign]},
                }
            )
        )
        platform = _platform(tmp_path, hooks_file=path)
        platform.install()
        data = json.loads(path.read_text())
        assert data["theme"] == "dark"
        assert foreign in data["hooks"]["sessionStart"]
        assert _expected_entry(platform, "sessionStart") in data["hooks"]["sessionStart"]

    def test_replaces_stale_owned_command_without_duplicating(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        stale = {
            "type": "command",
            "bash": "/old/path/thirdeye-copilot-hook sessionStart",
            "timeoutSec": HOOK_TIMEOUT_S,
        }
        path.write_text(
            json.dumps({"version": HOOK_CONFIG_VERSION, "hooks": {"sessionStart": [stale]}})
        )
        platform = _platform(tmp_path, hooks_file=path, entrypoint="/new/path/thirdeye-copilot-hook")
        platform.install()
        data = json.loads(path.read_text())
        commands = [entry.get("bash") for entry in data["hooks"]["sessionStart"]]
        assert commands.count(_expected_entry(platform, "sessionStart")["bash"]) == 1
        assert stale["bash"] not in commands

    def test_upgrades_partial_install_to_full_event_set(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path)
        partial = {
            "version": HOOK_CONFIG_VERSION,
            "hooks": {
                "sessionStart": [_expected_entry(platform, "sessionStart")],
                "sessionEnd": [_expected_entry(platform, "sessionEnd")],
            },
        }
        path.write_text(json.dumps(partial))
        platform.install()
        data = json.loads(path.read_text())
        assert set(data["hooks"]) == set(CLI_HOOK_EVENTS)


class TestUninstall:
    def test_removes_only_owned_entries_and_deletes_empty_file(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path)
        platform.install()
        platform.uninstall()
        assert not path.exists()

    def test_uninstall_preserves_foreign_hooks_and_extra_fields(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        foreign = {
            "type": "command",
            "bash": "/opt/foreign-hook agentStop",
            "timeoutSec": 12,
        }
        path.write_text(
            json.dumps(
                {
                    "version": HOOK_CONFIG_VERSION,
                    "notes": "keep me",
                    "hooks": {"agentStop": [foreign]},
                }
            )
        )
        platform = _platform(tmp_path, hooks_file=path)
        platform.install()
        platform.uninstall()
        data = json.loads(path.read_text())
        assert data["notes"] == "keep me"
        assert data["hooks"] == {"agentStop": [foreign]}

    def test_uninstall_on_missing_file_is_noop(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        _platform(tmp_path, hooks_file=path).uninstall()


class TestShellQuoting:
    def test_bash_quotes_paths_with_spaces(self, tmp_path: Path):
        entrypoint = "/opt/my tools/thirdeye-copilot-hook"
        platform = _platform(tmp_path, entrypoint=entrypoint, windows=False)
        _, command = platform._command("sessionStart")
        parts = shlex.split(command, posix=True)
        assert parts == [entrypoint, "sessionStart"]

    def test_powershell_quotes_paths_with_spaces(self, tmp_path: Path):
        entrypoint = r"C:\Users\First Last\tools\thirdeye-copilot-hook.exe"
        platform = _platform(tmp_path, entrypoint=entrypoint, windows=True)
        shell, command = platform._command("userPromptSubmitted")
        assert shell == "powershell"
        assert command.startswith("& ")
        assert entrypoint in command
        assert "'userPromptSubmitted'" in command

    def test_install_writes_bash_commands_on_posix(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        entrypoint = "/opt/my tools/thirdeye-copilot-hook"
        platform = _platform(tmp_path, hooks_file=path, entrypoint=entrypoint, windows=False)
        platform.install()
        entry = json.loads(path.read_text())["hooks"]["preToolUse"][0]
        assert "bash" in entry
        assert "powershell" not in entry
        assert shlex.split(entry["bash"], posix=True) == [entrypoint, "preToolUse"]

    def test_install_writes_powershell_commands_on_windows(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        entrypoint = r"C:\Program Files\Thirdeye\thirdeye-copilot-hook.exe"
        platform = _platform(tmp_path, hooks_file=path, entrypoint=entrypoint, windows=True)
        platform.install()
        entry = json.loads(path.read_text())["hooks"]["postToolUse"][0]
        assert entry["powershell"].startswith("& ")
        assert entrypoint in entry["powershell"]


class TestMalformedConfig:
    @pytest.mark.parametrize(
        "payload, needle",
        [
            ("{not json", "not valid JSON"),
            (json.dumps([]), "must be a JSON object"),
            (json.dumps({"version": 2, "hooks": {}}), "expected version"),
            (json.dumps({"version": 1, "hooks": []}), "'hooks' must be an object"),
            (
                json.dumps({"version": 1, "hooks": {"sessionStart": "nope"}}),
                "hooks.sessionStart must be a list",
            ),
        ],
    )
    def test_install_refuses_malformed_document_and_preserves_bytes(
        self,
        tmp_path: Path,
        payload: str,
        needle: str,
    ):
        path = tmp_path / OWNED_HOOK_FILENAME
        path.write_text(payload)
        before = path.read_bytes()
        with pytest.raises(click.ClickException) as exc_info:
            _platform(tmp_path, hooks_file=path).install()
        assert needle in str(exc_info.value)
        assert path.read_bytes() == before

    def test_is_installed_returns_false_for_malformed_document(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        path.write_text("{broken")
        platform = _platform(tmp_path, hooks_file=path)
        assert platform.is_installed() is False


class TestSourceHomeScope:
    def test_resolves_hooks_file_under_source_home(self, tmp_path: Path):
        source_home = tmp_path / "custom-copilot-home"
        platform = CopilotPlatform(
            source_home=source_home,
            entrypoint="/opt/bin/thirdeye-copilot-hook",
            windows=False,
        )
        expected = source_home / HOOKS_DIRECTORY_NAME / OWNED_HOOK_FILENAME
        assert platform.hooks_file == expected.resolve()
        platform.install()
        assert expected.exists()

    def test_explicit_hooks_file_overrides_source_home(self, tmp_path: Path):
        override = tmp_path / "override" / OWNED_HOOK_FILENAME
        platform = CopilotPlatform(
            source_home=tmp_path / "ignored-home",
            hooks_file=override,
            entrypoint="/opt/bin/thirdeye-copilot-hook",
            windows=False,
        )
        assert platform.hooks_file == override
        platform.install()
        assert override.exists()
        assert not (tmp_path / "ignored-home").exists()


class TestInstallStateChecks:
    def test_requires_every_event_to_be_installed(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path)
        platform.install()
        data = json.loads(path.read_text())
        data["hooks"].pop(CLI_HOOK_EVENTS[0])
        path.write_text(json.dumps(data))
        assert platform.is_installed() is False

    def test_is_installed_false_when_timeout_differs(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path)
        platform.install()
        data = json.loads(path.read_text())
        data["hooks"]["notification"][0]["timeoutSec"] = 99
        path.write_text(json.dumps(data))
        assert platform.is_installed() is False


class TestWindowsOwnershipRecognition:
    def test_replaces_stale_powershell_owned_command(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        stale = {
            "type": "command",
            "powershell": "& 'C:\\Old Path\\thirdeye-copilot-hook.exe' 'sessionStart'",
            "timeoutSec": HOOK_TIMEOUT_S,
        }
        path.write_text(
            json.dumps({"version": HOOK_CONFIG_VERSION, "hooks": {"sessionStart": [stale]}})
        )
        entrypoint = r"C:\New Path\thirdeye-copilot-hook.exe"
        platform = _platform(tmp_path, hooks_file=path, entrypoint=entrypoint, windows=True)
        platform.install()
        data = json.loads(path.read_text())
        assert data["hooks"]["sessionStart"] == [_expected_entry(platform, "sessionStart")]

    def test_recognizes_double_quoted_powershell_executable(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        owned = {
            "type": "command",
            "powershell": '& "C:\\Tools\\thirdeye-copilot-hook.exe" "agentStop"',
            "timeoutSec": HOOK_TIMEOUT_S,
        }
        path.write_text(
            json.dumps({"version": HOOK_CONFIG_VERSION, "hooks": {"agentStop": [owned]}})
        )
        platform = _platform(
            tmp_path,
            hooks_file=path,
            entrypoint=r"C:\Tools\thirdeye-copilot-hook.exe",
            windows=True,
        )
        platform.install()
        data = json.loads(path.read_text())
        assert len(data["hooks"]["agentStop"]) == 1
        assert platform.is_installed() is True


class TestUninstallEdgeCases:
    def test_leaves_invalid_unknown_event_shape_untouched(self, tmp_path: Path):
        path = tmp_path / OWNED_HOOK_FILENAME
        platform = _platform(tmp_path, hooks_file=path, windows=False)
        platform.install()
        data = json.loads(path.read_text())
        data["hooks"]["customFutureEvent"] = "not-a-list"
        path.write_text(json.dumps(data))
        platform.uninstall()
        remaining = json.loads(path.read_text())
        assert remaining["hooks"]["customFutureEvent"] == "not-a-list"
        assert "sessionStart" not in remaining["hooks"]
