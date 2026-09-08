from __future__ import annotations

import pytest

from thirdeye.platforms.base import Platform, command_matches, resolve_command


class TestPlatformIsAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Platform()


class TestSubclassMustImplement:
    def test_missing_install_and_uninstall(self):
        class Incomplete(Platform):
            name = "x"
            display_name = "X"

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_uninstall(self):
        class MissingUninstall(Platform):
            name = "x"
            display_name = "X"

            def install(self) -> None: ...

        with pytest.raises(TypeError):
            MissingUninstall()

    def test_missing_install(self):
        class MissingInstall(Platform):
            name = "x"
            display_name = "X"

            def uninstall(self) -> None: ...

        with pytest.raises(TypeError):
            MissingInstall()


class TestConcreteSubclass:
    def test_can_instantiate(self):
        class Concrete(Platform):
            name = "test"
            display_name = "Test Platform"

            def install(self) -> None: ...
            def uninstall(self) -> None: ...

        p = Concrete()
        assert p.name == "test"
        assert p.display_name == "Test Platform"

    def test_install_is_callable(self):
        class Concrete(Platform):
            name = "test"
            display_name = "Test Platform"

            def install(self) -> None: ...
            def uninstall(self) -> None: ...

        p = Concrete()
        assert p.install() is None

    def test_uninstall_is_callable(self):
        class Concrete(Platform):
            name = "test"
            display_name = "Test Platform"

            def install(self) -> None: ...
            def uninstall(self) -> None: ...

        p = Concrete()
        assert p.uninstall() is None


def test_command_matches_bare_and_absolute():
    bin_name = "thirdeye-claude-session-start"

    assert command_matches(bin_name, bin_name)
    assert command_matches(f"/usr/local/bin/{bin_name}", bin_name)


def test_command_matches_exe_only_on_windows(monkeypatch):
    command = r"C:\Users\thirdeye\Scripts\thirdeye-claude-session-start.exe"
    bin_name = "thirdeye-claude-session-start"

    monkeypatch.setattr("thirdeye._compat.IS_WINDOWS", True)
    assert command_matches(command, bin_name)

    monkeypatch.setattr("thirdeye._compat.IS_WINDOWS", False)
    assert not command_matches(command, bin_name)


def test_command_matches_rejects_shell_wrapper():
    assert not command_matches("thirdeye-claude-session-start.sh", "thirdeye-claude-session-start")


def test_command_matches_rejects_non_strings():
    assert not command_matches(None, "thirdeye-claude-session-start")
    assert not command_matches(42, "thirdeye-claude-session-start")


def test_resolve_command_falls_back_only_on_windows(monkeypatch):
    bin_name = "thirdeye-claude-session-start"
    resolved = f"/Users/First Last/.local/bin/{bin_name}"
    monkeypatch.setattr("thirdeye.platforms.base.shutil.which", lambda _: resolved)

    monkeypatch.setattr("thirdeye._compat.IS_WINDOWS", False)
    assert resolve_command(bin_name) == resolved

    monkeypatch.setattr("thirdeye._compat.IS_WINDOWS", True)
    assert resolve_command(bin_name) == bin_name
