import os
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.store import Store


@pytest.fixture(autouse=True)
def _never_touch_real_platform_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect every real home-directory path (thirdeye's own config and
    store, Codex's config.toml, hooks.json and sessions, Claude's
    settings.json, Cursor's hooks.json and history, Copilot's home) to this
    test's tmp_path, for every test in the suite — not just ones that
    remember to pass an explicit path to a Platform constructor.

    A CodexPlatform/ClaudePlatform built with no explicit override falls back
    to these module-level constants, so a test that forgets to override one
    (there is no way to statically catch that oversight) would otherwise
    install/uninstall against the developer's own real, live Codex/Claude
    config — as happened once already, wiping a real ~/.codex/hooks.json
    mid-test-run. Redirecting the constants themselves closes that off for
    every call site at once, present and future, rather than relying on each
    test to remember an override.
    """
    fake_home = tmp_path / "_fake_home"
    fake_home.mkdir()

    # Nothing from the shell running pytest may steer a test: a developer's
    # own THIRDEYE_CAPTURE_ENV, THIRDEYE_HOME or COPILOT_HOME would otherwise
    # change what Config.load() and the adapters read.
    for key in [key for key in os.environ if key.startswith("THIRDEYE_")]:
        monkeypatch.delenv(key)
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    # Every runtime `Path.home()` lookup, and thirdeye's own default store,
    # resolve inside the test instead of the developer's real home. Tests that
    # need a specific THIRDEYE_HOME still set their own over this one.
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("THIRDEYE_HOME", str(fake_home / ".thirdeye"))
    # Paths computed from the real home at import time, before this fixture
    # could redirect it. `codex.usage` binds its own copy of the sessions root.
    for module in ("rollout", "usage"):
        monkeypatch.setattr(
            f"thirdeye.platforms.codex.{module}.CODEX_SESSIONS_ROOT",
            fake_home / ".codex" / "sessions",
        )
    monkeypatch.setattr(
        "thirdeye.commands.add.ORPHAN_CONFIG_PATHS",
        (fake_home / ".gemini" / "settings.json",),
    )
    monkeypatch.setattr(
        "thirdeye.platforms.codex.install.CODEX_CONFIG_FILE", fake_home / ".codex" / "config.toml"
    )
    monkeypatch.setattr(
        "thirdeye.platforms.codex.install.CODEX_HOOKS_FILE", fake_home / ".codex" / "hooks.json"
    )
    monkeypatch.setattr(
        "thirdeye.platforms.claude.install.SETTINGS_FILE",
        fake_home / ".claude" / "settings.json",
    )
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.install.HOOKS_FILE",
        fake_home / ".cursor" / "hooks.json",
    )
    # Cursor turn recovery reads transcript and chat history separately from
    # hook configuration. Keep every test hermetic unless it deliberately
    # supplies synthetic roots.
    monkeypatch.setenv("THIRDEYE_CURSOR_TRANSCRIPT_ROOTS", str(fake_home / "transcripts"))
    monkeypatch.setenv("THIRDEYE_CURSOR_CHAT_ROOTS", str(fake_home / "chats"))


@pytest.fixture
def tmp_store(tmp_path: Path) -> Store:
    return Store(Config(root=tmp_path))


@pytest.fixture
def populated_store(tmp_store: Store) -> Store:
    """Two sessions: one for `claude`, one for `cursor`."""
    with tmp_store.open_session("01J9G7XK4P", platform="claude", cwd="/proj/a") as w:
        w.append("user_message", "hi from claude")
        w.append("assistant_message", "hello user")
    with tmp_store.open_session("02ABCDEF12", platform="cursor", cwd="/proj/b") as w:
        w.append("user_message", "hi from cursor")
    return tmp_store
