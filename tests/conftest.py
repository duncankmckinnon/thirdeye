from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.store import Store


@pytest.fixture(autouse=True)
def _never_touch_real_platform_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect every platform's real home-directory config path (Codex's
    config.toml and hooks.json, Claude's settings.json) to this test's
    tmp_path, for every test in the suite — not just ones that remember to
    pass an explicit path to a Platform constructor.

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
