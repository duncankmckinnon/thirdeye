"""Install / uninstall Grok Bot — arms FS store observer (no Cursor hooks)."""

from __future__ import annotations

import os
from pathlib import Path

from thirdeye.platforms.base import Platform
from thirdeye.platforms.grok_bot.constants import (
    DISPLAY_NAME,
    INSTALLED_MARKER,
    PLATFORM_NAME,
    default_state_dir,
)

# Marker-gated action kick (Duncan Q1) — not a boot/pidfile daemon SoT.
KICK_ENABLED = "kick.enabled"
WATCHER_ENABLED = "watcher.enabled"
AGENTS_ROOT_ENV = "THIRDEYE_GROK_BOT_AGENTS_ROOT"


class GrokBotPlatform(Platform):
    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(
        self,
        state_dir: Path | None = None,
        agents_root: Path | str | None = None,
    ) -> None:
        self._state_dir = state_dir or default_state_dir()
        env_root = os.environ.get(AGENTS_ROOT_ENV)
        if agents_root is not None:
            self._agents_root: Path | None = Path(agents_root)
        elif env_root:
            self._agents_root = Path(env_root)
        else:
            self._agents_root = None
        self._observer = None

    @property
    def _marker(self) -> Path:
        return self._state_dir / INSTALLED_MARKER

    @property
    def _kick_flag(self) -> Path:
        return self._state_dir / KICK_ENABLED

    @property
    def _watcher_flag(self) -> Path:
        return self._state_dir / WATCHER_ENABLED

    def _resolve_agents_root(self) -> Path | None:
        if self._agents_root is not None:
            return self._agents_root
        env_root = os.environ.get(AGENTS_ROOT_ENV)
        if env_root:
            return Path(env_root)
        return None

    def install(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._marker.write_text("1\n", encoding="utf-8")
        self._kick_flag.write_text("1\n", encoding="utf-8")
        self._watcher_flag.write_text("1\n", encoding="utf-8")
        # Arm real FS/mtime observer when agents root is known (env or ctor).
        root = self._resolve_agents_root()
        if root is not None:
            from thirdeye.platforms.grok_bot import watch as watch_mod

            self._observer = watch_mod.start_store_observer(
                self, agents_root=root
            )

    def is_installed(self) -> bool:
        return self._marker.is_file()

    def is_store_kick_enabled(self) -> bool:
        return self._kick_flag.is_file() or self._watcher_flag.is_file()

    def is_kick_enabled(self) -> bool:
        return self.is_store_kick_enabled()

    def is_mutation_kick_enabled(self) -> bool:
        return self.is_store_kick_enabled()

    def is_action_indicator_enabled(self) -> bool:
        return self.is_store_kick_enabled()

    def is_watcher_running(self) -> bool:
        return self.is_store_kick_enabled()

    def is_passive_running(self) -> bool:
        return self.is_store_kick_enabled()

    def is_running(self) -> bool:
        return self.is_store_kick_enabled()

    def uninstall(self) -> None:
        from thirdeye.platforms.grok_bot import watch as watch_mod

        watch_mod.stop_store_observer(self)
        self._observer = None
        for path in (self._kick_flag, self._watcher_flag):
            if path.exists():
                path.unlink()
        watermark = self._state_dir / "watermarks.json"
        if watermark.exists():
            watermark.unlink()
        if self._marker.exists():
            self._marker.unlink()
        if self._state_dir.exists() and not any(self._state_dir.iterdir()):
            self._state_dir.rmdir()
