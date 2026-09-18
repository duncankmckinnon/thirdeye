"""Install / uninstall Grok Bot — arms store-mutation kick (no Cursor hooks)."""

from __future__ import annotations

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
# Legacy alias from interim Wave 4 library layer.
WATCHER_ENABLED = "watcher.enabled"


class GrokBotPlatform(Platform):
    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or default_state_dir()

    @property
    def _marker(self) -> Path:
        return self._state_dir / INSTALLED_MARKER

    @property
    def _kick_flag(self) -> Path:
        return self._state_dir / KICK_ENABLED

    @property
    def _watcher_flag(self) -> Path:
        # Keep path for back-compat with interim aa29a74 flag name.
        return self._state_dir / WATCHER_ENABLED

    def install(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._marker.write_text("1\n", encoding="utf-8")
        # Arm marker-gated store-mutation kick (idle-exiting observer contract).
        self._kick_flag.write_text("1\n", encoding="utf-8")
        self._watcher_flag.write_text("1\n", encoding="utf-8")

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
        # Interim alias — prefer is_store_kick_enabled for SoT naming.
        return self.is_store_kick_enabled()

    def is_passive_running(self) -> bool:
        return self.is_store_kick_enabled()

    def is_running(self) -> bool:
        return self.is_store_kick_enabled()

    def uninstall(self) -> None:
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
