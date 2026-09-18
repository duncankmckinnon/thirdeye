"""Install / uninstall the Grok Bot platform (marker only; no Cursor hooks)."""

from __future__ import annotations

from pathlib import Path

from thirdeye.platforms.base import Platform
from thirdeye.platforms.grok_bot.constants import (
    DISPLAY_NAME,
    INSTALLED_MARKER,
    PLATFORM_NAME,
    default_state_dir,
)


class GrokBotPlatform(Platform):
    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or default_state_dir()

    @property
    def _marker(self) -> Path:
        return self._state_dir / INSTALLED_MARKER

    def install(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._marker.write_text("1\n", encoding="utf-8")

    def is_installed(self) -> bool:
        return self._marker.is_file()

    def uninstall(self) -> None:
        if self._marker.exists():
            self._marker.unlink()
        if self._state_dir.exists() and not any(self._state_dir.iterdir()):
            self._state_dir.rmdir()
