"""Constants for the Grok Bot platform."""

from __future__ import annotations

from pathlib import Path

from thirdeye.config import default_root

PLATFORM_NAME = "grok_bot"
DISPLAY_NAME = "Grok Bot"

# Marker directory under THIRDEYE_HOME. Capture (Wave 2) polls agent store.db
# on each bot box; install only records that the platform is enabled.
def default_state_dir() -> Path:
    return default_root() / "platforms" / PLATFORM_NAME

INSTALLED_MARKER = "installed"
