from __future__ import annotations

from abc import ABC, abstractmethod


class Platform(ABC):
    name: str
    display_name: str

    @abstractmethod
    def install(self) -> None: ...

    @abstractmethod
    def uninstall(self) -> None: ...

    def is_installed(self) -> bool:
        """Return whether thirdeye's complete integration is configured."""
        return False
