from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_root() -> Path:
    env = os.environ.get("THIRDEYE_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".thirdeye"


def _parse_patterns(raw: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in raw.split(",") if p.strip())


@dataclass(frozen=True)
class Config:
    root: Path
    capture_env_patterns: tuple[str, ...] = ()

    @classmethod
    def load(cls) -> Config:
        return cls(
            root=default_root(),
            capture_env_patterns=_parse_patterns(os.environ.get("THIRDEYE_CAPTURE_ENV", "")),
        )

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"


def load() -> Config:
    return Config.load()
