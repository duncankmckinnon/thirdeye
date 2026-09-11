from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from thirdeye._compat import fsops


def default_root() -> Path:
    env = os.environ.get("THIRDEYE_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".thirdeye"


def _parse_patterns(raw: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _coerce_patterns(value: Any) -> tuple[str, ...]:
    """Normalize a config.yaml ``capture_env`` value to a pattern tuple.

    Accepts either a comma-separated string (``"WB_*, BUILD_LABEL"``) or a
    YAML list (``["WB_*", "BUILD_LABEL"]``); anything else yields ``()``.
    """
    if isinstance(value, str):
        return _parse_patterns(value)
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


@dataclass(frozen=True)
class LogfireSettings:
    """Persisted Logfire export settings, read from config.yaml's ``logfire`` key.

    ``token`` is the Logfire write token (called "gateway key" in the UI/CLI,
    since that's the term users reach for). Persisted indefinitely to disk, not
    just for the current environment/session, so ``thirdeye logfire enable`` is
    a one-time setup step. ``api_key`` is a separate project key with managed
    dataset write scope; keeping it separate prevents dataset setup from
    changing live trace ingestion.
    """

    enabled: bool = False
    token: str | None = None
    api_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "token": self.token, "api_key": self.api_key}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> LogfireSettings:
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            token=raw.get("token") or None,
            api_key=raw.get("api_key") or None,
        )


def _read_config_yaml(config_file: Path) -> dict[str, Any]:
    if not config_file.exists():
        return {}
    try:
        with open(config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_config_yaml(config_file: Path, data: dict[str, Any]) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_file.with_suffix(config_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    fsops.replace(tmp, config_file)


@dataclass(frozen=True)
class Config:
    root: Path
    capture_env_patterns: tuple[str, ...] = ()
    logfire: LogfireSettings = field(default_factory=LogfireSettings)

    @classmethod
    def load(cls) -> Config:
        root = default_root()
        raw = _read_config_yaml(root / "config.yaml")
        # THIRDEYE_CAPTURE_ENV wins when set, so a one-off run can still
        # override; otherwise fall back to config.yaml's ``capture_env`` so
        # capture does not depend on the launching shell exporting anything
        # (workbench dispatches agents from contexts that may not source rc).
        patterns = _parse_patterns(os.environ.get("THIRDEYE_CAPTURE_ENV", ""))
        if not patterns:
            patterns = _coerce_patterns(raw.get("capture_env"))
        return cls(
            root=root,
            capture_env_patterns=patterns,
            logfire=LogfireSettings.from_dict(raw.get("logfire")),
        )

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"

    def write_logfire_settings(self, settings: LogfireSettings) -> Config:
        """Persist ``settings`` to config.yaml, preserving other top-level keys.

        Returns a copy of this Config with the new settings applied, so callers
        don't need a second load() to see their own write.
        """
        data = _read_config_yaml(self.config_file)
        data["logfire"] = settings.to_dict()
        _write_config_yaml(self.config_file, data)
        return replace(self, logfire=settings)

    def write_capture_env_patterns(self, patterns: tuple[str, ...] | list[str]) -> Config:
        """Persist ``capture_env`` to config.yaml, preserving other top-level keys.

        An empty sequence removes the key. Returns a copy of this Config with
        the new patterns applied. ``THIRDEYE_CAPTURE_ENV`` still overrides this
        at load time when set.
        """
        cleaned = tuple(str(p).strip() for p in patterns if str(p).strip())
        data = _read_config_yaml(self.config_file)
        if cleaned:
            data["capture_env"] = list(cleaned)
        else:
            data.pop("capture_env", None)
        _write_config_yaml(self.config_file, data)
        return replace(self, capture_env_patterns=cleaned)


def load() -> Config:
    return Config.load()
