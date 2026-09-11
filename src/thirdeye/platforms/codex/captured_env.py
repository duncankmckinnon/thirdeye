"""Resolve opted-in environment context for Codex turn export.

Codex's turn export runs in ``thirdeye-codex-notify``, an argv-invoked
callback Codex spawns detached from the agent process that actually held the
opted-in vars (``WB_*`` and friends), so ``os.environ`` there is unreliable.
Codex's ``hooks.json`` ``SessionStart`` hook — a normal child of that agent
process — captures the same env and persists it into meta; the notify path
reads it back here when its own live capture comes up empty.
"""

from __future__ import annotations

from pathlib import Path

from thirdeye.config import Config


def resolve_captured_env(config: Config, session_dir_: Path) -> dict[str, str]:
    """Raw ``{ENV_VAR: value}`` for this session: live capture, else the
    session-start snapshot persisted in meta.
    """
    from thirdeye.env_capture import capture_env

    captured = capture_env(config.capture_env_patterns)
    if captured:
        return captured
    return _persisted_captured_env(session_dir_)


def _persisted_captured_env(session_dir_: Path) -> dict[str, str]:
    try:
        from thirdeye.meta import read_meta
        from thirdeye.paths import meta_path

        meta = read_meta(meta_path(session_dir_))
    except Exception:
        return {}
    if meta is None:
        return {}
    raw = meta.extra.get("captured_env")
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if value is not None}
