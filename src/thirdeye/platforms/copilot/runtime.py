"""Runtime composition for archive reconciliation after V1 capture.

Capture remains the durable, local V1 boundary while reconciliation and
export eligibility are V2 derived work. A projection failure must never turn
a successful hook receipt or source import into a failed capture.
"""

from __future__ import annotations

from thirdeye.config import Config
from thirdeye.paths import platform_dir

from .constants import PLATFORM_NAME
from .identity import stored_session_id
from .reconcile import reconcile_archive
from .state import read_json, state_path
from .types import SourcePaths


def exports_configured(config: Config) -> bool:
    """Whether runtime activity may queue detached live export work."""
    return bool(config.logfire.enabled and config.logfire.token)


def archived_session_ids(config: Config, paths: SourcePaths) -> list[str]:
    """Return this source home's retained V1 sessions without reading sources."""
    root = platform_dir(config.root, PLATFORM_NAME)
    prefix = f"copilot-{paths['source_key'][:16]}-"
    try:
        directories = sorted(entry for entry in root.iterdir() if entry.is_dir())
    except OSError:
        return []
    result: list[str] = []
    for directory in directories:
        if not directory.name.startswith(prefix):
            continue
        try:
            state = read_json(state_path(directory)) or {}
        except ValueError:
            continue
        if state.get("source_key") == paths["source_key"]:
            result.append(directory.name)
    return result


def reconcile_session(
    config: Config,
    paths: SourcePaths,
    native_session_id: str,
    *,
    export: bool = False,
    include_history: bool = False,
) -> dict[str, int]:
    """Derive one retained session, optionally queueing detached live export."""
    return reconcile_archive(
        config,
        stored_session_id(paths, native_session_id),
        export=export and exports_configured(config),
        include_history=include_history,
    )


def reconcile_archived_sessions(
    config: Config,
    paths: SourcePaths,
    *,
    export: bool = False,
    include_history: bool = False,
) -> dict[str, dict[str, int]]:
    """Replay retained sessions even after their live sources disappear."""
    return {
        session_id: reconcile_archive(
            config,
            session_id,
            export=export and exports_configured(config),
            include_history=include_history,
        )
        for session_id in archived_session_ids(config, paths)
    }


__all__ = ["archived_session_ids", "exports_configured", "reconcile_archived_sessions", "reconcile_session"]
