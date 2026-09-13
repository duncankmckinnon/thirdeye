"""Runtime composition for archive reconciliation after V1 capture.

Capture remains the durable, local V1 boundary while reconciliation and
export eligibility are V2 derived work. A projection failure must never turn
a successful hook receipt or source import into a failed capture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.paths import platform_dir, session_dir
from thirdeye.usage.errlog import log_capture_error

from .constants import PLATFORM_NAME
from .identity import stored_session_id, validate_stored_session_id
from .jsonio import atomic_write_json, read_json_object
from .reconcile import reconcile_archive
from .state import read_json, state_path
from .types import SourcePaths

RUNTIME_STATUS_FILENAME = "copilot.runtime.json"


def exports_configured(config: Config) -> bool:
    """Whether runtime activity may queue detached live export work."""
    return bool(config.logfire.enabled and config.logfire.token)


def runtime_status_path(directory: Path) -> Path:
    return directory / RUNTIME_STATUS_FILENAME


def stored_session_directory(config: Config, stored_id: str) -> Path:
    """Return the archive directory for a stored ID after rejecting path escapes."""

    validate_stored_session_id(stored_id)
    root = platform_dir(config.root, PLATFORM_NAME).resolve()
    directory = (root / stored_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError("stored session ID escapes the Copilot archive") from exc
    return directory


def load_runtime_status(config: Config, stored_session_id: str) -> dict[str, Any]:
    """Return the last recorded derived-work error for a stored session."""

    try:
        directory = stored_session_directory(config, stored_session_id)
        raw = read_json_object(
            runtime_status_path(directory),
            invalid_message="invalid Copilot runtime status",
        )
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _note_reconcile_result(
    config: Config, stored_id: str, result: dict[str, int]
) -> dict[str, int]:
    """Persist/log derived failures without raising into capture."""

    directory = session_dir(config.root, PLATFORM_NAME, stored_id)
    path = runtime_status_path(directory)
    if result.get("errors"):
        log_capture_error(
            thirdeye_home=config.root,
            phase="copilot_reconcile",
            message=f"errors={result.get('errors')}",
            platform=PLATFORM_NAME,
            session_id=stored_id,
            silent_fallback=True,
        )
        if directory.is_dir():
            try:
                atomic_write_json(
                    path,
                    {
                        "last_error": {
                            "phase": "copilot_reconcile",
                            "errors": int(result.get("errors") or 0),
                        }
                    },
                )
            except OSError:
                pass
        return result
    if directory.is_dir():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return result


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
            validate_stored_session_id(directory.name)
            state = read_json(state_path(directory)) or {}
        except ValueError:
            continue
        if state.get("source_key") == paths["source_key"]:
            result.append(directory.name)
    return result


def all_archived_session_ids(config: Config) -> list[str]:
    """Return every retained Copilot archive under this Thirdeye home."""

    root = platform_dir(config.root, PLATFORM_NAME)
    try:
        directories = sorted(entry for entry in root.iterdir() if entry.is_dir())
    except OSError:
        return []
    result: list[str] = []
    for directory in directories:
        try:
            validate_stored_session_id(directory.name)
            state = read_json(state_path(directory))
        except ValueError:
            continue
        if state is not None:
            result.append(directory.name)
    return result


def reconcile_stored_session(
    config: Config,
    stored_session_id: str,
    *,
    rebuild: bool = False,
    export: bool = False,
    include_history: bool = False,
) -> dict[str, int]:
    """Derive one stored archive after validating its identity."""

    stored_session_directory(config, stored_session_id)
    return _note_reconcile_result(
        config,
        stored_session_id,
        reconcile_archive(
            config,
            stored_session_id,
            rebuild=rebuild,
            export=export,
            include_history=include_history,
        ),
    )


def reconcile_session(
    config: Config,
    paths: SourcePaths,
    native_session_id: str,
    *,
    export: bool = False,
    include_history: bool = False,
) -> dict[str, int]:
    """Derive one retained session, optionally queueing detached live export.

    ``export`` is forwarded even when Logfire is not configured: export
    assembly records the activation boundary before it decides whether any
    jobs can be dispatched.
    """
    return reconcile_stored_session(
        config,
        stored_session_id(paths, native_session_id),
        export=export,
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
        session_id: reconcile_stored_session(
            config,
            session_id,
            export=export,
            include_history=include_history,
        )
        for session_id in archived_session_ids(config, paths)
    }


__all__ = [
    "all_archived_session_ids",
    "archived_session_ids",
    "exports_configured",
    "load_runtime_status",
    "reconcile_archived_sessions",
    "reconcile_session",
    "reconcile_stored_session",
    "runtime_status_path",
    "stored_session_directory",
]
