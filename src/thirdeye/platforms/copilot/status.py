"""Local health reporting for the Copilot CLI capture archive."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.paths import platform_dir
from thirdeye.reader import SessionReader

from .archive import _record_from_event
from .constants import PLATFORM_NAME
from .install import CopilotPlatform
from .spool import read_spool
from .state import journal_path, read_json, state_path
from .types import SourcePaths, SourceRecord


def _path_capability(path: Path, *, directory: bool) -> dict[str, Any]:
    """Describe a local source without treating an absent Copilot install as an error."""

    try:
        exists = path.exists()
        kind_matches = path.is_dir() if directory else path.is_file()
        readable = kind_matches and path.stat() is not None
    except OSError as error:
        return {
            "path": str(path),
            "exists": False,
            "readable": False,
            "error": {"kind": "source_unreadable", "path": str(path), "reason": type(error).__name__},
        }
    return {"path": str(path), "exists": exists, "readable": readable}


def _archive_directories(config: Config, paths: SourcePaths) -> list[Path]:
    root = platform_dir(config.root, PLATFORM_NAME)
    prefix = f"copilot-{paths['source_key'][:16]}-"
    try:
        return sorted(
            (entry for entry in root.iterdir() if entry.is_dir() and entry.name.startswith(prefix)),
            key=lambda entry: entry.name,
        )
    except OSError:
        return []


def _record_hook(record: SourceRecord, latest: SourceRecord | None) -> SourceRecord | None:
    if record["source_kind"] != "hook":
        return latest
    if latest is None or record["observed_at"] > latest["observed_at"]:
        return record
    return latest


def _spool_sessions(config: Config, paths: SourcePaths) -> tuple[int, list[str], SourceRecord | None]:
    root = Path(config.root) / "spool" / "copilot" / paths["source_key"]
    count = 0
    sessions: list[str] = []
    latest: SourceRecord | None = None
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return count, sessions, latest
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            records = read_spool(config, paths, entry.name)
        except ValueError:
            continue
        if records:
            sessions.append(entry.name)
        count += len(records)
        for record in records:
            latest = _record_hook(record, latest)
    return count, sessions, latest


def _archive_status(
    config: Config, paths: SourcePaths
) -> tuple[list[dict[str, Any]], SourceRecord | None, list[dict[str, Any]], int, int]:
    """Read archive health/state without attempting an import or source read."""

    sessions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    latest_hook: SourceRecord | None = None
    pending_followup = 0
    active_leases = 0
    for directory in _archive_directories(config, paths):
        try:
            state = read_json(state_path(directory))
        except ValueError as error:
            errors.append({"kind": "invalid_archive_state", "session": directory.name, "reason": str(error)})
            continue
        if state is None:
            state = {}
        health = state.get("health") if isinstance(state.get("health"), dict) else {}
        diagnostics = health.get("diagnostics") if isinstance(health.get("diagnostics"), list) else []
        followup = state.get("followup", state.get("pending_followup", False))
        lease = state.get("lease", state.get("leases", None))
        pending_followup += int(bool(followup))
        active_leases += len(lease) if isinstance(lease, list) else int(bool(lease))
        sessions.append(
            {
                "stored_session_id": directory.name,
                "native_session_id": state.get("native_session_id"),
                "cursor": state.get("cursor", {}),
                "last_successful_import": health.get("last_successful_import"),
                "diagnostics": diagnostics,
                "journal_pending": journal_path(directory).is_file(),
            }
        )
        for diagnostic in diagnostics:
            if isinstance(diagnostic, dict):
                errors.append({"session": directory.name, **diagnostic})
        try:
            for event in SessionReader(directory).iter_events(types=("copilot_hook",)):
                record = _record_from_event(event)
                if record is not None:
                    latest_hook = _record_hook(record, latest_hook)
        except (OSError, ValueError):
            errors.append({"kind": "archive_events_unreadable", "session": directory.name})
    return sessions, latest_hook, errors, pending_followup, active_leases


def capture_status(config: Config, paths: SourcePaths) -> dict:
    """Return local capture installation, progress, and health information.

    An absent Copilot home/database and uninstalled owned hooks are reported as
    capabilities/configuration, not errors.  Archive/source diagnostics remain
    in ``errors`` so a command layer can choose a nonzero status for them.
    """

    home = Path(paths["home"])
    session_root = _path_capability(Path(paths["session_root"]), directory=True)
    database = _path_capability(Path(paths["database"]), directory=False)
    wal = _path_capability(Path(paths["database"]).with_name(f"{Path(paths['database']).name}-wal"), directory=False)
    sessions, archived_hook, archive_errors, pending_followup, active_leases = _archive_status(
        config, paths
    )
    spool_count, spool_sessions, spooled_hook = _spool_sessions(config, paths)
    last_hook = archived_hook
    if spooled_hook is not None:
        last_hook = _record_hook(spooled_hook, last_hook)

    source_errors = [
        item["error"]
        for item in (session_root, database, wal)
        if isinstance(item.get("error"), dict)
    ]
    last_successful = max(
        (
            value
            for session in sessions
            if isinstance((value := session.get("last_successful_import")), str)
        ),
        default=None,
    )
    installer = CopilotPlatform(source_home=home)
    return {
        "paths": dict(paths),
        "installation": {
            "configured": installer.is_installed(),
            "hooks_file": str(installer.hooks_file),
        },
        "capabilities": {
            "transcripts": session_root,
            "database": database,
            "database_wal": wal,
        },
        "last_observed_hook": last_hook,
        "last_successful_import": last_successful,
        "sessions": sessions,
        "pending": {
            "spool_records": spool_count,
            "spool_sessions": spool_sessions,
            "followup": pending_followup,
            "leases": active_leases,
            "journals": sum(1 for session in sessions if session["journal_pending"]),
        },
        "errors": [*source_errors, *archive_errors],
    }


__all__ = ["capture_status"]
