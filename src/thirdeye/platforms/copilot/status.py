"""Local health reporting for the Copilot CLI capture archive."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.paths import otel_jobs_dir, platform_dir, usage_log_path
from thirdeye.reader import SessionReader

from . import export_transport
from .archive import _record_from_event
from .constants import FOLLOWUP_LEASE_FILENAME, PLATFORM_NAME
from .database import read_database
from .export_state import load_export_state
from .install import CopilotPlatform
from .projection_store import read_projection_status
from .runtime import load_runtime_status
from .spool import read_spool
from .state import journal_path, read_json, state_path
from .types import SourcePaths, SourceRecord

_STATUS_PROBE_ID = "copilot-status-probe"
_FILE_LEVEL_DATABASE_CODES = frozenset(
    {
        "copilot_database_unreadable",
        "copilot_database_busy",
        "copilot_database_incompatible",
        "copilot_database_read_failed",
    }
)
_WORKER_TURN_KINDS = frozenset({"turn", "spans", "subagent_turn"})


def _error_capability(path: Path, *, exists: bool, reason: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": exists,
        "readable": False,
        "error": {"kind": "source_unreadable", "path": str(path), "reason": reason},
    }


def _missing_capability(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": False, "readable": False}


def _directory_capability(path: Path) -> dict[str, Any]:
    """Describe a directory by enumerating it, not just by a successful stat()."""

    try:
        exists = path.exists()
    except OSError as error:
        return _error_capability(path, exists=False, reason=type(error).__name__)
    if not exists:
        return _missing_capability(path)
    if not path.is_dir():
        return _error_capability(path, exists=True, reason="not a directory")
    try:
        next(iter(path.iterdir()), None)
    except OSError as error:
        return _error_capability(path, exists=True, reason=type(error).__name__)
    return {"path": str(path), "exists": True, "readable": True}


def _file_capability(path: Path) -> dict[str, Any]:
    """Describe a regular file by opening it for a bounded read."""

    try:
        exists = path.exists()
    except OSError as error:
        return _error_capability(path, exists=False, reason=type(error).__name__)
    if not exists:
        return _missing_capability(path)
    if not path.is_file():
        return _error_capability(path, exists=True, reason="not a file")
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as error:
        return _error_capability(path, exists=True, reason=type(error).__name__)
    return {"path": str(path), "exists": True, "readable": True}


def _database_capability(paths: SourcePaths) -> dict[str, Any]:
    """Open the session database read-only; absence is not a source error."""

    path = Path(paths["database"])
    try:
        exists = path.exists()
    except OSError as error:
        return _error_capability(path, exists=False, reason=type(error).__name__)
    if not exists:
        return _missing_capability(path)
    try:
        probe = read_database(paths, _STATUS_PROBE_ID, {})
    except (OSError, ValueError) as error:
        return _error_capability(path, exists=True, reason=type(error).__name__)
    for diagnostic in probe["diagnostics"]:
        code = diagnostic.get("code")
        if code == "copilot_database_missing":
            return _missing_capability(path)
        if isinstance(code, str) and code in _FILE_LEVEL_DATABASE_CODES:
            return _error_capability(path, exists=True, reason=code)
    if not path.is_file():
        return _error_capability(path, exists=True, reason="not a file")
    return {"path": str(path), "exists": True, "readable": True}


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


def _spool_file_errors(directory: Path) -> list[dict[str, Any]]:
    """Surface spool .diag files (locations/reasons only, never payloads)."""

    errors: list[dict[str, Any]] = []
    try:
        diagnostics = sorted(directory.glob("*.diag"))
    except OSError as error:
        return [
            {
                "kind": "spool_unreadable",
                "session": directory.name,
                "path": str(directory),
                "reason": type(error).__name__,
            }
        ]
    for diag_path in diagnostics:
        try:
            payload = json.loads(diag_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append(
                {
                    "kind": "spool_unreadable",
                    "session": directory.name,
                    "path": diag_path.name,
                }
            )
            continue
        reason = payload.get("reason") if isinstance(payload, dict) else None
        loc = payload.get("path") if isinstance(payload, dict) else None
        errors.append(
            {
                "kind": "spool_unreadable",
                "session": directory.name,
                "path": loc if isinstance(loc, str) else diag_path.name,
                "reason": reason if isinstance(reason, str) else "invalid spool diagnostic",
            }
        )
    return errors


def _spool_sessions(
    config: Config, paths: SourcePaths
) -> tuple[int, list[str], SourceRecord | None, list[dict[str, Any]]]:
    root = Path(config.root) / "spool" / "copilot" / paths["source_key"]
    count = 0
    sessions: list[str] = []
    latest: SourceRecord | None = None
    errors: list[dict[str, Any]] = []
    try:
        entries = sorted(root.iterdir())
    except FileNotFoundError:
        return count, sessions, latest, errors
    except OSError as error:
        return (
            count,
            sessions,
            latest,
            [
                {
                    "kind": "spool_unreadable",
                    "path": str(root),
                    "reason": type(error).__name__,
                }
            ],
        )
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            records = read_spool(config, paths, entry.name)
            json_files = list(entry.glob("*.json"))
        except (OSError, ValueError) as error:
            errors.append(
                {
                    "kind": "spool_unreadable",
                    "session": entry.name,
                    "reason": str(error) if isinstance(error, ValueError) else type(error).__name__,
                }
            )
            continue
        if json_files:
            sessions.append(entry.name)
        count += len(json_files)
        for record in records:
            latest = _record_hook(record, latest)
        errors.extend(_spool_file_errors(entry))
    return count, sessions, latest, errors


def _followup_lease_pending(directory: Path) -> bool:
    """True when a live follow-up lease file exists beside the session."""

    try:
        payload = json.loads((directory / FOLLOWUP_LEASE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return False
    return float(expires_at) > time.time()


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iter_json_jobs(directory: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    try:
        entries = list(directory.iterdir())
    except OSError:
        return payloads
    for path in entries:
        if not path.is_file() or path.suffix != ".json" or path.name.endswith(".claim"):
            continue
        payload = _json_object(path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _worker_kind(message: object) -> str:
    text = str(message or "")
    prefix = "kind="
    if not text.startswith(prefix):
        return ""
    return text[len(prefix) :].split(None, 1)[0]


def _iter_worker_turn_failures(config: Config, stored_session_id: str) -> list[dict[str, Any]]:
    """Read deleted whole-turn export failures from the worker error log.

    Generic turn/spans/subagent jobs are unlinked before delivery. A crash or
    export failure therefore leaves no job and no sent claim; the durable
    breadcrumb is ``usage-errors.jsonl``.
    """

    path = usage_log_path(config.root)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    failures: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("phase") != "otel_worker_export_failed":
            continue
        if entry.get("session_id") != stored_session_id:
            continue
        if _worker_kind(entry.get("message")) not in _WORKER_TURN_KINDS:
            continue
        failures.append(entry)
    return failures


def _export_health(config: Config, stored_session_id: str, directory: Path) -> dict[str, Any]:
    """Summarize queue/delivery health from claims, jobs, and worker logs."""

    ledger = load_export_state(config, stored_session_id)
    placements = ledger.get("placements") if isinstance(ledger.get("placements"), dict) else {}
    conflicts = ledger.get("conflicts") if isinstance(ledger.get("conflicts"), dict) else {}
    turn_errors = ledger.get("turn_errors") if isinstance(ledger.get("turn_errors"), dict) else {}

    delivered: set[str] = set()
    queued: set[str] = set()
    errored: set[str] = set()

    for accounting_id, item in placements.items():
        if not isinstance(accounting_id, str) or not isinstance(item, dict):
            continue
        key = f"acct:{accounting_id}"
        span_id = item.get("span_id")
        sent = export_transport.delivery_sent(directory, accounting_id)
        job = (
            export_transport.status(config.root, span_id)
            if isinstance(span_id, str) and span_id
            else None
        )
        job_state = job.get("state") if job else None
        if sent or item.get("emitted") or job_state == "emitted":
            delivered.add(key)
        elif job_state == "failed":
            queued.add(key)
            errored.add(key)
        elif job_state in {"queued", "claimed", "retrying"}:
            queued.add(key)
        else:
            queued.add(key)
        if item.get("last_error") or (job and job.get("last_error")):
            errored.add(key)

    for payload in _iter_json_jobs(export_transport.jobs_dir(config.root)):
        if payload.get("session_id") != stored_session_id:
            continue
        accounting_id = payload.get("accounting_id")
        key = (
            f"acct:{accounting_id}"
            if isinstance(accounting_id, str) and accounting_id
            else f"acct-job:{payload.get('job_id')}"
        )
        if key in delivered:
            continue
        state = payload.get("state")
        if state == "emitted" or (
            isinstance(accounting_id, str)
            and export_transport.delivery_sent(directory, accounting_id)
        ):
            delivered.add(key)
            queued.discard(key)
            continue
        queued.add(key)
        if state == "failed" or payload.get("last_error"):
            errored.add(key)

    sent_dir = directory / "otel-turns-sent"
    try:
        claim_files = list(sent_dir.iterdir()) if sent_dir.is_dir() else []
    except OSError:
        claim_files = []
    for path in claim_files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        key = f"turn-claim:{path.name}"
        if text == "sent":
            delivered.add(key)
        else:
            queued.add(key)

    for payload in _iter_json_jobs(otel_jobs_dir(config.root)):
        if payload.get("session_id") != stored_session_id:
            continue
        if payload.get("kind") not in {"turn", "spans", "subagent_turn"}:
            continue
        queued.add(f"otel:{payload.get('job_id') or id(payload)}")
        if payload.get("last_error"):
            errored.add(f"otel:{payload.get('job_id') or id(payload)}")

    for index, entry in enumerate(_iter_worker_turn_failures(config, stored_session_id)):
        key = f"otel-fail:{entry.get('ts') or index}:{entry.get('message')}"
        queued.add(key)
        errored.add(key)

    queued -= delivered

    runtime_status = load_runtime_status(config, stored_session_id)
    last_error = runtime_status.get("last_error")
    if isinstance(last_error, dict):
        errored.add("runtime")
    else:
        last_error = None

    return {
        "activated": bool(ledger.get("activated")),
        "queued": len(queued),
        "delivered": len(delivered),
        "errors": len(errored) + len(conflicts) + len(turn_errors),
        "last_error": last_error,
    }


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
            errors.append(
                {"kind": "invalid_archive_state", "session": directory.name, "reason": str(error)}
            )
            continue
        if state is None:
            state = {}
        stored_key = state.get("source_key")
        if isinstance(stored_key, str) and stored_key != paths["source_key"]:
            errors.append(
                {
                    "kind": "source_key_collision",
                    "session": directory.name,
                    "reason": "Copilot source-key prefix collision for stored session ID",
                }
            )
            continue
        health = state.get("health") if isinstance(state.get("health"), dict) else {}
        diagnostics = (
            health.get("diagnostics") if isinstance(health.get("diagnostics"), list) else []
        )
        if _followup_lease_pending(directory):
            pending_followup += 1
            active_leases += 1
        session = {
            "stored_session_id": directory.name,
            "native_session_id": state.get("native_session_id"),
            "cursor": state.get("cursor", {}),
            "last_successful_import": health.get("last_successful_import"),
            "diagnostics": diagnostics,
            "journal_pending": journal_path(directory).is_file(),
        }
        try:
            session["projection"] = read_projection_status(config, directory.name)
        except Exception as error:
            session["projection"] = {"errors": 1}
            errors.append(
                {
                    "kind": "copilot_projection_unreadable",
                    "session": directory.name,
                    "reason": type(error).__name__,
                }
            )
        try:
            session["export"] = _export_health(config, directory.name, directory)
            last_error = session["export"].get("last_error")
            if isinstance(last_error, dict):
                errors.append(
                    {
                        "kind": "copilot_reconcile_error",
                        "session": directory.name,
                        "reason": last_error.get("phase") or "copilot_reconcile",
                        "errors": last_error.get("errors"),
                    }
                )
        except Exception as error:
            session["export"] = {"activated": False, "queued": 0, "delivered": 0, "errors": 1}
            errors.append(
                {
                    "kind": "copilot_export_ledger_unreadable",
                    "session": directory.name,
                    "reason": type(error).__name__,
                }
            )
        sessions.append(session)
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
    session_root = _directory_capability(Path(paths["session_root"]))
    database = _database_capability(paths)
    wal = _file_capability(Path(paths["database"]).with_name(f"{Path(paths['database']).name}-wal"))
    sessions, archived_hook, archive_errors, pending_followup, active_leases = _archive_status(
        config, paths
    )
    spool_count, spool_sessions, spooled_hook, spool_errors = _spool_sessions(config, paths)
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
        "errors": [*source_errors, *archive_errors, *spool_errors],
    }


__all__ = ["capture_status"]
