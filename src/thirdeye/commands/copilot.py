from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import click

from thirdeye.config import Config
from thirdeye.platforms.copilot.capture import (
    _is_absence_diagnostic,
    _is_retryable_code,
)
from thirdeye.platforms.copilot.capture import sync as capture_sync
from thirdeye.platforms.copilot.constants import COPILOT_HOME_ENV
from thirdeye.platforms.copilot.identity import resolve_sources, validate_native_id
from thirdeye.platforms.copilot.status import capture_status
from thirdeye.platforms.copilot.types import SourcePaths, SyncResult
from thirdeye.platforms.copilot.watch import watch as watch_loop

_MIN_WATCH_INTERVAL = 0.1
_LOCATOR_KEYS = (
    "file",
    "path",
    "table",
    "row_id",
    "offset",
    "byte_offset",
    "generation",
    "file_generation",
)


def _source_home_option(fn):
    return click.option(
        "--source-home",
        type=click.Path(path_type=Path),
        default=None,
        help="Copilot home directory. Overrides COPILOT_HOME and ~/.copilot.",
    )(fn)


def _validate_interval(_ctx: click.Context, _param: click.Parameter, value: float) -> float:
    if not math.isfinite(value) or value < _MIN_WATCH_INTERVAL:
        raise click.BadParameter(
            f"interval must be finite and at least {_MIN_WATCH_INTERVAL} seconds"
        )
    return value


def _requested_home(source_home: Path | None) -> Path:
    if source_home is not None:
        return source_home
    configured = os.environ.get(COPILOT_HOME_ENV)
    if configured:
        return Path(configured)
    return Path.home() / ".copilot"


def _resolve_paths(source_home: Path | None) -> SourcePaths:
    try:
        return resolve_sources(source_home)
    except ValueError as exc:
        raise click.ClickException(
            f"Cannot resolve Copilot sources at {_requested_home(source_home)}: {exc}"
        ) from exc


def _counts_line(result: SyncResult) -> str:
    return (
        f"sessions={result['sessions']} "
        f"records_written={result['records_written']} "
        f"duplicate_records={result['duplicate_records']} "
        f"pending={result['pending']} "
        f"errors={result['errors']}"
    )


def _format_locator(locator: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for key in _LOCATOR_KEYS:
        value = locator.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            parts.append(f"{key}={value}")
    return ",".join(parts) if parts else None


def _format_diagnostic(error: object) -> str:
    if not isinstance(error, dict):
        return str(error)
    code = error.get("code") or error.get("kind") or "error"
    if not isinstance(code, str) or not code:
        code = "error"
    message = error.get("message") or error.get("reason")
    bits = [f"{code}: {message}" if isinstance(message, str) and message else code]
    session = error.get("session") or error.get("native_session_id")
    if isinstance(session, str) and session:
        bits.append(f"session={session}")
    path = error.get("path")
    if isinstance(path, str) and path:
        bits.append(f"path={path}")
    source_id = error.get("source_id")
    if isinstance(source_id, str) and source_id:
        bits.append(f"source_id={source_id}")
    locator = error.get("locator")
    if isinstance(locator, dict):
        formatted = _format_locator(locator)
        if formatted:
            bits.append(f"locator={formatted}")
    return " ".join(bits)


def _format_last_hook(record: object) -> str:
    if not isinstance(record, dict):
        return "none"
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    locator = record.get("locator") if isinstance(record.get("locator"), dict) else {}
    event = payload.get("event") or locator.get("event") or "unknown"
    observed_at = record.get("observed_at")
    session = record.get("native_session_id")
    observed = observed_at if isinstance(observed_at, str) and observed_at else "unknown"
    native_id = session if isinstance(session, str) and session else "unknown"
    return f"observed_at={observed} event={event} session={native_id}"


def _format_timestamp(value: object) -> str:
    return value if isinstance(value, str) and value else "none"


def _print_capability(label: str, capability: object) -> None:
    data = capability if isinstance(capability, dict) else {}
    path = data.get("path", "")
    click.echo(
        f"{label}: exists={data.get('exists', False)} "
        f"readable={data.get('readable', False)} path={path}"
    )


def _status_error_affects_exit(error: object) -> bool:
    """Reuse capture's absence/retryable classifiers; missing timestamps are diagnoses."""

    if not isinstance(error, dict):
        return True
    if _is_absence_diagnostic(error):
        return False
    if _is_retryable_code(error.get("code")):
        return False
    return error.get("kind") != "missing_source_time"


def _print_sync_result(result: SyncResult) -> None:
    click.echo(_counts_line(result))


def _print_sync_followup(config: Config, paths: SourcePaths, result: SyncResult) -> None:
    if result["errors"] == 0 and result["pending"] == 0:
        return
    if result["errors"] > 0 and result["sessions"] > 0:
        click.echo(f"imported with {result['errors']} source diagnostics")
    if result["errors"] > 0:
        status = capture_status(config, paths)
        for error in status.get("errors") or []:
            click.echo(f"  {_format_diagnostic(error)}")
        click.echo("See `thirdeye copilot status` for source diagnostics.")
    if result["pending"] > 0:
        click.echo("Pending work remains; rerun sync or use `thirdeye copilot watch`.")


def _print_status(status: dict[str, Any]) -> None:
    paths = status.get("paths") or {}
    installation = status.get("installation") or {}
    pending = status.get("pending") or {}
    capabilities = status.get("capabilities") or {}
    configured = bool(installation.get("configured"))
    click.echo(f"Copilot home: {paths.get('home', '')}")
    click.echo(f"Session root: {paths.get('session_root', '')}")
    click.echo(f"Database: {paths.get('database', '')}")
    click.echo(f"Hooks file: {installation.get('hooks_file', '')}")
    if configured:
        click.echo("Hooks: configured")
    else:
        click.echo("Hooks: not configured (informational; persisted import still works)")
        click.echo("Install hooks with: thirdeye add --copilot")
    _print_capability("Transcripts", capabilities.get("transcripts"))
    _print_capability("Database", capabilities.get("database"))
    _print_capability("Database WAL", capabilities.get("database_wal"))
    click.echo(f"Last observed hook: {_format_last_hook(status.get('last_observed_hook'))}")
    click.echo(f"Last successful import: {_format_timestamp(status.get('last_successful_import'))}")
    click.echo(
        "Pending: "
        f"spool_records={pending.get('spool_records', 0)} "
        f"followup={pending.get('followup', 0)} "
        f"leases={pending.get('leases', 0)} "
        f"journals={pending.get('journals', 0)}"
    )
    errors = status.get("errors") or []
    informational = [error for error in errors if not _status_error_affects_exit(error)]
    blocking = [error for error in errors if _status_error_affects_exit(error)]
    if informational:
        click.echo("Informational:")
        for error in informational:
            click.echo(f"  {_format_diagnostic(error)}")
    if blocking:
        click.echo("Source errors:")
        for error in blocking:
            click.echo(f"  {_format_diagnostic(error)}")
    else:
        click.echo("Source errors: none")
    click.echo(
        "Start an interactive Copilot CLI session in a trusted folder, then rerun; "
        "'Last observed hook' should update."
    )


@click.group(name="copilot", help="Ingest local GitHub Copilot CLI recordings.")
def copilot_group() -> None:
    pass


@copilot_group.command("sync", help="One-shot local ingestion of Copilot CLI recordings.")
@click.option("--session-id", default=None, help="Exact native Copilot session ID.")
@_source_home_option
def sync_cmd(session_id: str | None, source_home: Path | None) -> None:
    config = Config.load()
    paths = _resolve_paths(source_home)
    try:
        if session_id is not None:
            validate_native_id(session_id)
        result = capture_sync(config, paths, session_id=session_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_sync_result(result)
    _print_sync_followup(config, paths, result)
    if session_id is not None and result["sessions"] == 0:
        raise click.ClickException(
            f"Copilot session {session_id} was not found or could not be imported"
        )


@copilot_group.command("watch", help="Poll local Copilot CLI recordings until interrupted.")
@_source_home_option
@click.option(
    "--interval",
    type=float,
    default=1.0,
    show_default=True,
    callback=_validate_interval,
    help="Seconds between source polls. Must be finite and at least 0.1.",
)
def watch_cmd(source_home: Path | None, interval: float) -> None:
    config = Config.load()
    paths = _resolve_paths(source_home)
    click.echo(
        f"Watching Copilot home {paths['home']} every {interval}s (local-only). Ctrl-C to stop."
    )
    try:
        watch_loop(config, paths, interval=interval)
    except KeyboardInterrupt:
        pass
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Stopped watching Copilot recordings.")


@copilot_group.command("status", help="Show Copilot CLI capture paths, hooks, and health.")
@_source_home_option
def status_cmd(source_home: Path | None) -> None:
    config = Config.load()
    paths = _resolve_paths(source_home)
    status = capture_status(config, paths)
    _print_status(status)
    if any(_status_error_affects_exit(error) for error in status.get("errors") or []):
        raise click.ClickException("Copilot source errors prevent a healthy status")
