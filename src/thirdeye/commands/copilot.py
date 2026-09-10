from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import click

from thirdeye.config import Config
from thirdeye.platforms.copilot.capture import sync as capture_sync
from thirdeye.platforms.copilot.identity import resolve_sources, validate_native_id
from thirdeye.platforms.copilot.status import capture_status
from thirdeye.platforms.copilot.types import SyncResult
from thirdeye.platforms.copilot.watch import watch as watch_loop

_MIN_WATCH_INTERVAL = 0.1


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


def _print_sync_result(result: SyncResult) -> None:
    click.echo(
        f"sessions={result['sessions']} "
        f"records_written={result['records_written']} "
        f"duplicate_records={result['duplicate_records']} "
        f"pending={result['pending']} "
        f"errors={result['errors']}"
    )


def _print_status(status: dict[str, Any]) -> None:
    paths = status.get("paths") or {}
    installation = status.get("installation") or {}
    pending = status.get("pending") or {}
    configured = bool(installation.get("configured"))
    click.echo(f"Copilot home: {paths.get('home', '')}")
    click.echo(f"Session root: {paths.get('session_root', '')}")
    click.echo(f"Database: {paths.get('database', '')}")
    click.echo(f"Hooks file: {installation.get('hooks_file', '')}")
    if configured:
        click.echo("Hooks: configured")
        click.echo("Restart Copilot CLI or start a new interactive session if hooks were just installed.")
    else:
        click.echo("Hooks: not configured (informational; persisted import still works)")
        click.echo("Install hooks with: thirdeye add --copilot")
    click.echo(f"Last observed hook: {status.get('last_observed_hook')}")
    click.echo(f"Last successful import: {status.get('last_successful_import')}")
    click.echo(
        "Pending: "
        f"spool_records={pending.get('spool_records', 0)} "
        f"followup={pending.get('followup', 0)} "
        f"leases={pending.get('leases', 0)} "
        f"journals={pending.get('journals', 0)}"
    )
    errors = status.get("errors") or []
    if errors:
        click.echo("Source errors:")
        for error in errors:
            if isinstance(error, dict):
                kind = error.get("kind") or error.get("code") or "error"
                message = error.get("message") or error.get("reason") or error
                click.echo(f"  {kind}: {message}")
            else:
                click.echo(f"  {error}")
    else:
        click.echo("Source errors: none")
    click.echo("Verify receipt with: thirdeye copilot status")


@click.group(name="copilot", help="Ingest local GitHub Copilot CLI recordings.")
def copilot_group() -> None:
    pass


@copilot_group.command("sync", help="One-shot local ingestion of Copilot CLI recordings.")
@click.option("--session-id", default=None, help="Exact native Copilot session ID.")
@_source_home_option
def sync_cmd(session_id: str | None, source_home: Path | None) -> None:
    config = Config.load()
    paths = resolve_sources(source_home)
    try:
        if session_id is not None:
            validate_native_id(session_id)
        result = capture_sync(config, paths, session_id=session_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_sync_result(result)
    if session_id is not None and result["errors"] > 0:
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
    paths = resolve_sources(source_home)
    watch_loop(config, paths, interval=interval)


@copilot_group.command("status", help="Show Copilot CLI capture paths, hooks, and health.")
@_source_home_option
def status_cmd(source_home: Path | None) -> None:
    config = Config.load()
    paths = resolve_sources(source_home)
    status = capture_status(config, paths)
    _print_status(status)
    if status.get("errors"):
        raise click.ClickException("Copilot source errors prevent a healthy status")
