from __future__ import annotations

import click

from thirdeye.config import Config, LogfireSettings
from thirdeye.logfire_auth import LogfireAuthError, mint_write_token
from thirdeye.otel_export import is_available, status


def _mask(token: str | None) -> str:
    return "********" if token else "(none)"


def enable_with_token(token: str) -> None:
    """Persist a Logfire write token and enable live export."""
    config = Config.load()
    config.write_logfire_settings(
        LogfireSettings(enabled=True, token=token, api_key=config.logfire.api_key)
    )


@click.group(name="logfire", help="Export thirdeye sessions to Pydantic Logfire, live.")
def logfire_group() -> None:
    pass


@logfire_group.command("enable", help="Turn on Logfire export and persist the write token.")
@click.option(
    "--auth",
    "force_auth",
    is_flag=True,
    help="Re-authenticate with Logfire and mint a new write token.",
)
def enable(force_auth: bool) -> None:
    if not is_available():
        raise click.ClickException(
            "the `logfire` package is not installed. Install with: pip install 'thrdi[logfire]'"
        )
    config = Config.load()
    token = config.logfire.token
    if (
        token
        and not force_auth
        and click.confirm("Use the saved Logfire write token?", default=True)
    ):
        enable_with_token(token)
        click.echo(f"logfire export enabled (token {_mask(token)})")
        return
    try:
        token = mint_write_token(force_auth=True) if force_auth else mint_write_token()
    except LogfireAuthError as exc:
        raise click.ClickException(str(exc)) from exc
    enable_with_token(token)
    click.echo(f"logfire export enabled (token {_mask(token)})")


@logfire_group.command("disable", help="Turn off Logfire export. Keeps the saved token.")
def disable() -> None:
    config = Config.load()
    settings = config.logfire
    config.write_logfire_settings(
        LogfireSettings(enabled=False, token=settings.token, api_key=settings.api_key)
    )
    click.echo("logfire export disabled")


@logfire_group.command("status", help="Show whether Logfire export is configured and active.")
def show_status() -> None:
    config = Config.load()
    s = status(config)
    click.echo(f"package installed : {s['package_installed']}")
    click.echo(f"enabled           : {s['enabled']}")
    click.echo(f"token             : {_mask(config.logfire.token)}")
    click.echo(f"dataset API key   : {_mask(config.logfire.api_key)}")
    active = s["package_installed"] and s["enabled"] and s["has_token"]
    click.echo(f"active            : {active}")
