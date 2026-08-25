from __future__ import annotations

import click

from thirdeye.config import Config, LogfireSettings
from thirdeye.otel_export import is_available, status


def _mask(token: str | None) -> str:
    return "********" if token else "(none)"


@click.group(name="logfire", help="Export thirdeye sessions to Pydantic Logfire, live.")
def logfire_group() -> None:
    pass


@logfire_group.command("enable", help="Turn on Logfire export and persist the write token.")
def enable() -> None:
    if not is_available():
        raise click.ClickException(
            "the `logfire` package is not installed. Install with: pip install 'thrdi[logfire]'"
        )
    token = click.prompt("Logfire write token (gateway key)", hide_input=True)
    config = Config.load()
    config.write_logfire_settings(LogfireSettings(enabled=True, token=token))
    click.echo(f"logfire export enabled (token {_mask(token)})")


@logfire_group.command("disable", help="Turn off Logfire export. Keeps the saved token.")
def disable() -> None:
    config = Config.load()
    settings = config.logfire
    config.write_logfire_settings(
        LogfireSettings(enabled=False, token=settings.token)
    )
    click.echo("logfire export disabled")


@logfire_group.command("status", help="Show whether Logfire export is configured and active.")
def show_status() -> None:
    config = Config.load()
    s = status(config)
    click.echo(f"package installed : {s['package_installed']}")
    click.echo(f"enabled           : {s['enabled']}")
    click.echo(f"token             : {_mask(config.logfire.token)}")
    active = s["package_installed"] and s["enabled"] and s["has_token"]
    click.echo(f"active            : {active}")
