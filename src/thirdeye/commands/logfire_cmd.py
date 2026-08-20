from __future__ import annotations

import click

from thirdeye.config import Config, LogfireSettings
from thirdeye.otel_export import is_available, status


def _mask(token: str | None) -> str:
    if not token:
        return "(none)"
    return f"...{token[-4:]}" if len(token) > 4 else "***"


@click.group(name="logfire", help="Export thirdeye sessions to Pydantic Logfire, live.")
def logfire_group() -> None:
    pass


@logfire_group.command("enable", help="Turn on Logfire export and persist the write token.")
@click.option("--token", required=True, help="Logfire write token (gateway key) for a project.")
@click.option("--project", default=None, help="Logfire project name, for your own reference.")
def enable(token: str, project: str | None) -> None:
    if not is_available():
        raise click.ClickException(
            "the `logfire` package is not installed. Install with: pip install 'thrdi[logfire]'"
        )
    config = Config.load()
    config.write_logfire_settings(LogfireSettings(enabled=True, token=token, project=project))
    click.echo(f"logfire export enabled (token {_mask(token)}, project={project or '(unset)'})")


@logfire_group.command("disable", help="Turn off Logfire export. Keeps the saved token.")
def disable() -> None:
    config = Config.load()
    settings = config.logfire
    config.write_logfire_settings(
        LogfireSettings(enabled=False, token=settings.token, project=settings.project)
    )
    click.echo("logfire export disabled")


@logfire_group.command("status", help="Show whether Logfire export is configured and active.")
def show_status() -> None:
    config = Config.load()
    s = status(config)
    click.echo(f"package installed : {s['package_installed']}")
    click.echo(f"enabled           : {s['enabled']}")
    click.echo(f"token             : {_mask(config.logfire.token)}")
    click.echo(f"project           : {s['project'] or '(unset)'}")
    active = s["package_installed"] and s["enabled"] and s["has_token"]
    click.echo(f"active            : {active}")
