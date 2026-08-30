from __future__ import annotations

from pathlib import Path

import click

from thirdeye.commands import add as add_commands
from thirdeye.commands import logfire_cmd, skill

_PLATFORM_LABELS = {
    "claude": "Claude Code",
    "codex": "Codex CLI",
    "cursor": "Cursor",
}


def _install_tracing(platform_name: str) -> None:
    platform = add_commands._resolve_platform(platform_name)
    platform.install()
    click.echo(f"  Installed tracing for {platform.display_name}")


def _skill_targets(platforms: list[str]) -> list[Path]:
    targets: list[Path] = []
    if "claude" in platforms:
        targets.append(skill.CLAUDE_TARGET)
    if "codex" in platforms:
        targets.append(skill.CODEX_TARGET)
    if "cursor" in platforms or not targets:
        targets.append(skill.DEFAULT_TARGET)
    return targets


def _install_skills(platforms: list[str]) -> None:
    names = skill._list_bundled_skills()
    if not names:
        raise click.ClickException("no bundled skills found")
    for target in _skill_targets(platforms):
        for name in names:
            click.echo(f"  {skill._install_one(name, target / name, force=False)}")


def _enable_logfire() -> bool:
    if not logfire_cmd.is_available():
        click.echo(
            "  Logfire support is not installed; run "
            "pip install 'thrdi[logfire]' and then thirdeye logfire enable."
        )
        return False

    token = click.prompt("Logfire write token (gateway key)", hide_input=True)
    logfire_cmd.enable_with_token(token)
    click.echo("  Logfire export enabled (token ********)")
    return True


@click.command(help="Interactively configure tracing, agent skills, and Logfire export.")
def setup() -> None:
    """Walk through the complete thirdeye setup flow."""
    click.echo("Set up thirdeye\n")
    click.echo("Agent tracing")

    platforms: list[str] = []
    for name, label in _PLATFORM_LABELS.items():
        if click.confirm(f"Configure tracing for {label}?", default=False):
            _install_tracing(name)
            platforms.append(name)

    click.echo("\nAgent skills")
    skills_installed = click.confirm("Install thirdeye's bundled agent skills?", default=True)
    if skills_installed:
        _install_skills(platforms)
    else:
        click.echo("  Skipped agent skills")

    click.echo("\nPydantic Logfire")
    logfire_enabled = False
    if click.confirm("Enable live Logfire export?", default=False):
        logfire_enabled = _enable_logfire()
    else:
        click.echo("  Skipped Logfire export")

    click.echo("\nSetup complete.")
    if platforms:
        labels = ", ".join(_PLATFORM_LABELS[name] for name in platforms)
        click.echo(f"  Tracing: {labels}")
    else:
        click.echo("  Tracing: no platforms configured")
    click.echo(f"  Skills: {'installed' if skills_installed else 'skipped'}")
    click.echo(f"  Logfire: {'enabled' if logfire_enabled else 'skipped'}")
