from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from thirdeye.commands import add as add_commands
from thirdeye.commands import logfire_cmd, skill
from thirdeye.config import Config
from thirdeye.platforms.base import Platform
from thirdeye.platforms.codex.install import CodexPlatform

_PLATFORM_LABELS = {
    "claude": "Claude Code",
    "codex": "Codex CLI",
    "cursor": "Cursor",
}

_SKILL_TARGET_LABELS = {
    skill.CLAUDE_TARGET: "Claude Code",
    skill.CODEX_TARGET: "Codex CLI",
    skill.DEFAULT_TARGET: "shared agent skills",
}


def _prompt_multiselect(
    prompt: str,
    choices: list[tuple[str, str]],
    *,
    default: str,
) -> list[str]:
    """Prompt for comma-separated choice numbers (plus ``all``/``none``)."""
    for index, (_, label) in enumerate(choices, start=1):
        click.echo(f"  {index}. {label}")

    by_number = {str(index): key for index, (key, _) in enumerate(choices, start=1)}
    valid_keys = {key.lower(): key for key, _ in choices}
    while True:
        raw = click.prompt(prompt, default=default).strip().lower()
        if raw in {"", "none"}:
            return []
        if raw == "all":
            return [key for key, _ in choices]
        selected: list[str] = []
        invalid: list[str] = []
        for token in (part.strip() for part in raw.split(",")):
            key = by_number.get(token) or valid_keys.get(token)
            if key is None:
                invalid.append(token)
            elif key not in selected:
                selected.append(key)
        if not invalid:
            return selected
        click.echo(
            f"  Invalid selection: {', '.join(invalid)}. "
            "Enter comma-separated numbers, all, or none."
        )


def _install_tracing(platform_name: str, platform: Platform | None = None) -> bool:
    platform = platform or add_commands._resolve_platform(platform_name)
    if platform_name == "codex":
        incumbent = cast(CodexPlatform, platform).notify_conflict()
        if incumbent:
            click.echo(f"  Codex notify is already used by {incumbent!r}.")
            click.echo(
                "  Codex supports one notify program; replacing it may disable "
                "features that depend on the current notifier."
            )
            if not click.confirm(
                "Replace the existing Codex notifier and install thirdeye?",
                default=False,
            ):
                click.echo("  Skipped Codex tracing; existing notifier was not changed")
                return False
            platform = add_commands._resolve_platform(platform_name, force=True)
    platform.install()
    click.echo(f"  Installed tracing for {platform.display_name}")
    return True


def _skill_targets(platforms: list[str]) -> list[Path]:
    targets: list[Path] = []
    if "claude" in platforms:
        targets.append(skill.CLAUDE_TARGET)
    if "codex" in platforms:
        targets.append(skill.CODEX_TARGET)
    if "cursor" in platforms or not targets:
        targets.append(skill.DEFAULT_TARGET)
    return targets


def _install_new_skills(platforms: list[str]) -> str:
    names = skill._list_bundled_skills()
    if not names:
        raise click.ClickException("no bundled skills found")

    missing_by_key: dict[str, tuple[Path, list[str]]] = {}
    conflicts: list[Path] = []
    for target in _skill_targets(platforms):
        missing: list[str] = []
        for name in names:
            state = skill._install_state(name, target / name)
            if state == "missing":
                missing.append(name)
            elif state == "conflict":
                conflicts.append(target / name)
        if missing:
            missing_by_key[str(target)] = (target, missing)
        elif not any((target / name) in conflicts for name in names):
            click.echo(f"  {_SKILL_TARGET_LABELS[target]}: already up to date")

    for path in conflicts:
        click.echo(f"  Skipping {path}: an existing non-thirdeye entry is in the way")

    if not missing_by_key:
        return "up to date" if not conflicts else "no new skills installed"

    choices = [
        (key, f"{_SKILL_TARGET_LABELS[target]} ({len(missing)} new)")
        for key, (target, missing) in missing_by_key.items()
    ]
    selected = _prompt_multiselect("Install new skills in", choices, default="all")
    if not selected:
        click.echo("  Skipped new agent skills")
        return "skipped"

    installed = 0
    for key in selected:
        target, missing = missing_by_key[key]
        for name in missing:
            click.echo(f"  {skill._install_one(name, target / name, force=False)}")
            installed += 1
    return f"installed {installed} new"


def _configure_logfire() -> str:
    settings = Config.load().logfire

    if settings.token:
        state = "enabled" if settings.enabled else "disabled"
        click.echo(f"  Logfire export is {state} with a saved token (********)")
        if click.confirm("Change the saved Logfire write token?", default=False):
            token = click.prompt("New Logfire write token (gateway key)", hide_input=True)
            logfire_cmd.enable_with_token(token)
            click.echo("  Updated the token and enabled Logfire export (********)")
            return "token updated"
        if settings.enabled:
            return "already configured"
        if click.confirm("Enable Logfire export with the saved token?", default=True):
            logfire_cmd.enable_with_token(settings.token)
            click.echo("  Logfire export enabled (token ********)")
            return "enabled"
        return "skipped"

    if not click.confirm("Enable live Logfire export?", default=False):
        click.echo("  Skipped Logfire export")
        return "skipped"
    token = click.prompt("Logfire write token (gateway key)", hide_input=True)
    logfire_cmd.enable_with_token(token)
    click.echo("  Logfire export enabled (token ********)")
    return "enabled"


@click.command(help="Interactively configure tracing, agent skills, and Logfire export.")
def setup() -> None:
    """Walk through the complete thirdeye setup flow."""
    click.echo("Set up thirdeye\n")
    click.echo("Agent tracing")

    platform_objects = {
        name: add_commands._resolve_platform(name) for name in _PLATFORM_LABELS
    }
    configured = [name for name, platform in platform_objects.items() if platform.is_installed()]
    for name in configured:
        click.echo(f"  {_PLATFORM_LABELS[name]}: already configured")

    available = [name for name in _PLATFORM_LABELS if name not in configured]
    if available:
        selected = _prompt_multiselect(
            "Configure tracing for",
            [(name, _PLATFORM_LABELS[name]) for name in available],
            default="none",
        )
    else:
        selected = []
        click.echo("  All supported agents are already configured")

    newly_configured: list[str] = []
    for name in selected:
        if _install_tracing(name, platform_objects[name]):
            newly_configured.append(name)
    all_configured = configured + newly_configured

    click.echo("\nAgent skills")
    skills_status = _install_new_skills(all_configured)

    if logfire_cmd.is_available():
        click.echo("\nPydantic Logfire")
        logfire_status = _configure_logfire()
    else:
        logfire_status = "extension not installed"

    click.echo("\nSetup complete.")
    if all_configured:
        labels = ", ".join(_PLATFORM_LABELS[name] for name in all_configured)
        click.echo(f"  Tracing: {labels}")
    else:
        click.echo("  Tracing: no platforms configured")
    click.echo(f"  Skills: {skills_status}")
    click.echo(f"  Logfire: {logfire_status}")
