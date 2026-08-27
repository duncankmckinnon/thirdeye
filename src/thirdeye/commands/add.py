from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import click

from thirdeye.platforms.base import Platform
from thirdeye.platforms.claude.install import ClaudePlatform
from thirdeye.platforms.codex.install import CodexPlatform
from thirdeye.platforms.cursor.install import CursorPlatform

PLATFORMS: dict[str, type[Platform]] = {
    "claude": ClaudePlatform,
    "codex": CodexPlatform,
    "cursor": CursorPlatform,
}

# Config files that may still reference console scripts for platforms this
# version no longer installs (Gemini). Deleting those platforms
# leaves their hook entries orphaned in other tools' config, firing a missing
# binary on every event, so `thirdeye add --list` warns about them.
ORPHAN_CONFIG_PATHS: tuple[Path, ...] = (
    Path.home() / ".gemini" / "settings.json",
)


def _platform_options(fn):
    fn = click.option("--cursor", "platform_flag", flag_value="cursor", help="Cursor.")(fn)
    fn = click.option("--codex", "platform_flag", flag_value="codex", help="Codex CLI.")(fn)
    fn = click.option("--claude", "platform_flag", flag_value="claude", help="Claude Code.")(fn)
    return fn


def _resolve_platform(platform_flag: str | None, *, force: bool = False) -> Platform:
    if not platform_flag:
        raise click.UsageError("Pick a platform: --claude, --codex, --cursor")
    platform_cls = PLATFORMS[platform_flag]
    if platform_flag == "codex" and force:
        return platform_cls(force=True)
    return platform_cls()


def _is_stale_command(command: str) -> bool:
    """True if a hook command targets a removed-platform console script."""
    name = Path(command).name
    return name.startswith("thirdeye-gemini-")


def find_orphaned_hooks(
    config_paths: Iterable[Path] = ORPHAN_CONFIG_PATHS,
) -> list[tuple[Path, str]]:
    """Return (config_file, stale_command) for each removed-platform hook found.

    A command is stale when its basename starts with "thirdeye-gemini-".
    Missing or malformed files yield nothing.
    """
    found: list[tuple[Path, str]] = []
    for path in config_paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for command in _iter_commands(data):
            if _is_stale_command(command):
                found.append((path, command))
    return found


def _iter_commands(node: object) -> Iterable[str]:
    """Yield every string under a "command" key anywhere in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "command" and isinstance(value, str):
                yield value
            else:
                yield from _iter_commands(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_commands(item)


@click.command(help="Install tracing hooks for an agentic platform.")
@click.option(
    "--list",
    "list_platforms",
    is_flag=True,
    help="List supported platforms and warn about orphaned hooks from removed platforms.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Codex only: take over an existing 'notify' program instead of erroring.",
)
@_platform_options
def add(platform_flag: str | None, list_platforms: bool, force: bool) -> None:
    if list_platforms:
        click.echo("Supported platforms:")
        for name in PLATFORMS:
            click.echo(f"  {name}")
        for path, command in find_orphaned_hooks(ORPHAN_CONFIG_PATHS):
            click.echo(
                f"Warning: {path} still references removed hook {command!r}. "
                "Remove it from that tool's config to stop the missing-binary error.",
            )
        return
    platform = _resolve_platform(platform_flag, force=force)
    platform.install()
    click.echo(f"Installed tracing for {platform.display_name}")


@click.command(help="Remove tracing hooks for an agentic platform.")
@_platform_options
def remove(platform_flag: str | None) -> None:
    platform = _resolve_platform(platform_flag)
    platform.uninstall()
    click.echo(f"Removed tracing for {platform.display_name}")
