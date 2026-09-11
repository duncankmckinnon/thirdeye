from __future__ import annotations

import click

from thirdeye.config import Config


@click.group(
    name="capture-env",
    help="Manage which environment variables thirdeye records as span attributes and tags.",
)
def capture_env_group() -> None:
    pass


@capture_env_group.command(
    "show", help="Show the active capture patterns and where they come from."
)
def show() -> None:
    import os

    import yaml

    config = Config.load()
    env_raw = os.environ.get("THIRDEYE_CAPTURE_ENV", "")
    try:
        persisted = yaml.safe_load(config.config_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        persisted = {}
    if env_raw.strip():
        source = "THIRDEYE_CAPTURE_ENV (overrides config.yaml)"
    elif isinstance(persisted, dict) and "capture_env" in persisted:
        source = f"config.yaml ({config.config_file})"
    else:
        source = "built-in default"
    patterns = ", ".join(config.capture_env_patterns) or "(none)"
    click.echo(f"patterns : {patterns}")
    click.echo(f"source   : {source}")


@capture_env_group.command("set", help="Persist capture patterns to config.yaml, e.g. 'WB_*'.")
@click.argument("patterns", nargs=-1, required=True)
def set_patterns(patterns: tuple[str, ...]) -> None:
    # Accept both `set WB_* BUILD_LABEL` and `set 'WB_*,BUILD_LABEL'`.
    flat = tuple(p.strip() for chunk in patterns for p in chunk.split(",") if p.strip())
    config = Config.load().write_capture_env_patterns(flat)
    click.echo(f"capture_env set to: {', '.join(config.capture_env_patterns)}")
    click.echo(f"written to {config.config_file}")


@capture_env_group.command("clear", help="Turn capture off by persisting an empty capture_env.")
def clear() -> None:
    Config.load().write_capture_env_patterns(())
    click.echo("capture_env cleared (capture disabled)")
