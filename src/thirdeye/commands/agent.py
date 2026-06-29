from __future__ import annotations

import sys
from pathlib import Path

import click

from thirdeye.agent.exec import run_agent_streaming
from thirdeye.agent.harness import AgentHarness
from thirdeye.agent.prompt import (
    VALID_SKILLS,
    build_agent_prompt,
    load_builtin_skills,
    load_skill_file,
)
from thirdeye.config import Config
from thirdeye.eval.agents import get_adapter, list_agent_names


@click.command(
    name="agent",
    help=(
        "Run an AI agent against your thirdeye sessions.\n\n"
        "The agent is given all available skills by default and runs in read-only mode "
        "unless --fix is set. Use --skill to inject custom skills from local files, "
        "or --skills to list the built-in skills. Use --stream to see tool calls and "
        "results in real time."
    ),
)
@click.argument("task", required=False, default=None)
@click.option(
    "--agent",
    "agent_name",
    default="claude",
    show_default=True,
    help="Agent CLI to dispatch (claude, codex, gemini, or a custom name).",
)
@click.option(
    "--fix",
    "fix_mode",
    is_flag=True,
    default=False,
    help="Unlock full tool access so the agent can edit files (default: read-only).",
)
@click.option(
    "--skill",
    "skill_paths",
    multiple=True,
    metavar="PATH",
    help=(
        "Path to a skill file to inject into the prompt (repeatable). "
        "When omitted, all built-in skills are used. "
        "Use --skills to see available built-in skills."
    ),
)
@click.option(
    "--skills",
    "list_skills",
    is_flag=True,
    default=False,
    help="List the built-in skills available to the agent and exit.",
)
@click.option(
    "--cwd",
    "cwd",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Working directory context injected into the prompt (default: current dir).",
)
@click.option(
    "--stream",
    "stream",
    is_flag=True,
    default=False,
    help="Show tool calls and results as the agent explores (streams intermediate events).",
)
def agent_cmd(
    task: str | None,
    agent_name: str,
    fix_mode: bool,
    skill_paths: tuple[str, ...],
    list_skills: bool,
    cwd: Path | None,
    stream: bool,
) -> None:
    if list_skills:
        click.echo("Available skills:")
        for name in sorted(VALID_SKILLS):
            click.echo(f"  {name}")
        return

    if task is None:
        raise click.UsageError("Missing argument 'TASK'.")

    config = Config.load()

    available = list_agent_names(config.root)
    if agent_name not in available:
        raise click.ClickException(
            f"unknown agent {agent_name!r} — available: {', '.join(available)}"
        )

    try:
        if skill_paths:
            bodies = load_builtin_skills() + [load_skill_file(Path(p)) for p in skill_paths]
            prompt = build_agent_prompt(
                task,
                skill_bodies=bodies,
                cwd=cwd or Path.cwd(),
                thirdeye_home=config.root,
            )
        else:
            prompt = build_agent_prompt(
                task,
                cwd=cwd or Path.cwd(),
                thirdeye_home=config.root,
            )
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e)) from e

    try:
        adapter = get_adapter(agent_name, thirdeye_home=config.root)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    mode = "fix" if fix_mode else "review"
    harness = AgentHarness(adapter, mode, streaming=stream)

    try:
        returncode, _ = run_agent_streaming(
            harness,
            prompt,
            cwd=cwd or Path.cwd(),
            thirdeye_home=config.root,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    if returncode != 0:
        sys.exit(returncode)
