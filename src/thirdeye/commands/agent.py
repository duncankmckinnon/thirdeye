from __future__ import annotations

import sys
from pathlib import Path

import click

from thirdeye.agent.exec import run_agent_streaming
from thirdeye.agent.harness import AgentHarness
from thirdeye.agent.prompt import VALID_SKILLS, build_agent_prompt
from thirdeye.config import Config
from thirdeye.eval.agents import get_adapter, list_agent_names


@click.command(name="agent", help="Run an AI agent against your thirdeye sessions.")
@click.argument("task")
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
    "skills",
    multiple=True,
    metavar="SKILL",
    help=(
        "Skill to inject into the prompt (repeatable). "
        f"Valid: {', '.join(sorted(VALID_SKILLS))}. "
        "Defaults to use-thirdeye and thirdeye-review."
    ),
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
    help="Allocate a pseudo-terminal so the agent streams output in real time (Unix only).",
)
def agent_cmd(
    task: str,
    agent_name: str,
    fix_mode: bool,
    skills: tuple[str, ...],
    cwd: Path | None,
    stream: bool,
) -> None:
    config = Config.load()

    available = list_agent_names(config.root)
    if agent_name not in available:
        raise click.ClickException(
            f"unknown agent {agent_name!r} — available: {', '.join(available)}"
        )

    skill_list = list(skills) if skills else None  # None → DEFAULT_SKILLS in build_agent_prompt

    try:
        prompt = build_agent_prompt(
            task,
            skills=skill_list,
            cwd=cwd or Path.cwd(),
            thirdeye_home=config.root,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    try:
        adapter = get_adapter(agent_name, thirdeye_home=config.root)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    mode = "fix" if fix_mode else "review"
    harness = AgentHarness(adapter, mode)

    use_pty = stream
    if stream and sys.platform == "win32":
        click.echo(
            "Warning: --stream uses a pseudo-terminal which is not supported on Windows; "
            "falling back to standard pipe output.",
            err=False,
        )
        use_pty = False

    try:
        returncode, _ = run_agent_streaming(
            harness,
            prompt,
            cwd=cwd or Path.cwd(),
            thirdeye_home=config.root,
            use_pty=use_pty,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    if returncode != 0:
        sys.exit(returncode)
