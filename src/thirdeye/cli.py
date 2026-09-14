from __future__ import annotations

import click

from thirdeye import __version__
from thirdeye._compat.streams import force_utf8_stdio
from thirdeye.commands.add import add, remove
from thirdeye.commands.agent import agent_cmd
from thirdeye.commands.capture_env import capture_env_group
from thirdeye.commands.copilot import copilot_group
from thirdeye.commands.eval import eval_group
from thirdeye.commands.ingest import ingest
from thirdeye.commands.logfire_cmd import logfire_group
from thirdeye.commands.reads import event, events, list_sessions, search, show, stats, tail
from thirdeye.commands.setup import setup
from thirdeye.commands.skill import skills_group
from thirdeye.commands.tags import tag, tags
from thirdeye.commands.ui import serve, ui
from thirdeye.commands.usage import usage
from thirdeye.commands.views import views_group


@click.group(name="thirdeye", help="Trace agentic CLIs to a unified local store.")
@click.version_option(__version__, prog_name="thirdeye")
def main() -> None:
    pass


main.add_command(add)
main.add_command(remove)
main.add_command(ingest)
main.add_command(copilot_group)
main.add_command(list_sessions)
main.add_command(show)
main.add_command(events)
main.add_command(tail)
main.add_command(event)
main.add_command(search)
main.add_command(skills_group)
main.add_command(tag)
main.add_command(tags)
main.add_command(stats)
main.add_command(usage)
main.add_command(eval_group)
main.add_command(ui)
main.add_command(serve)
main.add_command(views_group)
main.add_command(agent_cmd)
main.add_command(logfire_group)
main.add_command(capture_env_group)
main.add_command(setup)


def run() -> None:
    """Console-script entry point.

    Wraps the ``main`` group rather than folding the stdio setup into it so the
    encoding is fixed before Click can emit anything -- ``--help`` and
    ``--version`` are handled during parsing, before a group callback would run
    -- and so in-process callers (tests using Click's ``CliRunner``) keep
    invoking ``main`` against the streams they installed.
    """
    force_utf8_stdio()
    main()
