from __future__ import annotations

import click

from thirdeye import __version__
from thirdeye.commands.add import add, remove
from thirdeye.commands.eval import eval_group
from thirdeye.commands.ingest import ingest
from thirdeye.commands.reads import event, events, list_sessions, search, show, stats, tail
from thirdeye.commands.skill import skill
from thirdeye.commands.tags import tag, tags
from thirdeye.commands.ui import ui
from thirdeye.commands.usage import usage
from thirdeye.commands.views import views_group


@click.group(name="thirdeye", help="Trace agentic CLIs to a unified local store.")
@click.version_option(__version__, prog_name="thirdeye")
def main() -> None:
    pass


main.add_command(add)
main.add_command(remove)
main.add_command(ingest)
main.add_command(list_sessions)
main.add_command(show)
main.add_command(events)
main.add_command(tail)
main.add_command(event)
main.add_command(search)
main.add_command(skill)
main.add_command(tag)
main.add_command(tags)
main.add_command(stats)
main.add_command(usage)
main.add_command(eval_group)
main.add_command(ui)
main.add_command(views_group)
