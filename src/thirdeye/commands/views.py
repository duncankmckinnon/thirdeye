from __future__ import annotations

import click

from thirdeye.config import Config
from thirdeye.web.views_store import ALLOWED_PAGES, SavedView, ViewStore, now_iso


@click.group(name="views", help="Manage saved UI filter views.")
def views_group() -> None:
    pass


@views_group.command("list", help="List saved views for a page.")
@click.option(
    "--page",
    required=True,
    type=click.Choice(sorted(ALLOWED_PAGES)),
    help="Page these views belong to.",
)
def list_cmd(page: str) -> None:
    config = Config.load()
    for v in ViewStore(config.root, page).list():
        click.echo(f"{v.name}\t{v.path}\t{v.created_at}")


@views_group.command("save", help="Save a view by name.")
@click.option(
    "--page",
    required=True,
    type=click.Choice(sorted(ALLOWED_PAGES)),
)
@click.option("--name", required=True)
@click.option(
    "--path",
    required=True,
    help="Query string starting with '?', e.g. '?since=7d&order=newest'",
)
def save_cmd(page: str, name: str, path: str) -> None:
    config = Config.load()
    try:
        ViewStore(config.root, page).save(
            SavedView(name=name, path=path, created_at=now_iso())
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"saved {name}")


@views_group.command("delete", help="Delete a saved view by name.")
@click.option(
    "--page",
    required=True,
    type=click.Choice(sorted(ALLOWED_PAGES)),
)
@click.option("--name", required=True)
def delete_cmd(page: str, name: str) -> None:
    config = Config.load()
    ViewStore(config.root, page).delete(name)
    click.echo(f"deleted {name}")
