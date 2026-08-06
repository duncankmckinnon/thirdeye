from __future__ import annotations

import json
import sys

import click

from thirdeye.config import Config
from thirdeye.ids import new_ulid
from thirdeye.store import Store


@click.command(help="Read newline-delimited JSON events from stdin and append them to a session.")
@click.option("--platform", required=True, help="Platform name (e.g. claude, codex).")
@click.option("--session-id", default=None, help="Session ID. Generated if omitted.")
@click.option("--cwd", default=None, help="Working directory for the session.")
def ingest(platform: str, session_id: str | None, cwd: str | None) -> None:
    sid = session_id or new_ulid()
    cwd_val = cwd or "."
    store = Store(Config.load())
    written = 0
    with store.open_session(sid, platform=platform, cwd=cwd_val) as w:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.pop("t", "event")
            data = obj.pop("data", None)
            w.append(t, data)
            written += 1
    click.echo(f"{sid}\t{written} events", err=True)
    click.echo(sid)
