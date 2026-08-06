from __future__ import annotations

import json
import time
from datetime import datetime

import click

from thirdeye.config import Config
from thirdeye.paths import session_dir, usage_log_path
from thirdeye.timeparse import parse_when
from thirdeye.usage.index import UsageIndex


def _parse_window(value: str | None, flag: str) -> datetime | None:
    if value is None:
        return None
    try:
        return parse_when(value)
    except ValueError as e:
        raise click.ClickException(f"could not parse {flag} {value!r}: {e}") from e


def _resolve_session(config: Config, prefix: str) -> tuple[str, str]:
    from thirdeye.store import Store

    try:
        return Store(config).resolve_session_id(prefix)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


class _UsageGroup(click.Group):
    """Route non-subcommand args to the default `show` subcommand.

    Lets `thirdeye usage [SESSION_PREFIX] [OPTIONS]` work alongside
    `thirdeye usage reindex` and `thirdeye usage errors` without the
    positional-vs-subcommand parsing collision that arises from putting
    a positional argument directly on an `invoke_without_command` group.
    """

    def parse_args(self, ctx, args):
        if not args:
            args = ["show"]
        elif args[0] not in self.commands and args[0] not in ("--help", "-h"):
            args = ["show", *args]
        return super().parse_args(ctx, args)


@click.group(
    cls=_UsageGroup,
    name="usage",
    help="Per-event model and token usage.",
)
def usage():
    pass


@usage.command(name="show")
@click.argument("session_prefix", required=False)
@click.option("--json", "as_json", is_flag=True, help="JSONL output.")
@click.option("--platform", "platform_filter", default=None)
@click.option(
    "--harness",
    "harness_filter",
    default=None,
    help="Alias for --platform.",
)
@click.option(
    "--model",
    "model_filter",
    default=None,
    help="Filter rows where `model` contains this substring.",
)
@click.option("--since", default=None, help="Time window lower bound.")
@click.option("--until", default=None, help="Time window upper bound.")
@click.option(
    "--top",
    type=int,
    default=None,
    help="Rollup mode: keep top N sessions by total_tokens.",
)
@click.option(
    "--sort",
    type=click.Choice(["total", "input", "output", "ts"]),
    default=None,
)
def show_cmd(
    session_prefix,
    as_json,
    platform_filter,
    harness_filter,
    model_filter,
    since,
    until,
    top,
    sort,
):
    """Render per-session or rollup usage view."""
    _run_show(
        session_prefix=session_prefix,
        as_json=as_json,
        platform_filter=platform_filter or harness_filter,
        model_filter=model_filter,
        since=since,
        until=until,
        top=top,
        sort=sort,
    )


@usage.command(name="reindex")
@click.argument("session_prefix", required=False)
def reindex_cmd(session_prefix):
    """Force-rebuild usage.db from sidecars."""
    config = Config.load()
    idx = UsageIndex(config.root)
    conn = idx.connect()
    t0 = time.monotonic()
    if session_prefix:
        platform, sid = _resolve_session(config, session_prefix)
        conn.execute("DELETE FROM usage WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM usage_sync WHERE session_id = ?", (sid,))
        conn.commit()
        sd = session_dir(config.root, platform, sid)
        n = idx.refresh_session(conn, sid, sd)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        click.echo(f"Indexed {n} rows from 1 session in {elapsed_ms} ms")
    else:
        conn.execute("DELETE FROM usage")
        conn.execute("DELETE FROM usage_sync")
        conn.commit()
        n = idx.refresh(conn)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM usage").fetchone()[0]
        click.echo(f"Indexed {n} rows from {sessions} sessions in {elapsed_ms} ms")


@usage.command(name="errors")
@click.option("-n", "n", type=int, default=20, help="Last N entries.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--platform", "platform_filter", default=None)
@click.option("--phase", default=None)
@click.option("--since", default=None)
@click.option("--until", default=None)
def errors_cmd(n, as_json, platform_filter, phase, since, until):
    """Show entries from <thirdeye_home>/logs/usage-errors.jsonl."""
    config = Config.load()
    log = usage_log_path(config.root)
    if not log.exists():
        click.echo("No usage errors logged.")
        return

    since_dt = _parse_window(since, "--since")
    until_dt = _parse_window(until, "--until")

    entries: list[dict] = []
    with log.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if platform_filter and e.get("platform") != platform_filter:
                continue
            if phase and e.get("phase") != phase:
                continue
            if since_dt or until_dt:
                ts_raw = e.get("ts", "")
                try:
                    iso = ts_raw[:-1] + "+00:00" if ts_raw.endswith("Z") else ts_raw
                    e_ts = datetime.fromisoformat(iso)
                except (TypeError, ValueError):
                    continue
                if since_dt and e_ts < since_dt:
                    continue
                if until_dt and e_ts > until_dt:
                    continue
            entries.append(e)

    entries = entries[-n:]
    if as_json:
        for e in entries:
            click.echo(json.dumps(e, separators=(",", ":")))
        return
    if not entries:
        click.echo("No matching entries.")
        return
    for e in entries:
        click.echo(
            f"{e.get('ts', '')}  {e.get('level', '?'):<5}  "
            f"{e.get('platform', '?'):<7}  {e.get('phase', '?'):<20}  "
            f"{e.get('session_id', '')[:12]:<12}  {e.get('message', '')}"
        )


def _run_show(
    *,
    session_prefix,
    as_json,
    platform_filter,
    model_filter,
    since,
    until,
    top,
    sort,
):
    config = Config.load()
    idx = UsageIndex(config.root)
    conn = idx.connect()
    idx.refresh(conn)

    since_dt = _parse_window(since, "--since")
    until_dt = _parse_window(until, "--until")

    if session_prefix:
        platform, sid = _resolve_session(config, session_prefix)
        _render_session(
            conn,
            sid,
            platform_filter,
            model_filter,
            since_dt,
            until_dt,
            sort or "ts",
            as_json,
        )
    else:
        _render_rollup(
            conn,
            platform_filter,
            model_filter,
            since_dt,
            until_dt,
            top,
            sort or "total",
            as_json,
        )


def _render_session(
    conn,
    sid,
    platform_filter,
    model_filter,
    since_dt,
    until_dt,
    sort,
    as_json,
):
    sql = [
        "SELECT seq, ts, platform, model, input_tokens, output_tokens, "
        "total_tokens FROM usage WHERE session_id = ?"
    ]
    params: list = [sid]
    if platform_filter:
        sql.append("AND platform = ?")
        params.append(platform_filter)
    if model_filter:
        sql.append("AND model LIKE ?")
        params.append(f"%{model_filter}%")
    if since_dt:
        sql.append("AND ts >= ?")
        params.append(since_dt.isoformat())
    if until_dt:
        sql.append("AND ts <= ?")
        params.append(until_dt.isoformat())
    sort_col = {
        "total": "total_tokens DESC",
        "input": "input_tokens DESC",
        "output": "output_tokens DESC",
        "ts": "ts ASC",
    }[sort]
    sql.append(f"ORDER BY {sort_col}")
    rows = conn.execute(" ".join(sql), params).fetchall()

    if as_json:
        for r in rows:
            click.echo(
                json.dumps(
                    {
                        "session_id": sid,
                        "seq": r[0],
                        "ts": r[1],
                        "platform": r[2],
                        "model": r[3],
                        "input_tokens": r[4],
                        "output_tokens": r[5],
                        "total_tokens": r[6],
                    },
                    separators=(",", ":"),
                )
            )
        return

    if not rows:
        click.echo(f"No usage data for session {sid}.")
        return
    click.echo(f"{'SEQ':<5} {'TS':<26} {'MODEL':<25} {'INPUT':>10} {'OUTPUT':>8} {'TOTAL':>10}")
    tot_in = tot_out = tot = 0
    for r in rows:
        click.echo(f"{r[0]:<5} {r[1]:<26} {r[3][:25]:<25} {r[4]:>10,} {r[5]:>8,} {r[6]:>10,}")
        tot_in += r[4]
        tot_out += r[5]
        tot += r[6]
    click.echo(f"\n{len(rows)} turns · {tot_in:,} input · {tot_out:,} output · {tot:,} total")


def _render_rollup(
    conn,
    platform_filter,
    model_filter,
    since_dt,
    until_dt,
    top,
    sort,
    as_json,
):
    sql = [
        "SELECT session_id, platform, "
        "COUNT(*) AS turns, "
        "SUM(input_tokens) AS in_tok, "
        "SUM(output_tokens) AS out_tok, "
        "SUM(total_tokens) AS total_tok "
        "FROM usage WHERE 1=1"
    ]
    params: list = []
    if platform_filter:
        sql.append("AND platform = ?")
        params.append(platform_filter)
    if model_filter:
        sql.append("AND model LIKE ?")
        params.append(f"%{model_filter}%")
    if since_dt:
        sql.append("AND ts >= ?")
        params.append(since_dt.isoformat())
    if until_dt:
        sql.append("AND ts <= ?")
        params.append(until_dt.isoformat())
    sql.append("GROUP BY session_id, platform")
    sort_col = {
        "total": "total_tok DESC",
        "input": "in_tok DESC",
        "output": "out_tok DESC",
        "ts": "session_id ASC",
    }[sort]
    sql.append(f"ORDER BY {sort_col}")
    if top is not None:
        sql.append("LIMIT ?")
        params.append(top)
    rows = conn.execute(" ".join(sql), params).fetchall()

    if as_json:
        for r in rows:
            click.echo(
                json.dumps(
                    {
                        "session_id": r[0],
                        "platform": r[1],
                        "turns": r[2],
                        "input_tokens": r[3],
                        "output_tokens": r[4],
                        "total_tokens": r[5],
                    },
                    separators=(",", ":"),
                )
            )
        return

    if not rows:
        click.echo("No usage data.")
        return
    click.echo(
        f"{'SESSION':<14} {'PLATFORM':<9} {'TURNS':>5} {'INPUT':>12} {'OUTPUT':>10} {'TOTAL':>12}"
    )
    tot_in = tot_out = tot = 0
    for r in rows:
        click.echo(f"{r[0][:14]:<14} {r[1]:<9} {r[2]:>5} {r[3]:>12,} {r[4]:>10,} {r[5]:>12,}")
        tot_in += r[3]
        tot_out += r[4]
        tot += r[5]
    click.echo(f"\n{len(rows)} sessions · {tot_in:,} input · {tot_out:,} output · {tot:,} total")


__all__ = ["usage"]
