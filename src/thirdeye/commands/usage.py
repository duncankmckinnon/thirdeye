from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import click

from thirdeye.commands.add import find_orphaned_hooks
from thirdeye.config import Config
from thirdeye.paths import (
    session_dir,
    sessions_root,
    usage_db_path,
    usage_jsonl_path,
    usage_log_path,
    usage_state_path,
)
from thirdeye.timeparse import parse_when
from thirdeye.usage.index import UsageIndex
from thirdeye.usage.read import iter_calls
from thirdeye.usage.types import UsageRow


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


def _row_ts(row: UsageRow) -> datetime | None:
    s = row.ts
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    # Normalize to UTC-aware so comparisons against the aware --since/--until
    # bounds never raise on a naive timestamp lacking an offset.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _keep_row(
    row: UsageRow,
    *,
    platform_filter: str | None,
    model_filter: str | None,
    since_dt: datetime | None,
    until_dt: datetime | None,
) -> bool:
    if platform_filter and row.platform != platform_filter:
        return False
    if model_filter and model_filter not in row.response_model:
        return False
    if since_dt or until_dt:
        dt = _row_ts(row)
        if dt is None:
            return False
        if since_dt and dt < since_dt:
            return False
        if until_dt and dt > until_dt:
            return False
    return True


def _iter_session_dirs(root: Path):
    """Yield (platform, session_id, session_dir) for every captured session."""
    troot = sessions_root(root)
    if not troot.exists():
        return
    for platform_dir_ in sorted(troot.iterdir()):
        if not platform_dir_.is_dir():
            continue
        for sd in sorted(platform_dir_.iterdir()):
            if not sd.is_dir():
                continue
            yield platform_dir_.name, sd.name, sd


def _fmt_cache(value: int | None) -> str:
    """Render an absent cache attribute as '-', distinct from a reported 0."""
    return "-" if value is None else f"{value:,}"


class _UsageGroup(click.Group):
    """Route non-subcommand args to the default `show` subcommand.

    Lets `thirdeye usage [SESSION_PREFIX] [OPTIONS]` work alongside
    `thirdeye usage reindex`, `thirdeye usage reset`, and `thirdeye usage
    errors` without the positional-vs-subcommand parsing collision that arises
    from putting a positional argument directly on an `invoke_without_command`
    group.
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
    help="Per-call model and token usage.",
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
    help="Filter rows where the response model contains this substring.",
)
@click.option("--since", default=None, help="Time window lower bound.")
@click.option("--until", default=None, help="Time window upper bound.")
@click.option(
    "--top",
    type=int,
    default=None,
    help="Rollup mode: keep top N sessions by total tokens.",
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


@usage.command(name="reset")
@click.option("--yes", is_flag=True, help="Required to actually delete anything.")
def reset_cmd(yes):
    """Delete all captured usage data (sidecars + usage.db).

    Leaves events.alog, events.idx, tags.jsonl, meta.yaml, and every upstream
    transcript untouched.
    """
    config = Config.load()
    _run_reset(config.root, yes=yes)


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
    since_dt = _parse_window(since, "--since")
    until_dt = _parse_window(until, "--until")

    if session_prefix:
        platform, sid = _resolve_session(config, session_prefix)
        _render_session(
            session_dir(config.root, platform, sid),
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
            config.root,
            platform_filter,
            model_filter,
            since_dt,
            until_dt,
            top,
            sort or "total",
            as_json,
        )


def _render_session(
    session_dir_,
    sid,
    platform_filter,
    model_filter,
    since_dt,
    until_dt,
    sort,
    as_json,
):
    rows = [
        r
        for r in iter_calls(session_dir_)
        if _keep_row(
            r,
            platform_filter=platform_filter,
            model_filter=model_filter,
            since_dt=since_dt,
            until_dt=until_dt,
        )
    ]
    sort_key = {
        "total": lambda r: -r.total_tokens,
        "input": lambda r: -r.input_tokens,
        "output": lambda r: -r.output_tokens,
        "ts": lambda r: r.ts,
    }[sort]
    rows.sort(key=sort_key)

    if as_json:
        for r in rows:
            click.echo(json.dumps(r.to_dict(), separators=(",", ":")))
        return

    if not rows:
        click.echo(f"No usage data for session {sid}.")
        return
    click.echo(
        f"{'SEQ':<5} {'TS':<26} {'MODEL':<25} {'INPUT':>10} {'OUTPUT':>8} "
        f"{'CACHE_R':>12} {'CACHE_C':>12} {'TOTAL':>10}"
    )
    tot_in = tot_out = tot = 0
    for r in rows:
        click.echo(
            f"{r.seq:<5} {r.ts:<26} {r.response_model[:25]:<25} "
            f"{r.input_tokens:>10,} {r.output_tokens:>8,} "
            f"{_fmt_cache(r.cache_read_input_tokens):>12} "
            f"{_fmt_cache(r.cache_creation_input_tokens):>12} "
            f"{r.total_tokens:>10,}"
        )
        tot_in += r.input_tokens
        tot_out += r.output_tokens
        tot += r.total_tokens
    click.echo(f"\n{len(rows)} calls · {tot_in:,} input · {tot_out:,} output · {tot:,} total")


def _render_rollup(
    root,
    platform_filter,
    model_filter,
    since_dt,
    until_dt,
    top,
    sort,
    as_json,
):
    # (session_id, platform) -> [turns, input_tokens, output_tokens]
    agg: dict[tuple[str, str], list[int]] = {}
    for platform, sid, sd in _iter_session_dirs(root):
        for r in iter_calls(sd):
            if not _keep_row(
                r,
                platform_filter=platform_filter,
                model_filter=model_filter,
                since_dt=since_dt,
                until_dt=until_dt,
            ):
                continue
            bucket = agg.setdefault((sid, platform), [0, 0, 0])
            bucket[0] += 1
            bucket[1] += r.input_tokens
            bucket[2] += r.output_tokens

    rows = [
        (sid, platform, turns, in_tok, out_tok, in_tok + out_tok)
        for (sid, platform), (turns, in_tok, out_tok) in agg.items()
    ]
    sort_key = {
        "total": lambda r: -r[5],
        "input": lambda r: -r[3],
        "output": lambda r: -r[4],
        "ts": lambda r: r[0],
    }[sort]
    rows.sort(key=sort_key)
    if top is not None:
        rows = rows[:top]

    if as_json:
        for sid, platform, turns, in_tok, out_tok, _total in rows:
            click.echo(
                json.dumps(
                    {
                        "session_id": sid,
                        "platform": platform,
                        "calls": turns,
                        "gen_ai.usage.input_tokens": in_tok,
                        "gen_ai.usage.output_tokens": out_tok,
                    },
                    separators=(",", ":"),
                )
            )
        return

    if not rows:
        click.echo("No usage data.")
        return
    click.echo(
        f"{'SESSION':<14} {'PLATFORM':<9} {'CALLS':>5} {'INPUT':>12} {'OUTPUT':>10} {'TOTAL':>12}"
    )
    tot_in = tot_out = tot = 0
    for sid, platform, turns, in_tok, out_tok, total in rows:
        click.echo(
            f"{sid[:14]:<14} {platform:<9} {turns:>5} {in_tok:>12,} {out_tok:>10,} {total:>12,}"
        )
        tot_in += in_tok
        tot_out += out_tok
        tot += total
    click.echo(f"\n{len(rows)} sessions · {tot_in:,} input · {tot_out:,} output · {tot:,} total")


def _run_reset(root: Path, *, yes: bool) -> None:
    """Destroy every usage sidecar and usage.db under `root`.

    Counts are computed and reported before anything is removed. Without
    `--yes` the command refuses and exits non-zero, deleting nothing.
    """
    sidecar_files: list[Path] = []
    sessions_affected: set[Path] = set()
    for _platform, _sid, sd in _iter_session_dirs(root):
        jsonl = usage_jsonl_path(sd)
        state = usage_state_path(sd)
        if jsonl.exists():
            sidecar_files.append(jsonl)
            sessions_affected.add(sd)
        if state.exists():
            sidecar_files.append(state)
            sessions_affected.add(sd)

    db = usage_db_path(root)
    db_rows = 0
    if db.exists():
        conn = None
        try:
            conn = sqlite3.connect(db)
            db_rows = conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0]
        except sqlite3.Error:
            db_rows = 0
        finally:
            if conn is not None:
                conn.close()

    click.echo("usage reset will destroy:")
    click.echo(f"  sidecar files:     {len(sidecar_files)}")
    click.echo(f"  sessions affected: {len(sessions_affected)}")
    click.echo(f"  usage.db rows:     {db_rows}")

    if not yes:
        raise click.ClickException("refusing to delete without --yes; nothing was removed.")

    for f in sidecar_files:
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    # Remove usage.db and its WAL/SHM companions if present.
    for p in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    click.echo(
        f"Deleted {len(sidecar_files)} sidecar file(s) across "
        f"{len(sessions_affected)} session(s) and {db_rows} usage.db row(s)."
    )

    orphans = find_orphaned_hooks()
    for path, command in orphans:
        click.echo(
            f"Warning: {path} still references removed hook {command!r}. "
            "Remove it from that tool's config (detection only — not edited here)."
        )


__all__ = ["usage"]
