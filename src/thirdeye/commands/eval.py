from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import click

from thirdeye.config import Config
from thirdeye.eval.agents import list_agent_names
from thirdeye.eval.definition import (
    SHIPPED_NAMES,
    EvalDefinition,
    delete_definition,
    list_definitions,
    load_definition,
    save_definition,
)
from thirdeye.eval.result import EvalResult
from thirdeye.eval.runner import run_eval, run_eval_background
from thirdeye.eval.store import EvalStore, iter_results_by_definition
from thirdeye.paths import session_dir
from thirdeye.store import Store


@click.group(name="eval", help="Run and manage session evaluations.")
def eval_group() -> None:
    pass


# --- run ---


@eval_group.command(name="run")
@click.argument("session_prefix")
@click.option("--using", default="default", show_default=True, help="Eval definition name.")
@click.option("--agent", required=True, help="Agent CLI to dispatch.")
@click.option("--background", "background", is_flag=True, help="Detach. Print job_id, exit 0.")
@click.option("--json", "as_json", is_flag=True, help="Print result as JSON.")
@click.option(
    "--save/--no-save", default=True, show_default=True, help="Persist to <session>/evals.jsonl."
)
def run_cmd(session_prefix, using, agent, background, as_json, save):
    config = Config.load()
    try:
        platform, sid = Store(config).resolve_session_id(session_prefix)
    except (ValueError, KeyError) as e:
        raise click.ClickException(str(e)) from e

    if agent not in list_agent_names(config.root):
        raise click.ClickException(
            f"unknown agent {agent!r} — choose one of "
            f"{', '.join(list_agent_names(config.root))}"
        )

    if background:
        try:
            job_id = run_eval_background(
                thirdeye_home=config.root,
                platform=platform,
                session_id=sid,
                definition_name=using,
                agent_name=agent,
            )
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(str(e)) from e
        click.echo(job_id)
        return

    try:
        result = run_eval(
            thirdeye_home=config.root,
            platform=platform,
            session_id=sid,
            definition_name=using,
            agent_name=agent,
            save=save,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e

    if as_json:
        import json

        click.echo(json.dumps(result.to_dict(), separators=(",", ":")))
    else:
        _render_result(result)


# --- _run-worker (hidden background worker entry point) ---


@eval_group.command(name="_run-worker", hidden=True)
@click.argument("run_id")
@click.argument("platform")
@click.argument("session_id")
@click.argument("definition_name")
@click.argument("agent_name")
def _run_worker(run_id, platform, session_id, definition_name, agent_name):
    """Internal: detached worker that finishes a background eval."""
    config = Config.load()
    sd = session_dir(config.root, platform, session_id)
    store = EvalStore(sd)
    try:
        run_eval(
            thirdeye_home=config.root,
            platform=platform,
            session_id=session_id,
            definition_name=definition_name,
            agent_name=agent_name,
            save=True,
            run_id=run_id,
        )
        store.remove_job(run_id)
    except Exception as e:
        existing = store.read_job(run_id) or {}
        existing["status"] = "failed"
        existing["error"] = f"{type(e).__name__}: {e}"
        store.write_job(run_id, existing)
        raise


# --- show ---


@eval_group.command(name="show")
@click.argument("session_prefix")
@click.option("--id", "eval_id", default=None, help="Pin to a specific past result.")
@click.option("--using", default=None, help="Latest result of this definition.")
@click.option("--json", "as_json", is_flag=True)
def show_cmd(session_prefix, eval_id, using, as_json):
    config = Config.load()
    try:
        platform, sid = Store(config).resolve_session_id(session_prefix)
    except (ValueError, KeyError) as e:
        raise click.ClickException(str(e)) from e
    store = EvalStore(session_dir(config.root, platform, sid))
    if eval_id:
        result = store.find_by_id(eval_id)
        if result is None:
            raise click.ClickException(f"no eval result with id {eval_id!r}")
    else:
        result = store.latest(definition=using)
        if result is None:
            raise click.ClickException(f"no eval results for session {sid}")
    if as_json:
        import json

        click.echo(json.dumps(result.to_dict(), separators=(",", ":")))
    else:
        _render_result(result)


# --- list ---


@eval_group.command(name="list")
@click.argument("session_prefix", required=False)
@click.option("--using", default=None)
@click.option("--agent", default=None)
@click.option("--verdict", default=None, type=click.Choice(["pass", "warn", "fail", "unknown"]))
@click.option("--since", default=None)
@click.option("--until", default=None)
@click.option("--json", "as_json", is_flag=True)
def list_cmd(session_prefix, using, agent, verdict, since, until, as_json):
    from thirdeye.timeparse import parse_when

    config = Config.load()
    since_dt = parse_when(since) if since else None
    until_dt = parse_when(until) if until else None

    sessions: list[tuple[str, str]] = []
    if session_prefix:
        try:
            platform, sid = Store(config).resolve_session_id(session_prefix)
        except (ValueError, KeyError) as e:
            raise click.ClickException(str(e)) from e
        sessions.append((platform, sid))
    else:
        for s in Store(config).list_sessions():
            sessions.append((s.platform, s.session_id))

    rows: list[EvalResult] = []
    for platform, sid in sessions:
        sd = session_dir(config.root, platform, sid)
        for r in EvalStore(sd).iter_results():
            if using and r.definition != using:
                continue
            if agent and r.agent != agent:
                continue
            if verdict and r.verdict != verdict:
                continue
            if since_dt or until_dt:
                try:
                    ts_raw = r.started_at
                    iso = ts_raw[:-1] + "+00:00" if ts_raw.endswith("Z") else ts_raw
                    e_ts = datetime.fromisoformat(iso)
                except (TypeError, ValueError):
                    continue
                if since_dt and e_ts < since_dt:
                    continue
                if until_dt and e_ts > until_dt:
                    continue
            rows.append(r)
    rows.sort(key=lambda r: r.started_at, reverse=True)

    if as_json:
        import json

        for r in rows:
            click.echo(json.dumps(r.to_dict(), separators=(",", ":")))
        return
    if not rows:
        click.echo("No eval results.")
        return
    click.echo(
        f"{'TS':<26} {'SESSION':<14} {'USING':<18} {'AGENT':<8} "
        f"{'VERDICT':<8} {'OVERALL':<7} {'DURATION':<10}"
    )
    for r in rows:
        overall = r.scores.get("overall")
        overall_str = f"{overall:g}/10" if overall is not None else "-"
        click.echo(
            f"{r.started_at:<26} {r.session_id[:14]:<14} {r.definition[:18]:<18} "
            f"{r.agent[:8]:<8} {r.verdict:<8} {overall_str:<7} "
            f"{r.duration_ms / 1000:.1f}s"
        )


# --- status ---


@eval_group.command(name="status")
@click.argument("session_prefix", required=False)
@click.option("--json", "as_json", is_flag=True)
def status_cmd(session_prefix, as_json):
    config = Config.load()
    sessions: list[tuple[str, str]] = []
    if session_prefix:
        try:
            platform, sid = Store(config).resolve_session_id(session_prefix)
        except (ValueError, KeyError) as e:
            raise click.ClickException(str(e)) from e
        sessions.append((platform, sid))
    else:
        for s in Store(config).list_sessions():
            sessions.append((s.platform, s.session_id))

    jobs: list[dict] = []
    for platform, sid in sessions:
        sd = session_dir(config.root, platform, sid)
        for job in EvalStore(sd).iter_jobs():
            if job.get("status") == "running" and "pid" in job:
                pid = int(job["pid"])
                if not _pid_alive(pid):
                    job = {**job, "status": "orphaned"}
                    EvalStore(sd).write_job(job["job_id"], job)
            jobs.append(job)

    if as_json:
        import json

        for j in jobs:
            click.echo(json.dumps(j, separators=(",", ":")))
        return
    if not jobs:
        click.echo("No background eval jobs.")
        return
    click.echo(
        f"{'JOB':<28} {'SESSION':<14} {'USING':<18} {'AGENT':<8} " f"{'STATUS':<10} {'STARTED':<26}"
    )
    for j in jobs:
        click.echo(
            f"{j.get('job_id', '')[:28]:<28} "
            f"{j.get('session_id', '')[:14]:<14} "
            f"{j.get('using', '')[:18]:<18} "
            f"{j.get('agent', '')[:8]:<8} "
            f"{j.get('status', ''):<10} "
            f"{j.get('started_at', ''):<26}"
        )


# --- batch ---


@eval_group.command(name="batch", help="Dispatch an eval on multiple sessions.")
@click.argument("definition_name")
@click.argument("session_prefixes", nargs=-1, required=True)
@click.option("--agent", required=True, help="Agent CLI to dispatch.")
def batch_cmd(
    definition_name: str, session_prefixes: tuple[str, ...], agent: str
) -> None:
    config = Config.load()
    store = Store(config)
    if len(session_prefixes) > 25:
        raise click.UsageError("batch capped at 25 sessions")
    for prefix in session_prefixes:
        try:
            platform, sid = store.resolve_session_id(prefix)
        except (KeyError, ValueError) as e:
            click.echo(f"skip {prefix}: {e}", err=True)
            continue
        try:
            run_id = run_eval_background(
                thirdeye_home=config.root,
                platform=platform,
                session_id=sid,
                definition_name=definition_name,
                agent_name=agent,
            )
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"skip {prefix}: {e}", err=True)
            continue
        click.echo(f"{sid}\t{run_id}\tdispatched")


# --- results ---


@eval_group.command(name="results", help="List runs for an eval definition across sessions.")
@click.option("--def", "definition_name", required=True, help="Eval definition name.")
@click.option("--as-json", "as_json", is_flag=True)
def results_cmd(definition_name: str, as_json: bool) -> None:
    import json

    config = Config.load()
    rows = list(iter_results_by_definition(config, definition_name))
    if as_json:
        for meta, r in rows:
            click.echo(
                json.dumps(
                    {
                        "session_id": meta.session_id,
                        "platform": meta.platform,
                        "run_id": r.id,
                        "agent": r.agent,
                        "verdict": r.verdict,
                        "started_at": r.started_at,
                        "duration_ms": r.duration_ms,
                    },
                    separators=(",", ":"),
                )
            )
        return
    for meta, r in rows:
        click.echo(
            f"{meta.session_id}\t{r.id[:8]}\t{r.agent}\t{r.verdict}\t{r.started_at}"
        )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- def subgroup ---


@eval_group.group(name="def")
def def_group() -> None:
    """Manage eval definitions."""
    pass


@def_group.command(name="list")
def def_list_cmd():
    config = Config.load()
    for n in list_definitions(config.root):
        suffix = " (shipped)" if n in SHIPPED_NAMES else ""
        click.echo(f"{n}{suffix}")


@def_group.command(name="show")
@click.argument("name")
def def_show_cmd(name):
    config = Config.load()
    try:
        defn = load_definition(config.root, name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    click.echo(defn.to_yaml())


@def_group.command(name="create")
@click.argument("name")
@click.option("--directive", default=None, help="Inline directive text.")
@click.option(
    "--directive-file",
    "directive_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Read directive from FILE.",
)
@click.option(
    "--from", "from_name", default=None, help="Copy directive from an existing definition."
)
@click.option("--description", default="", help="Short description.")
@click.option("--default-agent", default="claude", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite if exists.")
def def_create_cmd(name, directive, directive_file, from_name, description, default_agent, force):
    sources = [bool(directive), bool(directive_file), bool(from_name)]
    if sum(sources) != 1:
        raise click.ClickException(
            "exactly one of --directive, --directive-file, --from is required"
        )
    config = Config.load()
    if directive_file:
        text = directive_file.read_text(encoding="utf-8")
    elif from_name:
        try:
            base = load_definition(config.root, from_name)
            text = base.directive
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
    else:
        text = directive
    defn = EvalDefinition(
        name=name,
        description=description,
        directive=text,
        default_agent=default_agent,
        output_schema="v1",
    )
    try:
        path = save_definition(config.root, defn, force=force)
    except FileExistsError as e:
        raise click.ClickException(str(e) + " (use --force to overwrite)") from e
    click.echo(f"Created {path}")


@def_group.command(name="edit")
@click.argument("name")
def def_edit_cmd(name):
    config = Config.load()
    try:
        load_definition(config.root, name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    from thirdeye.paths import eval_def_path

    path = eval_def_path(config.root, name)
    click.edit(filename=str(path))


@def_group.command(name="rm")
@click.argument("name")
def def_rm_cmd(name):
    config = Config.load()
    removed = delete_definition(config.root, name)
    if not removed:
        raise click.ClickException(f"no user copy of definition {name!r}")
    if name in SHIPPED_NAMES:
        click.echo(f"Removed user copy of '{name}'; shipped version restored on next load.")
    else:
        click.echo(f"Removed '{name}'.")


# --- shared rendering ---


def _render_result(result: EvalResult) -> None:
    click.echo(
        f"EVAL: {result.definition} · agent: {result.agent} · "
        f"{result.started_at} · {result.duration_ms / 1000:.1f}s"
    )
    click.echo(f"\nVERDICT: {result.verdict}")
    click.echo(f"SUMMARY: {result.summary}")
    if result.scores:
        click.echo("\nSCORES:")
        for k, v in result.scores.items():
            click.echo(f"  {k:<20} {v:g}/10")
    if result.findings:
        click.echo("\nFINDINGS:")
        for f in result.findings:
            glyph = {"info": "·", "warn": "⚠", "error": "✖"}.get(f.severity, "·")
            seq = f"seq={f.seq}" if f.seq is not None else "seq=null"
            cat = f.category or ""
            click.echo(f"  {glyph}  {seq:<10} {cat:<14} {f.note}")
    if result.markdown:
        click.echo("\n--- Narrative ---")
        click.echo(result.markdown)
