from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.eval._ulid import ulid_now
from thirdeye.eval.agents import get_adapter
from thirdeye.eval.agents.exec import AgentInvocation
from thirdeye.eval.agents.exec import invoke_agent as _invoke_agent
from thirdeye.eval.definition import EvalDefinition, load_definition
from thirdeye.eval.prompt import build_prompt
from thirdeye.eval.result import VALID_VERDICTS, EvalResult, Finding, parse_envelope
from thirdeye.eval.store import EvalStore
from thirdeye.paths import session_dir
from thirdeye.reader import SessionReader
from thirdeye.store import Store

__all__ = ["AgentInvocation", "run_eval", "run_eval_background"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _condense_event_line(seq: int, t: str, data: dict[str, Any]) -> str:
    """Build one-line timeline entry."""
    snippet = ""
    if t == "tool_call":
        tool = data.get("tool_name") or data.get("name") or "?"
        cmd = (data.get("tool_input") or {}).get("command") or ""
        snippet = f"{tool}: {str(cmd)[:80]}"
    elif t == "user_message":
        snippet = str(data.get("prompt") or data.get("content") or "")[:80]
    elif t == "assistant_message":
        snippet = str(data.get("last_assistant_message") or "")[:80]
    elif t == "tool_result":
        tool = data.get("tool_name") or "?"
        snippet = f"{tool}"
    return f"{seq:<4} {t:<20} {snippet}"


def _read_timeline(thirdeye_home: Path, platform: str, sid: str, *, limit: int = 500) -> list[str]:
    """Read up to ``limit`` events from the session log and condense each line.

    Best-effort; on read error returns ``[]``.
    """
    sd = session_dir(thirdeye_home, platform, sid)
    try:
        reader = SessionReader(sd)
    except Exception:
        return []
    lines: list[str] = []
    try:
        for ev in reader.iter_events():
            seq = int(ev.get("seq") or 0)
            t = str(ev.get("t") or "")
            data = ev.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            lines.append(_condense_event_line(seq, t, data))
            if len(lines) >= limit:
                break
    except Exception:
        pass
    return lines


def _list_session_ids_on_platform(thirdeye_home: Path, platform: str) -> set[str]:
    """Return the set of session ids currently recorded for ``platform``."""
    try:
        config = Config(root=thirdeye_home)
        return {s.session_id for s in Store(config).list_sessions(platform=platform)}
    except Exception:
        return set()


def _build_eval_result(
    *,
    eval_id: str,
    session_id: str,
    definition: EvalDefinition,
    agent_name: str,
    agent_session_id: str | None,
    started_at: str,
    ended_at: str,
    duration_ms: int,
    envelope: dict[str, Any] | None,
    narrative: str,
    cost: dict[str, Any],
    agent_model: str = "",
) -> EvalResult:
    if envelope is None:
        return EvalResult(
            id=eval_id,
            session_id=session_id,
            definition=definition.name,
            agent=agent_name,
            agent_model=agent_model,
            agent_session_id=agent_session_id,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            verdict="unknown",
            summary="(agent emitted no parseable JSON envelope)",
            scores={},
            findings=[],
            markdown=narrative,
            cost=cost,
        )
    verdict = str(envelope.get("verdict", "unknown"))
    if verdict not in VALID_VERDICTS:
        verdict = "unknown"
    scores = {str(k): float(v) for k, v in (envelope.get("scores") or {}).items()}
    findings = [
        Finding.from_dict(f) for f in (envelope.get("findings") or []) if isinstance(f, dict)
    ]
    return EvalResult(
        id=eval_id,
        session_id=session_id,
        definition=definition.name,
        agent=agent_name,
        agent_model=agent_model,
        agent_session_id=agent_session_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        verdict=verdict,
        summary=str(envelope.get("summary", "")),
        scores=scores,
        findings=findings,
        markdown=narrative,
        cost=cost,
    )


def run_eval(
    *,
    thirdeye_home: Path,
    platform: str,
    session_id: str,
    definition_name: str = "default",
    agent_name: str,
    cwd: Path | None = None,
    save: bool = True,
    run_id: str | None = None,
) -> EvalResult:
    """Run one eval synchronously, returning the persisted EvalResult.

    Raises ``FileNotFoundError`` if the agent binary is missing or the
    definition is unknown. Raises ``RuntimeError`` if the agent returns a
    non-zero exit. Raises ``ValueError`` for an unknown agent name.
    """
    definition = load_definition(thirdeye_home, definition_name)
    adapter = get_adapter(agent_name, thirdeye_home=thirdeye_home)
    sd = session_dir(thirdeye_home, platform, session_id)

    timeline = _read_timeline(thirdeye_home, platform, session_id)
    prompt = build_prompt(
        thirdeye_home=thirdeye_home,
        platform=platform,
        session_id=session_id,
        definition=definition,
        event_lines=timeline,
    )

    pre_ids = _list_session_ids_on_platform(thirdeye_home, adapter.name)
    started = _now_iso()
    invocation = _invoke_agent(adapter, prompt, cwd or sd)
    ended = _now_iso()
    post_ids = _list_session_ids_on_platform(thirdeye_home, adapter.name)
    new_ids = post_ids - pre_ids
    agent_session_id = next(iter(new_ids)) if len(new_ids) == 1 else None

    if invocation.returncode != 0:
        tail = invocation.stderr.strip().splitlines()[-5:]
        raise RuntimeError(
            f"agent {adapter.name!r} exited {invocation.returncode}: "
            + ("; ".join(tail) if tail else "(no stderr)")
        )

    agent_text, cost = adapter.parse_output(invocation.stdout)
    envelope, narrative = parse_envelope(agent_text)

    result = _build_eval_result(
        eval_id=run_id or ulid_now(),
        session_id=session_id,
        definition=definition,
        agent_name=adapter.name,
        agent_session_id=agent_session_id,
        started_at=started,
        ended_at=ended,
        duration_ms=invocation.duration_ms,
        envelope=envelope,
        narrative=narrative,
        cost=cost,
        agent_model=str(cost.get("model", "")) if isinstance(cost, dict) else "",
    )

    if save:
        EvalStore(sd).append(result)
    return result


def run_eval_background(
    *,
    thirdeye_home: Path,
    platform: str,
    session_id: str,
    definition_name: str = "default",
    agent_name: str,
    cwd: Path | None = None,
    thirdeye_bin: str | None = None,
) -> str:
    """Start a detached worker. Return the new run_id.

    The worker is launched as ``thirdeye eval _run-worker <run_id> ...``
    (an undocumented subcommand on the same binary). The same ULID is
    used for both the on-disk job stub and the eventual EvalResult.id.
    """
    sd = session_dir(thirdeye_home, platform, session_id)
    store = EvalStore(sd)
    run_id = ulid_now()
    log_path = sd / "evals.jobs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    binary = thirdeye_bin or shutil.which("thirdeye") or sys.executable
    worker_cmd = [
        binary,
        "eval",
        "_run-worker",
        run_id,
        platform,
        session_id,
        definition_name,
        agent_name,
    ]
    log_file = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            worker_cmd,
            cwd=cwd or sd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_file.close()

    store.write_job(
        run_id,
        {
            "job_id": run_id,
            "session_id": session_id,
            "using": definition_name,
            "agent": agent_name,
            "platform": platform,
            "status": "running",
            "started_at": _now_iso(),
            "pid": proc.pid,
            "log_path": str(log_path),
        },
    )
    return run_id
