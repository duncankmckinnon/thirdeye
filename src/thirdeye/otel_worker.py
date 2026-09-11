"""Detached worker process for `thirdeye.otel_export`.

`otel_export.export_turn` and `otel_export.export_spans` write small job files
describing completed turns or live span batches and spawn this module
(``python -m thirdeye.otel_worker <job_path>``) as a detached, unwaited-for
child. All the actual Logfire work — configuring the SDK, building the turn's
whole span subtree, and flushing (a real network round trip) — happens here,
off the hook process's critical path.

Run standalone, never imported by anything that cares about its return value:
every failure mode ends in a clean process exit, never a traceback on stderr
(this can run well after the parent hook process, and possibly the coding
harness itself, is gone — there's no one left to usefully see it). A failure
is instead logged, best-effort, to the same breadcrumb log hook invocations
use -- without that, a span that silently never lands in Logfire is
otherwise indistinguishable from one that was never attempted.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from thirdeye._compat import fsops

_JOB_CLAIM_STALE_S = 30.0
_JOB_MAX_ATTEMPTS = 5
_ACCOUNTING_KINDS = frozenset({"session_accounting", "turn_accounting"})


def _write_job_state(job_path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a claim/retry state without partial JSON."""
    temporary = job_path.with_name(f".{job_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str), encoding="utf-8", newline="\n")
    fsops.replace(temporary, job_path)
    fsops.sync_directory(job_path.parent)


def _job_claim_path(job_path: Path) -> Path:
    return job_path.with_suffix(f"{job_path.suffix}.claim")


def _create_job_claim(path: Path) -> bool:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        return False
    return True


def _claim_job(job_path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recover a claimed-but-unsent job and claim it for this worker.

    Remote flush and local deletion cannot be a single transaction. A crash in
    between can retry a deterministic span, reducing but not eliminating
    remote duplicates.
    """
    state = payload.get("state")
    if state == "emitted":
        fsops.unlink(job_path, missing_ok=True)
        _release_job_claim(job_path)
        return None
    if state == "failed":
        _release_job_claim(job_path)
        return None
    claim_path = _job_claim_path(job_path)
    if not _create_job_claim(claim_path):
        try:
            stale = time.time() - claim_path.stat().st_mtime > _JOB_CLAIM_STALE_S
        except OSError:
            stale = True
        if not stale:
            return None
        fsops.unlink(claim_path, missing_ok=True)
        if not _create_job_claim(claim_path):
            return None
    claimed = dict(payload)
    claimed["state"] = "claimed"
    claimed["attempt"] = int(payload.get("attempt", 0))
    _write_job_state(job_path, claimed)
    return claimed


def _release_job_claim(job_path: Path) -> None:
    fsops.unlink(_job_claim_path(job_path), missing_ok=True)


def _retry_job(job_path: Path, payload: dict[str, Any]) -> None:
    next_attempt = int(payload.get("attempt", 0)) + 1
    retry = dict(payload)
    retry["attempt"] = next_attempt
    retry["state"] = "failed" if next_attempt >= _JOB_MAX_ATTEMPTS else "queued"
    _write_job_state(job_path, retry)


def _mark_emitted(job_path: Path, payload: dict[str, Any]) -> None:
    emitted = dict(payload)
    emitted["state"] = "emitted"
    _write_job_state(job_path, emitted)


def _accounting_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "accounting_id": payload["accounting_id"],
        "usage": payload["usage"],
        "attribution_status": payload["attribution_status"],
        "agent_id": payload.get("agent_id"),
        "attributes": payload.get("attributes") or {},
        "call_id": payload.get("call_id"),
    }


def _run_accounting_job(job_path: Path, payload: dict[str, Any]) -> None:
    """Claim, export, and ack one accounting job. Retryable on failure."""
    try:
        claimed = _claim_job(job_path, payload)
    except Exception as exc:
        _log_worker_failure(kind="job_claim", payload=payload, error=exc)
        return
    if claimed is None:
        return

    from thirdeye.otel_export import _captured_attributes

    token = _captured_attributes.set(claimed.get("captured_attributes") or {})
    kind = claimed.get("kind")
    delivered = False
    try:
        from thirdeye.config import Config
        from thirdeye.otel_export import (
            _export_session_accounting_inner,
            _export_turn_accounting_inner,
        )

        config = Config.load()
        if kind == "session_accounting":
            _export_session_accounting_inner(
                config=config,
                session_dir_=Path(claimed["session_dir"]),
                session_id=claimed["session_id"],
                platform=claimed["platform"],
                cwd=claimed["cwd"],
                accounting=_accounting_from_payload(claimed),
            )
        elif kind == "turn_accounting":
            _export_turn_accounting_inner(
                config=config,
                session_dir_=Path(claimed["session_dir"]),
                session_id=claimed["session_id"],
                platform=claimed["platform"],
                cwd=claimed["cwd"],
                turn_id=str(claimed["turn_id"]),
                accounting=_accounting_from_payload(claimed),
                turn_span_id=claimed.get("turn_span_id"),
            )
        else:
            raise RuntimeError(f"unhandled accounting job kind {kind!r}")
        _mark_emitted(job_path, claimed)
        delivered = True
    except Exception as exc:
        try:
            _retry_job(job_path, claimed)
        except Exception:
            pass
        _log_worker_failure(kind=str(kind or ""), payload=claimed, error=exc)
    finally:
        _captured_attributes.reset(token)
        _release_job_claim(job_path)
    if delivered:
        fsops.unlink(job_path, missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return
    job_path = Path(argv[0])
    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log_worker_failure(kind="job_read", payload={}, error=exc)
        fsops.unlink(job_path, missing_ok=True)
        return

    kind = payload.get("kind")
    if kind in _ACCOUNTING_KINDS:
        _run_accounting_job(job_path, payload)
        return

    # Existing platforms delete the ULID job as soon as it is readable. There
    # is no scanner to respawn retained turn/spans/subagent jobs, so a crash
    # or export failure must not leave poison pills on disk.
    fsops.unlink(job_path, missing_ok=True)

    from thirdeye.otel_export import _captured_attributes

    token = _captured_attributes.set(payload.get("captured_attributes") or {})
    try:
        from thirdeye.config import Config

        config = Config.load()
        if kind == "turn":
            from thirdeye.otel_export import _export_turn_inner

            _export_turn_inner(
                config=config,
                session_dir_=Path(payload["session_dir"]),
                session_id=payload["session_id"],
                platform=payload["platform"],
                cwd=payload["cwd"],
                turn=payload["turn"],
            )
        elif kind == "spans":
            from thirdeye.otel_export import _export_spans_batch

            _export_spans_batch(
                config=config,
                session_dir_=Path(payload["session_dir"]),
                session_id=payload["session_id"],
                platform=payload["platform"],
                cwd=payload["cwd"],
                trace_id=payload["trace_id"],
                spans=payload["spans"],
            )
        elif kind == "subagent_turn":
            from thirdeye.otel_export import _export_subagent_turn_inner

            _export_subagent_turn_inner(
                config=config,
                session_dir_=Path(payload["session_dir"]),
                session_id=payload["session_id"],
                platform=payload["platform"],
                cwd=payload["cwd"],
                trace_id=payload["trace_id"],
                parent_span_id=payload["parent_span_id"],
                turn=payload["turn"],
            )
    except Exception as exc:
        _log_worker_failure(kind=str(kind or ""), payload=payload, error=exc)
    finally:
        _captured_attributes.reset(token)


def _log_worker_failure(*, kind: str, payload: dict[str, Any], error: Exception) -> None:
    """Best-effort breadcrumb for an export that silently failed. Never
    raises -- if even this fails, there is genuinely no one left to tell.
    """
    try:
        from thirdeye.config import Config
        from thirdeye.usage.errlog import log_capture_error

        log_capture_error(
            thirdeye_home=Config.load().root,
            phase="otel_worker_export_failed",
            level="error",
            platform=str(payload.get("platform") or ""),
            session_id=str(payload.get("session_id") or ""),
            error=error,
            message=f"kind={kind}",
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
