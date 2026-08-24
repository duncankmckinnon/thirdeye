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
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return
    job_path = Path(argv[0])
    try:
        payload = json.loads(job_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _log_worker_failure(kind="job_read", payload={}, error=exc)
        return
    finally:
        job_path.unlink(missing_ok=True)

    kind = payload.get("kind")
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
