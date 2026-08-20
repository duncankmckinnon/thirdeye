"""Detached worker process for `thirdeye.otel_export`.

`otel_export.export_event` writes a small job file describing one thirdeye
event, and `otel_export.export_llm_calls` writes one describing a whole batch
of LLM calls (tagged ``"kind": "llm_calls"``); either way they spawn this
module (``python -m thirdeye.otel_worker <job_path>``) as a detached,
unwaited-for child. All the actual Logfire work — configuring the SDK,
building the span(s), and flushing (a real network round trip) — happens
here, off the hook process's critical path.

Run standalone, never imported by anything that cares about its return value:
every failure mode ends in a clean process exit, never a traceback on stderr
(this can run well after the parent hook process, and possibly Claude Code
itself, is gone — there's no one left to usefully see it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return
    job_path = Path(argv[0])
    try:
        payload = json.loads(job_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    finally:
        job_path.unlink(missing_ok=True)

    try:
        from thirdeye.config import Config

        config = Config.load()
        if payload.get("kind") == "codex_turn":
            from thirdeye.otel_export import _export_codex_turn_inner

            state_path = Path(payload["state_path"])
            try:
                exported = _export_codex_turn_inner(
                    config=config,
                    session_dir_=Path(payload["session_dir"]),
                    session_id=payload["session_id"],
                    cwd=payload["cwd"],
                    seq=payload["seq"],
                    turn=payload["turn"],
                )
                if not exported:
                    raise RuntimeError("Codex turn export was not flushed")
                state_path.write_text("sent")
            except Exception:
                state_path.unlink(missing_ok=True)
                raise
        elif payload.get("kind") == "llm_calls":
            from thirdeye.otel_export import _export_llm_calls_inner

            _export_llm_calls_inner(
                config=config,
                session_dir_=Path(payload["session_dir"]),
                session_id=payload["session_id"],
                platform=payload["platform"],
                cwd=payload["cwd"],
                calls=payload["calls"],
            )
        else:
            from thirdeye.otel_export import _export_event_inner

            _export_event_inner(
                config=config,
                session_dir_=Path(payload["session_dir"]),
                session_id=payload["session_id"],
                platform=payload["platform"],
                cwd=payload["cwd"],
                t=payload["t"],
                seq=payload["seq"],
                ts=payload["ts"],
                data=payload.get("data"),
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
