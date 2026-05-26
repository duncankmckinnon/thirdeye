from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from thirdeye.eval.result import EvalResult
from thirdeye.paths import (
    eval_job_path,
    evals_jobs_dir,
    evals_jsonl_path,
    session_dir as _session_dir,
)

if TYPE_CHECKING:
    from thirdeye.config import Config
    from thirdeye.meta import SessionMeta


class EvalStore:
    """Sidecar I/O for one session directory.

    - evals.jsonl   — append-only, one EvalResult per line
    - evals.jobs/   — one <job_id>.json per background invocation
    """

    def __init__(self, session_dir_: Path) -> None:
        self.session_dir = session_dir_

    # --- evals.jsonl ---

    @property
    def jsonl_path(self) -> Path:
        return evals_jsonl_path(self.session_dir)

    def append(self, result: EvalResult) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), separators=(",", ":")) + "\n")

    def iter_results(self) -> Iterator[EvalResult]:
        """Yield all results in file order. Skips malformed lines silently."""
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield EvalResult.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    def latest(self, *, definition: str | None = None) -> EvalResult | None:
        """Return the most recently appended result, optionally filtered by definition."""
        latest: EvalResult | None = None
        for r in self.iter_results():
            if definition is not None and r.definition != definition:
                continue
            latest = r
        return latest

    def find_by_id(self, eval_id: str) -> EvalResult | None:
        for r in self.iter_results():
            if r.id == eval_id:
                return r
        return None

    # --- evals.jobs/<job_id>.json ---

    @property
    def jobs_dir(self) -> Path:
        return evals_jobs_dir(self.session_dir)

    def write_job(self, job_id: str, payload: dict[str, Any]) -> Path:
        """Atomically write a job stub. Overwrites if present."""
        path = eval_job_path(self.session_dir, job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def read_job(self, job_id: str) -> dict[str, Any] | None:
        path = eval_job_path(self.session_dir, job_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def remove_job(self, job_id: str) -> bool:
        path = eval_job_path(self.session_dir, job_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def iter_jobs(self) -> Iterator[dict[str, Any]]:
        """Yield each job stub in this session directory."""
        if not self.jobs_dir.is_dir():
            return
        for f in sorted(self.jobs_dir.glob("*.json")):
            try:
                yield json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue


def iter_results_by_definition(
    config: Config, definition_name: str
) -> Iterator[tuple[SessionMeta, EvalResult]]:
    """Walk every session, yield (meta, result) pairs whose
    result.definition == definition_name. Ordered by started_at DESC."""
    from thirdeye.store import Store

    rows: list[tuple[SessionMeta, EvalResult]] = []
    store = Store(config)
    for meta in store.list_sessions():
        sd = _session_dir(config.root, meta.platform, meta.session_id)
        es = EvalStore(sd)
        for result in es.iter_results():
            if result.definition == definition_name:
                rows.append((meta, result))
    rows.sort(key=lambda pair: pair[1].started_at, reverse=True)
    yield from rows
