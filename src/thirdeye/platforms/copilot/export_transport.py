"""Copilot-owned transport for deterministic accounting exports.

Copilot accounting can move as transcript and usage evidence is reconciled.
Its jobs therefore need an atomic cancel/claim protocol and observable retry
state that the generic fire-and-forget OTel transport does not promise.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from thirdeye import otel_export
from thirdeye._compat import fsops, proc
from thirdeye.config import Config
from thirdeye.usage.errlog import log_capture_error

from .jsonio import atomic_write_json

_CLAIM_STALE_S = 30.0
_MAX_ATTEMPTS = 5
_KINDS = frozenset({"session_accounting", "turn_accounting"})


def jobs_dir(root: Path) -> Path:
    return root / "logs" / "copilot-otel-jobs"


def job_path(root: Path, job_id: str) -> Path:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return jobs_dir(root) / f"accounting-{digest}.json"


def claim_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.claim")


def delivery_claim_path(session_dir: Path, accounting_id: str) -> Path:
    digest = hashlib.sha256(accounting_id.encode("utf-8")).hexdigest()
    return session_dir / "copilot-accounting-sent" / f"{digest}.json"


def delivery_sent(session_dir: Path, accounting_id: str) -> bool:
    try:
        return delivery_claim_path(session_dir, accounting_id).read_text(encoding="utf-8") == "sent"
    except OSError:
        return False


def _mark_delivered(session_dir: Path, accounting_id: str) -> None:
    path = delivery_claim_path(session_dir, accounting_id)
    if _atomic_create(path, "sent"):
        fsops.sync_directory(path.parent)


def _atomic_create(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(descriptor, text.encode("utf-8"))
    except BaseException:
        fsops.unlink(path, missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    return True


def _take_claim(path: Path, text: str, *, recover_stale: bool) -> bool:
    if _atomic_create(path, text):
        return True
    if not recover_stale:
        return False
    try:
        stale = time.time() - path.stat().st_mtime > _CLAIM_STALE_S
    except OSError:
        stale = True
    if not stale:
        return False
    fsops.unlink(path, missing_ok=True)
    return _atomic_create(path, text)


def _read_job(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_job(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _create_job(path: Path, payload: dict[str, Any]) -> bool:
    """Publish a complete job atomically, preserving the first writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, default=str, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        fsops.sync_directory(path.parent)
        return True
    finally:
        fsops.unlink(temporary, missing_ok=True)


def status(root: Path, job_id: str) -> dict[str, Any] | None:
    path = job_path(root, job_id)
    payload = _read_job(path)
    if payload is None:
        return None
    state = "claimed" if claim_path(path).exists() else payload.get("state")
    return {
        "state": state,
        "attempt": payload.get("attempt"),
        "last_error": payload.get("last_error"),
    }


def cancel(root: Path, job_id: str) -> dict[str, Any]:
    """Cancel only after acquiring the same exclusive claim as the worker."""
    path = job_path(root, job_id)
    owner = claim_path(path)
    if not _atomic_create(owner, f"cancel:{os.getpid()}"):
        return {**(status(root, job_id) or {}), "state": "claimed"}
    try:
        payload = _read_job(path)
        if payload is None:
            return {"state": "cancelled", "attempt": None, "last_error": None}
        result = {
            "state": payload.get("state"),
            "attempt": payload.get("attempt"),
            "last_error": payload.get("last_error"),
        }
        if result["state"] == "emitted":
            return result
        fsops.unlink(path, missing_ok=True)
        return {**result, "state": "cancelled"}
    finally:
        fsops.unlink(owner, missing_ok=True)


def _spawn(path: Path) -> None:
    proc.spawn_detached(
        [sys.executable, "-m", "thirdeye.platforms.copilot.export_transport", str(path)]
    )


def queue_turn(
    config: Config,
    session_dir: Path,
    session_id: str,
    cwd: str,
    turn: dict[str, Any],
) -> bool:
    """Queue a Copilot turn while preserving the generic worker job shape."""
    if not config.logfire.enabled or not config.logfire.token:
        return False
    try:
        path = otel_export._write_job(
            config.root,
            {
                "kind": "turn",
                "captured_attributes": otel_export._resolve_captured_attributes(config, None),
                "session_dir": str(session_dir),
                "session_id": session_id,
                "platform": "copilot",
                "cwd": cwd,
                "turn": turn,
            },
        )
        otel_export._spawn(path)
        return True
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="copilot_turn_export_spawn",
            error=exc,
            platform="copilot",
            session_id=session_id,
        )
        return False


def _placement_is_current(config: Config, payload: dict[str, Any]) -> bool:
    from .export_state import load_export_state

    state = load_export_state(config, str(payload["session_id"]))
    placement = (state.get("placements") or {}).get(str(payload["accounting_id"])) or {}
    return placement.get("span_id") == payload.get("job_id")


def _queue(config: Config, payload: dict[str, Any]) -> bool:
    if not config.logfire.enabled or not config.logfire.token:
        return False
    try:
        if payload.get("kind") not in _KINDS:
            raise ValueError(f"unsupported Copilot accounting job kind: {payload.get('kind')!r}")
        path = job_path(config.root, str(payload["job_id"]))
        queued = {
            **payload,
            "captured_attributes": otel_export._resolve_captured_attributes(config, None),
            "state": "queued",
            "attempt": 0,
        }
        owner = claim_path(path)
        if not _take_claim(owner, f"queue:{os.getpid()}", recover_stale=True):
            return True
        dispatch = False
        try:
            if not _placement_is_current(config, payload):
                return True
            delivered = delivery_sent(
                Path(str(payload["session_dir"])), str(payload["accounting_id"])
            )
            if not delivered:
                _create_job(path, queued)
                dispatch = True
        finally:
            fsops.unlink(owner, missing_ok=True)
        if dispatch:
            _spawn(path)
        return True
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="copilot_accounting_export_spawn",
            error=exc,
            platform="copilot",
            session_id=str(payload.get("session_id") or ""),
        )
        return False


def queue_session_accounting(
    config: Config,
    session_dir: Path,
    session_id: str,
    cwd: str,
    accounting: dict[str, Any],
) -> bool:
    accounting_id = str(accounting["accounting_id"])
    logical_span_id = f"accounting:{session_id}:{accounting_id}"
    return _queue(
        config,
        {
            "job_id": logical_span_id,
            "kind": "session_accounting",
            "session_dir": str(session_dir),
            "session_id": session_id,
            "cwd": cwd,
            "accounting_id": accounting_id,
            "usage": dict(accounting["usage"]),
            "attribution_status": str(accounting["attribution_status"]),
            "agent_id": accounting.get("agent_id"),
            "attributes": dict(accounting.get("attributes") or {}),
        },
    )


def queue_turn_accounting(
    config: Config,
    session_dir: Path,
    session_id: str,
    cwd: str,
    turn_id: str,
    accounting: dict[str, Any],
    *,
    turn_span_id: str,
) -> bool:
    accounting_id = str(accounting["accounting_id"])
    logical_span_id = f"accounting:{session_id}:{turn_id}:{accounting_id}"
    return _queue(
        config,
        {
            "job_id": logical_span_id,
            "kind": "turn_accounting",
            "session_dir": str(session_dir),
            "session_id": session_id,
            "cwd": cwd,
            "turn_id": turn_id,
            "turn_span_id": turn_span_id,
            "accounting_id": accounting_id,
            "usage": dict(accounting["usage"]),
            "attribution_status": str(accounting["attribution_status"]),
            "agent_id": accounting.get("agent_id"),
            "attributes": dict(accounting.get("attributes") or {}),
        },
    )


def _acquire(path: Path) -> dict[str, Any] | None:
    owner = claim_path(path)
    if not _take_claim(owner, str(os.getpid()), recover_stale=True):
        return None

    # Re-read only after owning the claim. A cancellation can happen after a
    # worker's initial argv read but before this point; stale in-memory bytes
    # must never resurrect the cancelled job.
    payload = _read_job(path)
    if payload is None or payload.get("state") in {"failed", "emitted"}:
        fsops.unlink(owner, missing_ok=True)
        return None
    if delivery_sent(Path(str(payload["session_dir"])), str(payload["accounting_id"])):
        fsops.unlink(path, missing_ok=True)
        fsops.unlink(owner, missing_ok=True)
        return None
    claimed = {**payload, "state": "claimed"}
    _write_job(path, claimed)
    return claimed


def _accounting(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "accounting_id": payload["accounting_id"],
        "usage": payload["usage"],
        "attribution_status": payload["attribution_status"],
        "agent_id": payload.get("agent_id"),
        "attributes": payload.get("attributes") or {},
        "call_id": payload.get("call_id"),
    }


def _deliver(config: Config, payload: dict[str, Any]) -> None:
    token = otel_export._captured_attributes.set(payload.get("captured_attributes") or {})
    try:
        common = {
            "config": config,
            "session_dir_": Path(payload["session_dir"]),
            "session_id": payload["session_id"],
            "platform": "copilot",
            "cwd": payload["cwd"],
            "accounting": _accounting(payload),
        }
        if payload["kind"] == "session_accounting":
            otel_export._export_session_accounting_inner(**common)
        elif payload["kind"] == "turn_accounting":
            otel_export._export_turn_accounting_inner(
                **common,
                turn_id=str(payload["turn_id"]),
                turn_span_id=payload.get("turn_span_id"),
            )
        else:
            raise ValueError(f"unsupported Copilot accounting job kind: {payload.get('kind')!r}")
    finally:
        otel_export._captured_attributes.reset(token)


def run(path: Path) -> None:
    root = path.parents[2] if len(path.parents) > 2 else path.parent
    try:
        claimed = _acquire(path)
    except Exception as exc:
        log_capture_error(
            thirdeye_home=root,
            phase="copilot_accounting_claim_failed",
            error=exc,
            platform="copilot",
        )
        return
    if claimed is None:
        return
    owner = claim_path(path)
    try:
        _deliver(Config.load(), claimed)
    except Exception as exc:
        attempt = int(claimed.get("attempt", 0)) + 1
        retry = {
            **claimed,
            "state": "failed" if attempt >= _MAX_ATTEMPTS else "retrying",
            "attempt": attempt,
            "last_error": f"{type(exc).__name__}: {exc}",
        }
        try:
            _write_job(path, retry)
        except Exception as state_error:
            log_capture_error(
                thirdeye_home=root,
                phase="copilot_accounting_state_failed",
                error=state_error,
                platform="copilot",
                session_id=str(claimed.get("session_id") or ""),
            )
        log_capture_error(
            thirdeye_home=root,
            phase="copilot_accounting_export_failed",
            error=exc,
            platform="copilot",
            session_id=str(claimed.get("session_id") or ""),
            message=f"accounting_id={claimed.get('accounting_id')} kind={claimed.get('kind')}",
        )
    else:
        try:
            _mark_delivered(Path(str(claimed["session_dir"])), str(claimed["accounting_id"]))
            _write_job(path, {**claimed, "state": "emitted", "last_error": None})
            fsops.unlink(path, missing_ok=True)
        except Exception as exc:
            log_capture_error(
                thirdeye_home=root,
                phase="copilot_accounting_ack_failed",
                error=exc,
                platform="copilot",
                session_id=str(claimed.get("session_id") or ""),
            )
    finally:
        fsops.unlink(owner, missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    values = sys.argv[1:] if argv is None else argv
    if values:
        run(Path(values[0]))


if __name__ == "__main__":
    main()
