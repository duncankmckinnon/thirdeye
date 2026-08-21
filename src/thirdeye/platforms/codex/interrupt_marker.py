"""Fallback export path for a Codex turn that never gets an
``agent-turn-complete`` ``notify`` call at all (as opposed to one that gets a
``notify`` call but whose rollout carries a ``turn_aborted`` frame -- that
case is handled entirely by ``turn.py``'s own status detection). Whether
Codex's ``notify`` ever skips firing for an aborted turn could not be
empirically confirmed here (no live Codex CLI, no persistent log of raw
``notify`` invocations, no network access) -- a real ``turn_aborted`` frame
pulled from a genuine rollout on this machine confirmed the *shape* `turn.py`
parses is right (see its test fixture), but not this module's own necessity.
Kept because the failure mode of being wrong the other way -- deleting this
and silently losing every interrupted turn's trace -- is worse than the dead
code. Isolated in its own module so it can be deleted later without touching
``turn.py``'s primary path.

Two hook mechanisms touch the marker for one session concurrently:
``codex/hooks.py``'s argv-invoked ``notify`` (clears it once a turn's real
completion is known) and ``codex/hooks_json.py``'s JSON-stdin hooks (open it
at turn start; every other hook call checks it). There's no id shared
between the two -- ``notify``'s own ``turn-id`` isn't known until the turn is
already finishing -- so correctness relies on two things instead:

- All reads/writes/clears go through a single, always-open, never-unlinked
  inode (truncated-and-rewritten in place) under ``fcntl.flock`` for the
  whole read-decide-write section -- unlink+recreate would let two racing
  opens land on different inodes and defeat the lock.
- ``clear_marker_not_after`` only clears a marker opened *at or before* the
  completing turn's own ``end_ts``, so a delayed ``notify`` for turn A can't
  clobber a newer marker turn B's ``UserPromptSubmit`` opened while A was
  still in flight -- B can only have started after A ended.

Hooks that fire *mid-turn* (SubagentStart/Stop, PermissionRequest,
PreCompact/PostCompact, SessionStart) reap only once a marker is old enough
to be genuinely abandoned (``_ABANDONED_AFTER_SECONDS``), since an open
marker at that point is normally just the still-running current turn.
``UserPromptSubmit``/``SessionEnd`` reap immediately -- either firing means
the turn that opened the marker is unambiguously over.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from thirdeye.ids import new_ulid
from thirdeye.tracing.model import TurnSpanDict
from thirdeye.usage.errlog import log_capture_error
from thirdeye.writer import utc_iso_ms

if TYPE_CHECKING:
    from thirdeye.config import Config

_PLATFORM = "codex"

# Real Codex turns don't run for hours; a marker still open past this is
# abandoned (crashed process, no SessionEnd ever arrived), not a currently
# running turn, so mid-turn hooks may safely reap it.
_ABANDONED_AFTER_SECONDS = 6 * 3600


def _marker_path(session_dir_: Path) -> Path:
    return session_dir_ / "codex-open-turn.json"


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)


@contextlib.contextmanager
def _locked_marker(session_dir_: Path) -> Iterator[int]:
    session_dir_.mkdir(parents=True, exist_ok=True)
    fd = os.open(_marker_path(session_dir_), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_locked(fd: int, marker: dict[str, Any] | None) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    if marker is not None:
        os.write(fd, json.dumps(marker).encode())


def has_open_marker(session_dir_: Path) -> bool:
    try:
        with _locked_marker(session_dir_) as fd:
            os.lseek(fd, 0, os.SEEK_SET)
            return os.read(fd, 1) != b""
    except OSError:
        return False


def mark_turn_open(session_dir_: Path, *, prompt: str) -> None:
    marker = {"turn_id": new_ulid(), "start_ts": utc_iso_ms(), "input_message": prompt}
    try:
        with _locked_marker(session_dir_) as fd:
            _write_locked(fd, marker)
    except OSError:
        pass


def clear_marker_not_after(session_dir_: Path, *, not_after_ts: str) -> None:
    """Called by ``notify()`` once a turn's real completion is known. Never
    raises: marker cleanup must never block the rollout/usage capture that
    runs after it in the same hook invocation.
    """
    try:
        with _locked_marker(session_dir_) as fd:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 1 << 20)
            if not raw:
                return
            try:
                marker = json.loads(raw)
            except json.JSONDecodeError:
                _write_locked(fd, None)
                return
            marker_start = str(marker.get("start_ts") or "")
            try:
                if marker_start and _parse_ts(marker_start) > _parse_ts(not_after_ts):
                    return  # opened after this turn ended -- a newer turn's marker
            except ValueError:
                pass
            _write_locked(fd, None)
    except OSError:
        pass


def _reap(
    config: Config, session_dir_: Path, session_id: str, cwd: str, *, min_age_seconds: float
) -> None:
    turn: TurnSpanDict | None = None
    try:
        with _locked_marker(session_dir_) as fd:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 1 << 20)
            if not raw:
                return
            try:
                marker = json.loads(raw)
            except json.JSONDecodeError:
                _write_locked(fd, None)  # corrupt -- quarantine rather than leave it wedged
                return
            turn_id = marker.get("turn_id")
            start_ts = marker.get("start_ts")
            if not turn_id or not start_ts:
                _write_locked(fd, None)
                return
            try:
                age = (datetime.now(UTC) - _parse_ts(str(start_ts))).total_seconds()
            except ValueError:
                _write_locked(fd, None)
                return
            if age < min_age_seconds:
                return
            turn = {
                "turn_id": str(turn_id),
                "start_ts": str(start_ts),
                "end_ts": utc_iso_ms(),
                "input_message": str(marker.get("input_message") or ""),
                "output_message": "",
                "status": "interrupted",
                "llm_calls": [],
                "permission_requests": [],
                "subagents": [],
                "attributes": {},
            }
            _write_locked(fd, None)
    except OSError:
        return

    if turn is None:
        return
    try:
        from thirdeye.otel_export import export_turn

        export_turn(config, session_dir_, session_id, _PLATFORM, cwd, turn)
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="codex_interrupt_marker_export",
            error=exc,
            platform=_PLATFORM,
            session_id=session_id,
        )


def close_stale_turn_if_open(
    config: Config, session_dir_: Path, session_id: str, cwd: str
) -> None:
    """A new turn is starting (``UserPromptSubmit``) or the session is
    ending (``SessionEnd``) -- either way, a marker still open at this point
    unambiguously belongs to a now-finished-one-way-or-another previous
    turn, so it's closed out immediately, no age threshold needed.
    """
    _reap(config, session_dir_, session_id, cwd, min_age_seconds=0)


def reap_abandoned_marker(config: Config, session_dir_: Path, session_id: str, cwd: str) -> None:
    """Safety-net check for hooks that legitimately fire *mid-turn*. See the
    module docstring for why this needs an age threshold and
    ``close_stale_turn_if_open`` doesn't.
    """
    _reap(config, session_dir_, session_id, cwd, min_age_seconds=_ABANDONED_AFTER_SECONDS)
