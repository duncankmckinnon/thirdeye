"""Fallback for interrupted Codex turns that never reach ``notify``."""

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


def mark_turn_open(session_dir_: Path, *, prompt: str, prompt_id: str | None = None) -> None:
    marker = {
        "turn_id": new_ulid(),
        "start_ts": utc_iso_ms(),
        "input_message": prompt,
        "prompt_id": prompt_id,
    }
    try:
        with _locked_marker(session_dir_) as fd:
            os.lseek(fd, 0, os.SEEK_SET)
            if not os.read(fd, 1):
                _write_locked(fd, marker)
    except OSError:
        pass


def clear_marker_not_after(session_dir_: Path, *, not_after_ts: str) -> None:
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
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    *,
    min_age_seconds: float,
    prompt_id: str | None = None,
    replacement: dict[str, Any] | None = None,
) -> None:
    turn: TurnSpanDict | None = None
    try:
        with _locked_marker(session_dir_) as fd:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 1 << 20)
            if not raw:
                if replacement is not None:
                    _write_locked(fd, replacement)
                return
            try:
                marker = json.loads(raw)
            except json.JSONDecodeError:
                _write_locked(fd, replacement)
                return
            if replacement is not None:
                old_seq = marker.get("turn_seq")
                new_seq = replacement.get("turn_seq")
                if isinstance(old_seq, int) and isinstance(new_seq, int) and old_seq > new_seq:
                    return
            turn_id = marker.get("turn_id")
            start_ts = marker.get("start_ts")
            if not turn_id or not start_ts:
                _write_locked(fd, replacement)
                return
            marker_prompt_id = marker.get("prompt_id")
            different_prompt = bool(
                prompt_id and marker_prompt_id and prompt_id != marker_prompt_id
            )
            try:
                age = (datetime.now(UTC) - _parse_ts(str(start_ts))).total_seconds()
            except ValueError:
                _write_locked(fd, replacement)
                return
            if not different_prompt and age < min_age_seconds:
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
            _write_locked(fd, replacement)
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
    _reap(config, session_dir_, session_id, cwd, min_age_seconds=0)


def reap_abandoned_marker(config: Config, session_dir_: Path, session_id: str, cwd: str) -> None:
    _reap(config, session_dir_, session_id, cwd, min_age_seconds=_ABANDONED_AFTER_SECONDS)


def reap_marker_for_event(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    *,
    prompt_id: str | None,
) -> None:
    _reap(
        config,
        session_dir_,
        session_id,
        cwd,
        min_age_seconds=_ABANDONED_AFTER_SECONDS,
        prompt_id=prompt_id,
    )


def replace_open_turn(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    *,
    prompt: str,
    prompt_id: str | None,
    turn_seq: int,
) -> None:
    replacement = {
        "turn_id": new_ulid(),
        "start_ts": utc_iso_ms(),
        "input_message": prompt,
        "prompt_id": prompt_id,
        "turn_seq": turn_seq,
    }
    _reap(
        config,
        session_dir_,
        session_id,
        cwd,
        min_age_seconds=0,
        replacement=replacement,
    )
