"""Fallback export path for a Codex turn that never gets an
``agent-turn-complete`` ``notify`` call at all (as opposed to one that gets a
``notify`` call but whose rollout carries a ``turn_aborted`` frame — that
case is handled entirely by ``turn.py``'s own status detection and needs no
help from this module).

Codex's older, argv-invoked ``notify`` callback only ever fires with type
``agent-turn-complete`` — there is no "turn started" signal on that channel to
mark against, and if ``notify`` is simply never called for an aborted turn,
nothing else would ever export it. ``hooks.json``'s ``UserPromptSubmit``
handler (``hooks_json.py``) does fire reliably at the start of every turn, so
it is the one place that can plant an open-turn marker; any later Codex hook
call for the same session (typically the very next ``UserPromptSubmit``,
``SessionEnd`` as a catch-all) closes it out as interrupted if ``notify``
never got the chance to clear it first.

Deliberately isolated in its own module: if it turns out ``notify`` always
fires even for aborted turns, making this fallback unreachable in practice,
this whole file can be deleted without touching ``turn.py``'s primary,
rollout-based ``turn_aborted`` path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from thirdeye.ids import new_ulid
from thirdeye.tracing.model import TurnSpanDict
from thirdeye.writer import utc_iso_ms

if TYPE_CHECKING:
    from thirdeye.config import Config


def _marker_path(session_dir_: Path) -> Path:
    return session_dir_ / "codex-open-turn.json"


def mark_turn_open(session_dir_: Path, *, prompt: str) -> None:
    marker = {"turn_id": new_ulid(), "start_ts": utc_iso_ms(), "input_message": prompt}
    try:
        _marker_path(session_dir_).write_text(json.dumps(marker))
    except OSError:
        pass


def clear_turn_marker(session_dir_: Path) -> None:
    _marker_path(session_dir_).unlink(missing_ok=True)


def close_stale_turn_if_open(
    config: Config, session_dir_: Path, session_id: str, cwd: str
) -> None:
    """If a prior turn was left open (its ``notify`` call never arrived to
    clear the marker), export it now as interrupted. Never raises: this runs
    inside hook subprocesses, same safety contract as ``otel_export``.
    """
    from thirdeye.otel_export import export_turn

    path = _marker_path(session_dir_)
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    turn_id = marker.get("turn_id")
    start_ts = marker.get("start_ts")
    if not turn_id or not start_ts:
        path.unlink(missing_ok=True)
        return
    turn: TurnSpanDict = {
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
    try:
        export_turn(config, session_dir_, session_id, "codex", cwd, turn)
    finally:
        path.unlink(missing_ok=True)
