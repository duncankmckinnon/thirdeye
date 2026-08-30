"""Join Cursor IDE child conversations to the parent Task that dispatched them.

Backgrounded IDE subagents often never fire ``subagentStop``. Their transcript
under ``agent-transcripts/<parent>/subagents/<child>.jsonl`` still ends with
``{"type": "turn_ended"}``. That record is the completion signal.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.subagents import _data, _seq, _string, cursor_subagent_windows
from thirdeye.reader import SessionReader


def transcript_roots() -> list[Path]:
    override = os.environ.get("THIRDEYE_CURSOR_TRANSCRIPT_ROOTS", "")
    if override:
        return [Path(part) for part in override.split(os.pathsep) if part]
    projects = Path.home() / ".cursor" / "projects"
    if not projects.is_dir():
        return []
    return [path / "agent-transcripts" for path in projects.iterdir() if path.is_dir()]


def transcript_turn_ended(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("type") == "turn_ended":
            return True
        break
    return False


def parent_session_for_child(child_session_id: str, roots: list[Path] | None = None) -> str:
    if not child_session_id:
        return ""
    for root in roots if roots is not None else transcript_roots():
        if not root.is_dir():
            continue
        matches = list(root.glob(f"*/subagents/{child_session_id}.jsonl"))
        if len(matches) == 1:
            return matches[0].parent.parent.name
    return ""


def ended_child_ids(parent_session_id: str, roots: list[Path] | None = None) -> list[str]:
    ended: list[str] = []
    for root in roots if roots is not None else transcript_roots():
        folder = root / parent_session_id / "subagents"
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            if transcript_turn_ended(path):
                ended.append(path.stem)
    return ended


def unmatched_task_ids(events: list[dict[str, Any]]) -> list[str]:
    paired = {
        window.subagent_id for window in cursor_subagent_windows(events) if window.start_event
    }
    seen: list[str] = []
    for event in sorted(events, key=_seq):
        if event.get("t") != "subagent_start":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        subagent_id = _string(data, "subagent_id", "subagentId")
        if subagent_id and subagent_id not in paired and subagent_id not in seen:
            seen.append(subagent_id)
    return seen


def pending_ended_child_ids(parent_session_id: str, events: list[dict[str, Any]]) -> list[str]:
    exported = _exported_child_ids(events)
    return [child_id for child_id in ended_child_ids(parent_session_id) if child_id not in exported]


def _exported_child_ids(events: list[dict[str, Any]]) -> set[str]:
    exported = {_string(_data(event), "child_session_id") for event in events}
    exported.discard("")
    return exported


def _unmatched_start_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = unmatched_task_ids(events)
    found: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=_seq):
        if event.get("t") != "subagent_start":
            continue
        subagent_id = _string(_data(event), "subagent_id", "subagentId")
        if subagent_id in wanted and subagent_id not in found:
            found[subagent_id] = event
    return [found[subagent_id] for subagent_id in wanted if subagent_id in found]


def child_started_ts(
    parent_session_id: str,
    child_session_id: str,
    *,
    thirdeye_home: Path | None = None,
) -> str:
    if thirdeye_home is not None:
        child_dir = session_dir(thirdeye_home, "cursor", child_session_id)
        if child_dir.is_dir():
            child_events = list(SessionReader(child_dir).iter_events())
            if child_events:
                return str(child_events[0].get("ts") or "")
    for root in transcript_roots():
        path = root / parent_session_id / "subagents" / f"{child_session_id}.jsonl"
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        return datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return ""


def task_id_for_child(
    events: list[dict[str, Any]],
    child_session_id: str,
    child_ts: str,
) -> str:
    """Bind one IDE child conversation to one unmatched Task start.

    Resume starts already store ``cursor.subagent.agent_id``. First-dispatch
    children are matched to the unmatched start whose timestamp window
    contains ``child_ts`` — never by list length or zip order.
    """
    if not child_session_id:
        return ""
    starts = _unmatched_start_events(events)
    for start in starts:
        if _string(_data(start), "cursor.subagent.agent_id") == child_session_id:
            return _string(_data(start), "subagent_id", "subagentId")
    if not child_ts:
        return ""
    for index, start in enumerate(starts):
        start_ts = str(start.get("ts") or "")
        if not start_ts or child_ts < start_ts:
            continue
        next_ts = str(starts[index + 1].get("ts") or "") if index + 1 < len(starts) else ""
        if next_ts and child_ts >= next_ts:
            continue
        return _string(_data(start), "subagent_id", "subagentId")
    return ""


def exportable_child_pairs(
    parent_session_id: str,
    events: list[dict[str, Any]],
    *,
    only_child: str = "",
    thirdeye_home: Path | None = None,
    ended: list[str] | None = None,
    child_started: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    ended_ids = list(ended) if ended is not None else ended_child_ids(parent_session_id)
    exported = _exported_child_ids(events)
    unmatched = set(unmatched_task_ids(events))
    candidates = [only_child] if only_child else ended_ids
    pairs: list[tuple[str, str]] = []
    used_tasks: set[str] = set()
    for child_id in candidates:
        if not child_id or child_id not in ended_ids or child_id in exported:
            continue
        child_ts = (child_started or {}).get(child_id) or child_started_ts(
            parent_session_id, child_id, thirdeye_home=thirdeye_home
        )
        task_id = task_id_for_child(events, child_id, child_ts)
        if not task_id or task_id not in unmatched or task_id in used_tasks:
            continue
        pairs.append((task_id, child_id))
        used_tasks.add(task_id)
    return pairs
