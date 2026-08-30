"""Join Cursor IDE child conversations to the parent Task that dispatched them.

Backgrounded IDE subagents often never fire ``subagentStop``. Their transcript
under ``agent-transcripts/<parent>/subagents/<child>.jsonl`` still ends with
``{"type": "turn_ended"}``. That record is the completion signal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from thirdeye.platforms.cursor.subagents import _seq, _string, cursor_subagent_windows


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
    exported = {
        _string(event.get("data") if isinstance(event.get("data"), dict) else {}, "child_session_id")
        for event in events
    }
    exported.discard("")
    return [child_id for child_id in ended_child_ids(parent_session_id) if child_id not in exported]
