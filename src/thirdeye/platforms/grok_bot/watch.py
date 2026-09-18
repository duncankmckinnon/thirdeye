"""Passive Grok Bot watcher — install-enabled stamp/poll → otel_export.

User happy path: ``thirdeye add --grok-bot`` (or Cursor co-install) enables the
watcher. Tests drive one cycle via ``tick`` / ``run_once``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.platforms.grok_bot.capture import build_turns_from_entries
from thirdeye.platforms.grok_bot.constants import PLATFORM_NAME

_WATERMARK_FILE = "watermarks.json"


def is_watcher_running(platform: Any = None) -> bool:
    if platform is not None and hasattr(platform, "is_watcher_running"):
        return bool(platform.is_watcher_running())
    return False


def is_running(platform: Any = None) -> bool:
    return is_watcher_running(platform)


def status(platform: Any = None) -> dict[str, Any]:
    running = is_watcher_running(platform)
    return {"running": running, "enabled": running}


def _state_dir(platform: Any) -> Path | None:
    if platform is None:
        return None
    return getattr(platform, "_state_dir", None) or getattr(platform, "state_dir", None)


def _watermark_path(platform: Any) -> Path | None:
    root = _state_dir(platform)
    if root is None:
        return None
    return Path(root) / _WATERMARK_FILE


def _load_watermarks(platform: Any) -> dict[str, int]:
    path = _watermark_path(platform)
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in data.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _save_watermarks(platform: Any, marks: dict[str, int]) -> None:
    path = _watermark_path(platform)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(marks, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def _read_entries_since(db_path: Path, after_seq: int) -> list[tuple[int, dict[str, Any]]] | None:
    """Return (seq, entry) rows with seq > after_seq, or None on fail-open."""
    import sqlite3

    try:
        conn = sqlite3.connect(str(db_path), timeout=0.1)
    except sqlite3.Error:
        return None
    try:
        try:
            rows = conn.execute(
                "SELECT seq, id, entry FROM transcript_entries "
                "WHERE seq > ? ORDER BY seq",
                (after_seq,),
            ).fetchall()
        except sqlite3.Error:
            return None
        out: list[tuple[int, dict[str, Any]]] = []
        for seq, _eid, body in rows:
            try:
                parsed = json.loads(body)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                out.append((int(seq), parsed))
        return out
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _discover_stores(
    *,
    agents_root: Path | None,
    store_path: Path | str | None,
    agent_id: str,
) -> list[tuple[str, Path]]:
    if store_path is not None:
        path = Path(store_path)
        aid = agent_id or path.parent.name
        return [(aid, path)]
    if agents_root is None:
        return []
    root = Path(agents_root)
    stores: list[tuple[str, Path]] = []
    try:
        children = [c for c in root.iterdir() if c.is_dir()]
    except OSError:
        children = []
    if agent_id:
        stores.append((agent_id, root / agent_id / "store.db"))
    for child in children:
        if agent_id and child.name == agent_id:
            continue  # already added
        stores.append((child.name, child / "store.db"))
    if not stores and agent_id:
        stores.append((agent_id, root / agent_id / "store.db"))
    seen: set[str] = set()
    unique: list[tuple[str, Path]] = []
    for aid, path in stores:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((aid, path))
    return unique


def tick(
    platform: Any = None,
    *,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    store_path: Path | str | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run one passive poll cycle (test harness / watcher loop body)."""
    from thirdeye import otel_export

    result: dict[str, Any] = {"exported": 0, "skipped": True}
    root = Path(agents_root) if agents_root is not None else None
    stores = _discover_stores(
        agents_root=root, store_path=store_path, agent_id=agent_id
    )
    if not stores:
        return result

    marks = _load_watermarks(platform)
    exported = 0
    try:
        config = Config.load()
    except Exception:
        config = None  # type: ignore[assignment]

    for aid, db_path in stores:
        after = marks.get(str(db_path), marks.get(aid, 0))
        rows = _read_entries_since(db_path, after)
        if rows is None:
            continue
        if not rows:
            continue
        max_seq = max(seq for seq, _ in rows)
        entries = [entry for _seq, entry in rows]
        conv = conversation_id or aid
        turns = build_turns_from_entries(
            entries,
            conversation_id=conv,
            agent_id=aid or agent_id,
            agent_name=agent_name,
        )
        session_dir = Path(cwd) if cwd else db_path.parent
        for turn in turns:
            if not turn:
                continue
            if not (turn.get("input_message") or turn.get("output_message")):
                continue
            try:
                otel_export.export_turn(
                    config,  # type: ignore[arg-type]
                    session_dir,
                    conv,
                    PLATFORM_NAME,
                    cwd or str(session_dir),
                    turn,
                )
                exported += 1
            except Exception:
                continue
        # Advance watermark even when no complete turns (avoid re-reading forever).
        marks[str(db_path)] = max_seq
        marks[aid] = max_seq

    _save_watermarks(platform, marks)
    return {"exported": exported, "skipped": exported == 0}


def run_once(
    platform: Any = None,
    *,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    store_path: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return tick(
        platform,
        agents_root=agents_root,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
        store_path=store_path,
        **kwargs,
    )


def poll_once(
    platform: Any = None,
    *,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    store_path: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return tick(
        platform,
        agents_root=agents_root,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
        store_path=store_path,
        **kwargs,
    )
