"""Grok Bot store-mutation kick — install-armed, idle-exiting → otel_export.

Duncan Q1: action indicator is ``store.db`` mutation, not a boot/pidfile daemon.
Install arms the kick; ``on_store_mutation`` (aliases) runs a short sync via
``tick`` then returns. ``tick``/``run_once`` remain library helpers.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.platforms.grok_bot.capture import build_turns_from_entries
from thirdeye.platforms.grok_bot.constants import PLATFORM_NAME

_WATERMARK_FILE = "watermarks.json"


def _kick_flag_set(platform: Any = None) -> bool:
    if platform is not None:
        for name in (
            "is_store_kick_enabled",
            "is_kick_enabled",
            "is_mutation_kick_enabled",
            "is_action_indicator_enabled",
        ):
            fn = getattr(platform, name, None)
            if callable(fn):
                try:
                    return bool(fn())
                except TypeError:
                    continue
        # Fall back to flag files under state dir.
        root = getattr(platform, "_state_dir", None)
        if root is not None:
            from pathlib import Path as _P

            r = _P(root)
            if (r / "kick.enabled").is_file() or (r / "watcher.enabled").is_file():
                return True
    return False


def is_store_kick_enabled(platform: Any = None) -> bool:
    return _kick_flag_set(platform)


def is_kick_enabled(platform: Any = None) -> bool:
    return is_store_kick_enabled(platform)


def kick_enabled(platform: Any = None) -> bool:
    return is_store_kick_enabled(platform)


def is_action_indicator_enabled(platform: Any = None) -> bool:
    return is_store_kick_enabled(platform)


def is_watcher_running(platform: Any = None) -> bool:
    """Interim alias for kick-enabled (not boot-daemon SoT)."""
    if platform is not None and hasattr(platform, "is_watcher_running"):
        try:
            return bool(platform.is_watcher_running())
        except TypeError:
            pass
    return is_store_kick_enabled(platform)


def is_running(platform: Any = None) -> bool:
    return is_store_kick_enabled(platform)


def status(platform: Any = None) -> dict[str, Any]:
    enabled = is_store_kick_enabled(platform)
    return {"running": enabled, "enabled": enabled, "kick": enabled}


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


def on_store_mutation(
    platform: Any = None,
    *,
    store_path: Path | str | None = None,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Action-indicator entry: store.db mutation → short sync → idle-exit.

    Marker-gated: if kick is not enabled, no-op (fail-open). Reuses ``tick``
    for watermarked export via shared ``otel_export``.
    """
    if platform is not None and not is_store_kick_enabled(platform):
        return {"exported": 0, "skipped": True, "reason": "kick_disabled"}
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


def notify_store_change(
    platform: Any = None,
    *,
    store_path: Path | str | None = None,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return on_store_mutation(
        platform,
        store_path=store_path,
        agents_root=agents_root,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
        **kwargs,
    )


def handle_store_kick(
    platform: Any = None,
    *,
    store_path: Path | str | None = None,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return on_store_mutation(
        platform,
        store_path=store_path,
        agents_root=agents_root,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
        **kwargs,
    )


def kick_from_store_change(
    platform: Any = None,
    *,
    store_path: Path | str | None = None,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return on_store_mutation(
        platform,
        store_path=store_path,
        agents_root=agents_root,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
        **kwargs,
    )


def on_action_indicator(
    platform: Any = None,
    *,
    store_path: Path | str | None = None,
    agents_root: Path | str | None = None,
    conversation_id: str = "",
    agent_id: str = "",
    agent_name: str = "",
    cwd: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return on_store_mutation(
        platform,
        store_path=store_path,
        agents_root=agents_root,
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        cwd=cwd,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Real FS/mtime observer (Wave 4 RC) — armed by install, stopped by uninstall.
# Short-lived / idle-capable; not a boot/pidfile daemon SoT.
# ---------------------------------------------------------------------------

_OBSERVERS: dict[int, "_StoreMtimeObserver"] = {}
_OBSERVERS_LOCK = threading.Lock()


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return (st.st_size, st.st_mtime_ns)


class _StoreMtimeObserver:
    """Poll agents/*/store.db (+ WAL) mtime; on change run on_store_mutation."""

    def __init__(self, platform: Any, agents_root: Path, *, interval: float = 0.05) -> None:
        self._platform = platform
        self._agents_root = Path(agents_root)
        self._interval = interval
        self._stop = threading.Event()
        self._stamps: dict[str, tuple[int, int] | None] = {}
        self._thread = threading.Thread(
            target=self._loop,
            name="thirdeye-grok-bot-store-observer",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not is_store_kick_enabled(self._platform):
                    break
                self._scan_once()
            except Exception:
                # Fail-open: never raise into the agent process.
                pass
            if self._stop.wait(self._interval):
                break

    def _iter_store_paths(self) -> list[Path]:
        root = self._agents_root
        out: list[Path] = []
        try:
            if not root.is_dir():
                return out
            for child in root.iterdir():
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    continue
                db = child / "store.db"
                out.append(db)
                # WAL changes count as activity even if DB mtime lags.
                out.append(Path(str(db) + "-wal"))
        except OSError:
            return out
        return out

    def _scan_once(self) -> None:
        seen_keys: set[str] = set()
        changed_dbs: set[Path] = set()
        for path in self._iter_store_paths():
            key = str(path)
            seen_keys.add(key)
            stamp = _file_stamp(path)
            prev = self._stamps.get(key, object())
            if stamp != prev:
                self._stamps[key] = stamp
                # Map WAL path back to store.db
                db = path if path.name == "store.db" else path.parent / "store.db"
                if stamp is not None or path.name == "store.db":
                    if db.exists() or path.name == "store.db":
                        changed_dbs.add(db)
        # Drop stamps for vanished paths
        for key in list(self._stamps):
            if key not in seen_keys:
                del self._stamps[key]
        for db in changed_dbs:
            if not db.exists():
                continue
            agent_id = db.parent.name
            try:
                on_store_mutation(
                    self._platform,
                    store_path=db,
                    agents_root=self._agents_root,
                    conversation_id=agent_id,
                    agent_id=agent_id,
                    agent_name="",
                    cwd=str(self._agents_root.parent),
                )
            except Exception:
                continue


def start_store_observer(
    platform: Any,
    *,
    agents_root: Path | str,
    interval: float = 0.05,
) -> _StoreMtimeObserver:
    """Arm a box-level mtime observer for ``agents/*/store.db``."""
    root = Path(agents_root)
    key = id(platform)
    with _OBSERVERS_LOCK:
        old = _OBSERVERS.pop(key, None)
        if old is not None:
            old.stop()
        obs = _StoreMtimeObserver(platform, root, interval=interval)
        _OBSERVERS[key] = obs
        obs.start()
        return obs


def stop_store_observer(platform: Any = None) -> None:
    """Disarm observer for ``platform`` (or all if platform is None)."""
    with _OBSERVERS_LOCK:
        if platform is None:
            items = list(_OBSERVERS.items())
            _OBSERVERS.clear()
        else:
            key = id(platform)
            obs = _OBSERVERS.pop(key, None)
            items = [(key, obs)] if obs is not None else []
    for _key, obs in items:
        if obs is not None:
            obs.stop()


def is_store_observer_running(platform: Any = None) -> bool:
    with _OBSERVERS_LOCK:
        if platform is None:
            return any(
                obs._thread.is_alive() for obs in _OBSERVERS.values()  # noqa: SLF001
            )
        obs = _OBSERVERS.get(id(platform))
        return bool(obs is not None and obs._thread.is_alive())  # noqa: SLF001

