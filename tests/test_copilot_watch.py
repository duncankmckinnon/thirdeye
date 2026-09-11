"""Behavioral tests for Copilot foreground watch polling."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import thirdeye.platforms.copilot.watch as watch_mod
from thirdeye.config import Config
from thirdeye.platforms.copilot.hook_payload import parse_hook
from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.spool import enqueue_hook
from thirdeye.platforms.copilot.types import SourcePaths, SyncResult
from thirdeye.platforms.copilot.watch import _changed_sessions, watch

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
CLI_FIXTURE = FIXTURES / "cli-1.0.83"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
OTHER_SESSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
OBSERVED_AT = "2026-09-10T17:08:25.626Z"


@pytest.fixture
def copilot_env(tmp_path: Path) -> tuple[Config, SourcePaths]:
    home = tmp_path / "copilot-home"
    home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(home)
    return config, paths


def _empty_result(**overrides: int) -> SyncResult:
    result: SyncResult = {
        "sessions": 0,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 0,
    }
    result.update(overrides)  # type: ignore[typeddict-item]
    return result


def _write_transcript(home: Path, native_id: str, *, events_path: Path | None = None) -> Path:
    session_dir = home / "session-state" / native_id
    session_dir.mkdir(parents=True, exist_ok=True)
    source = events_path or (CLI_FIXTURE / "events.jsonl")
    destination = session_dir / "events.jsonl"
    shutil.copy(source, destination)
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")
    return destination


def _write_database(home: Path, *, session_id: str = NATIVE_SESSION_ID) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, created_at TEXT);
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER,
                content TEXT,
                updated_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions (id, cwd, created_at) VALUES (?, ?, ?)",
            (session_id, "/tmp/probe", "2026-09-10T17:08:00.000Z"),
        )
        connection.execute(
            "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, session_id, 0, "turn-0", "2026-09-10T17:08:10.000Z"),
        )
        connection.commit()
    finally:
        connection.close()
    return database


def _hook_record(*, observation_id: str, session_id: str = NATIVE_SESSION_ID) -> dict[str, Any]:
    return parse_hook(
        "agentStop",
        {
            "sessionId": session_id,
            "timestamp": 1789060105626,
            "cwd": "/fixture/workspace",
            "stopReason": "end_turn",
        },
        {"env": {"WB_PLAN": "p"}},
        observed_at=OBSERVED_AT,
        observation_id=observation_id,
    )


def _snapshot(
    *,
    sessions: set[str] | None = None,
    database_sessions: set[str] | None = None,
    transcripts: dict[str, object] | None = None,
    database: tuple[object, object, object] | None = None,
    spool: dict[str, tuple[tuple[str, object], ...]] | None = None,
) -> dict[str, Any]:
    session_set = sessions or set()
    return {
        "sessions": session_set,
        "database_sessions": database_sessions or set(),
        "transcripts": transcripts or {},
        "database": database or (None, None, None),
        "spool": spool or {},
    }


def test_watch_module_has_no_command_or_hook_runtime_imports() -> None:
    source = Path(watch_mod.__file__).read_text(encoding="utf-8")
    assert "thirdeye.commands" not in source
    assert "hook_lifecycle" not in source
    assert "followup" not in source


@pytest.mark.parametrize(
    "interval",
    [0.09, -1.0, float("inf"), float("nan"), True, "1"],
)
def test_watch_rejects_invalid_interval(interval: object) -> None:
    config = Config(root=Path("/tmp/thirdeye"))
    paths = resolve_sources(Path("/tmp/copilot"))
    with pytest.raises(ValueError, match="interval must be a finite number"):
        watch(config, paths, interval=interval)  # type: ignore[arg-type]


def test_changed_sessions_detects_new_session() -> None:
    before = _snapshot(sessions=set())
    after = _snapshot(sessions={NATIVE_SESSION_ID})
    assert _changed_sessions(before, after) == {NATIVE_SESSION_ID}


def test_changed_sessions_detects_transcript_stamp_change() -> None:
    stamp_a = (1, 2, 100, 200)
    stamp_b = (1, 2, 150, 200)
    before = _snapshot(
        sessions={NATIVE_SESSION_ID},
        transcripts={NATIVE_SESSION_ID: stamp_a},
    )
    after = _snapshot(
        sessions={NATIVE_SESSION_ID},
        transcripts={NATIVE_SESSION_ID: stamp_b},
    )
    assert _changed_sessions(before, after) == {NATIVE_SESSION_ID}


def test_changed_sessions_detects_spool_stamp_change() -> None:
    before = _snapshot(
        sessions={NATIVE_SESSION_ID},
        spool={NATIVE_SESSION_ID: (("a.json", (1, 2, 3, 4)),)},
    )
    after = _snapshot(
        sessions={NATIVE_SESSION_ID},
        spool={NATIVE_SESSION_ID: (("a.json", (1, 2, 5, 4)),)},
    )
    assert _changed_sessions(before, after) == {NATIVE_SESSION_ID}


def test_changed_sessions_database_change_includes_prior_database_sessions() -> None:
    before = _snapshot(
        sessions={NATIVE_SESSION_ID, OTHER_SESSION_ID},
        database_sessions={NATIVE_SESSION_ID},
        database=((1, 2, 3, 4), None, None),
    )
    after = _snapshot(
        sessions={NATIVE_SESSION_ID, OTHER_SESSION_ID},
        database_sessions={OTHER_SESSION_ID},
        database=((1, 2, 3, 4), (5, 6, 7, 8), None),
    )
    changed = _changed_sessions(before, after)
    assert NATIVE_SESSION_ID in changed
    assert OTHER_SESSION_ID in changed


def test_watch_performs_initial_full_sync(
    monkeypatch: pytest.MonkeyPatch, copilot_env: tuple[Config, SourcePaths]
) -> None:
    config, paths = copilot_env
    calls: list[str | None] = []

    def tracking_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append(session_id)
        return _empty_result()

    def stop_immediately(_interval: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", tracking_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", stop_immediately)

    watch(config, paths, interval=0.1)

    assert calls == [None]


def test_watch_skips_per_session_sync_when_sources_are_quiet(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    calls: list[str | None] = []
    cycle = {"count": 0}

    def tracking_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append(session_id)
        return _empty_result()

    def sleep_then_interrupt(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", tracking_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", sleep_then_interrupt)

    watch(config, paths, interval=0.1)

    assert calls == [None]


def test_watch_syncs_only_changed_transcript_session(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_transcript(home, OTHER_SESSION_ID)
    events_path = home / "session-state" / NATIVE_SESSION_ID / "events.jsonl"
    calls: list[str | None] = []
    cycle = {"count": 0}

    def tracking_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append(session_id)
        return _empty_result()

    def append_during_poll(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write('{"type":"synthetic.append"}\n')
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", tracking_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", append_during_poll)

    watch(config, paths, interval=0.1)

    assert calls[0] is None
    assert calls.count(NATIVE_SESSION_ID) == 1
    assert OTHER_SESSION_ID not in calls


def test_watch_detects_database_wal_change(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    database = _write_database(home, session_id=NATIVE_SESSION_ID)
    calls: list[str | None] = []
    cycle = {"count": 0}

    def tracking_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append(session_id)
        return _empty_result()

    def mutate_database(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (2, NATIVE_SESSION_ID, 1, "late-row", "2026-09-10T17:08:12.000Z"),
                )
                connection.commit()
            finally:
                connection.close()
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", tracking_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", mutate_database)

    watch(config, paths, interval=0.1)

    assert calls[0] is None
    assert calls.count(NATIVE_SESSION_ID) >= 1


def test_watch_detects_spool_change(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    calls: list[str | None] = []
    cycle = {"count": 0}

    def tracking_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append(session_id)
        return _empty_result()

    def enqueue_during_poll(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            enqueue_hook(config, paths, _hook_record(observation_id="obs-watch-spool"))
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", tracking_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", enqueue_during_poll)

    watch(config, paths, interval=0.1)

    assert calls[0] is None
    assert calls.count(NATIVE_SESSION_ID) == 1


def test_watch_retries_sessions_with_pending_or_errors(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    attempts: dict[str, int] = {}
    cycle = {"count": 0}

    def flaky_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        if session_id is None:
            return _empty_result(pending=1)
        attempts[session_id] = attempts.get(session_id, 0) + 1
        if attempts[session_id] == 1:
            return _empty_result(pending=1)
        return _empty_result()

    def three_poll_cycles(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", flaky_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", three_poll_cycles)

    watch(config, paths, interval=0.1)

    assert attempts.get(NATIVE_SESSION_ID, 0) >= 2


def test_watch_exits_cleanly_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env

    monkeypatch.setattr(watch_mod, "sync", lambda *args, **kwargs: _empty_result())
    monkeypatch.setattr(
        watch_mod, "_SLEEP", lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    watch(config, paths, interval=0.1)


def test_watch_integration_captures_transcript_append(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    from thirdeye.platforms.copilot.capture import iter_captured_records
    from thirdeye.platforms.copilot.identity import stored_session_id

    config, paths = copilot_env
    home = Path(paths["home"])
    events_path = _write_transcript(home, NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    cycle = {"count": 0}

    def append_then_stop(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write('{"type":"synthetic.append"}\n')
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "_SLEEP", append_then_stop)

    before = len(list(iter_captured_records(config, stored)))
    watch(config, paths, interval=0.1)
    after = len(list(iter_captured_records(config, stored)))

    assert after > before


def _unlink_database(database: Path) -> None:
    database.unlink()
    for suffix in ("-wal", "-shm"):
        database.with_name(database.name + suffix).unlink(missing_ok=True)


def test_changed_sessions_database_deletion_includes_prior_ids() -> None:
    before = _snapshot(
        sessions={NATIVE_SESSION_ID},
        database_sessions={NATIVE_SESSION_ID},
        database=((1, 2, 3, 4), None, None),
    )
    after = _snapshot(sessions=set(), database_sessions=set(), database=(None, None, None))
    assert _changed_sessions(before, after) == {NATIVE_SESSION_ID}


def test_watch_does_not_retry_deleted_database_session(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    database = _write_database(home, session_id=NATIVE_SESSION_ID)
    calls: list[str | None] = []
    cycle = {"count": 0}

    def tracking_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append(session_id)
        if session_id is None:
            return _empty_result()
        if not Path(paths["database"]).is_file():
            return _empty_result(errors=1)
        return _empty_result()

    def delete_then_poll(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            _unlink_database(database)
        elif cycle["count"] >= 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", tracking_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", delete_then_poll)

    watch(config, paths, interval=0.1)

    assert calls[0] is None
    assert calls.count(NATIVE_SESSION_ID) == 1


def test_watch_syncs_recreated_database_after_deletion(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    database = _write_database(home, session_id=NATIVE_SESSION_ID)
    calls: list[str | None] = []
    cycle = {"count": 0}

    def tracking_sync(cfg: Config, p: SourcePaths, *, session_id: str | None = None) -> SyncResult:
        calls.append(session_id)
        return _empty_result()

    def delete_then_recreate(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            _unlink_database(database)
        elif cycle["count"] == 2:
            _write_database(home, session_id=NATIVE_SESSION_ID)
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "sync", tracking_sync)
    monkeypatch.setattr(watch_mod, "_SLEEP", delete_then_recreate)

    watch(config, paths, interval=0.1)

    assert calls[0] is None
    assert calls.count(NATIVE_SESSION_ID) >= 2


def test_watch_restart_then_captures_later_append(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    from thirdeye.platforms.copilot.capture import iter_captured_records
    from thirdeye.platforms.copilot.identity import stored_session_id

    config, paths = copilot_env
    home = Path(paths["home"])
    events_path = _write_transcript(home, NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)

    monkeypatch.setattr(
        watch_mod, "_SLEEP", lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    watch(config, paths, interval=0.1)
    first = len(list(iter_captured_records(config, stored)))
    assert first > 0

    cycle = {"count": 0}

    def append_then_stop(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write('{"type":"synthetic.restart-append"}\n')
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "_SLEEP", append_then_stop)
    watch(config, paths, interval=0.1)
    second = len(list(iter_captured_records(config, stored)))
    assert second > first


def test_watch_integration_captures_late_database_rows_then_survives_deletion(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    from thirdeye.platforms.copilot.capture import iter_captured_records
    from thirdeye.platforms.copilot.identity import stored_session_id

    config, paths = copilot_env
    home = Path(paths["home"])
    database = _write_database(home, session_id=NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    cycle = {"count": 0}

    def late_row_then_delete(_interval: float) -> None:
        cycle["count"] += 1
        if cycle["count"] == 1:
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "INSERT INTO turns (id, session_id, turn_index, content, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (2, NATIVE_SESSION_ID, 1, "late-row", "2026-09-10T17:08:12.000Z"),
                )
                connection.commit()
            finally:
                connection.close()
        elif cycle["count"] == 2:
            _unlink_database(database)
        elif cycle["count"] >= 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod, "_SLEEP", late_row_then_delete)
    watch(config, paths, interval=0.1)

    payloads = [record["payload"] for record in iter_captured_records(config, stored)]
    assert any(
        isinstance(payload, dict) and payload.get("row", {}).get("content") == "late-row"
        for payload in payloads
    )
