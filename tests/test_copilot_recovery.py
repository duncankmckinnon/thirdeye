"""Crash recovery, journal replay, and stale-cursor contention for Copilot archive."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import thirdeye.platforms.copilot.archive as archive_mod
import thirdeye.platforms.copilot.state as state_mod
from thirdeye._compat import fsops
from thirdeye.config import Config
from thirdeye.meta import read_meta
from thirdeye.paths import meta_path
from thirdeye.platforms.copilot.archive import commit_batch, iter_captured_records, load_cursor
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.state import journal_path, read_json, state_path
from thirdeye.platforms.copilot.types import SourceBatch, SourcePaths, SourceRecord

NATIVE_ID = "session-a"


def _record(
    source_id: str,
    *,
    native_session_id: str = NATIVE_ID,
    source_kind: str = "transcript",
    payload: dict[str, Any] | None = None,
) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "native_session_id": native_session_id,
        "ts": "2026-09-10T17:08:24.000Z",
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": payload or {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _batch(
    paths: SourcePaths,
    records: list[SourceRecord],
    *,
    next_cursor: dict[str, Any],
    base_cursor: dict[str, Any] | None = None,
) -> SourceBatch:
    cursor = dict(next_cursor)
    if base_cursor is not None:
        cursor["base_cursor"] = base_cursor
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/proj",
        "records": records,
        "next_cursor": cursor,
        "diagnostics": [],
    }


@contextmanager
def fault_at(point: str) -> Iterator[None]:
    seen: list[str] = []

    def injector(name: str) -> None:
        seen.append(name)
        if name == point:
            raise RuntimeError(f"injected fault at {point}")

    archive_mod._fault_injector = injector
    state_mod._fault_injector = injector
    try:
        yield seen
    finally:
        archive_mod._fault_injector = None
        state_mod._fault_injector = None


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(root=tmp_path / "thirdeye")


@pytest.fixture
def paths(tmp_path) -> SourcePaths:
    home = tmp_path / "copilot-home"
    home.mkdir()
    return resolve_sources(home)


def _session_dir(config: Config, paths: SourcePaths):
    from thirdeye.paths import session_dir
    from thirdeye.platforms.copilot.constants import PLATFORM_NAME

    return session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, NATIVE_ID))


@pytest.mark.parametrize(
    "fault_point",
    [
        "after_journal",
        "after_append",
        "after_checkpoint",
        "after_journal_clear",
    ],
)
def test_crash_at_journal_boundary_recovers_on_next_commit(
    config: Config,
    paths: SourcePaths,
    fault_point: str,
) -> None:
    directory = _session_dir(config, paths)
    records = [_record("key/a/recover-1"), _record("key/a/recover-2")]

    with fault_at(fault_point):
        with pytest.raises(RuntimeError, match="injected fault"):
            commit_batch(
                config,
                paths,
                _batch(paths, records, next_cursor={"generation": 1}),
            )

    if fault_point == "after_journal":
        assert journal_path(directory).is_file()

    if fault_point in {"after_append", "after_checkpoint", "after_journal_clear"}:
        captured_before = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
        assert len(captured_before) == 2

    recovery = commit_batch(
        config,
        paths,
        _batch(
            paths,
            [_record("key/a/recover-3")],
            next_cursor={"generation": 2},
            base_cursor={"generation": 1},
        ),
    )

    assert recovery["records_written"] >= 1
    assert not journal_path(directory).exists()
    cursor = load_cursor(config, paths, NATIVE_ID)
    assert cursor == {"generation": 2}
    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
    assert {r["source_id"] for r in captured} >= {
        "key/a/recover-1",
        "key/a/recover-2",
        "key/a/recover-3",
    }


def test_recovery_after_journal_only_replays_uncommitted_records(
    config: Config,
    paths: SourcePaths,
) -> None:
    directory = _session_dir(config, paths)
    records = [_record("key/a/journal-only")]

    with fault_at("after_journal"):
        with pytest.raises(RuntimeError):
            commit_batch(config, paths, _batch(paths, records, next_cursor={"generation": 1}))

    assert journal_path(directory).is_file()
    result = commit_batch(
        config,
        paths,
        _batch(paths, [], next_cursor={"generation": 1}, base_cursor={}),
    )
    assert result["records_written"] == 1
    assert result["duplicate_records"] == 0
    assert (
        list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))[0]["source_id"]
        == "key/a/journal-only"
    )


def test_stale_cursor_rejects_competing_batch(config: Config, paths: SourcePaths) -> None:
    first = commit_batch(
        config,
        paths,
        _batch(paths, [_record("key/a/first")], next_cursor={"generation": 1}),
    )
    assert first["errors"] == 0

    stale = commit_batch(
        config,
        paths,
        _batch(
            paths,
            [_record("key/a/stale")],
            next_cursor={"generation": 99},
            base_cursor={},
        ),
    )
    assert stale == {
        "sessions": 1,
        "records_written": 0,
        "duplicate_records": 0,
        "pending": 1,
        "errors": 1,
    }
    assert load_cursor(config, paths, NATIVE_ID) == {"generation": 1}
    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
    assert {r["source_id"] for r in captured} == {"key/a/first"}

    state = read_json(state_path(_session_dir(config, paths)))
    assert state is not None
    assert any(item.get("kind") == "stale_cursor" for item in state["health"]["diagnostics"])


def test_two_competing_batches_second_wins_first_becomes_stale(
    config: Config,
    paths: SourcePaths,
) -> None:
    batch_a = _batch(
        paths,
        [_record("key/a/contender-a")],
        next_cursor={"generation": 1},
        base_cursor={},
    )
    batch_b = _batch(
        paths,
        [_record("key/a/contender-b")],
        next_cursor={"generation": 1},
        base_cursor={},
    )

    winner = commit_batch(config, paths, batch_b)
    assert winner["records_written"] == 1

    loser = commit_batch(config, paths, batch_a)
    assert loser["errors"] == 1
    assert loser["pending"] == 1
    assert loser["records_written"] == 0

    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
    assert {r["source_id"] for r in captured} == {"key/a/contender-b"}


def test_recovery_replay_is_idempotent_when_events_already_committed(
    config: Config,
    paths: SourcePaths,
) -> None:
    directory = _session_dir(config, paths)
    records = [_record("key/a/idempotent")]

    with fault_at("after_append"):
        with pytest.raises(RuntimeError):
            commit_batch(config, paths, _batch(paths, records, next_cursor={"generation": 1}))

    assert journal_path(directory).is_file()
    first_recovery = commit_batch(
        config,
        paths,
        _batch(paths, [], next_cursor={"generation": 1}, base_cursor={}),
    )
    assert first_recovery["records_written"] == 0
    assert first_recovery["duplicate_records"] == 1
    assert not journal_path(directory).exists()

    second_recovery = commit_batch(
        config,
        paths,
        _batch(
            paths,
            [_record("key/a/next")],
            next_cursor={"generation": 2},
            base_cursor={"generation": 1},
        ),
    )
    assert second_recovery["records_written"] == 1
    assert second_recovery["duplicate_records"] == 0


def _session_end_record(source_id: str = "key/a/close") -> SourceRecord:
    return _record(
        source_id,
        source_kind="hook",
        payload={"event": "sessionEnd", "context": {}},
    )


def _resume_record(source_id: str = "key/a/resume") -> SourceRecord:
    return _record(
        source_id,
        source_kind="hook",
        payload={"event": "resume", "context": {}},
    )


@pytest.mark.parametrize(
    "fault_point",
    [
        "after_journal",
        "after_append",
        "after_lifecycle",
        "after_checkpoint",
        "after_journal_clear",
    ],
)
def test_session_end_survives_crash_at_every_boundary(
    config: Config,
    paths: SourcePaths,
    fault_point: str,
) -> None:
    directory = _session_dir(config, paths)
    with fault_at(fault_point):
        with pytest.raises(RuntimeError, match="injected fault"):
            commit_batch(
                config,
                paths,
                _batch(paths, [_session_end_record()], next_cursor={"generation": 1}),
            )

    cursor = load_cursor(config, paths, NATIVE_ID)
    assert cursor == {"generation": 1}
    assert not journal_path(directory).exists()
    meta = read_meta(meta_path(directory))
    assert meta is not None
    assert meta.status == "closed"
    assert meta.ended_at is not None


@pytest.mark.parametrize(
    "fault_point",
    [
        "after_journal",
        "after_append",
        "after_lifecycle",
        "after_checkpoint",
        "after_journal_clear",
    ],
)
def test_resume_survives_crash_at_every_boundary(
    config: Config,
    paths: SourcePaths,
    fault_point: str,
) -> None:
    directory = _session_dir(config, paths)
    commit_batch(
        config,
        paths,
        _batch(paths, [_session_end_record()], next_cursor={"generation": 1}),
    )
    closed = read_meta(meta_path(directory))
    assert closed is not None
    assert closed.status == "closed"

    with fault_at(fault_point):
        with pytest.raises(RuntimeError, match="injected fault"):
            commit_batch(
                config,
                paths,
                _batch(
                    paths,
                    [_resume_record()],
                    next_cursor={"generation": 2},
                    base_cursor={"generation": 1},
                ),
            )

    cursor = load_cursor(config, paths, NATIVE_ID)
    assert cursor == {"generation": 2}
    assert not journal_path(directory).exists()
    meta = read_meta(meta_path(directory))
    assert meta is not None
    assert meta.status == "open"
    assert meta.ended_at is None


def test_load_cursor_completes_recovery_without_new_batch(
    config: Config,
    paths: SourcePaths,
) -> None:
    directory = _session_dir(config, paths)
    with fault_at("after_journal"):
        with pytest.raises(RuntimeError):
            commit_batch(
                config,
                paths,
                _batch(paths, [_record("key/a/journal-only")], next_cursor={"generation": 1}),
            )

    assert journal_path(directory).is_file()
    cursor = load_cursor(config, paths, NATIVE_ID)
    assert cursor == {"generation": 1}
    assert not journal_path(directory).exists()
    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
    assert [record["source_id"] for record in captured] == ["key/a/journal-only"]


def test_iter_captured_records_completes_recovery_without_new_batch(
    config: Config,
    paths: SourcePaths,
) -> None:
    directory = _session_dir(config, paths)
    with fault_at("after_journal"):
        with pytest.raises(RuntimeError):
            commit_batch(
                config,
                paths,
                _batch(paths, [_record("key/a/via-iter")], next_cursor={"generation": 1}),
            )

    assert journal_path(directory).is_file()
    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
    assert [record["source_id"] for record in captured] == ["key/a/via-iter"]
    assert load_cursor(config, paths, NATIVE_ID) == {"generation": 1}
    assert not journal_path(directory).exists()


def test_state_publication_fsyncs_directory_after_replace_and_unlink(
    config: Config,
    paths: SourcePaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    real_replace = fsops.replace
    real_unlink = fsops.unlink
    real_sync = fsops.sync_directory

    def replace(src: Path | str, dst: Path | str) -> None:
        order.append(f"replace:{Path(dst).name}")
        real_replace(src, dst)

    def unlink(path: Path, *, missing_ok: bool = False) -> None:
        order.append(f"unlink:{path.name}")
        real_unlink(path, missing_ok=missing_ok)

    def sync_directory(path: Path) -> None:
        order.append(f"dirsync:{path.name}")
        real_sync(path)

    monkeypatch.setattr(fsops, "replace", replace)
    monkeypatch.setattr(fsops, "unlink", unlink)
    monkeypatch.setattr(fsops, "sync_directory", sync_directory)

    commit_batch(
        config,
        paths,
        _batch(paths, [_record("key/a/durable")], next_cursor={"generation": 1}),
    )

    pairs = list(zip(order, order[1:], strict=False))
    assert ("replace:copilot.journal.json", f"dirsync:{_session_dir(config, paths).name}") in pairs
    assert ("replace:copilot.state.json", f"dirsync:{_session_dir(config, paths).name}") in pairs
    assert ("unlink:copilot.journal.json", f"dirsync:{_session_dir(config, paths).name}") in pairs


def test_crash_between_replace_and_directory_sync_still_publishes_state(
    config: Config,
    paths: SourcePaths,
) -> None:
    directory = _session_dir(config, paths)
    with fault_at("after_state_replace"):
        with pytest.raises(RuntimeError, match="injected fault"):
            commit_batch(
                config,
                paths,
                _batch(paths, [_record("key/a/dirsync")], next_cursor={"generation": 1}),
            )

    assert state_path(directory).is_file()
    cursor = load_cursor(config, paths, NATIVE_ID)
    assert cursor == {"generation": 1}
    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
    assert [record["source_id"] for record in captured] == ["key/a/dirsync"]
