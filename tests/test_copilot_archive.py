"""Behavioral tests for the Copilot evidence archive (commit, cursor, retrieval)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import thirdeye.platforms.copilot.archive as archive_mod
from thirdeye.config import Config
from thirdeye.meta import read_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.platforms.copilot.archive import commit_batch, iter_captured_records, load_cursor
from thirdeye.platforms.copilot.constants import PLATFORM_NAME, SOURCE_SCHEMA_VERSION
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.state import read_json, state_path
from thirdeye.platforms.copilot.types import SourceBatch, SourcePaths, SourceRecord
from thirdeye.reader import SessionReader

FIXTURES = Path(__file__).parent / "fixtures" / "copilot" / "v1-cases"
NATIVE_ID = "session-a"


def _record(
    source_id: str,
    *,
    source_kind: str = "transcript",
    native_session_id: str = NATIVE_ID,
    ts: str | None = "2026-09-10T17:08:24.000Z",
    observed_at: str = "2026-09-10T17:08:25.000Z",
    payload: dict[str, Any] | None = None,
) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "native_session_id": native_session_id,
        "ts": ts,
        "observed_at": observed_at,
        "payload": payload or {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _batch(
    paths: SourcePaths,
    records: list[SourceRecord],
    *,
    next_cursor: dict[str, Any] | None = None,
    base_cursor: dict[str, Any] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    cwd: str | None = "/proj",
) -> SourceBatch:
    cursor = dict(next_cursor or {"generation": 1})
    if base_cursor is not None:
        cursor["base_cursor"] = base_cursor
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": cwd,
        "records": records,
        "next_cursor": cursor,
        "diagnostics": diagnostics or [],
    }


@pytest.fixture
def copilot_home(tmp_path: Path) -> Path:
    home = tmp_path / "copilot-home"
    home.mkdir()
    return home


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(root=tmp_path / "thirdeye")


@pytest.fixture
def paths(copilot_home: Path) -> SourcePaths:
    return resolve_sources(copilot_home)


def _session_directory(config: Config, paths: SourcePaths) -> Path:
    stored = stored_session_id(paths, NATIVE_ID)
    return session_dir(config.root, PLATFORM_NAME, stored)


def test_commit_batch_writes_records_and_advances_cursor(
    config: Config, paths: SourcePaths
) -> None:
    records = [_record("key/a/event-1"), _record("key/a/event-2")]
    result = commit_batch(config, paths, _batch(paths, records, next_cursor={"offset": 2}))

    assert result == {
        "sessions": 1,
        "records_written": 2,
        "duplicate_records": 0,
        "pending": 0,
        "errors": 0,
    }
    assert load_cursor(config, paths, NATIVE_ID) == {"offset": 2}
    captured = list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))
    assert {r["source_id"] for r in captured} == {"key/a/event-1", "key/a/event-2"}


def test_load_cursor_returns_empty_for_unknown_session(config: Config, paths: SourcePaths) -> None:
    assert load_cursor(config, paths, NATIVE_ID) == {}


def test_load_cursor_returns_defensive_copy(config: Config, paths: SourcePaths) -> None:
    commit_batch(
        config, paths, _batch(paths, [_record("key/a/event-1")], next_cursor={"offset": 1})
    )
    cursor = load_cursor(config, paths, NATIVE_ID)
    cursor["offset"] = 999
    assert load_cursor(config, paths, NATIVE_ID) == {"offset": 1}


def test_iter_captured_records_empty_when_session_missing(config: Config) -> None:
    assert list(iter_captured_records(config, "copilot-missing-session")) == []


@pytest.mark.parametrize(
    ("source_kind", "event_type"),
    [
        ("transcript", "copilot_transcript"),
        ("database", "copilot_database"),
        ("hook", "copilot_hook"),
        ("metadata", "copilot_metadata"),
        ("unknown", "copilot_metadata"),
    ],
)
def test_event_type_mapping(
    config: Config,
    paths: SourcePaths,
    source_kind: str,
    event_type: str,
) -> None:
    record = _record("key/a/typed", source_kind=source_kind)
    commit_batch(config, paths, _batch(paths, [record]))
    events = list(SessionReader(_session_directory(config, paths)).iter_events())
    assert len(events) == 1
    assert events[0]["t"] == event_type
    envelope = events[0]["data"]
    assert envelope["schema_version"] == SOURCE_SCHEMA_VERSION
    assert envelope["source_record"]["source_id"] == "key/a/typed"


def test_original_source_timestamp_is_retained(config: Config, paths: SourcePaths) -> None:
    source_ts = "2026-09-10T17:08:24.000Z"
    commit_batch(
        config,
        paths,
        _batch(paths, [_record("key/a/ts", ts=source_ts, observed_at="2026-09-10T17:08:25.000Z")]),
    )
    event = SessionReader(_session_directory(config, paths)).get_event(0)
    assert event["ts"] == source_ts


def test_invalid_observed_at_is_rejected_even_when_source_ts_is_valid(
    config: Config, paths: SourcePaths
) -> None:
    result = commit_batch(
        config,
        paths,
        _batch(
            paths,
            [
                _record(
                    "key/a/bad-observed",
                    ts="2026-09-10T17:08:24.000Z",
                    observed_at="2026-09-10T17:08:99.000Z",
                )
            ],
            next_cursor={"offset": 1},
        ),
    )
    assert result["records_written"] == 0
    assert result["pending"] == 1
    assert result["errors"] == 1
    assert list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID))) == []


def test_missing_source_time_falls_back_to_observed_at(config: Config, paths: SourcePaths) -> None:
    observed = "2026-09-10T17:08:25.000Z"
    commit_batch(
        config, paths, _batch(paths, [_record("key/a/no-ts", ts=None, observed_at=observed)])
    )
    event = SessionReader(_session_directory(config, paths)).get_event(0)
    assert event["ts"] == observed

    state = read_json(state_path(_session_directory(config, paths)))
    assert state is not None
    kinds = {item["kind"] for item in state["health"]["diagnostics"]}
    assert "missing_source_time" in kinds


def test_duplicate_source_ids_are_skipped(config: Config, paths: SourcePaths) -> None:
    record = _record("key/a/dup")
    first = commit_batch(config, paths, _batch(paths, [record], next_cursor={"offset": 1}))
    second = commit_batch(
        config,
        paths,
        _batch(paths, [record], next_cursor={"offset": 2}, base_cursor={"offset": 1}),
    )
    assert first["records_written"] == 1
    assert second["records_written"] == 0
    assert second["duplicate_records"] == 1
    assert len(list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID)))) == 1


def test_batch_source_key_must_match_paths(config: Config, paths: SourcePaths) -> None:
    batch = _batch(paths, [_record("key/a/event-1")])
    batch["source_key"] = "0" * 64
    with pytest.raises(ValueError, match="source_key does not match"):
        commit_batch(config, paths, batch)


def test_batch_rejects_foreign_native_session_id(config: Config, paths: SourcePaths) -> None:
    foreign = _record("key/a/foreign", native_session_id="other-session")
    with pytest.raises(ValueError, match="another native session"):
        commit_batch(config, paths, _batch(paths, [foreign]))


def test_source_key_prefix_collision_is_rejected(
    config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive reuse must compare full source_key metadata, not just stored ID prefix."""
    import hashlib

    from thirdeye.platforms.copilot import identity as identity_mod
    from thirdeye.store import Store

    case = json.loads((FIXTURES / "source-key-prefix-collision.json").read_text(encoding="utf-8"))
    second_home = tmp_path / "home-b"
    second_home.mkdir()
    second_key = case["homes"][1]["source_key"]

    original_source_key = identity_mod._source_key

    def keyed_source_key(home: Path) -> str:
        if home.resolve() == second_home.resolve():
            return second_key
        return original_source_key(home)

    monkeypatch.setattr(identity_mod, "_source_key", keyed_source_key)

    stored_id = case["colliding_stored_session_id"]
    first_key = case["homes"][0]["source_key"]
    Store(config).open_session(
        stored_id,
        platform=PLATFORM_NAME,
        cwd="/proj",
        extra={
            "copilot": {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "source_key": first_key,
                "source_home": case["homes"][0]["home"],
                "native_session_id": NATIVE_ID,
            }
        },
    ).flush_and_detach()

    second_paths: SourcePaths = {
        "home": str(second_home.resolve()),
        "source_key": second_key,
        "session_root": str((second_home / "session-state").resolve()),
        "database": str((second_home / "session-store.db").resolve()),
    }
    assert hashlib.sha256(str(second_home.resolve()).encode()).hexdigest() != second_key
    with pytest.raises(ValueError, match="source-key prefix collision"):
        commit_batch(config, second_paths, _batch(second_paths, [_record("key/b/second")]))


def test_provisional_session_metadata_is_created(config: Config, paths: SourcePaths) -> None:
    commit_batch(config, paths, _batch(paths, [_record("key/a/hook", source_kind="hook")]))
    meta = read_meta(meta_path(_session_directory(config, paths)))
    assert meta is not None
    assert meta.platform == PLATFORM_NAME
    identity = meta.extra["copilot"]
    assert identity["source_key"] == paths["source_key"]
    assert identity["native_session_id"] == NATIVE_ID


def test_committed_cursor_strips_optimistic_base_marker(config: Config, paths: SourcePaths) -> None:
    commit_batch(config, paths, _batch(paths, [_record("key/a/one")], next_cursor={"offset": 1}))
    assert load_cursor(config, paths, NATIVE_ID) == {"offset": 1}

    commit_batch(
        config,
        paths,
        _batch(paths, [_record("key/a/two")], next_cursor={"offset": 2}, base_cursor={"offset": 1}),
    )
    cursor = load_cursor(config, paths, NATIVE_ID)
    assert cursor == {"offset": 2}
    state = read_json(state_path(_session_directory(config, paths)))
    assert state is not None
    assert "base_cursor" not in state["cursor"]
    assert "_base_cursor" not in state["cursor"]


def test_append_does_not_reopen_closed_session(config: Config, paths: SourcePaths) -> None:
    close_record = _record(
        "key/a/close",
        source_kind="hook",
        payload={"event": "sessionEnd", "context": {}},
    )
    commit_batch(config, paths, _batch(paths, [close_record], next_cursor={"generation": 1}))
    closed = read_meta(meta_path(_session_directory(config, paths)))
    assert closed is not None
    assert closed.status == "closed"
    ended_at = closed.ended_at
    assert ended_at is not None

    commit_batch(
        config,
        paths,
        _batch(
            paths,
            [_record("key/a/after-close")],
            next_cursor={"generation": 2},
            base_cursor={"generation": 1},
        ),
    )
    meta = read_meta(meta_path(_session_directory(config, paths)))
    assert meta is not None
    assert meta.status == "closed"
    assert meta.ended_at == ended_at


def test_invalid_timestamps_are_rejected_without_inventing_time(
    config: Config, paths: SourcePaths
) -> None:
    result = commit_batch(
        config,
        paths,
        _batch(
            paths,
            [_record("key/a/bad-ts", ts="not-a-timestamp", observed_at="also-invalid")],
            next_cursor={"offset": 1},
        ),
    )
    assert result["records_written"] == 0
    assert result["pending"] == 1
    assert result["errors"] == 1
    assert load_cursor(config, paths, NATIVE_ID) == {}
    assert list(iter_captured_records(config, stored_session_id(paths, NATIVE_ID))) == []

    state = read_json(state_path(_session_directory(config, paths)))
    assert state is not None
    assert any(item.get("kind") == "invalid_observed_at" for item in state["health"]["diagnostics"])


def test_session_end_closes_and_resume_reopens(config: Config, paths: SourcePaths) -> None:
    close_record = _record(
        "key/a/close",
        source_kind="hook",
        payload={"event": "sessionEnd", "context": {}},
    )
    commit_batch(config, paths, _batch(paths, [close_record]))
    meta = read_meta(meta_path(_session_directory(config, paths)))
    assert meta is not None
    assert meta.status == "closed"
    assert meta.ended_at is not None

    reopen_record = _record(
        "key/a/reopen",
        source_kind="hook",
        payload={"event": "resume", "context": {}},
    )
    commit_batch(
        config,
        paths,
        _batch(
            paths, [reopen_record], next_cursor={"generation": 2}, base_cursor={"generation": 1}
        ),
    )
    meta = read_meta(meta_path(_session_directory(config, paths)))
    assert meta is not None
    assert meta.status == "open"
    assert meta.ended_at is None


def test_last_lifecycle_event_in_batch_wins(config: Config, paths: SourcePaths) -> None:
    start = _record(
        "key/a/start",
        source_kind="hook",
        payload={"event": "sessionStart", "hook_payload": {}, "context": {}},
    )
    end = _record(
        "key/a/end",
        source_kind="hook",
        payload={"event": "sessionEnd", "hook_payload": {}, "context": {}},
    )
    commit_batch(config, paths, _batch(paths, [start, end]))
    meta = read_meta(meta_path(_session_directory(config, paths)))
    assert meta is not None
    assert meta.status == "closed"
    assert meta.ended_at is not None

    later_end = _record(
        "key/a/end-again",
        source_kind="hook",
        payload={"event": "sessionEnd", "hook_payload": {}, "context": {}},
    )
    resume = _record(
        "key/a/resume-after-end",
        source_kind="hook",
        payload={"event": "resume", "hook_payload": {}, "context": {}},
    )
    commit_batch(
        config,
        paths,
        _batch(
            paths,
            [later_end, resume],
            next_cursor={"generation": 2},
            base_cursor={"generation": 1},
        ),
    )
    meta = read_meta(meta_path(_session_directory(config, paths)))
    assert meta is not None
    assert meta.status == "open"
    assert meta.ended_at is None


def test_child_stop_does_not_close_session(config: Config, paths: SourcePaths) -> None:
    child_stop = _record(
        "key/a/child-stop",
        source_kind="hook",
        payload={
            "event": "sessionEnd",
            "hook_payload": {"agentId": "child-agent", "parentToolCallId": "tool-1"},
            "context": {},
        },
    )
    commit_batch(config, paths, _batch(paths, [child_stop]))
    meta = read_meta(meta_path(_session_directory(config, paths)))
    assert meta is not None
    assert meta.status == "open"
    assert meta.ended_at is None


def test_archive_module_has_no_usage_store_or_otel_imports() -> None:
    source = Path(archive_mod.__file__).read_text(encoding="utf-8")
    assert "UsageStore" not in source
    assert "usage_store" not in source
    assert "otel" not in source.lower()


def test_fixture_source_batch_can_be_committed(config: Config, copilot_home: Path) -> None:
    raw = json.loads((FIXTURES / "source-batch.json").read_text(encoding="utf-8"))
    paths = resolve_sources(copilot_home)
    raw["source_key"] = paths["source_key"]
    result = commit_batch(config, paths, raw)
    assert result["records_written"] == 1
    assert result["errors"] == 0
    record = next(iter_captured_records(config, stored_session_id(paths, raw["native_session_id"])))
    assert record["source_kind"] == "database"
    assert record["ts"] is None
