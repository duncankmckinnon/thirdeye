"""Migration, journal recovery, rebuild equivalence, and competing projection commits."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import thirdeye.platforms.copilot.projection_state as projection_state_mod
from thirdeye._compat import fsops
from thirdeye.config import Config
from thirdeye.paths import session_dir, usage_jsonl_path
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection_state import (
    empty_projection_state,
    projection_journal_path,
    projection_state_path,
    publish_projection_document,
    read_projection_document,
)
from thirdeye.platforms.copilot.projection_store import (
    commit_projection,
    load_projection_state,
    read_projected_turns,
    reset_projection_state,
)
from thirdeye.platforms.copilot.types import (
    PROJECTION_SCHEMA_VERSION,
    Projection,
    SourceBatch,
    SourcePaths,
    SourceRecord,
)
from thirdeye.usage.index import UsageIndex
from thirdeye.usage.read import iter_calls
from thirdeye.usage.types import UsageRow

NATIVE_ID = "session-migrate"
INTERACTION_ID = "interaction-migrate-1"


def _record(source_id: str, *, ts: str = "2026-09-10T17:08:24.000Z") -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": "transcript",
        "native_session_id": NATIVE_ID,
        "ts": ts,
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _batch(paths: SourcePaths, records: list[SourceRecord]) -> SourceBatch:
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/proj",
        "records": records,
        "next_cursor": {"generation": 1},
        "diagnostics": [],
    }


def _usage_row(*, call_id: str, input_tokens: int, session_id: str) -> UsageRow:
    return UsageRow(
        session_id=session_id,
        seq=0,
        call_id=call_id,
        ts="2026-09-10T17:08:30.000Z",
        platform=PLATFORM_NAME,
        provider_name="openai",
        response_model="gpt-5.6-luna",
        input_tokens=input_tokens,
        output_tokens=10,
    )


def _main_turn(*, source_ids: list[str]) -> dict[str, Any]:
    return {
        "turn_id": "copilot:turn:key:session:interaction-migrate-1",
        "start_ts": "2026-09-10T17:08:24.000Z",
        "end_ts": "2026-09-10T17:08:28.000Z",
        "input_message": "hello",
        "output_message": "done",
        "status": "completed",
        "llm_calls": [],
        "permission_requests": [],
        "subagents": [],
        "attributes": {"interaction_id": INTERACTION_ID},
        "source_ids": source_ids,
    }


def _normalized_event(event_id: str, *, source_ids: list[str]) -> dict[str, Any]:
    return {
        "id": event_id,
        "kind": "user_prompt",
        "classification": "main",
        "initiator": "user",
        "source_ids": source_ids,
        "attributes": {"interaction_id": INTERACTION_ID},
    }


def _projection(**overrides: Any) -> Projection:
    base: Projection = {
        "normalized_events": [],
        "turns": [],
        "usage_rows": [],
        "attributions": [],
        "pending": [],
        "diagnostics": [],
    }
    base.update(overrides)
    return base


@contextmanager
def fault_at(point: str) -> Iterator[list[str]]:
    seen: list[str] = []

    def injector(name: str) -> None:
        seen.append(name)
        if name == point:
            raise RuntimeError(f"injected fault at {point}")

    projection_state_mod._fault_injector = injector
    try:
        yield seen
    finally:
        projection_state_mod._fault_injector = None


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(root=tmp_path / "thirdeye")


@pytest.fixture
def paths(tmp_path: Path) -> SourcePaths:
    home = tmp_path / "copilot-home"
    home.mkdir()
    return resolve_sources(home)


def _stored(_config: Config, paths: SourcePaths) -> str:
    return stored_session_id(paths, NATIVE_ID)


def _directory(config: Config, stored: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored)


def _seed_v1_archive(config: Config, paths: SourcePaths) -> str:
    commit_batch(
        config,
        paths,
        _batch(
            paths, [_record("key/a/v1-one"), _record("key/a/v1-two", ts="2026-09-10T17:08:26.000Z")]
        ),
    )
    return _stored(config, paths)


def _sample_projection(stored: str) -> Projection:
    return _projection(
        normalized_events=[
            _normalized_event("evt-one", source_ids=["key/a/v1-one"]),
            _normalized_event("evt-two", source_ids=["key/a/v1-two"]),
        ],
        turns=[_main_turn(source_ids=["key/a/v1-one", "key/a/v1-two"])],
        usage_rows=[_usage_row(call_id="usage-src-1", input_tokens=111, session_id=stored)],
        attributions=[
            {
                "usage_source_id": "usage-src-1",
                "logical_call_id": "logical-usage-1",
                "stored_turn_id": "copilot:turn:key:session:interaction-migrate-1",
                "agent_id": None,
                "call_id": None,
                "status": "matched",
                "join_kind": "direct",
                "evidence": [],
            }
        ],
    )


def test_v1_archive_first_projection_commit_is_upgrade_safe(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_v1_archive(config, paths)
    directory = _directory(config, stored)

    assert not projection_state_path(directory).exists()
    counts = commit_projection(config, stored, _sample_projection(stored), empty_projection_state())

    assert counts["events"] == 2
    assert counts["turns"] == 1
    assert counts["usage"] == 1
    assert projection_state_path(directory).is_file()
    assert len(read_projected_turns(config, stored)) == 1


@pytest.mark.parametrize(
    "fault_point",
    [
        "after_projection_journal",
        "after_projection_state",
        "after_projection_journal_clear",
    ],
)
def test_projection_journal_crash_recovers_on_next_read(
    config: Config, paths: SourcePaths, fault_point: str
) -> None:
    stored = _seed_v1_archive(config, paths)
    directory = _directory(config, stored)
    projection = _sample_projection(stored)

    with fault_at(fault_point):
        with pytest.raises(RuntimeError, match="injected fault"):
            commit_projection(config, stored, projection, empty_projection_state())

    if fault_point == "after_projection_journal":
        assert projection_journal_path(directory).is_file()

    state = load_projection_state(config, stored)
    assert state["index_totals"]["events"] == 2
    assert not projection_journal_path(directory).exists()
    turns = read_projected_turns(config, stored)
    assert len(turns) == 1
    assert len(turns[0]["events"]) == 2


def test_orphan_journal_recovers_without_new_commit(config: Config, paths: SourcePaths) -> None:
    stored = _seed_v1_archive(config, paths)
    directory = _directory(config, stored)
    document = read_projection_document(directory)
    document["indexes"]["events"] = {
        "evt-orphan": _normalized_event("evt-orphan", source_ids=["key/a/v1-one"])
    }
    publish_projection_document(directory, document)

    with fault_at("after_projection_journal"):
        with pytest.raises(RuntimeError):
            publish_projection_document(directory, document)

    assert projection_journal_path(directory).is_file()
    recovered = read_projection_document(directory)
    assert "evt-orphan" in recovered["indexes"]["events"]
    assert not projection_journal_path(directory).exists()


def test_load_projection_state_recovers_missing_usage_sidecar(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_v1_archive(config, paths)
    commit_projection(config, stored, _sample_projection(stored), empty_projection_state())
    directory = _directory(config, stored)
    fsops.unlink(usage_jsonl_path(directory), missing_ok=True)
    assert not usage_jsonl_path(directory).exists()

    load_projection_state(config, stored)

    rows = list(iter_calls(directory))
    assert len(rows) == 1
    assert rows[0].input_tokens == 111


def _indexed_usage(config: Config, stored: str) -> list[tuple[str, int]]:
    index = UsageIndex(config.root)
    connection = index.connect()
    try:
        index.refresh(connection)
        rows = connection.execute(
            "SELECT call_id, gen_ai_usage_input_tokens FROM usage WHERE session_id = ? "
            "ORDER BY call_id",
            (stored,),
        ).fetchall()
    finally:
        connection.close()
    return [(str(call_id), int(tokens)) for call_id, tokens in rows]


def test_load_clears_stale_usage_index_after_sidecar_unlink_crash(
    config: Config, paths: SourcePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _seed_v1_archive(config, paths)
    commit_projection(config, stored, _sample_projection(stored), empty_projection_state())
    assert _indexed_usage(config, stored) == [("logical-usage-1", 111)]

    import thirdeye.platforms.copilot.projection_store as store_mod

    monkeypatch.setattr(store_mod, "_invalidate_usage_index", lambda *_args, **_kwargs: None)
    reset_projection_state(config, stored)
    directory = _directory(config, stored)
    assert not usage_jsonl_path(directory).exists()
    assert _indexed_usage(config, stored) == [("logical-usage-1", 111)]

    monkeypatch.undo()
    load_projection_state(config, stored)
    assert _indexed_usage(config, stored) == []
    assert list(iter_calls(directory)) == []


def test_rebuild_after_reset_matches_original_projection(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_v1_archive(config, paths)
    projection = _sample_projection(stored)
    commit_projection(config, stored, projection, empty_projection_state())

    before_turns = read_projected_turns(config, stored)
    before_state = load_projection_state(config, stored)

    reset_projection_state(config, stored)
    commit_projection(config, stored, projection, empty_projection_state())

    after_turns = read_projected_turns(config, stored)
    after_state = load_projection_state(config, stored)

    assert after_turns == before_turns
    assert after_state["index_totals"] == before_state["index_totals"]


def test_competing_projection_commits_merge_indexes(config: Config, paths: SourcePaths) -> None:
    stored = _seed_v1_archive(config, paths)
    start_signal = config.root.parent / "start-projection-writers"
    script = r"""
from pathlib import Path
import sys
import time

from thirdeye.config import Config
from thirdeye.platforms.copilot.projection_state import empty_projection_state
from thirdeye.platforms.copilot.projection_store import commit_projection
from thirdeye.usage.types import UsageRow

root = Path(sys.argv[1])
stored = sys.argv[2]
writer_id = sys.argv[3]
start_signal = Path(sys.argv[4])
while not start_signal.exists():
    time.sleep(0.001)

source_id = f"key/a/v1-{'one' if writer_id == 'a' else 'two'}"
event_id = f"evt-{writer_id}"
logical_id = f"logical-{writer_id}"
usage_source = f"usage-{writer_id}"
config = Config(root=root)
row = UsageRow(
    session_id=stored,
    seq=0,
    call_id=usage_source,
    ts="2026-09-10T17:08:30.000Z",
    platform="copilot",
    provider_name="openai",
    response_model="gpt-5.6-luna",
    input_tokens=50 if writer_id == "a" else 75,
    output_tokens=10,
)
commit_projection(
    config,
    stored,
    {
        "normalized_events": [
            {
                "id": event_id,
                "kind": "user_prompt",
                "classification": "main",
                "initiator": "user",
                "source_ids": [source_id],
                "attributes": {"interaction_id": "interaction-migrate-1"},
            }
        ],
        "turns": [],
        "usage_rows": [row],
        "attributions": [
            {
                "usage_source_id": usage_source,
                "logical_call_id": logical_id,
                "stored_turn_id": None,
                "agent_id": None,
                "call_id": None,
                "status": "pending",
                "join_kind": None,
                "evidence": [],
            }
        ],
        "pending": [],
        "diagnostics": [],
    },
    empty_projection_state(),
)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(config.root), stored, writer_id, str(start_signal)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        for writer_id in ("a", "b")
    ]
    start_signal.touch()
    deadline = time.monotonic() + 30
    failures: list[str] = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=max(0.1, deadline - time.monotonic()))
        if process.returncode != 0:
            failures.append(f"exit={process.returncode} stdout={stdout!r} stderr={stderr!r}")
    assert not failures, failures

    state = load_projection_state(config, stored)
    assert state["index_totals"]["events"] == 2
    assert state["index_totals"]["usage"] == 2

    document = json.loads(projection_state_path(_directory(config, stored)).read_text())
    assert set(document["indexes"]["events"]) == {"evt-a", "evt-b"}
    assert set(document["indexes"]["usage"]) == {"logical-a", "logical-b"}


def test_replay_same_event_id_replaces_without_duplicating(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_v1_archive(config, paths)
    first = _projection(
        normalized_events=[
            {
                **_normalized_event("evt-stable", source_ids=["key/a/v1-one"]),
                "attributes": {"interaction_id": INTERACTION_ID, "note": "first"},
            }
        ],
    )
    second = _projection(
        normalized_events=[
            {
                **_normalized_event("evt-stable", source_ids=["key/a/v1-one"]),
                "attributes": {"interaction_id": INTERACTION_ID, "note": "second"},
            }
        ],
    )

    commit_projection(config, stored, first, empty_projection_state())
    commit_projection(config, stored, second, empty_projection_state())

    document = json.loads(projection_state_path(_directory(config, stored)).read_text())
    events = document["indexes"]["events"]
    assert len(events) == 1
    assert events["evt-stable"]["attributes"]["note"] == "second"


def test_usage_sidecar_rewrite_after_state_publication_crash(
    config: Config, paths: SourcePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _seed_v1_archive(config, paths)
    directory = _directory(config, stored)
    writes: list[int] = []

    import thirdeye.platforms.copilot.projection_store as store_mod

    original_write_usage = store_mod._write_usage_index

    def counting_write_usage(path: Path, usage_index: dict[str, Any]) -> None:
        writes.append(len(usage_index))
        if len(writes) == 2:
            raise RuntimeError("crash while rewriting usage sidecar")
        original_write_usage(path, usage_index)

    monkeypatch.setattr(store_mod, "_write_usage_index", counting_write_usage)

    commit_projection(config, stored, _sample_projection(stored), empty_projection_state())

    with pytest.raises(RuntimeError, match="crash while rewriting usage sidecar"):
        commit_projection(
            config,
            stored,
            _projection(
                usage_rows=[_usage_row(call_id="usage-src-2", input_tokens=222, session_id=stored)],
                attributions=[
                    {
                        "usage_source_id": "usage-src-2",
                        "logical_call_id": "logical-usage-1",
                        "stored_turn_id": None,
                        "agent_id": None,
                        "call_id": None,
                        "status": "matched",
                        "join_kind": "direct",
                        "evidence": [],
                    }
                ],
            ),
            empty_projection_state(),
        )

    load_projection_state(config, stored)
    rows = list(iter_calls(directory))
    assert len(rows) == 1
    assert rows[0].input_tokens == 222


def test_stale_projection_schema_is_discarded_for_rebuild(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_v1_archive(config, paths)
    directory = _directory(config, stored)
    stale = {
        "schema_version": PROJECTION_SCHEMA_VERSION - 1,
        "state": {
            "projection_schema_version": PROJECTION_SCHEMA_VERSION - 1,
            "archive_source_ids": ["stale"],
            "semantic_state": {"open_interactions": {"old": {}}},
            "accounting_state": {"logical_calls": {}},
            "projection_revision": "stale",
        },
        "indexes": {"events": {"old": {"id": "old"}}},
    }
    projection_state_path(directory).write_text(json.dumps(stale), encoding="utf-8")

    state = load_projection_state(config, stored)
    assert state["archive_source_ids"] == []
    assert state["semantic_state"]["open_interactions"] == {}
    assert not projection_state_path(directory).exists()
    assert read_projected_turns(config, stored) == []


def test_stale_journal_schema_is_discarded_without_raising(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_v1_archive(config, paths)
    directory = _directory(config, stored)
    journal = {
        "schema_version": 0,
        "document": {
            "schema_version": 1,
            "state": {"projection_schema_version": 1},
            "indexes": {},
        },
    }
    projection_journal_path(directory).write_text(json.dumps(journal) + "\n", encoding="utf-8")

    state = load_projection_state(config, stored)
    assert state["projection_schema_version"] == PROJECTION_SCHEMA_VERSION
    assert not projection_journal_path(directory).exists()


def test_corrupt_journal_is_discarded_and_snapshot_used(config: Config, paths: SourcePaths) -> None:
    stored = _seed_v1_archive(config, paths)
    commit_projection(config, stored, _sample_projection(stored), empty_projection_state())
    directory = _directory(config, stored)
    projection_journal_path(directory).write_text("{not-json", encoding="utf-8")

    state = load_projection_state(config, stored)
    assert state["index_totals"]["events"] == 2
    assert not projection_journal_path(directory).exists()
    assert len(read_projected_turns(config, stored)) == 1
