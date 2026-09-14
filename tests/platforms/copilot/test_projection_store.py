"""Behavioral tests for Copilot V2 projection commit, load, and turn reads."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir, usage_db_path, usage_jsonl_path
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.constants import PLATFORM_NAME, SOURCE_SCHEMA_VERSION
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection_state import (
    empty_projection_state,
    projection_journal_path,
    projection_state_path,
)
from thirdeye.platforms.copilot.projection_store import (
    commit_projection,
    load_projection_state,
    read_projected_turns,
    reset_projection_state,
)
from thirdeye.platforms.copilot.types import Projection, SourceBatch, SourcePaths, SourceRecord
from thirdeye.reader import SessionReader
from thirdeye.usage.index import UsageIndex
from thirdeye.usage.read import iter_calls
from thirdeye.usage.types import UsageRow

NATIVE_ID = "session-proj"
INTERACTION_ID = "interaction-main-1"


def _record(
    source_id: str,
    *,
    native_session_id: str = NATIVE_ID,
    source_kind: str = "transcript",
    ts: str = "2026-09-10T17:08:24.000Z",
) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "native_session_id": native_session_id,
        "ts": ts,
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _batch(
    paths: SourcePaths,
    records: list[SourceRecord],
    *,
    next_cursor: dict[str, Any] | None = None,
) -> SourceBatch:
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/proj",
        "records": records,
        "next_cursor": dict(next_cursor or {"generation": 1}),
        "diagnostics": [],
    }


def _usage_row(
    *,
    call_id: str,
    input_tokens: int = 100,
    output_tokens: int = 10,
    session_id: str = "stored-session",
    seq: int = 0,
) -> UsageRow:
    return UsageRow(
        session_id=session_id,
        seq=seq,
        call_id=call_id,
        ts="2026-09-10T17:08:30.000Z",
        platform=PLATFORM_NAME,
        provider_name="openai",
        response_model="gpt-5.6-luna",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _main_turn(
    *,
    turn_id: str = "copilot:turn:key:session:interaction-main-1",
    interaction_id: str = INTERACTION_ID,
    start_ts: str = "2026-09-10T17:08:24.000Z",
    end_ts: str = "2026-09-10T17:08:28.000Z",
    source_ids: list[str] | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"interaction_id": interaction_id}
    if agent_id is not None:
        attributes["agent_id"] = agent_id
    turn: dict[str, Any] = {
        "turn_id": turn_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "input_message": "hello",
        "output_message": "done",
        "status": "completed",
        "llm_calls": [],
        "permission_requests": [],
        "subagents": [],
        "attributes": attributes,
    }
    if source_ids:
        turn["source_ids"] = source_ids
    return turn


def _normalized_event(
    event_id: str,
    *,
    interaction_id: str = INTERACTION_ID,
    source_ids: list[str],
    stored_turn_id: str | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"interaction_id": interaction_id}
    if stored_turn_id is not None:
        attributes["stored_turn_id"] = stored_turn_id
    return {
        "id": event_id,
        "kind": "user_prompt",
        "classification": "main",
        "initiator": "user",
        "source_ids": source_ids,
        "source_references": [
            {"source_id": source_id, "source_kind": "transcript", "role": "user_prompt"}
            for source_id in source_ids
        ],
        "attributes": attributes,
    }


def _attribution(
    *,
    logical_call_id: str,
    usage_source_id: str,
    status: str = "matched",
) -> dict[str, Any]:
    return {
        "usage_source_id": usage_source_id,
        "logical_call_id": logical_call_id,
        "stored_turn_id": "copilot:turn:key:session:interaction-main-1",
        "agent_id": None,
        "call_id": None,
        "status": status,
        "join_kind": "direct",
        "evidence": ["direct:usage_source_id"],
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


def _seed_archive(config: Config, paths: SourcePaths, records: list[SourceRecord]) -> str:
    commit_batch(config, paths, _batch(paths, records))
    return _stored(config, paths)


def test_commit_projection_persists_indexes_and_usage_sidecar(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(
        config,
        paths,
        [_record("key/a/event-1"), _record("key/a/event-2", ts="2026-09-10T17:08:26.000Z")],
    )
    directory = _directory(config, stored)

    projection = _projection(
        normalized_events=[
            _normalized_event("copilot:event:key/a/event-1", source_ids=["key/a/event-1"]),
            _normalized_event("copilot:event:key/a/event-2", source_ids=["key/a/event-2"]),
        ],
        turns=[_main_turn(source_ids=["key/a/event-1", "key/a/event-2"])],
        usage_rows=[_usage_row(call_id="src-usage-1", session_id=stored)],
        attributions=[_attribution(logical_call_id="logical-1", usage_source_id="src-usage-1")],
        pending=[{"id": "pending-1", "kind": "open_interaction", "message": "unfinished"}],
        diagnostics=[
            {
                "code": "missing_capability",
                "severity": "info",
                "message": "no direct join id",
                "source_ids": [],
                "details": {},
            }
        ],
    )
    next_state = empty_projection_state()
    next_state["archive_source_ids"] = ["key/a/event-1", "key/a/event-2"]

    counts = commit_projection(config, stored, projection, next_state)

    assert counts == {
        "events": 2,
        "turns": 1,
        "usage": 1,
        "attributions": 1,
        "pending": 1,
        "diagnostics": 1,
    }
    assert projection_state_path(directory).is_file()
    assert not projection_journal_path(directory).exists()

    usage_rows = list(iter_calls(directory))
    assert len(usage_rows) == 1
    assert usage_rows[0].call_id == "logical-1"

    state = load_projection_state(config, stored)
    assert state["archive_source_ids"] == ["key/a/event-1", "key/a/event-2"]
    assert state["index_totals"]["events"] == 2
    assert state["projection_revision"].startswith("sha256:")


def test_read_projected_turns_join_archived_store_events(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(
        config,
        paths,
        [_record("key/a/user"), _record("key/a/assistant", ts="2026-09-10T17:08:26.000Z")],
    )
    turn_id = "copilot:turn:key:session:interaction-main-1"

    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[
                _normalized_event("evt-user", source_ids=["key/a/user"]),
                _normalized_event("evt-assistant", source_ids=["key/a/assistant"]),
            ],
            turns=[_main_turn(turn_id=turn_id)],
        ),
        empty_projection_state(),
    )

    turns = read_projected_turns(config, stored)
    assert len(turns) == 1
    turn = turns[0]
    assert turn["turn_id"] == turn_id
    assert turn["session_id"] == stored
    assert turn["platform"] == PLATFORM_NAME
    assert turn["cwd"] == "/proj"
    assert [event["data"]["source_record"]["source_id"] for event in turn["events"]] == [
        "key/a/user",
        "key/a/assistant",
    ]
    assert turn["start_seq"] == 0
    assert turn["end_seq"] == 1


def test_child_agent_top_level_turns_are_excluded(config: Config, paths: SourcePaths) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/main")])
    main_turn = _main_turn()
    child_turn = _main_turn(
        turn_id="copilot:turn:key:session:child",
        interaction_id="child-interaction",
        agent_id="subagent-123",
    )

    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[_normalized_event("evt-main", source_ids=["key/a/main"])],
            turns=[main_turn, child_turn],
        ),
        empty_projection_state(),
    )

    turns = read_projected_turns(config, stored)
    assert [turn["turn_id"] for turn in turns] == [main_turn["turn_id"]]


def test_nested_child_archive_events_stay_on_main_turn(config: Config, paths: SourcePaths) -> None:
    stored = _seed_archive(
        config,
        paths,
        [_record("key/a/main"), _record("key/a/child", ts="2026-09-10T17:08:26.000Z")],
    )
    main_turn = _main_turn(source_ids=["key/a/main"])
    main_turn["subagents"] = [
        {
            "turn_id": "copilot:turn:key:session:child",
            "start_ts": "2026-09-10T17:08:26.000Z",
            "end_ts": "2026-09-10T17:08:27.000Z",
            "input_message": "explore",
            "output_message": "found",
            "status": "completed",
            "llm_calls": [],
            "permission_requests": [],
            "subagents": [],
            "attributes": {"interaction_id": INTERACTION_ID, "agent_id": "subagent-123"},
            "source_ids": ["key/a/child"],
        }
    ]
    child_event = _normalized_event("evt-child", source_ids=["key/a/child"])
    child_event["attributes"]["agent_id"] = "subagent-123"

    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[
                _normalized_event("evt-main", source_ids=["key/a/main"]),
                child_event,
            ],
            turns=[main_turn],
        ),
        empty_projection_state(),
    )

    turns = read_projected_turns(config, stored)
    assert len(turns) == 1
    assert [event["data"]["source_record"]["source_id"] for event in turns[0]["events"]] == [
        "key/a/main",
        "key/a/child",
    ]


def test_usage_revision_replaces_under_stable_logical_identity(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/usage")])

    first = _projection(
        usage_rows=[_usage_row(call_id="src-rev-1", input_tokens=100, session_id=stored)],
        attributions=[_attribution(logical_call_id="logical-1", usage_source_id="src-rev-1")],
    )
    commit_projection(config, stored, first, empty_projection_state())

    second = _projection(
        usage_rows=[_usage_row(call_id="src-rev-2", input_tokens=150, session_id=stored, seq=1)],
        attributions=[_attribution(logical_call_id="logical-1", usage_source_id="src-rev-2")],
    )
    counts = commit_projection(config, stored, second, empty_projection_state())

    assert counts["usage"] == 1
    directory = _directory(config, stored)
    rows = list(iter_calls(directory))
    assert len(rows) == 1
    assert rows[0].input_tokens == 150
    assert rows[0].call_id == "logical-1"

    sidecar_lines = usage_jsonl_path(directory).read_text(encoding="utf-8").splitlines()
    assert len(sidecar_lines) == 1


def test_next_state_is_authoritative_and_closes_interactions(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/one"), _record("key/a/two")])

    first_state = empty_projection_state()
    first_state["archive_source_ids"] = ["key/a/one"]
    first_state["semantic_state"] = {
        "open_interactions": {
            INTERACTION_ID: {
                "interaction_id": INTERACTION_ID,
                "agent_id": None,
                "stored_turn_id": "turn-open",
                "source_ids": ["key/a/one"],
                "last_event_source_id": "key/a/one",
                "start_ts": "2026-09-10T17:08:24.000Z",
                "pending_tool_call_ids": [],
            }
        }
    }
    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[_normalized_event("evt-1", source_ids=["key/a/one"])],
        ),
        first_state,
    )

    second_state = empty_projection_state()
    second_state["archive_source_ids"] = ["key/a/one", "key/a/two"]
    second_state["semantic_state"] = {"open_interactions": {}}
    second_state["accounting_state"] = {
        "logical_calls": {
            "logical-1": {
                "logical_call_id": "logical-1",
                "generation": "gen-a",
                "content_revision": "rev-a",
                "metrics_digest": "sha256:abc",
                "usage_source_id": "src-rev-1",
            }
        }
    }
    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[_normalized_event("evt-2", source_ids=["key/a/two"])],
        ),
        second_state,
    )

    replaced = load_projection_state(config, stored)
    assert replaced["archive_source_ids"] == ["key/a/one", "key/a/two"]
    assert replaced["semantic_state"]["open_interactions"] == {}
    assert "logical-1" in replaced["accounting_state"]["logical_calls"]

    document_events = json.loads(projection_state_path(_directory(config, stored)).read_text())[
        "indexes"
    ]["events"]
    assert set(document_events) == {"evt-1", "evt-2"}


def test_load_projection_state_returns_defensive_copy(config: Config, paths: SourcePaths) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/event")])
    state = empty_projection_state()
    state["archive_source_ids"] = ["key/a/event"]
    commit_projection(config, stored, _projection(), state)

    loaded = load_projection_state(config, stored)
    loaded["archive_source_ids"].append("mutated")
    again = load_projection_state(config, stored)
    assert again["archive_source_ids"] == ["key/a/event"]


def test_commit_projection_requires_usage_row_instances(config: Config, paths: SourcePaths) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/event")])
    bad: Projection = _projection(usage_rows=[{"call_id": "not-a-row"}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="UsageRow"):
        commit_projection(config, stored, bad, empty_projection_state())


def test_projection_reads_do_not_require_live_copilot_sources(
    config: Config, paths: SourcePaths, tmp_path: Path
) -> None:
    stored = _seed_archive(
        config,
        paths,
        [_record("key/a/user"), _record("key/a/assistant", ts="2026-09-10T17:08:26.000Z")],
    )
    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[
                _normalized_event("evt-user", source_ids=["key/a/user"]),
                _normalized_event("evt-assistant", source_ids=["key/a/assistant"]),
            ],
            turns=[_main_turn()],
        ),
        empty_projection_state(),
    )

    copilot_home = tmp_path / "copilot-home"
    session_root = copilot_home / "session-state" / NATIVE_ID
    session_root.mkdir(parents=True)
    (session_root / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (copilot_home / "session-store.db").write_bytes(b"sqlite")
    assert any(copilot_home.iterdir())
    shutil.rmtree(session_root)
    (copilot_home / "session-store.db").unlink()
    assert not session_root.exists()
    assert not (copilot_home / "session-store.db").exists()

    turns = read_projected_turns(config, stored)
    assert len(turns) == 1
    assert len(turns[0]["events"]) == 2
    archived = list(SessionReader(_directory(config, stored)).iter_events())
    assert len(archived) == 2


def test_unfinished_projection_pending_does_not_block_later_capture(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/first")])
    commit_projection(
        config,
        stored,
        _projection(
            pending=[{"id": "pending-open", "kind": "open_interaction", "message": "still open"}],
            diagnostics=[
                {
                    "code": "missing_capability",
                    "severity": "warning",
                    "message": "partial semantics",
                    "source_ids": ["key/a/first"],
                    "details": {},
                }
            ],
        ),
        empty_projection_state(),
    )

    commit_batch(
        config, paths, _batch(paths, [_record("key/a/second")], next_cursor={"generation": 2})
    )

    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[_normalized_event("evt-second", source_ids=["key/a/second"])],
        ),
        empty_projection_state(),
    )

    state = load_projection_state(config, stored)
    assert state["index_totals"]["events"] == 1
    assert state["index_totals"]["pending"] == 0
    assert state["index_totals"]["diagnostics"] == 0
    captured = {
        event["data"]["source_record"]["source_id"]
        for event in SessionReader(_directory(config, stored)).iter_events(
            types={"copilot_transcript"}
        )
    }
    assert captured == {"key/a/first", "key/a/second"}


def test_reset_projection_state_removes_only_derived_files(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/persist")])
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="usage-1", session_id=stored)],
            normalized_events=[_normalized_event("evt", source_ids=["key/a/persist"])],
            turns=[_main_turn(source_ids=["key/a/persist"])],
        ),
        empty_projection_state(),
    )
    directory = _directory(config, stored)

    reset_projection_state(config, stored)

    assert not projection_state_path(directory).exists()
    assert not usage_jsonl_path(directory).exists()
    archived = list(SessionReader(directory).iter_events())
    assert len(archived) == 1
    assert archived[0]["data"]["schema_version"] == SOURCE_SCHEMA_VERSION


def test_reset_projection_state_leaves_export_ledger_untouched(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/persist")])
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="usage-1", session_id=stored)],
            turns=[_main_turn(source_ids=["key/a/persist"])],
        ),
        empty_projection_state(),
    )
    directory = _directory(config, stored)
    ledger = directory / "copilot.export.ledger.json"
    payload = '{"schema_version":1,"jobs":[]}\n'
    ledger.write_text(payload, encoding="utf-8")

    reset_projection_state(config, stored)

    assert ledger.is_file()
    assert ledger.read_text(encoding="utf-8") == payload
    assert not projection_state_path(directory).exists()
    assert not usage_jsonl_path(directory).exists()


def _indexed_usage(config: Config, stored: str) -> list[tuple[str, int, int]]:
    index = UsageIndex(config.root)
    connection = index.connect()
    try:
        index.refresh(connection)
        rows = connection.execute(
            "SELECT call_id, gen_ai_usage_input_tokens, seq FROM usage WHERE session_id = ? "
            "ORDER BY call_id",
            (stored,),
        ).fetchall()
    finally:
        connection.close()
    return [(str(call_id), int(tokens), int(seq)) for call_id, tokens, seq in rows]


def test_incremental_turn_commit_retains_prior_partition_evidence(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(
        config,
        paths,
        [_record("key/a/e1"), _record("key/a/e2", ts="2026-09-10T17:08:26.000Z")],
    )
    turn = _main_turn(source_ids=["key/a/e1"])
    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[_normalized_event("evt-1", source_ids=["key/a/e1"])],
            turns=[turn],
        ),
        empty_projection_state(),
    )
    first = read_projected_turns(config, stored)
    assert [event["data"]["source_record"]["source_id"] for event in first[0]["events"]] == [
        "key/a/e1"
    ]

    later = _main_turn(source_ids=["key/a/e2"])
    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[_normalized_event("evt-2", source_ids=["key/a/e2"])],
            turns=[later],
        ),
        empty_projection_state(),
    )
    turns = read_projected_turns(config, stored)
    assert len(turns) == 1
    assert [event["data"]["source_record"]["source_id"] for event in turns[0]["events"]] == [
        "key/a/e1",
        "key/a/e2",
    ]
    assert turns[0]["start_seq"] == 0
    assert turns[0]["end_seq"] == 1


def test_derived_state_does_not_duplicate_raw_archive_payloads(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(
        config,
        paths,
        [_record("key/a/user"), _record("key/a/assistant", ts="2026-09-10T17:08:26.000Z")],
    )
    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[
                _normalized_event("evt-user", source_ids=["key/a/user"]),
                _normalized_event("evt-assistant", source_ids=["key/a/assistant"]),
            ],
            turns=[_main_turn(source_ids=["key/a/user", "key/a/assistant"])],
        ),
        empty_projection_state(),
    )
    document = json.loads(projection_state_path(_directory(config, stored)).read_text())
    dumped = json.dumps(document)
    assert "user.message" not in dumped
    assert document["indexes"]["turns"]
    turn_record = next(iter(document["indexes"]["turns"].values()))
    assert "events" not in turn_record
    assert turn_record["span"]["input_message"] == "hello"
    assert set(turn_record["source_ids"]) == {"key/a/user", "key/a/assistant"}


def test_unresolved_turn_evidence_does_not_invent_store_events(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/other")])
    commit_projection(
        config,
        stored,
        _projection(turns=[_main_turn()]),
        empty_projection_state(),
    )
    turns = read_projected_turns(config, stored)
    assert len(turns) == 1
    assert turns[0]["events"] == []
    assert turns[0]["start_seq"] is None
    assert turns[0]["end_seq"] is None


def test_in_progress_turns_are_omitted_from_projected_reads(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/open")])
    open_turn = _main_turn(source_ids=["key/a/open"])
    open_turn["status"] = "in_progress"
    commit_projection(
        config,
        stored,
        _projection(
            normalized_events=[_normalized_event("evt-open", source_ids=["key/a/open"])],
            turns=[open_turn],
        ),
        empty_projection_state(),
    )
    assert read_projected_turns(config, stored) == []


def test_commit_projection_rejects_unknown_session(config: Config) -> None:
    with pytest.raises(ValueError, match="unknown Copilot session"):
        commit_projection(config, "ghost-session", _projection(), empty_projection_state())
    assert not (config.root / "traces" / PLATFORM_NAME / "ghost-session").exists()


def test_read_and_reset_unknown_session_do_not_create_paths(config: Config) -> None:
    ghost = "ghost-session"
    state = load_projection_state(config, ghost)
    assert state["archive_source_ids"] == []
    assert state["semantic_state"]["open_interactions"] == {}
    assert read_projected_turns(config, ghost) == []
    reset_projection_state(config, ghost)
    assert not (config.root / "traces" / PLATFORM_NAME / ghost).exists()
    assert not usage_db_path(config.root).exists()


def test_commit_projection_rejects_usage_row_for_another_session(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/event")])
    with pytest.raises(ValueError, match="session_id"):
        commit_projection(
            config,
            stored,
            _projection(usage_rows=[_usage_row(call_id="logical-1", session_id="other")]),
            empty_projection_state(),
        )


def test_usage_seq_is_stamped_from_archived_revision(config: Config, paths: SourcePaths) -> None:
    stored = _seed_archive(
        config,
        paths,
        [
            _record("key/a/prompt"),
            _record("db-src-1", source_kind="database", ts="2026-09-10T17:08:30.000Z"),
        ],
    )
    next_state = empty_projection_state()
    next_state["accounting_state"] = {
        "logical_calls": {
            "logical-1": {
                "logical_call_id": "logical-1",
                "generation": "gen-a",
                "content_revision": "rev-a",
                "metrics_digest": "sha256:abc",
                "usage_source_id": "db-src-1",
            }
        }
    }
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="db-src-1", session_id=stored)],
            attributions=[_attribution(logical_call_id="logical-1", usage_source_id="db-src-1")],
        ),
        next_state,
    )
    rows = list(iter_calls(_directory(config, stored)))
    assert len(rows) == 1
    assert rows[0].seq == 1
    assert rows[0].call_id == "logical-1"
    assert _indexed_usage(config, stored) == [("logical-1", 100, 1)]


def test_usage_correction_is_visible_through_usage_index(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/usage")])
    first = _projection(
        usage_rows=[_usage_row(call_id="logical-1", input_tokens=100, session_id=stored)],
    )
    commit_projection(config, stored, first, empty_projection_state())
    assert _indexed_usage(config, stored) == [("logical-1", 100, 0)]

    second = _projection(
        usage_rows=[_usage_row(call_id="logical-1", input_tokens=150, session_id=stored, seq=1)],
    )
    commit_projection(config, stored, second, empty_projection_state())
    assert _indexed_usage(config, stored) == [("logical-1", 150, 0)]


def test_delayed_attribution_rekeys_source_id_without_double_counting(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/usage")])
    first_state = empty_projection_state()
    first_state["accounting_state"] = {
        "logical_calls": {
            "copilot:usage:logical-1": {
                "logical_call_id": "copilot:usage:logical-1",
                "generation": "gen-a",
                "content_revision": "rev-1",
                "metrics_digest": "sha256:abc",
                "usage_source_id": "db-src-rev1",
            }
        }
    }
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="db-src-rev1", input_tokens=100, session_id=stored)]
        ),
        first_state,
    )

    second_state = empty_projection_state()
    second_state["accounting_state"] = {
        "logical_calls": {
            "copilot:usage:logical-1": {
                "logical_call_id": "copilot:usage:logical-1",
                "generation": "gen-a",
                "content_revision": "rev-2",
                "metrics_digest": "sha256:def",
                "usage_source_id": "db-src-rev2",
            }
        }
    }
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="db-src-rev2", input_tokens=150, session_id=stored)],
            attributions=[
                _attribution(
                    logical_call_id="copilot:usage:logical-1", usage_source_id="db-src-rev2"
                )
            ],
        ),
        second_state,
    )

    rows = list(iter_calls(_directory(config, stored)))
    assert [(row.call_id, row.input_tokens) for row in rows] == [("copilot:usage:logical-1", 150)]
    assert _indexed_usage(config, stored) == [("copilot:usage:logical-1", 150, 0)]
    document = json.loads(projection_state_path(_directory(config, stored)).read_text())
    assert set(document["indexes"]["usage"]) == {"copilot:usage:logical-1"}


def test_identity_only_commit_rekeys_existing_usage_without_a_row(
    config: Config, paths: SourcePaths
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/usage")])
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="db-src-rev1", input_tokens=100, session_id=stored)]
        ),
        empty_projection_state(),
    )
    directory = _directory(config, stored)
    assert [(row.call_id, row.input_tokens) for row in iter_calls(directory)] == [
        ("db-src-rev1", 100)
    ]

    later_state = empty_projection_state()
    later_state["accounting_state"] = {
        "logical_calls": {
            "copilot:usage:logical-1": {
                "logical_call_id": "copilot:usage:logical-1",
                "generation": "gen-a",
                "content_revision": "rev-1",
                "metrics_digest": "sha256:abc",
                "usage_source_id": "db-src-rev1",
            }
        }
    }
    counts = commit_projection(
        config,
        stored,
        _projection(
            attributions=[
                _attribution(
                    logical_call_id="copilot:usage:logical-1", usage_source_id="db-src-rev1"
                )
            ]
        ),
        later_state,
    )

    assert counts["usage"] == 1
    rows = list(iter_calls(directory))
    assert [(row.call_id, row.input_tokens) for row in rows] == [("copilot:usage:logical-1", 100)]
    assert _indexed_usage(config, stored) == [("copilot:usage:logical-1", 100, 0)]
    document = json.loads(projection_state_path(directory).read_text())
    assert set(document["indexes"]["usage"]) == {"copilot:usage:logical-1"}


def test_load_repairs_stale_usage_index_when_sidecar_already_matches(
    config: Config, paths: SourcePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _seed_archive(config, paths, [_record("key/a/usage")])
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="logical-1", input_tokens=100, session_id=stored)]
        ),
        empty_projection_state(),
    )
    assert _indexed_usage(config, stored) == [("logical-1", 100, 0)]

    import thirdeye.platforms.copilot.projection_store as store_mod

    monkeypatch.setattr(store_mod, "_invalidate_usage_index", lambda *_args, **_kwargs: None)
    commit_projection(
        config,
        stored,
        _projection(
            usage_rows=[_usage_row(call_id="logical-1", input_tokens=150, session_id=stored)]
        ),
        empty_projection_state(),
    )

    directory = _directory(config, stored)
    sidecar_rows = [
        json.loads(line) for line in usage_jsonl_path(directory).read_text().splitlines() if line
    ]
    assert sidecar_rows[0]["gen_ai.usage.input_tokens"] == 150
    assert _indexed_usage(config, stored) == [("logical-1", 100, 0)]

    monkeypatch.undo()
    load_projection_state(config, stored)
    assert _indexed_usage(config, stored) == [("logical-1", 150, 0)]
