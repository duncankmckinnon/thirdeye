"""Behavioral tests for archive-only Copilot reconciliation."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config
from thirdeye.platforms.copilot.archive import commit_batch, iter_captured_records
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection_store import (
    load_projection_state,
    read_projected_turns,
)
from thirdeye.platforms.copilot.reconcile import reconcile_archive
from thirdeye.platforms.copilot.transcript import read_transcript
from thirdeye.platforms.copilot.types import SourceBatch, SourcePaths, SourceRecord
from thirdeye.reader import SessionReader

FIXTURES = Path(__file__).parent / "fixtures"
RECON = FIXTURES / "reconciliation-cases"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
SOURCE_KEY = "a" * 64
GENERATION = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
OBSERVED_AT = "2026-09-10T17:09:00.000Z"
RESULT_KEYS = (
    "events",
    "usage",
    "turns",
    "exports",
    "pending",
    "ambiguous",
    "conflicting",
    "errors",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _drain_cli_transcript(home: Path) -> list[SourceRecord]:
    session_dir = home / "session-state" / NATIVE_SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "events.jsonl", session_dir / "events.jsonl")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")
    paths = resolve_sources(home)
    cursor: dict[str, Any] = {}
    records: list[SourceRecord] = []
    while True:
        slice_ = read_transcript(paths, NATIVE_SESSION_ID, cursor)
        records.extend(slice_["records"])
        cursor = slice_["next_cursor"]
        if slice_["exhausted"]:
            break
    return [record for record in records if record["source_kind"] == "transcript"]


def _source_key(records: list[SourceRecord]) -> str:
    for record in records:
        if record["source_kind"] == "transcript":
            return record["source_id"].split("/", 1)[0]
    pytest.fail("expected at least one transcript record")


def _usage_record(
    row: dict[str, Any],
    *,
    content_revision: str,
    generation: str = GENERATION,
    source_key: str = SOURCE_KEY,
    observed_at: str = OBSERVED_AT,
) -> SourceRecord:
    primary_key = row["id"]
    source_id = (
        f"copilot-db:{source_key}:{NATIVE_SESSION_ID}:assistant_usage_events:"
        f"{primary_key}:{content_revision}"
    )
    return {
        "source_id": source_id,
        "source_kind": "database",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": row.get("created_at"),
        "observed_at": observed_at,
        "payload": {"table": "assistant_usage_events", "row": row},
        "locator": {
            "database": "/example/.copilot/session-store.db",
            "table": "assistant_usage_events",
            "primary_key": primary_key,
            "content_revision": content_revision,
            "generation": generation,
        },
    }


def _six_call_records(*, source_key: str = SOURCE_KEY) -> list[SourceRecord]:
    rows = _load_json(FIXTURES / "assistant-usage-events.json")
    revisions = {
        call["row_id"]: call["usage_source_id"].rsplit(":", 1)[-1]
        for call in _load_json(RECON / "observed-six-calls.json")["calls"]
    }
    return [
        _usage_record(row, content_revision=revisions[row["id"]], source_key=source_key)
        for row in rows
    ]


def _batch(paths: SourcePaths, records: list[SourceRecord]) -> SourceBatch:
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_SESSION_ID,
        "cwd": "/proj",
        "records": records,
        "next_cursor": {"generation": 1},
        "diagnostics": [],
    }


def _seed_archive(config: Config, paths: SourcePaths, records: list[SourceRecord]) -> str:
    commit_batch(config, paths, _batch(paths, records))
    return stored_session_id(paths, NATIVE_SESSION_ID)


def _substitute_source_key(value: str, source_key: str) -> str:
    return value.replace(SOURCE_KEY, source_key)


def _rewrite_source_key(value: Any, source_key: str) -> Any:
    if isinstance(value, str):
        return _substitute_source_key(value, source_key)
    if isinstance(value, list):
        return [_rewrite_source_key(item, source_key) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_source_key(item, source_key) for key, item in value.items()}
    return value


def _append_archive(
    config: Config, paths: SourcePaths, records: list[SourceRecord], *, generation: int
) -> None:
    commit_batch(
        config,
        paths,
        {
            **_batch(paths, records),
            "next_cursor": {"generation": generation},
        },
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(root=tmp_path / "thirdeye")


@pytest.fixture
def copilot_home(tmp_path: Path) -> Path:
    home = tmp_path / "copilot-home"
    home.mkdir()
    return home


@pytest.fixture
def paths(copilot_home: Path) -> SourcePaths:
    return resolve_sources(copilot_home)


@pytest.fixture
def cli_transcript_records(copilot_home: Path) -> list[SourceRecord]:
    return _drain_cli_transcript(copilot_home)


def _full_corpus_records(cli_transcript_records: list[SourceRecord]) -> list[SourceRecord]:
    source_key = _source_key(cli_transcript_records)
    return cli_transcript_records + _six_call_records(source_key=source_key)


def _seed_full_corpus(
    config: Config, paths: SourcePaths, cli_transcript_records: list[SourceRecord]
) -> str:
    return _seed_archive(config, paths, _full_corpus_records(cli_transcript_records))


def test_reconcile_archive_projects_six_call_corpus(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)

    result = reconcile_archive(config, stored)

    assert set(result.keys()) == set(RESULT_KEYS)
    assert result["exports"] == 0
    assert result["errors"] == 0
    assert result["usage"] == 6
    assert result["turns"] == 2
    assert result["events"] > 0
    assert result["ambiguous"] == 0
    assert result["conflicting"] == 0
    turns = read_projected_turns(config, stored)
    assert len(turns) == 2
    state = load_projection_state(config, stored)
    assert len(state["archive_source_ids"]) == len(list(iter_captured_records(config, stored)))


def test_reconcile_archive_is_idempotent(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)

    first = reconcile_archive(config, stored)
    second = reconcile_archive(config, stored)

    assert first == second
    assert read_projected_turns(config, stored) == read_projected_turns(config, stored)


def test_rebuild_is_idempotent_and_matches_initial_reconcile(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    initial = reconcile_archive(config, stored)
    first_rebuild = reconcile_archive(config, stored, rebuild=True)
    second_rebuild = reconcile_archive(config, stored, rebuild=True)

    assert first_rebuild == second_rebuild
    assert first_rebuild["events"] == initial["events"]
    assert first_rebuild["usage"] == initial["usage"]
    assert first_rebuild["turns"] == initial["turns"]
    assert first_rebuild["exports"] == 0


def test_incremental_reconcile_matches_full_replay(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
) -> None:
    source_key = _source_key(cli_transcript_records)
    usage = _six_call_records(source_key=source_key)
    partial = cli_transcript_records[: len(cli_transcript_records) // 2]
    remainder = cli_transcript_records[len(cli_transcript_records) // 2 :]

    stored_incremental = _seed_archive(config, paths, partial)
    reconcile_archive(config, stored_incremental)
    _append_archive(config, paths, remainder + usage, generation=2)
    incremental = reconcile_archive(config, stored_incremental)
    incremental_turns = read_projected_turns(config, stored_incremental)

    stored_full = _seed_full_corpus(config, paths, cli_transcript_records)
    full = reconcile_archive(config, stored_full)
    full_turns = read_projected_turns(config, stored_full)

    assert incremental["turns"] == full["turns"]
    assert incremental["usage"] == full["usage"]
    assert incremental["events"] == full["events"]
    assert [turn["turn_id"] for turn in incremental_turns] == [
        turn["turn_id"] for turn in full_turns
    ]


def test_reconcile_default_does_not_export(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    imports: list[str] = []
    original = __import__

    def tracking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        imports.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", tracking_import)

    result = reconcile_archive(config, stored)

    assert result["exports"] == 0
    assert not any("export" in item for item in imports)


def test_export_failure_does_not_roll_back_projection(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)

    def boom(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("export unavailable")

    monkeypatch.setattr(
        "thirdeye.platforms.copilot.reconcile.queue_exports",
        boom,
    )

    result = reconcile_archive(config, stored, export=True)

    assert result["turns"] == 2
    assert result["usage"] == 6
    assert result["exports"] == 0
    assert result["errors"] == 1
    assert len(read_projected_turns(config, stored)) == 2


def test_projection_failure_preserves_prior_derived_state(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    baseline = reconcile_archive(config, stored)
    prior_turns = read_projected_turns(config, stored)

    def explode(*_args: Any, **_kwargs: Any) -> tuple[Any, dict[str, Any]]:
        raise RuntimeError("projection failed")

    monkeypatch.setattr("thirdeye.platforms.copilot.reconcile.build_projection", explode)

    result = reconcile_archive(config, stored)

    assert result["errors"] == 1
    assert result["events"] == baseline["events"]
    assert result["usage"] == baseline["usage"]
    assert result["turns"] == baseline["turns"]
    assert read_projected_turns(config, stored) == prior_turns


def test_build_failure_during_rebuild_preserves_prior_projection(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    reconcile_archive(config, stored)
    prior_turns = read_projected_turns(config, stored)

    def explode(*_args: Any, **_kwargs: Any) -> tuple[Any, dict[str, Any]]:
        raise RuntimeError("projection failed")

    monkeypatch.setattr("thirdeye.platforms.copilot.reconcile.build_projection", explode)

    result = reconcile_archive(config, stored, rebuild=True)

    assert result["errors"] == 1
    assert read_projected_turns(config, stored) == prior_turns


def test_reconcile_unknown_session_reports_error_without_creating_paths(
    config: Config,
) -> None:
    ghost = "copilot-ghost-session"

    result = reconcile_archive(config, ghost)

    assert result["errors"] == 1
    assert result["exports"] == 0
    assert all(result[key] == 0 for key in RESULT_KEYS if key not in {"errors", "exports"})
    assert not (config.root / "traces" / "copilot" / ghost).exists()


def test_reconcile_works_without_live_copilot_source_files(
    config: Config,
    copilot_home: Path,
    cli_transcript_records: list[SourceRecord],
) -> None:
    paths = resolve_sources(copilot_home)
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    shutil.rmtree(copilot_home)

    result = reconcile_archive(config, stored, rebuild=True)

    assert result["errors"] == 0
    assert result["turns"] == 2
    assert result["usage"] == 6
    assert len(list(iter_captured_records(config, stored))) > 0


def test_rebuild_preserves_immutable_archive_records(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    before = {
        event["data"]["source_record"]["source_id"]
        for event in SessionReader(
            config.root / "traces" / "copilot" / stored
        ).iter_events()
    }

    reconcile_archive(config, stored, rebuild=True)

    after = {
        event["data"]["source_record"]["source_id"]
        for event in SessionReader(
            config.root / "traces" / "copilot" / stored
        ).iter_events()
    }
    assert before == after


def test_late_row_archive_reconcile_resolves_after_full_replay(
    config: Config,
    paths: SourcePaths,
) -> None:
    source_key = paths["source_key"]
    case = _load_json(RECON / "cases.json")["late_row"]
    partial = _rewrite_source_key(case["input_records"], source_key)
    final_answer = _rewrite_source_key(
        _load_json(RECON / "cases.json")["ambiguous"]["input_records"][1],
        source_key,
    )
    turn_end = copy.deepcopy(final_answer)
    turn_end.update(
        {
            "source_id": (
                f"{source_key}/{NATIVE_SESSION_ID}/4a386a37-ca7e-4ebc-a746-cdda20f2a4bb"
            ),
            "payload": {
                "type": "assistant.turn_end",
                "data": {"turnId": "1"},
                "id": "4a386a37-ca7e-4ebc-a746-cdda20f2a4bb",
                "timestamp": "2026-09-10T17:08:25.626Z",
                "parentId": "33cc6465-29e1-4a04-8bdb-00241474b4d2",
                "schema_version": 1,
            },
        }
    )
    full = partial + [final_answer, turn_end]

    stored = _seed_archive(config, paths, partial)
    pending = reconcile_archive(config, stored)
    assert pending["errors"] == 0
    assert pending["usage"] == 1
    assert pending["pending"] >= 1

    _append_archive(config, paths, [final_answer, turn_end], generation=2)
    resolved = reconcile_archive(config, stored)

    assert resolved["errors"] == 0
    assert resolved["usage"] == 1
    assert resolved["pending"] < pending["pending"]
    state = load_projection_state(config, stored)
    assert len(state["archive_source_ids"]) == len(full)
