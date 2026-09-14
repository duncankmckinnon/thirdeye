"""Behavioral tests for archive-only Copilot reconciliation."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.copilot import projection_store
from thirdeye.platforms.copilot.archive import commit_batch, iter_captured_records
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection import build_projection as _real_build_projection
from thirdeye.platforms.copilot.projection_state import projection_state_path
from thirdeye.platforms.copilot.projection_store import (
    ProjectionConflictError,
    commit_projection,
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
FULL_NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd7"
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


def _directory(config: Config, stored: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored)


def _document(config: Config, stored: str) -> dict[str, Any]:
    return json.loads(projection_state_path(_directory(config, stored)).read_text())


def _normalized_index_values(
    config: Config, stored: str, native_session_id: str, index_name: str
) -> list[Any]:
    index = _document(config, stored)["indexes"][index_name]
    normalized = [_rewrite_native_id(item, native_session_id) for item in index.values()]
    return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))


def _drain_cli_transcript(
    home: Path, *, native_session_id: str = NATIVE_SESSION_ID
) -> list[SourceRecord]:
    session_dir = home / "session-state" / native_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "events.jsonl", session_dir / "events.jsonl")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")
    paths = resolve_sources(home)
    cursor: dict[str, Any] = {}
    records: list[SourceRecord] = []
    while True:
        slice_ = read_transcript(paths, native_session_id, cursor)
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
    native_session_id: str = NATIVE_SESSION_ID,
    observed_at: str = OBSERVED_AT,
) -> SourceRecord:
    primary_key = row["id"]
    source_id = (
        f"copilot-db:{source_key}:{native_session_id}:assistant_usage_events:"
        f"{primary_key}:{content_revision}"
    )
    return {
        "source_id": source_id,
        "source_kind": "database",
        "native_session_id": native_session_id,
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


def _six_call_records(
    *, source_key: str = SOURCE_KEY, native_session_id: str = NATIVE_SESSION_ID
) -> list[SourceRecord]:
    rows = _load_json(FIXTURES / "assistant-usage-events.json")
    revisions = {
        call["row_id"]: call["usage_source_id"].rsplit(":", 1)[-1]
        for call in _load_json(RECON / "observed-six-calls.json")["calls"]
    }
    return [
        _usage_record(
            row,
            content_revision=revisions[row["id"]],
            source_key=source_key,
            native_session_id=native_session_id,
        )
        for row in rows
    ]


def _batch(
    paths: SourcePaths,
    records: list[SourceRecord],
    *,
    native_session_id: str = NATIVE_SESSION_ID,
) -> SourceBatch:
    return {
        "source_key": paths["source_key"],
        "native_session_id": native_session_id,
        "cwd": "/proj",
        "records": records,
        "next_cursor": {"generation": 1},
        "diagnostics": [],
    }


def _seed_archive(
    config: Config,
    paths: SourcePaths,
    records: list[SourceRecord],
    *,
    native_session_id: str = NATIVE_SESSION_ID,
) -> str:
    commit_batch(config, paths, _batch(paths, records, native_session_id=native_session_id))
    return stored_session_id(paths, native_session_id)


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


_VOLATILE_KEYS = frozenset({"observed_at", "locator"})


def _rewrite_native_id(value: Any, native_session_id: str, placeholder: str = "NATIVE") -> Any:
    """Normalize identity fields that legitimately differ between two

    independently seeded stored sessions built from the same fixture content:
    the native session ID baked into IDs/paths, each session's own wall-clock
    capture timestamp, and each archived copy's own filesystem locator (byte
    offsets/inode generation are per-file-instance, not semantic content).
    """
    if isinstance(value, str):
        for identity in {native_session_id, NATIVE_SESSION_ID, FULL_NATIVE_SESSION_ID}:
            value = value.replace(identity, placeholder)
        return value
    if isinstance(value, list):
        return [_rewrite_native_id(item, native_session_id, placeholder) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_native_id(item, native_session_id, placeholder)
            for key, item in value.items()
            if key not in _VOLATILE_KEYS
        }
    return value


def _append_archive(
    config: Config,
    paths: SourcePaths,
    records: list[SourceRecord],
    *,
    generation: int,
    native_session_id: str = NATIVE_SESSION_ID,
) -> None:
    commit_batch(
        config,
        paths,
        {
            **_batch(paths, records, native_session_id=native_session_id),
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


def _full_corpus_records(
    cli_transcript_records: list[SourceRecord], *, native_session_id: str = NATIVE_SESSION_ID
) -> list[SourceRecord]:
    source_key = _source_key(cli_transcript_records)
    return cli_transcript_records + _six_call_records(
        source_key=source_key, native_session_id=native_session_id
    )


def _seed_full_corpus(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    *,
    native_session_id: str = NATIVE_SESSION_ID,
) -> str:
    return _seed_archive(
        config,
        paths,
        _full_corpus_records(cli_transcript_records, native_session_id=native_session_id),
        native_session_id=native_session_id,
    )


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
    first_turns = read_projected_turns(config, stored)
    first_state = load_projection_state(config, stored)
    first_document = _document(config, stored)

    second = reconcile_archive(config, stored)
    second_turns = read_projected_turns(config, stored)
    second_state = load_projection_state(config, stored)
    second_document = _document(config, stored)

    assert first == second
    assert first_turns == second_turns
    # commit_sequence is a storage-owned counter that advances on every
    # successful commit_projection call, whether or not its content changed
    # -- it must not be idempotent, unlike everything else in builder state.
    assert second_state["commit_sequence"] == first_state["commit_sequence"] + 1
    assert {k: v for k, v in first_state.items() if k != "commit_sequence"} == {
        k: v for k, v in second_state.items() if k != "commit_sequence"
    }
    for turn in second_turns:
        seqs = [event.get("seq") for event in turn.get("events") or []]
        assert len(seqs) == len(set(seqs)), "re-reconciling must not duplicate turn events"
    # Full raw indexes (including each turn's nested accounting_calls, which
    # read_projected_turns intentionally does not surface) must be byte-for-
    # byte identical across the two re-derivations: re-running reconcile
    # must not accumulate duplicate accounting calls or usage/attribution
    # entries inside any index.
    normalized_first = {k: v for k, v in first_document["state"].items() if k != "commit_sequence"}
    normalized_second = {
        k: v for k, v in second_document["state"].items() if k != "commit_sequence"
    }
    assert normalized_first == normalized_second
    assert first_document["indexes"] == second_document["indexes"]
    for record in second_document["indexes"]["turns"].values():
        calls = record["span"].get("accounting_calls", [])
        call_ids = [call["accounting_id"] for call in calls]
        assert len(call_ids) == len(set(call_ids))


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
    copilot_home: Path,
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

    # A genuinely independent stored session -- same source home, but a
    # different native session ID -- committed in one shot from the same
    # fixture content.  Reusing the incremental session's own ID here would
    # make "full" just another reconcile of the archive the incremental case
    # already fully populated, which proves nothing about replay equivalence.
    full_transcript = _drain_cli_transcript(copilot_home, native_session_id=FULL_NATIVE_SESSION_ID)
    stored_full = _seed_full_corpus(
        config, paths, full_transcript, native_session_id=FULL_NATIVE_SESSION_ID
    )
    full = reconcile_archive(config, stored_full)
    full_turns = read_projected_turns(config, stored_full)
    incremental_document = _document(config, stored_incremental)
    full_document = _document(config, stored_full)

    assert incremental == full
    normalized_incremental = _rewrite_native_id(incremental_turns, NATIVE_SESSION_ID)
    normalized_full = _rewrite_native_id(full_turns, FULL_NATIVE_SESSION_ID)
    assert normalized_incremental == normalized_full

    # Builder state must agree independently of the storage-owned commit
    # counter and identity-dependent document digest.
    incremental_state = copy.deepcopy(incremental_document["state"])
    full_state = copy.deepcopy(full_document["state"])
    for state in (incremental_state, full_state):
        state.pop("commit_sequence", None)
        state.pop("projection_revision", None)
    assert _rewrite_native_id(incremental_state, NATIVE_SESSION_ID) == _rewrite_native_id(
        full_state, FULL_NATIVE_SESSION_ID
    )

    # Index keys can fall back to a content digest computed over pre-
    # normalization payloads (see _index_key), so two sessions built from the
    # same content under different native IDs are not guaranteed to share
    # digest-fallback keys even though their *content* is equivalent.
    # Comparing normalized values as an order-independent multiset avoids
    # that false negative while still catching a real divergence (a usage
    # row, attribution, pending item, or diagnostic present under one path
    # and not the other, or duplicated under either).
    # Comparing every index's normalized values includes raw stored turns and
    # their nested accounting_calls, plus events and the usage identity map.
    for name in incremental_document["indexes"]:
        incremental_values = _normalized_index_values(
            config, stored_incremental, NATIVE_SESSION_ID, name
        )
        full_values = _normalized_index_values(config, stored_full, FULL_NATIVE_SESSION_ID, name)
        assert incremental_values == full_values, f"{name} index diverged between replay paths"


def test_reconcile_default_does_not_export(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _seed_full_corpus(config, paths, cli_transcript_records)

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("reconcile_archive must not import export assembly by default")

    # Patch the name bound inside reconcile.py itself: `import_module` there
    # is `from importlib import import_module`, a local reference that a
    # patch on `importlib.import_module` would not intercept.
    monkeypatch.setattr("thirdeye.platforms.copilot.reconcile.import_module", explode)

    result = reconcile_archive(config, stored)

    assert result["exports"] == 0
    assert result["errors"] == 0


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


def test_commit_failure_during_rebuild_preserves_prior_projection(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild that fails inside commit_projection (after build succeeds)

    must not lose the previous projection.  This is the atomicity gap a
    delete-then-commit rebuild would have: this test forces the failure to
    happen after a valid projection has already been built, exercising the
    commit step itself rather than the build step.
    """
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    baseline = reconcile_archive(config, stored)
    prior_turns = read_projected_turns(config, stored)
    prior_state = load_projection_state(config, stored)

    def build_with_invalid_usage_row(
        records: list[SourceRecord], prior_state: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        projection, next_state = _real_build_projection(records, prior_state)
        broken = dict(projection)
        broken["usage_rows"] = [*projection["usage_rows"], {"not": "a UsageRow instance"}]
        return broken, next_state

    monkeypatch.setattr(
        "thirdeye.platforms.copilot.reconcile.build_projection", build_with_invalid_usage_row
    )

    result = reconcile_archive(config, stored, rebuild=True)

    assert result["errors"] == baseline["errors"] + 1
    assert result["turns"] == baseline["turns"]
    assert result["usage"] == baseline["usage"]
    assert read_projected_turns(config, stored) == prior_turns
    assert load_projection_state(config, stored) == prior_state


def test_commit_projection_rejects_a_stale_base_commit_sequence(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
) -> None:
    """A commit built from state read before a concurrent writer committed

    must be refused rather than silently overwriting that newer commit.
    This exercises commit_projection's optimistic-concurrency check
    directly: build a projection from state observed at commit_sequence 1,
    let another writer advance the stored session to commit_sequence 2, then
    attempt to commit the stale one with its now-outdated base sequence.
    """
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    reconcile_archive(config, stored)
    stale_state = load_projection_state(config, stored)
    assert stale_state["commit_sequence"] == 1
    records = list(iter_captured_records(config, stored))
    stale_projection, stale_next_state = _real_build_projection(records, stale_state)
    stale_next_state["archive_source_ids"] = [record["source_id"] for record in records]

    reconcile_archive(config, stored, rebuild=True)
    newer_turns = read_projected_turns(config, stored)
    newer_state = load_projection_state(config, stored)
    assert newer_state["commit_sequence"] == 2

    with pytest.raises(ProjectionConflictError):
        commit_projection(
            config,
            stored,
            stale_projection,
            stale_next_state,
            base_commit_sequence=stale_state["commit_sequence"],
        )

    assert read_projected_turns(config, stored) == newer_turns
    assert load_projection_state(config, stored) == newer_state


def test_reconcile_reports_error_when_projection_advances_concurrently(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reconcile_archive itself must surface a concurrent-writer conflict

    as an error rather than corrupt state, even though its own load, build,
    and commit are not one held lock.  A second writer (simulated here by
    recursively calling reconcile_archive from inside a patched
    commit_projection, guarded so it only races once) finishes a full
    rebuild between this call's load and its commit; the racing call's own
    stale commit must then be refused, and the session must be left exactly
    as the interleaved rebuild left it.
    """
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    reconcile_archive(config, stored)
    baseline_turns = read_projected_turns(config, stored)

    real_commit = projection_store.commit_projection
    raced = {"done": False}

    def racing_commit(
        cfg: Config, sid: str, projection: Any, next_state: dict[str, Any], **kwargs: Any
    ) -> dict[str, int]:
        if not raced["done"]:
            raced["done"] = True
            reconcile_archive(cfg, sid, rebuild=True)
        return real_commit(cfg, sid, projection, next_state, **kwargs)

    monkeypatch.setattr("thirdeye.platforms.copilot.reconcile.commit_projection", racing_commit)

    result = reconcile_archive(config, stored)

    assert result["errors"] == 1
    assert read_projected_turns(config, stored) == baseline_turns


def test_usage_sidecar_publish_failure_reports_the_committed_document(
    config: Config,
    paths: SourcePaths,
    cli_transcript_records: list[SourceRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-validation I/O failure while publishing the usage-sidecar

    mirror is a distinct failure mode from a validation failure: by the time
    it can happen, the projection document has already durably published
    (see commit_projection's docstring), so the counts reconcile_archive
    reports must reflect that new document -- not the prior one -- with the
    error counted on top.  The next reconcile call must self-heal the
    sidecar without reprocessing anything new.
    """
    stored = _seed_full_corpus(config, paths, cli_transcript_records)
    baseline = reconcile_archive(config, stored)
    prior_turns = read_projected_turns(config, stored)
    baseline_sequence = load_projection_state(config, stored)["commit_sequence"]

    original_publish_usage = projection_store._publish_usage

    def selective_boom(
        cfg: Config, sid: str, directory: Path, usage_index: dict[str, Any], *, force: bool = False
    ) -> None:
        # load_projection_state's self-heal always passes force=True; only
        # commit_projection's own (non-forced) publish should fail here, so
        # this reaches the specific "document committed, sidecar mirror
        # failed" state the docstring describes rather than failing before
        # commit_projection is ever entered.
        if force:
            original_publish_usage(cfg, sid, directory, usage_index, force=force)
            return
        raise OSError("disk full while rewriting usage sidecar")

    monkeypatch.setattr(
        "thirdeye.platforms.copilot.projection_store._publish_usage", selective_boom
    )

    result = reconcile_archive(config, stored, rebuild=True)

    assert result["errors"] == baseline["errors"] + 1
    assert result["turns"] == baseline["turns"]
    assert result["usage"] == baseline["usage"]
    # The document committed despite the sidecar failure -- proven by the
    # storage-owned commit_sequence advancing even though this call reported
    # an error -- so turns already reflect the (content-equivalent, since
    # this rebuilds the same archive) new projection rather than being stuck
    # on the old one.  Read the raw document rather than load_projection_state
    # here: that call would itself retry the still-patched, still-failing
    # sidecar publish as part of its own self-heal.
    assert _document(config, stored)["state"]["commit_sequence"] == baseline_sequence + 1
    assert read_projected_turns(config, stored) == prior_turns

    monkeypatch.undo()
    healed = reconcile_archive(config, stored)
    assert healed["errors"] == 0
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
        for event in SessionReader(config.root / "traces" / "copilot" / stored).iter_events()
    }

    reconcile_archive(config, stored, rebuild=True)

    after = {
        event["data"]["source_record"]["source_id"]
        for event in SessionReader(config.root / "traces" / "copilot" / stored).iter_events()
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
            "source_id": (f"{source_key}/{NATIVE_SESSION_ID}/4a386a37-ca7e-4ebc-a746-cdda20f2a4bb"),
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
