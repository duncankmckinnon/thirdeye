"""Integration tests for Copilot runtime reconciliation wiring."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import otel_jobs_dir, session_dir
from thirdeye.platforms.copilot import watch as watch_mod
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.export_state import (
    load_export_state,
    record_placement,
    update_export_state,
)
from thirdeye.platforms.copilot.export_transport import delivery_claim_path, job_path
from thirdeye.platforms.copilot.followup import _claim_lease, _run
from thirdeye.platforms.copilot.followup import main as followup_main
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection_store import read_projection_status
from thirdeye.platforms.copilot.runtime import (
    all_archived_session_ids,
    archived_session_ids,
    exports_configured,
    load_runtime_status,
    reconcile_archived_sessions,
    reconcile_session,
)
from thirdeye.platforms.copilot.state import state_path
from thirdeye.platforms.copilot.status import capture_status
from thirdeye.platforms.copilot.types import SourceBatch, SourcePaths, SourceRecord, SyncResult
from thirdeye.usage.errlog import log_capture_error

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
OTHER_SESSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


@pytest.fixture
def copilot_env(tmp_path: Path) -> tuple[Config, SourcePaths]:
    home = tmp_path / "copilot-home"
    home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(home)
    return config, paths


def _record(source_id: str, *, native_session_id: str = NATIVE_SESSION_ID) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": "transcript",
        "native_session_id": native_session_id,
        "ts": "2026-09-10T17:08:24.000Z",
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


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
    *,
    native_session_id: str = NATIVE_SESSION_ID,
) -> str:
    commit_batch(
        config,
        paths,
        _batch(
            paths,
            [_record("runtime/archived", native_session_id=native_session_id)],
            native_session_id=native_session_id,
        ),
    )
    return stored_session_id(paths, native_session_id)


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


def test_exports_configured_requires_enabled_logfire_with_token() -> None:
    assert exports_configured(Config(root=Path("/tmp/thirdeye"))) is False
    assert (
        exports_configured(
            Config(
                root=Path("/tmp/thirdeye"),
                logfire=LogfireSettings(enabled=True, token="fake-token"),
            )
        )
        is True
    )
    assert (
        exports_configured(
            Config(
                root=Path("/tmp/thirdeye"),
                logfire=LogfireSettings(enabled=True, token=""),
            )
        )
        is False
    )


def test_archived_session_ids_lists_sessions_for_source_home(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    _seed_archive(config, paths, native_session_id=OTHER_SESSION_ID)

    assert archived_session_ids(config, paths) == sorted(
        [stored, stored_session_id(paths, OTHER_SESSION_ID)]
    )


def test_archived_session_ids_ignores_other_source_keys(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    directory = session_dir(config.root, PLATFORM_NAME, stored)
    state = json.loads(state_path(directory).read_text(encoding="utf-8"))
    state["source_key"] = "x" * 64
    state_path(directory).write_text(json.dumps(state) + "\n", encoding="utf-8")

    assert archived_session_ids(config, paths) == []


def test_all_archived_session_ids_lists_every_copilot_archive(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    first = _seed_archive(config, paths)
    second = _seed_archive(config, paths, native_session_id=OTHER_SESSION_ID)
    assert all_archived_session_ids(config) == sorted([first, second])


def test_reconcile_session_maps_native_id_to_stored_session(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    calls: list[tuple[str, bool, bool]] = []

    def fake_reconcile_archive(
        _config: Config,
        stored_session_id: str,
        *,
        rebuild: bool = False,
        export: bool = False,
        include_history: bool = False,
    ) -> dict[str, int]:
        calls.append((stored_session_id, export, include_history))
        return {
            "events": 1,
            "usage": 0,
            "turns": 1,
            "exports": 0,
            "pending": 0,
            "ambiguous": 0,
            "conflicting": 0,
            "errors": 0,
        }

    monkeypatch.setattr("thirdeye.platforms.copilot.runtime.reconcile_archive", fake_reconcile_archive)
    result = reconcile_session(config, paths, NATIVE_SESSION_ID, export=True, include_history=True)

    assert result["turns"] == 1
    assert calls == [(stored, True, True)]


def test_reconcile_session_activates_eligibility_without_logfire(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)

    result = reconcile_session(config, paths, NATIVE_SESSION_ID, export=True)

    assert result["exports"] == 0
    state = load_export_state(config, stored)
    assert state["activated"] is True
    restarted = load_export_state(config, stored)
    assert restarted["activated"] is True


def test_reconcile_archived_sessions_replays_all_retained_sessions(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    first = _seed_archive(config, paths)
    second = _seed_archive(config, paths, native_session_id=OTHER_SESSION_ID)
    seen: list[str] = []

    def fake_reconcile_archive(
        _config: Config,
        stored_session_id: str,
        *,
        rebuild: bool = False,
        export: bool = False,
        include_history: bool = False,
    ) -> dict[str, int]:
        seen.append(stored_session_id)
        return {
            "events": 0,
            "usage": 0,
            "turns": 0,
            "exports": 0,
            "pending": 0,
            "ambiguous": 0,
            "conflicting": 0,
            "errors": 0,
        }

    monkeypatch.setattr("thirdeye.platforms.copilot.runtime.reconcile_archive", fake_reconcile_archive)
    results = reconcile_archived_sessions(config, paths)

    assert set(seen) == {first, second}
    assert set(results) == {first, second}


def test_reconcile_archived_sessions_works_after_source_removal(
    copilot_env: tuple[Config, SourcePaths],
    tmp_path: Path,
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    session_dir_path = home / "session-state" / NATIVE_SESSION_ID
    session_dir_path.mkdir(parents=True)
    shutil.copy(FIXTURES / "events.jsonl", session_dir_path / "events.jsonl")
    (session_dir_path / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")

    from thirdeye.platforms.copilot.capture import sync

    sync(config, paths, session_id=NATIVE_SESSION_ID)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)
    shutil.rmtree(home / "session-state")
    (home / "session-store.db").unlink(missing_ok=True)

    result = reconcile_archived_sessions(config, paths)[stored]

    assert result["errors"] == 0
    assert read_projection_status(config, stored)["errors"] == 0


def test_followup_reconciles_after_successful_capture(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None
    reconcile_calls: list[tuple[str, bool]] = []

    def fake_capture(_config: Config, _paths: SourcePaths, _native_id: str) -> SyncResult:
        return _empty_result(sessions=1, records_written=1)

    def fake_reconcile_session(
        _config: Config,
        _paths: SourcePaths,
        native_session_id: str,
        *,
        export: bool = False,
        include_history: bool = False,
    ) -> dict[str, int]:
        reconcile_calls.append((native_session_id, export))
        return {
            "events": 0,
            "usage": 0,
            "turns": 0,
            "exports": 0,
            "pending": 0,
            "ambiguous": 0,
            "conflicting": 0,
            "errors": 0,
        }

    monkeypatch.setattr("thirdeye.platforms.copilot.capture.capture_session", fake_capture)
    monkeypatch.setattr(
        "thirdeye.platforms.copilot.runtime.reconcile_session",
        fake_reconcile_session,
    )
    _run(config, paths, NATIVE_SESSION_ID, generation)

    assert reconcile_calls == [(NATIVE_SESSION_ID, True)]


def test_followup_reconcile_failure_does_not_block_capture_completion(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    from thirdeye.platforms.copilot.followup import _lease_path

    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None
    directory = session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, NATIVE_SESSION_ID))

    def fake_capture(_config: Config, _paths: SourcePaths, _native_id: str) -> SyncResult:
        return _empty_result(sessions=1)

    def boom(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        raise RuntimeError("projection failed")

    monkeypatch.setattr("thirdeye.platforms.copilot.capture.capture_session", fake_capture)
    monkeypatch.setattr("thirdeye.platforms.copilot.runtime.reconcile_session", boom)
    _run(config, paths, NATIVE_SESSION_ID, generation)

    assert not _lease_path(directory).exists()
    log = config.root / "logs" / "usage-errors.jsonl"
    assert log.is_file()
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(entry.get("phase") == "copilot_followup_reconcile" for entry in entries)


def test_followup_logs_returned_reconcile_errors(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None

    def fake_capture(_config: Config, _paths: SourcePaths, _native_id: str) -> SyncResult:
        return _empty_result(sessions=1)

    def fake_reconcile_archive(
        *_args: Any, **_kwargs: Any
    ) -> dict[str, int]:
        return {
            "events": 0,
            "usage": 0,
            "turns": 0,
            "exports": 0,
            "pending": 0,
            "ambiguous": 0,
            "conflicting": 0,
            "errors": 2,
        }

    monkeypatch.setattr("thirdeye.platforms.copilot.capture.capture_session", fake_capture)
    monkeypatch.setattr("thirdeye.platforms.copilot.runtime.reconcile_archive", fake_reconcile_archive)
    _run(config, paths, NATIVE_SESSION_ID, generation)

    status = load_runtime_status(config, stored)
    assert status["last_error"]["errors"] == 2
    log = config.root / "logs" / "usage-errors.jsonl"
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(entry.get("phase") == "copilot_reconcile" for entry in entries)


def test_followup_entrypoint_loads_persisted_logfire_settings(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    config.write_logfire_settings(LogfireSettings(enabled=True, token="live-token"))
    seen: list[Config] = []

    def fake_run(loaded: Config, _paths: SourcePaths, _native_id: str, _generation: str) -> None:
        seen.append(loaded)

    monkeypatch.setattr("thirdeye.platforms.copilot.followup._run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "followup",
            "--source-home",
            paths["home"],
            "--session-id",
            NATIVE_SESSION_ID,
            "--config-root",
            str(config.root),
            "--generation",
            "deadbeef",
        ],
    )
    followup_main()

    assert len(seen) == 1
    assert seen[0].logfire.enabled is True
    assert seen[0].logfire.token == "live-token"


def test_watch_activation_reconciles_archived_sessions(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    activation: list[bool] = []
    per_session: list[str] = []

    def fake_sync(
        _config: Config, _paths: SourcePaths, *, session_id: str | None = None
    ) -> SyncResult:
        return _empty_result()

    def fake_reconcile_archived(
        _config: Config,
        _paths: SourcePaths,
        *,
        export: bool = False,
        include_history: bool = False,
    ) -> dict[str, dict[str, int]]:
        activation.append(export)
        return {}

    def fake_reconcile_one(
        _config: Config,
        _paths: SourcePaths,
        native_session_id: str,
        *,
        export: bool = False,
        include_history: bool = False,
    ) -> dict[str, int]:
        per_session.append(native_session_id)
        return {
            "events": 0,
            "usage": 0,
            "turns": 0,
            "exports": 0,
            "pending": 0,
            "ambiguous": 0,
            "conflicting": 0,
            "errors": 0,
        }

    monkeypatch.setattr(watch_mod, "sync", fake_sync)
    monkeypatch.setattr(watch_mod, "reconcile_archived_sessions", fake_reconcile_archived)
    monkeypatch.setattr(watch_mod, "reconcile_session", fake_reconcile_one)
    monkeypatch.setattr(
        watch_mod, "_SLEEP", lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    watch_mod.watch(config, paths, interval=0.1)

    assert activation == [True]
    assert per_session == []


def test_status_counts_delivery_claim_as_delivered_not_queued(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    directory = session_dir(config.root, PLATFORM_NAME, stored)

    def _record(state: dict[str, Any]) -> dict[str, Any]:
        updated, _, _ = record_placement(
            state,
            accounting_id="acct-1",
            destination="session-accounting-span",
            span_id="accounting:stored:acct-1",
            usage={},
            delivered=False,
        )
        return updated

    update_export_state(config, stored, _record)
    claim = delivery_claim_path(directory, "acct-1")
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.write_text("sent", encoding="utf-8")

    status = capture_status(config, paths)
    export = status["sessions"][0]["export"]
    assert export["delivered"] >= 1
    assert export["queued"] == 0


def test_status_counts_failed_accounting_job_as_error_backlog(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    span_id = f"accounting:{stored}:acct-fail"
    path = job_path(config.root, span_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "job_id": span_id,
                "kind": "session_accounting",
                "session_id": stored,
                "accounting_id": "acct-fail",
                "state": "failed",
                "attempt": 5,
                "last_error": "TimeoutError: collector stalled",
            }
        ),
        encoding="utf-8",
    )

    status = capture_status(config, paths)
    export = status["sessions"][0]["export"]
    assert export["errors"] >= 1
    assert export["queued"] >= 1
    assert export["delivered"] == 0


def test_status_counts_turn_job_and_sent_claim(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    directory = session_dir(config.root, PLATFORM_NAME, stored)
    jobs = otel_jobs_dir(config.root)
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "turn-queued.json").write_text(
        json.dumps({"kind": "turn", "session_id": stored, "job_id": "turn-queued"}),
        encoding="utf-8",
    )
    sent_dir = directory / "otel-turns-sent"
    sent_dir.mkdir(parents=True, exist_ok=True)
    (sent_dir / "delivered.json").write_text("sent", encoding="utf-8")

    status = capture_status(config, paths)
    export = status["sessions"][0]["export"]
    assert export["queued"] >= 1
    assert export["delivered"] >= 1


def test_status_counts_failed_turn_export_from_worker_log(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    """A failed whole-turn export deletes the job before delivery.

    The only remaining evidence is usage-errors.jsonl. Status must still
    report backlog and errors without a surviving job or sent claim.
    """

    config, paths = copilot_env
    stored = _seed_archive(config, paths)
    log_capture_error(
        thirdeye_home=config.root,
        phase="otel_worker_export_failed",
        level="error",
        platform=PLATFORM_NAME,
        session_id=stored,
        error=RuntimeError("logfire flush failed"),
        message="kind=turn",
    )
    log_capture_error(
        thirdeye_home=config.root,
        phase="otel_worker_export_failed",
        level="error",
        platform=PLATFORM_NAME,
        session_id="copilot-other-session",
        error=RuntimeError("other session"),
        message="kind=turn",
    )

    jobs = otel_jobs_dir(config.root)
    assert not jobs.exists() or not any(path.suffix == ".json" for path in jobs.iterdir())
    assert not (session_dir(config.root, PLATFORM_NAME, stored) / "otel-turns-sent").exists()

    status = capture_status(config, paths)
    export = status["sessions"][0]["export"]
    assert export["errors"] == 1
    assert export["queued"] == 1
    assert export["delivered"] == 0


def test_status_exposes_persisted_reconcile_last_error(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    stored = _seed_archive(config, paths)

    monkeypatch.setattr(
        "thirdeye.platforms.copilot.runtime.reconcile_archive",
        lambda *_args, **_kwargs: {
            "events": 0,
            "usage": 0,
            "turns": 0,
            "exports": 0,
            "pending": 0,
            "ambiguous": 0,
            "conflicting": 0,
            "errors": 1,
        },
    )
    reconcile_session(config, paths, NATIVE_SESSION_ID)

    status = capture_status(config, paths)
    export = status["sessions"][0]["export"]
    assert export["errors"] >= 1
    assert export["last_error"]["errors"] == 1
    assert any(error.get("kind") == "copilot_reconcile_error" for error in status["errors"])
    assert stored in {session["stored_session_id"] for session in status["sessions"]}
