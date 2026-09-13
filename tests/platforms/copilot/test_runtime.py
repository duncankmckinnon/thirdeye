"""Integration tests for Copilot runtime reconciliation wiring."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import session_dir
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.followup import _claim_lease, _run
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection_store import read_projection_status
from thirdeye.platforms.copilot.runtime import (
    archived_session_ids,
    exports_configured,
    reconcile_archived_sessions,
    reconcile_session,
)
from thirdeye.platforms.copilot.state import state_path
from thirdeye.platforms.copilot import watch as watch_mod

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

    assert archived_session_ids(config, paths) == sorted([stored, stored_session_id(paths, OTHER_SESSION_ID)])


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
        return {"events": 1, "usage": 0, "turns": 1, "exports": 0, "pending": 0, "ambiguous": 0, "conflicting": 0, "errors": 0}

    monkeypatch.setattr("thirdeye.platforms.copilot.runtime.reconcile_archive", fake_reconcile_archive)
    result = reconcile_session(config, paths, NATIVE_SESSION_ID, export=True, include_history=True)

    assert result["turns"] == 1
    assert calls == [(stored, False, True)]


def test_reconcile_session_skips_export_when_logfire_not_configured(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    _seed_archive(config, paths)
    calls: list[bool] = []

    def fake_reconcile_archive(
        _config: Config,
        _stored_session_id: str,
        *,
        rebuild: bool = False,
        export: bool = False,
        include_history: bool = False,
    ) -> dict[str, int]:
        calls.append(export)
        return {"events": 0, "usage": 0, "turns": 0, "exports": 0, "pending": 0, "ambiguous": 0, "conflicting": 0, "errors": 0}

    monkeypatch.setattr("thirdeye.platforms.copilot.runtime.reconcile_archive", fake_reconcile_archive)
    reconcile_session(config, paths, NATIVE_SESSION_ID, export=True)

    assert calls == [False]


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
        return {"events": 0, "usage": 0, "turns": 0, "exports": 0, "pending": 0, "ambiguous": 0, "conflicting": 0, "errors": 0}

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
        return {"events": 0, "usage": 0, "turns": 0, "exports": 0, "pending": 0, "ambiguous": 0, "conflicting": 0, "errors": 0}

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


def test_watch_activation_reconciles_archived_sessions(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    activation: list[bool] = []
    per_session: list[str] = []

    def fake_sync(_config: Config, _paths: SourcePaths, *, session_id: str | None = None) -> SyncResult:
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
        return {"events": 0, "usage": 0, "turns": 0, "exports": 0, "pending": 0, "ambiguous": 0, "conflicting": 0, "errors": 0}

    monkeypatch.setattr(watch_mod, "sync", fake_sync)
    monkeypatch.setattr(watch_mod, "reconcile_archived_sessions", fake_reconcile_archived)
    monkeypatch.setattr(watch_mod, "reconcile_session", fake_reconcile_one)
    monkeypatch.setattr(
        watch_mod, "_SLEEP", lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    watch_mod.watch(config, paths, interval=0.1)

    assert activation == [True]
    assert per_session == []
