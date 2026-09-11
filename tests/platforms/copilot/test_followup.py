"""Behavioral tests for coalesced Copilot hook follow-up capture."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from thirdeye._compat.locking import LockMode, locked
from thirdeye.config import Config
from thirdeye.paths import session_dir, usage_log_path
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.followup import (
    _claim_lease,
    _lease_path,
    _owns_lease,
    _release_lease,
    _run,
    schedule_followup,
    try_capture_session,
)
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.state import lock_path
from thirdeye.platforms.copilot.types import SourcePaths, SyncResult

NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"


@pytest.fixture
def copilot_env(tmp_path: Path) -> tuple[Config, SourcePaths]:
    home = tmp_path / "copilot-home"
    home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(home)
    return config, paths


def _session_directory(config: Config, paths: SourcePaths) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, NATIVE_SESSION_ID))


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


def test_claim_lease_returns_generation(copilot_env: tuple[Config, SourcePaths]) -> None:
    config, paths = copilot_env
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert isinstance(generation, str)
    assert generation
    lease = json.loads(_lease_path(_session_directory(config, paths)).read_text(encoding="utf-8"))
    assert lease["generation"] == generation
    assert lease["expires_at"] > time.time()


def test_live_lease_coalesces_second_claim(copilot_env: tuple[Config, SourcePaths]) -> None:
    config, paths = copilot_env
    first = _claim_lease(config, paths, NATIVE_SESSION_ID)
    second = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert first is not None
    assert second is None


def test_expired_lease_can_be_reclaimed(copilot_env: tuple[Config, SourcePaths]) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    first = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert first is not None
    _lease_path(directory).write_text(
        json.dumps({"generation": first, "expires_at": time.time() - 1}),
        encoding="utf-8",
    )
    second = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert second is not None
    assert second != first


def test_release_lease_only_removes_own_generation(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None
    _release_lease(directory, "other-generation")
    assert _lease_path(directory).is_file()
    _release_lease(directory, generation)
    assert not _lease_path(directory).exists()


def test_schedule_followup_spawns_detached_worker(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        "thirdeye.platforms.copilot.followup.proc.spawn_detached",
        lambda argv: spawned.append(list(argv)),
    )
    assert schedule_followup(config, paths, NATIVE_SESSION_ID) is True
    assert len(spawned) == 1
    argv = spawned[0]
    assert argv[:3] == ["python", "-m", "thirdeye.platforms.copilot.followup"] or argv[1:4] == [
        "-m",
        "thirdeye.platforms.copilot.followup",
        "--source-home",
    ]
    assert "--source-home" in argv
    assert paths["home"] in argv
    assert "--session-id" in argv
    assert NATIVE_SESSION_ID in argv
    assert "--config-root" in argv
    assert str(config.root) in argv
    assert "--generation" in argv


def test_schedule_followup_coalesces_while_lease_live(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    spawn_count = {"n": 0}

    def count_spawn(_argv: list[str]) -> None:
        spawn_count["n"] += 1

    monkeypatch.setattr(
        "thirdeye.platforms.copilot.followup.proc.spawn_detached",
        count_spawn,
    )
    assert schedule_followup(config, paths, NATIVE_SESSION_ID) is True
    assert schedule_followup(config, paths, NATIVE_SESSION_ID) is False
    assert spawn_count["n"] == 1


def test_spawn_failure_releases_lease_and_logs(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)

    def boom(_argv: list[str]) -> None:
        raise OSError("spawn denied")

    monkeypatch.setattr("thirdeye.platforms.copilot.followup.proc.spawn_detached", boom)
    assert schedule_followup(config, paths, NATIVE_SESSION_ID) is False
    assert not _lease_path(directory).exists()
    log = usage_log_path(config.root)
    assert log.is_file()
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(entry["phase"] == "copilot_followup_spawn" for entry in entries)


def test_run_attempts_capture_until_complete(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None
    attempts: list[int] = []

    def fake_capture(_config: Config, _paths: SourcePaths, _native_id: str) -> SyncResult:
        attempts.append(1)
        if len(attempts) < 2:
            return _empty_result(pending=1, errors=1)
        return _empty_result()

    monkeypatch.setattr(
        "thirdeye.platforms.copilot.capture.capture_session",
        fake_capture,
    )
    monkeypatch.setattr("thirdeye.platforms.copilot.followup.time.sleep", lambda _s: None)
    _run(config, paths, NATIVE_SESSION_ID, generation)
    assert len(attempts) >= 2
    assert not _lease_path(directory).exists()


def test_run_releases_lease_even_when_capture_raises(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None

    def boom(_config: Config, _paths: SourcePaths, _native_id: str) -> SyncResult:
        raise RuntimeError("capture failed")

    monkeypatch.setattr("thirdeye.platforms.copilot.capture.capture_session", boom)
    monkeypatch.setattr("thirdeye.platforms.copilot.followup.time.sleep", lambda _s: None)
    _run(config, paths, NATIVE_SESSION_ID, generation)
    assert not _lease_path(directory).exists()


def test_try_capture_session_returns_none_when_archive_lock_held(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    archive_lock = lock_path(directory)
    with locked(archive_lock, LockMode.EXCLUSIVE):
        start = time.monotonic()
        result = try_capture_session(config, paths, NATIVE_SESSION_ID)
        assert time.monotonic() - start < 0.25
        assert result is None


def test_run_does_not_block_when_archive_lock_is_held(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None
    archive_lock = lock_path(directory)
    held = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with locked(archive_lock, LockMode.EXCLUSIVE):
            held.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert held.wait(timeout=2)
    monkeypatch.setattr("thirdeye.platforms.copilot.followup.time.sleep", lambda _s: None)
    times = iter([0.0, 0.0, 6.0])
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        "thirdeye.platforms.copilot.followup.time.monotonic",
        lambda: next(times, 6.0),
    )
    finished = threading.Event()
    try:

        def run_worker() -> None:
            _run(config, paths, NATIVE_SESSION_ID, generation)
            finished.set()

        worker = threading.Thread(target=run_worker)
        start = real_monotonic()
        worker.start()
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        assert finished.is_set()
        assert real_monotonic() - start < 1.0
        assert not _lease_path(directory).exists()
    finally:
        release.set()
        holder.join(timeout=2)


def test_stale_lease_does_not_block_future_schedule(
    copilot_env: tuple[Config, SourcePaths],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    first = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert first is not None
    _lease_path(directory).write_text(
        json.dumps({"generation": first, "expires_at": time.time() - 1}),
        encoding="utf-8",
    )
    spawn_count = {"n": 0}
    monkeypatch.setattr(
        "thirdeye.platforms.copilot.followup.proc.spawn_detached",
        lambda _argv: spawn_count.__setitem__("n", spawn_count["n"] + 1),
    )
    assert schedule_followup(config, paths, NATIVE_SESSION_ID) is True
    assert spawn_count["n"] == 1


def test_main_is_silent_on_worker_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from thirdeye.platforms.copilot.followup import main

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("worker failed")

    monkeypatch.setattr("thirdeye.platforms.copilot.followup._run", boom)
    monkeypatch.setattr(
        "sys.argv",
        [
            "followup",
            "--source-home",
            "/tmp/copilot-home",
            "--session-id",
            NATIVE_SESSION_ID,
            "--config-root",
            "/tmp/thirdeye",
            "--generation",
            "gen-1",
        ],
    )
    main()


def test_owns_lease_false_when_generation_mismatch(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    generation = _claim_lease(config, paths, NATIVE_SESSION_ID)
    assert generation is not None
    assert _owns_lease(directory, "wrong-generation") is False
    assert _owns_lease(directory, generation) is True


def test_claim_lease_returns_none_when_followup_lock_held(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    directory = _session_directory(config, paths)
    from thirdeye.platforms.copilot.followup import _lease_lock_path

    lease_lock = _lease_lock_path(directory)
    lease_lock.parent.mkdir(parents=True, exist_ok=True)
    with locked(lease_lock, LockMode.EXCLUSIVE):
        assert _claim_lease(config, paths, NATIVE_SESSION_ID) is None
