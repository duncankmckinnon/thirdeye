"""Behavioral tests for Copilot local capture status reporting."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

import thirdeye.platforms.copilot.status as status_mod
from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.constants import OWNED_HOOK_FILENAME, PLATFORM_NAME
from thirdeye.platforms.copilot.hook_payload import parse_hook
from thirdeye.platforms.copilot.identity import (
    SOURCE_KEY_PREFIX_LEN,
    resolve_sources,
    stored_session_id,
)
from thirdeye.platforms.copilot.install import CopilotPlatform
from thirdeye.platforms.copilot.spool import enqueue_hook
from thirdeye.platforms.copilot.state import (
    journal_path,
    read_json,
    state_path,
    write_journal,
    write_state,
)
from thirdeye.platforms.copilot.status import capture_status
from thirdeye.platforms.copilot.types import SourceBatch, SourcePaths, SourceRecord

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
CLI_FIXTURE = FIXTURES / "cli-1.0.83"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
OBSERVED_AT_EARLY = "2026-09-10T17:08:20.000Z"
OBSERVED_AT_LATE = "2026-09-10T17:08:30.000Z"


@pytest.fixture
def copilot_env(tmp_path: Path) -> tuple[Config, SourcePaths]:
    home = tmp_path / "copilot-home"
    home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(home)
    return config, paths


def _record(
    source_id: str,
    *,
    source_kind: str = "transcript",
    observed_at: str = OBSERVED_AT_EARLY,
) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "native_session_id": NATIVE_SESSION_ID,
        "ts": "2026-09-10T17:08:24.000Z",
        "observed_at": observed_at,
        "payload": {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _batch(paths: SourcePaths, records: list[SourceRecord]) -> SourceBatch:
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_SESSION_ID,
        "cwd": "/proj",
        "records": records,
        "next_cursor": {"generation": 1},
        "diagnostics": [],
    }


def _hook_record(*, observation_id: str, observed_at: str = OBSERVED_AT_LATE) -> SourceRecord:
    return parse_hook(
        "agentStop",
        {
            "sessionId": NATIVE_SESSION_ID,
            "timestamp": 1789060105626,
            "cwd": "/fixture/workspace",
            "stopReason": "end_turn",
        },
        {"env": {"WB_PLAN": "p"}},
        observed_at=observed_at,
        observation_id=observation_id,
    )


def _write_transcript(home: Path, native_id: str) -> None:
    session_dir = home / "session-state" / native_id
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CLI_FIXTURE / "events.jsonl", session_dir / "events.jsonl")


def _write_database(home: Path, *, session_id: str = NATIVE_SESSION_ID) -> None:
    database = home / "session-store.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, created_at TEXT);"
        )
        connection.execute(
            "INSERT INTO sessions (id, cwd, created_at) VALUES (?, ?, ?)",
            (session_id, "/tmp/probe", "2026-09-10T17:08:00.000Z"),
        )
        connection.commit()
    finally:
        connection.close()


def test_status_module_has_no_command_or_hook_runtime_imports() -> None:
    source = Path(status_mod.__file__).read_text(encoding="utf-8")
    assert "thirdeye.commands" not in source
    assert "hook_lifecycle" not in source
    assert "from .followup" not in source


def test_capture_status_reports_empty_installation_and_capabilities(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    status = capture_status(config, paths)

    assert status["paths"] == dict(paths)
    assert status["installation"]["configured"] is False
    assert status["installation"]["hooks_file"].endswith(OWNED_HOOK_FILENAME)
    assert status["capabilities"]["transcripts"]["exists"] is False
    assert status["capabilities"]["database"]["exists"] is False
    assert status["capabilities"]["database_wal"]["exists"] is False
    assert status["last_observed_hook"] is None
    assert status["last_successful_import"] is None
    assert status["sessions"] == []
    assert status["pending"] == {
        "spool_records": 0,
        "spool_sessions": [],
        "followup": 0,
        "leases": 0,
        "journals": 0,
    }
    assert status["errors"] == []


def test_capture_status_reports_readable_source_capabilities(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    home = Path(paths["home"])
    _write_transcript(home, NATIVE_SESSION_ID)
    _write_database(home, session_id=NATIVE_SESSION_ID)

    status = capture_status(config, paths)

    assert status["capabilities"]["transcripts"]["readable"] is True
    assert status["capabilities"]["database"]["readable"] is True
    assert status["errors"] == []


def test_capture_status_reports_installed_hooks(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    # Match capture_status's default entrypoint resolution (bare hook name when
    # the dispatcher is not on PATH in CI).
    platform = CopilotPlatform(
        source_home=Path(paths["home"]),
        entrypoint="thirdeye-copilot-hook",
    )
    platform.install()

    status = capture_status(config, paths)

    assert status["installation"]["configured"] is True
    assert status["errors"] == []


def test_capture_status_reports_archived_session_progress(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    commit_batch(config, paths, _batch(paths, [_record("status/archived-1")]))

    status = capture_status(config, paths)
    stored = stored_session_id(paths, NATIVE_SESSION_ID)

    assert len(status["sessions"]) == 1
    session = status["sessions"][0]
    assert session["stored_session_id"] == stored
    assert session["native_session_id"] == NATIVE_SESSION_ID
    assert session["journal_pending"] is False
    assert isinstance(status["last_successful_import"], str)


def test_capture_status_reports_pending_spool_and_latest_hook(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    record = _hook_record(observation_id="status-spool-hook")
    enqueue_hook(config, paths, record)

    status = capture_status(config, paths)

    assert status["pending"]["spool_records"] == 1
    assert status["pending"]["spool_sessions"] == [NATIVE_SESSION_ID]
    assert status["last_observed_hook"] is not None
    assert status["last_observed_hook"]["source_id"] == record["source_id"]


def test_capture_status_prefers_latest_hook_observed_at(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    archived = _record("status/archived-hook", source_kind="hook", observed_at=OBSERVED_AT_EARLY)
    commit_batch(config, paths, _batch(paths, [archived]))
    spooled = _hook_record(observation_id="status-newer-hook", observed_at=OBSERVED_AT_LATE)
    enqueue_hook(config, paths, spooled)

    status = capture_status(config, paths)

    assert status["last_observed_hook"] is not None
    assert status["last_observed_hook"]["source_id"] == spooled["source_id"]


def test_capture_status_reports_followup_leases_and_journal(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    commit_batch(config, paths, _batch(paths, [_record("status/pending-state")]))
    directory = session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, NATIVE_SESSION_ID))
    state = {
        "schema_version": 1,
        "source_key": paths["source_key"],
        "source_home": paths["home"],
        "native_session_id": NATIVE_SESSION_ID,
        "cursor": {},
        "followup": True,
        "lease": [{"owner": "test"}],
        "health": {
            "diagnostics": [{"kind": "test_diagnostic", "message": "retry later"}],
            "last_successful_import": "2026-09-10T17:08:25.000Z",
        },
    }
    write_state(directory, state)
    write_journal(directory, {"pending": True})

    status = capture_status(config, paths)

    assert status["pending"]["followup"] == 1
    assert status["pending"]["leases"] == 1
    assert status["pending"]["journals"] == 1
    assert journal_path(directory).is_file()
    assert any(error.get("kind") == "test_diagnostic" for error in status["errors"])


def test_capture_status_reports_invalid_archive_state(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    commit_batch(config, paths, _batch(paths, [_record("status/invalid-state")]))
    directory = session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, NATIVE_SESSION_ID))
    state_path(directory).write_text("{not valid json", encoding="utf-8")

    status = capture_status(config, paths)

    assert any(error.get("kind") == "invalid_archive_state" for error in status["errors"])


def test_capture_status_tolerates_missing_optional_followup_state(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    commit_batch(config, paths, _batch(paths, [_record("status/minimal-state")]))
    directory = session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, NATIVE_SESSION_ID))
    state_path(directory).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_key": paths["source_key"],
                "source_home": paths["home"],
                "native_session_id": NATIVE_SESSION_ID,
                "cursor": {},
                "health": {"diagnostics": [], "last_successful_import": None},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = capture_status(config, paths)

    assert status["pending"]["followup"] == 0
    assert status["pending"]["leases"] == 0
    assert status["errors"] == []


def test_capture_status_reports_source_key_prefix_collision(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    commit_batch(config, paths, _batch(paths, [_record("status/collision")]))
    directory = session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, NATIVE_SESSION_ID))
    state = read_json(state_path(directory))
    assert state is not None
    original = state["source_key"]
    replacement = "b" if original[SOURCE_KEY_PREFIX_LEN] != "b" else "a"
    colliding = original[:SOURCE_KEY_PREFIX_LEN] + replacement + original[SOURCE_KEY_PREFIX_LEN + 1 :]
    assert colliding != original
    state["source_key"] = colliding
    write_state(directory, state)

    status = capture_status(config, paths)

    assert status["sessions"] == []
    assert any(error.get("kind") == "source_key_collision" for error in status["errors"])


def test_capture_status_reports_unusable_database(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    Path(paths["database"]).write_text("not a sqlite database\n", encoding="utf-8")

    status = capture_status(config, paths)

    assert status["capabilities"]["database"]["exists"] is True
    assert status["capabilities"]["database"]["readable"] is False
    assert any(error.get("kind") == "source_unreadable" for error in status["errors"])


def test_capture_status_reports_unusable_transcript_root(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    Path(paths["session_root"]).write_text("not a directory\n", encoding="utf-8")

    status = capture_status(config, paths)

    assert status["capabilities"]["transcripts"]["exists"] is True
    assert status["capabilities"]["transcripts"]["readable"] is False
    assert any(error.get("kind") == "source_unreadable" for error in status["errors"])


def test_capture_status_reports_unreadable_transcript_directory(
    monkeypatch: pytest.MonkeyPatch,
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    session_root = Path(paths["session_root"])
    session_root.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def fake_iterdir(self: Path):
        if self == session_root:
            raise PermissionError("denied")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    status = capture_status(config, paths)

    assert status["capabilities"]["transcripts"]["exists"] is True
    assert status["capabilities"]["transcripts"]["readable"] is False
    assert any(error.get("kind") == "source_unreadable" for error in status["errors"])


def test_capture_status_reports_malformed_spool_without_payloads(
    copilot_env: tuple[Config, SourcePaths],
) -> None:
    config, paths = copilot_env
    spool_dir = Path(config.root) / "spool" / "copilot" / paths["source_key"] / NATIVE_SESSION_ID
    spool_dir.mkdir(parents=True)
    (spool_dir / "broken.json").write_text("{not valid json\n", encoding="utf-8")

    status = capture_status(config, paths)

    assert status["pending"]["spool_records"] >= 1
    assert NATIVE_SESSION_ID in status["pending"]["spool_sessions"]
    assert any(error.get("kind") == "spool_unreadable" for error in status["errors"])
    serialized = json.dumps(status["errors"])
    assert "prompt" not in serialized.lower()
