"""Behavioral tests for durable Copilot hook observation spooling."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config
from thirdeye.platforms.copilot.hook_payload import parse_hook
from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.spool import ack_spool, enqueue_hook, read_spool
from thirdeye.platforms.copilot.types import SourceRecord, SourcePaths

NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
OBSERVED_AT = "2026-09-10T17:08:25.626Z"
_CONCURRENT_WORKERS = 8
_CONCURRENT_RECORDS = 16


@pytest.fixture
def copilot_env(tmp_path: Path) -> tuple[Config, SourcePaths]:
    copilot_home = tmp_path / "copilot-home"
    copilot_home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(copilot_home)
    return config, paths


def _hook_record(
    *,
    observation_id: str,
    event: str = "agentStop",
    session_id: str = NATIVE_SESSION_ID,
    stop_reason: str = "end_turn",
) -> SourceRecord:
    return parse_hook(
        event,
        {
            "sessionId": session_id,
            "timestamp": 1789060105626,
            "cwd": "/fixture/workspace",
            "stopReason": stop_reason,
        },
        {"env": {"WB_PLAN": "p"}},
        observed_at=OBSERVED_AT,
        observation_id=observation_id,
    )


def test_enqueue_returns_spool_path_and_read_returns_complete_record(copilot_env: tuple[Config, SourcePaths]):
    config, paths = copilot_env
    record = _hook_record(observation_id="obs-enqueue-1")

    spool_path = enqueue_hook(config, paths, record)
    assert Path(spool_path).is_file()

    records = read_spool(config, paths, NATIVE_SESSION_ID)
    assert len(records) == 1
    assert records[0]["source_id"] == record["source_id"]
    assert records[0]["payload"]["hook_payload"]["stopReason"] == "end_turn"


def test_same_content_distinct_observation_ids_remain_separate(copilot_env: tuple[Config, SourcePaths]):
    config, paths = copilot_env
    first = _hook_record(observation_id="hook-observation-1")
    second = _hook_record(observation_id="hook-observation-2")

    enqueue_hook(config, paths, first)
    enqueue_hook(config, paths, second)

    records = read_spool(config, paths, NATIVE_SESSION_ID)
    assert len(records) == 2
    assert {record["source_id"] for record in records} == {first["source_id"], second["source_id"]}


def test_ack_spool_removes_only_committed_ids(copilot_env: tuple[Config, SourcePaths]):
    config, paths = copilot_env
    keep = _hook_record(observation_id="obs-keep")
    drop = _hook_record(observation_id="obs-drop")
    enqueue_hook(config, paths, keep)
    enqueue_hook(config, paths, drop)

    ack_spool(config, paths, [drop["source_id"]])

    remaining = read_spool(config, paths, NATIVE_SESSION_ID)
    assert len(remaining) == 1
    assert remaining[0]["source_id"] == keep["source_id"]


def test_ack_spool_unknown_ids_are_no_ops(copilot_env: tuple[Config, SourcePaths]):
    config, paths = copilot_env
    record = _hook_record(observation_id="obs-stable")
    enqueue_hook(config, paths, record)

    ack_spool(config, paths, ["nonexistent-source-id"])
    ack_spool(config, paths, [])

    remaining = read_spool(config, paths, NATIVE_SESSION_ID)
    assert len(remaining) == 1
    assert remaining[0]["source_id"] == record["source_id"]


def test_malformed_spool_neighbor_does_not_discard_valid_records(
    copilot_env: tuple[Config, SourcePaths],
):
    config, paths = copilot_env
    valid = _hook_record(observation_id="obs-valid-neighbor")
    spool_path = Path(enqueue_hook(config, paths, valid))

    (spool_path.parent / "000000-malformed.json").write_text("{not json", encoding="utf-8")

    records = read_spool(config, paths, NATIVE_SESSION_ID)
    assert len(records) == 1
    assert records[0]["source_id"] == valid["source_id"]


def test_concurrent_enqueue_preserves_all_records(tmp_path: Path):
    copilot_home = tmp_path / "copilot-home"
    copilot_home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(copilot_home)

    barrier = threading.Barrier(_CONCURRENT_WORKERS)
    errors: list[str] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            for index in range(_CONCURRENT_RECORDS):
                record = _hook_record(observation_id=f"obs-{worker_id}-{index}")
                enqueue_hook(config, paths, record)
        except Exception as exc:  # pragma: no cover - surfaced via errors list
            errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(_CONCURRENT_WORKERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    records = read_spool(config, paths, NATIVE_SESSION_ID)
    assert len(records) == _CONCURRENT_WORKERS * _CONCURRENT_RECORDS


def test_spool_survives_subprocess_isolation(tmp_path: Path):
    copilot_home = tmp_path / "copilot-home"
    copilot_home.mkdir()
    config_root = tmp_path / "thirdeye"
    script = f"""
from thirdeye.config import Config
from thirdeye.platforms.copilot.hook_payload import parse_hook
from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.spool import enqueue_hook, read_spool

config = Config(root={config_root!r})
paths = resolve_sources({copilot_home!r})
record = parse_hook(
    "sessionStart",
    {{"sessionId": {NATIVE_SESSION_ID!r}, "timestamp": 1789060102204}},
    {{}},
    observed_at={OBSERVED_AT!r},
    observation_id="obs-subprocess",
)
enqueue_hook(config, paths, record)
assert len(read_spool(config, paths, {NATIVE_SESSION_ID!r})) == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr

    config = Config(root=config_root)
    paths = resolve_sources(copilot_home)
    records = read_spool(config, paths, NATIVE_SESSION_ID)
    assert len(records) == 1
    assert records[0]["payload"]["hook_payload"]["sessionId"] == NATIVE_SESSION_ID


def test_child_session_hooks_spool_under_native_session_id(tmp_path: Path):
    child_id = "bf8cb9f3-2097-4db0-a3c8-78a2653b2106"
    copilot_home = tmp_path / "copilot-home"
    copilot_home.mkdir()
    config = Config(root=tmp_path / "thirdeye")
    paths = resolve_sources(copilot_home)

    parent = _hook_record(observation_id="obs-parent", session_id=NATIVE_SESSION_ID)
    child = _hook_record(observation_id="obs-child", session_id=child_id)
    enqueue_hook(config, paths, parent)
    enqueue_hook(config, paths, child)

    parent_records = read_spool(config, paths, NATIVE_SESSION_ID)
    child_records = read_spool(config, paths, child_id)
    assert len(parent_records) == 1
    assert len(child_records) == 1
    assert parent_records[0]["native_session_id"] == NATIVE_SESSION_ID
    assert child_records[0]["native_session_id"] == child_id
