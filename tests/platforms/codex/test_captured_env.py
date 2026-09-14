from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from thirdeye.config import Config, LogfireSettings
from thirdeye.meta import SessionMeta, write_meta
from thirdeye.paths import meta_path
from thirdeye.platforms.codex.captured_env import resolve_captured_env


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return replace(
        Config(root=tmp_path, logfire=LogfireSettings(enabled=True, token="t")),
        capture_env_patterns=("WB_*",),
    )


def _persist(session_dir_: Path, **captured: str) -> None:
    session_dir_.mkdir(parents=True, exist_ok=True)
    write_meta(
        meta_path(session_dir_),
        SessionMeta(
            session_id="s1",
            platform="codex",
            cwd="/repo",
            started_at="2026-01-01T00:00:00Z",
            ended_at=None,
            status="open",
            event_count=1,
            last_seq=0,
            last_ts=None,
            extra={"captured_env": dict(captured)},
        ),
    )


def test_live_env_wins(config, monkeypatch, tmp_path):
    monkeypatch.setenv("WB_PLAN", "live")
    _persist(tmp_path / "sd", WB_PLAN="stale")
    assert resolve_captured_env(config, tmp_path / "sd") == {"WB_PLAN": "live"}


def test_falls_back_to_persisted_snapshot_when_env_empty(config, monkeypatch, tmp_path):
    monkeypatch.delenv("WB_PLAN", raising=False)
    _persist(tmp_path / "sd", WB_PLAN="auth", WB_STEP="test#1")
    assert resolve_captured_env(config, tmp_path / "sd") == {
        "WB_PLAN": "auth",
        "WB_STEP": "test#1",
    }


def test_empty_when_neither_source_has_anything(config, monkeypatch, tmp_path):
    monkeypatch.delenv("WB_PLAN", raising=False)
    (tmp_path / "sd").mkdir()
    assert resolve_captured_env(config, tmp_path / "sd") == {}


def test_missing_meta_is_tolerated(config, monkeypatch, tmp_path):
    monkeypatch.delenv("WB_PLAN", raising=False)
    assert resolve_captured_env(config, tmp_path / "nonexistent") == {}


def test_non_dict_captured_env_in_meta_is_ignored(config, monkeypatch, tmp_path):
    monkeypatch.delenv("WB_PLAN", raising=False)
    sd = tmp_path / "sd"
    sd.mkdir()
    write_meta(
        meta_path(sd),
        SessionMeta(
            session_id="s1",
            platform="codex",
            cwd="/repo",
            started_at="2026-01-01T00:00:00Z",
            ended_at=None,
            status="open",
            event_count=1,
            last_seq=0,
            last_ts=None,
            extra={"captured_env": "not-a-dict"},
        ),
    )
    assert resolve_captured_env(config, sd) == {}
