from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from thirdeye.paths import (
    session_dir,
    usage_jsonl_path,
    usage_log_path,
    usage_state_path,
)
from thirdeye.platforms.codex.usage import capture_usage_codex

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.jsonl"


@pytest.fixture
def fake_codex_root(tmp_path: Path) -> Path:
    root = tmp_path / "codex_sessions"
    nested = root / "2026" / "05" / "15"
    nested.mkdir(parents=True)
    shutil.copy(FIXTURE, nested / "rollout-2026-05-15T10-00-00-abc.jsonl")
    return root


def test_capture_creates_rows_from_fixture(tmp_path: Path, fake_codex_root: Path) -> None:
    rows = capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="abc",
        triggering_seq=5,
        sessions_root=fake_codex_root,
    )
    assert rows == 2
    lines = usage_jsonl_path(session_dir(tmp_path, "codex", "abc")).read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["platform"] == "codex"
    assert first["seq"] == 5
    assert first["gen_ai.response.model"] == "gpt-5.5"
    assert first["gen_ai.usage.input_tokens"] == 1000
    assert first["gen_ai.usage.output_tokens"] == 50
    # Totals are derived, never stored or serialized.
    assert "total_tokens" not in first
    assert "gen_ai.usage.total_tokens" not in first


def test_capture_is_incremental(tmp_path: Path, fake_codex_root: Path) -> None:
    capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="abc",
        triggering_seq=1,
        sessions_root=fake_codex_root,
    )
    sd = session_dir(tmp_path, "codex", "abc")
    state = json.loads(usage_state_path(sd).read_text())
    assert state["rollout_offset"] > 0
    rows = capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="abc",
        triggering_seq=2,
        sessions_root=fake_codex_root,
    )
    assert rows == 0


def test_capture_caches_rollout_path(tmp_path: Path, fake_codex_root: Path) -> None:
    capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="abc",
        triggering_seq=1,
        sessions_root=fake_codex_root,
    )
    state = json.loads(usage_state_path(session_dir(tmp_path, "codex", "abc")).read_text())
    assert state["rollout_path"].endswith("-abc.jsonl")


def test_capture_no_rollout_logs_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    rows = capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="unknown",
        triggering_seq=1,
        sessions_root=empty,
    )
    assert rows == 0
    assert usage_log_path(tmp_path).exists()


def test_capture_skips_non_usage_frames(tmp_path: Path) -> None:
    root = tmp_path / "root" / "2026" / "05" / "15"
    root.mkdir(parents=True)
    rollout = root / "rollout-2026-05-15T10-00-00-sid.jsonl"
    rollout.write_text(
        '{"type":"session_meta","payload":{"id":"sid","model_provider":"openai"}}\n'
        '{"type":"turn_context","payload":{}}\n'
        '{"type":"response_item","payload":{"type":"message","content":"hi"}}\n'
    )
    rows = capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="sid",
        triggering_seq=1,
        sessions_root=tmp_path / "root",
    )
    assert rows == 0


def test_safe_capture_swallows_unexpected_error(
    tmp_path: Path, fake_codex_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thirdeye.platforms.codex.usage as mod

    monkeypatch.setattr(
        mod,
        "_extract_usage_row",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oops")),
    )
    result = capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="abc",
        triggering_seq=1,
        sessions_root=fake_codex_root,
    )
    assert result is None
    assert "RuntimeError" in usage_log_path(tmp_path).read_text()
