from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.paths import (
    session_dir,
    usage_jsonl_path,
    usage_log_path,
    usage_state_path,
)
from thirdeye.platforms.claude.usage import capture_usage_claude

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "claude_transcript.jsonl"


def test_capture_creates_rows_from_fixture(tmp_path: Path) -> None:
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc123",
        transcript_path=str(FIXTURE),
        triggering_seq=5,
    )
    assert rows == 2
    jsonl = usage_jsonl_path(session_dir(tmp_path, "claude", "abc123"))
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["platform"] == "claude"
    assert first["seq"] == 5
    assert first["model"]
    assert first["input_tokens"] == 12450
    assert first["output_tokens"] == 187
    assert first["total_tokens"] == 12637


def test_capture_is_incremental(tmp_path: Path) -> None:
    """Re-running against the same transcript at the saved offset produces 0 new rows."""
    capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(FIXTURE),
        triggering_seq=1,
    )
    sd = session_dir(tmp_path, "claude", "abc")
    state = json.loads(usage_state_path(sd).read_text())
    initial_offset = state["transcript_offset"]
    assert initial_offset > 0

    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(FIXTURE),
        triggering_seq=2,
    )
    assert rows == 0
    state2 = json.loads(usage_state_path(sd).read_text())
    assert state2["transcript_offset"] == initial_offset


def test_capture_missing_transcript_logs_error(tmp_path: Path) -> None:
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path="/nonexistent/path.jsonl",
        triggering_seq=1,
    )
    assert rows == 0
    log = usage_log_path(tmp_path)
    assert log.exists() and "open_source" in log.read_text()


def test_capture_with_no_transcript_path(tmp_path: Path) -> None:
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=None,
        triggering_seq=1,
    )
    assert rows == 0
    sd = session_dir(tmp_path, "claude", "abc")
    assert not usage_jsonl_path(sd).exists()


def test_capture_handles_corrupt_jsonl_lines(tmp_path: Path) -> None:
    transcript = tmp_path / "bad.jsonl"
    transcript.write_text(
        '{"type":"assistant","message":{"model":"claude-3","usage":{"input_tokens":10,"output_tokens":5}}}\n'
        "this is not json\n"
        '{"type":"assistant","message":{"model":"claude-3","usage":{"input_tokens":7,"output_tokens":3}}}\n'
    )
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=10,
    )
    assert rows == 2


def test_capture_skips_non_assistant_frames(tmp_path: Path) -> None:
    transcript = tmp_path / "mixed.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"hi"}}\n'
        '{"type":"assistant","message":{"model":"claude-3","usage":{"input_tokens":10,"output_tokens":5}}}\n'
        '{"type":"meta"}\n'
    )
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=1,
    )
    assert rows == 1


def test_safe_capture_swallows_unexpected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thirdeye.platforms.claude.usage as mod

    monkeypatch.setattr(
        mod, "_extract_row", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oops"))
    )
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"assistant","message":{"model":"c","usage":{"input_tokens":1,"output_tokens":1}}}\n'
    )
    result = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=1,
    )
    assert result is None
    assert "RuntimeError" in usage_log_path(tmp_path).read_text()


def test_capture_flat_shape(tmp_path: Path) -> None:
    """A frame without `message` wrapper but with `usage`/`model` at root is captured."""
    transcript = tmp_path / "flat.jsonl"
    transcript.write_text(
        '{"model":"claude-haiku","usage":{"input_tokens":50,"output_tokens":25},"timestamp":"2026-05-15T01:00:00Z"}\n'
    )
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=7,
    )
    assert rows == 1
    sd = session_dir(tmp_path, "claude", "abc")
    line = json.loads(usage_jsonl_path(sd).read_text().strip().splitlines()[0])
    assert line["model"] == "claude-haiku"
    assert line["input_tokens"] == 50
    assert line["output_tokens"] == 25
    assert line["total_tokens"] == 75
    assert line["ts"] == "2026-05-15T01:00:00Z"


def test_capture_skips_frame_with_missing_model(tmp_path: Path) -> None:
    transcript = tmp_path / "nomodel.jsonl"
    transcript.write_text(
        '{"type":"assistant","message":{"usage":{"input_tokens":10,"output_tokens":5}}}\n'
    )
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=1,
    )
    assert rows == 0


def test_capture_skips_frame_with_missing_token_field(tmp_path: Path) -> None:
    transcript = tmp_path / "partial.jsonl"
    transcript.write_text(
        '{"type":"assistant","message":{"model":"c","usage":{"input_tokens":10}}}\n'
        '{"type":"assistant","message":{"model":"c","usage":{"output_tokens":5}}}\n'
    )
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=1,
    )
    assert rows == 0


def test_capture_appends_only_new_rows_on_growth(tmp_path: Path) -> None:
    """When transcript grows between calls, only the appended portion is processed."""
    transcript = tmp_path / "grow.jsonl"
    transcript.write_text(
        '{"type":"assistant","message":{"model":"c","usage":{"input_tokens":10,"output_tokens":5}}}\n'
    )
    rows1 = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=1,
    )
    assert rows1 == 1

    with transcript.open("a") as f:
        f.write(
            '{"type":"assistant","message":{"model":"c","usage":{"input_tokens":20,"output_tokens":7}}}\n'
            '{"type":"assistant","message":{"model":"c","usage":{"input_tokens":30,"output_tokens":3}}}\n'
        )

    rows2 = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=2,
    )
    assert rows2 == 2

    sd = session_dir(tmp_path, "claude", "abc")
    lines = usage_jsonl_path(sd).read_text().strip().splitlines()
    assert len(lines) == 3
    second = json.loads(lines[1])
    third = json.loads(lines[2])
    assert second["seq"] == 2 and second["input_tokens"] == 20
    assert third["seq"] == 2 and third["input_tokens"] == 30


def test_capture_advances_offset_with_no_rows(tmp_path: Path) -> None:
    """Even when no assistant frames are found, the offset must advance past the read bytes."""
    transcript = tmp_path / "user_only.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"hi"}}\n{"type":"meta"}\n'
    )
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=3,
    )
    assert rows == 0
    sd = session_dir(tmp_path, "claude", "abc")
    state = json.loads(usage_state_path(sd).read_text())
    assert state["transcript_offset"] == transcript.stat().st_size
    # last_seq should remain at the default (-1) since no rows were appended.
    assert state["last_seq"] == -1


def test_capture_updates_last_seq_when_rows_appended(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"assistant","message":{"model":"c","usage":{"input_tokens":1,"output_tokens":1}}}\n'
    )
    capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=42,
    )
    sd = session_dir(tmp_path, "claude", "abc")
    state = json.loads(usage_state_path(sd).read_text())
    assert state["last_seq"] == 42


def test_stop_hook_invokes_capture_and_survives_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop hook must call capture_usage_claude with the unstripped transcript_path
    and must not raise even if capture errors internally."""
    from thirdeye.config import Config
    from thirdeye.platforms.claude import hooks
    from thirdeye.platforms.claude import usage as usage_mod

    home = tmp_path / "thirdeye"
    home.mkdir()
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(root=home)))

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"assistant","message":{"model":"claude-3","usage":{"input_tokens":4,"output_tokens":2}}}\n'
    )

    payload = {
        "session_id": "hooksid",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
        "extra": "kept-in-event",
    }
    monkeypatch.setattr(hooks, "_read_stdin", lambda: payload)

    captured: dict = {}
    original = usage_mod.capture_usage_claude

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(
        hooks,
        "_strip_payload",
        lambda p: {k: v for k, v in p.items() if k not in {"session_id", "cwd", "transcript_path"}},
    )
    # Patch where the function is looked up: it's imported inside `stop`.
    monkeypatch.setattr("thirdeye.platforms.claude.usage.capture_usage_claude", spy)

    hooks.stop()

    assert captured["session_id"] == "hooksid"
    assert captured["transcript_path"] == str(transcript)
    assert isinstance(captured["triggering_seq"], int)
    # Capture actually wrote a sidecar row
    sd = session_dir(home, "claude", "hooksid")
    assert usage_jsonl_path(sd).exists()


def test_stop_hook_with_no_session_id_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thirdeye.platforms.claude import hooks

    monkeypatch.setattr(hooks, "_read_stdin", lambda: {})
    called = {"count": 0}

    def fake(**kwargs):
        called["count"] += 1

    monkeypatch.setattr("thirdeye.platforms.claude.usage.capture_usage_claude", fake)
    # Should return without touching capture or raising
    hooks.stop()
    assert called["count"] == 0


def test_stop_hook_survives_capture_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If capture_usage_claude were ever to raise, the stop hook should still exit cleanly.
    (In production it's @safe_capture wrapped, so it can't — but the hook itself shouldn't
    add another try/except, and this guards that the wrapper is actually in place.)"""
    from thirdeye.config import Config
    from thirdeye.platforms.claude import hooks
    from thirdeye.platforms.claude import usage as usage_mod

    home = tmp_path / "thirdeye"
    home.mkdir()
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(root=home)))

    payload = {
        "session_id": "boom",
        "cwd": str(tmp_path),
        "transcript_path": "/does/not/exist.jsonl",
    }
    monkeypatch.setattr(hooks, "_read_stdin", lambda: payload)

    # Force the inner extractor to blow up; @safe_capture should swallow it.
    monkeypatch.setattr(
        usage_mod,
        "_extract_row",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )

    # Must not raise
    hooks.stop()
