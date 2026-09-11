from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from thirdeye.commands.reads import events as events_cmd
from thirdeye.eval.result import EvalResult, Finding
from thirdeye.eval.store import EvalStore
from thirdeye.paths import session_dir


@pytest.fixture
def home_with_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real session with a couple of recorded events, plus seeded findings."""
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    from thirdeye.config import Config
    from thirdeye.store import Store

    config = Config(root=tmp_path)
    store = Store(config)
    store.append_event(
        session_id="abc", platform="claude", cwd="/x", t="user_message", data={"prompt": "hello"}
    )
    store.append_event(
        session_id="abc",
        platform="claude",
        cwd="/x",
        t="tool_call",
        data={"tool_name": "Edit", "tool_input": {"old_string": "x"}},
    )
    sd = session_dir(tmp_path, "claude", "abc")
    EvalStore(sd).append(
        EvalResult(
            id="01J",
            session_id="abc",
            definition="default",
            agent="claude",
            agent_model="",
            agent_session_id=None,
            started_at="t",
            ended_at="t",
            duration_ms=0,
            verdict="warn",
            summary="ok",
            findings=[
                Finding(seq=1, severity="warn", note="redundant edit", category="tools"),
                Finding(seq=None, severity="info", note="session-level note"),
            ],
        )
    )
    return tmp_path


def test_events_default_shows_findings(home_with_events: Path):
    result = CliRunner().invoke(events_cmd, ["abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "redundant edit" in result.output
    assert "session-level note" in result.output


def test_events_no_findings_flag_hides_them(home_with_events: Path):
    result = CliRunner().invoke(events_cmd, ["abc", "--no-findings"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "redundant edit" not in result.output
    assert "session-level note" not in result.output


def test_events_eval_filter(home_with_events: Path):
    sd = session_dir(home_with_events, "claude", "abc")
    EvalStore(sd).append(
        EvalResult(
            id="01K",
            session_id="abc",
            definition="token-efficiency",
            agent="claude",
            agent_model="",
            agent_session_id=None,
            started_at="t",
            ended_at="t",
            duration_ms=0,
            verdict="pass",
            summary="",
            findings=[Finding(seq=1, severity="info", note="cache hit")],
        )
    )
    result = CliRunner().invoke(events_cmd, ["abc", "--eval", "default"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "redundant edit" in result.output
    assert "cache hit" not in result.output


def test_events_session_level_findings_under_header(home_with_events: Path):
    result = CliRunner().invoke(events_cmd, ["abc"], catch_exceptions=False)
    assert "Session-level findings" in result.output
