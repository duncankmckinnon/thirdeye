from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from thirdeye.eval.runner import (
    AgentInvocation,
    run_eval,
    run_eval_background,
)
from thirdeye.eval.store import EvalStore
from thirdeye.paths import eval_def_path, session_dir


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a thirdeye home with a session and a custom definition."""
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    sd = session_dir(tmp_path, "claude", "abc")
    sd.mkdir(parents=True)
    (sd / "meta.yaml").write_text(
        "schema_version: 2\n"
        "session_id: abc\n"
        "platform: claude\n"
        "cwd: /x\n"
        "started_at: 2026-05-16T00:00:00Z\n"
        "ended_at: null\n"
        "status: open\n"
        "event_count: 1\n"
        "last_seq: -1\n"
        "last_ts: null\n"
    )
    cfg = eval_def_path(tmp_path, "test")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "name: test\n"
        "description: t\n"
        "directive: |\n  evaluate session\n"
        "default_agent: claude\n"
        "output_schema: v1\n"
    )
    return tmp_path


def _make_canned_output(envelope: dict, narrative: str = "narrative body") -> str:
    """Build the JSON wrapper Claude would return."""
    inner = "```json\n" + json.dumps(envelope) + "\n```\n\n" + narrative
    return json.dumps(
        {
            "result": inner,
            "cost_usd": {
                "input_tokens": 100,
                "output_tokens": 50,
                "usd": 0.001,
            },
        }
    )


def test_run_eval_persists_and_returns_result(home: Path, monkeypatch: pytest.MonkeyPatch):
    canned = _make_canned_output(
        {
            "verdict": "pass",
            "summary": "ok",
            "scores": {"overall": 8},
            "findings": [{"seq": 2, "severity": "info", "note": "fine"}],
        },
    )

    def fake_invoke(adapter, prompt, cwd):
        return AgentInvocation(stdout=canned, stderr="", returncode=0, duration_ms=42)

    monkeypatch.setattr("thirdeye.eval.runner._invoke_agent", fake_invoke)
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", lambda h, p: set())

    result = run_eval(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
    )
    assert result.verdict == "pass"
    assert result.summary == "ok"
    assert result.scores == {"overall": 8.0}
    assert len(result.findings) == 1 and result.findings[0].seq == 2
    sd = session_dir(home, "claude", "abc")
    persisted = list(EvalStore(sd).iter_results())
    assert len(persisted) == 1
    assert persisted[0].id == result.id


def test_run_eval_no_save_skips_persistence(home: Path, monkeypatch: pytest.MonkeyPatch):
    canned = _make_canned_output(
        {"verdict": "pass", "summary": "ok", "scores": {}, "findings": []},
    )
    monkeypatch.setattr(
        "thirdeye.eval.runner._invoke_agent",
        lambda a, p, c: AgentInvocation(canned, "", 0, 10),
    )
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", lambda h, p: set())

    run_eval(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
        save=False,
    )
    sd = session_dir(home, "claude", "abc")
    assert list(EvalStore(sd).iter_results()) == []


def test_run_eval_unknown_definition_raises(home: Path):
    with pytest.raises(FileNotFoundError, match="no eval definition"):
        run_eval(
            thirdeye_home=home,
            platform="claude",
            session_id="abc",
            definition_name="nope",
            agent_name="claude",
        )


def test_run_eval_unknown_agent_raises(home: Path):
    with pytest.raises(ValueError, match="unknown agent"):
        run_eval(
            thirdeye_home=home,
            platform="claude",
            session_id="abc",
            definition_name="test",
            agent_name="not-an-agent",
        )


def test_run_eval_missing_binary_raises(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("shutil.which", lambda x: None)
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", lambda h, p: set())
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        run_eval(
            thirdeye_home=home,
            platform="claude",
            session_id="abc",
            definition_name="test",
            agent_name="claude",
        )


def test_run_eval_nonzero_exit_raises(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "thirdeye.eval.runner._invoke_agent",
        lambda a, p, c: AgentInvocation("", "bad things", 1, 10),
    )
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", lambda h, p: set())
    with pytest.raises(RuntimeError, match="exited 1"):
        run_eval(
            thirdeye_home=home,
            platform="claude",
            session_id="abc",
            definition_name="test",
            agent_name="claude",
        )


def test_run_eval_no_envelope_yields_unknown_verdict(home: Path, monkeypatch: pytest.MonkeyPatch):
    bare = json.dumps({"result": "just text, no fenced json"})
    monkeypatch.setattr(
        "thirdeye.eval.runner._invoke_agent",
        lambda a, p, c: AgentInvocation(bare, "", 0, 10),
    )
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", lambda h, p: set())
    result = run_eval(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
    )
    assert result.verdict == "unknown"
    assert "just text" in result.markdown


def test_run_eval_links_agent_session_id_when_unambiguous(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canned = _make_canned_output(
        {"verdict": "pass", "summary": "ok", "scores": {}, "findings": []},
    )
    monkeypatch.setattr(
        "thirdeye.eval.runner._invoke_agent",
        lambda a, p, c: AgentInvocation(canned, "", 0, 10),
    )
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    calls = {"n": 0}

    def fake_list(h, p):
        calls["n"] += 1
        return {"existing"} if calls["n"] == 1 else {"existing", "new-eval-sid"}

    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", fake_list)

    result = run_eval(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
    )
    assert result.agent_session_id == "new-eval-sid"


def test_run_eval_agent_session_id_null_when_ambiguous(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canned = _make_canned_output(
        {"verdict": "pass", "summary": "ok", "scores": {}, "findings": []},
    )
    monkeypatch.setattr(
        "thirdeye.eval.runner._invoke_agent",
        lambda a, p, c: AgentInvocation(canned, "", 0, 10),
    )
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    calls = {"n": 0}

    def fake_list(h, p):
        calls["n"] += 1
        return set() if calls["n"] == 1 else {"a", "b"}

    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", fake_list)

    result = run_eval(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
    )
    assert result.agent_session_id is None


def test_run_eval_uses_provided_run_id(home: Path, monkeypatch: pytest.MonkeyPatch):
    canned = _make_canned_output(
        {"verdict": "pass", "summary": "ok", "scores": {}, "findings": []},
    )
    monkeypatch.setattr(
        "thirdeye.eval.runner._invoke_agent",
        lambda a, p, c: AgentInvocation(canned, "", 0, 10),
    )
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", lambda h, p: set())

    result = run_eval(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
        run_id="01XYZTESTRUNID0000000000AB",
    )
    assert result.id == "01XYZTESTRUNID0000000000AB"
    sd = session_dir(home, "claude", "abc")
    persisted = list(EvalStore(sd).iter_results())
    assert persisted[0].id == "01XYZTESTRUNID0000000000AB"


def test_run_eval_background_run_id_matches_worker_result_id(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The ULID returned by run_eval_background is the one the worker uses for the result."""
    recorded: dict = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    run_id = run_eval_background(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
        thirdeye_bin="/usr/bin/thirdeye",
    )
    assert recorded["cmd"][3] == run_id

    canned = _make_canned_output(
        {"verdict": "pass", "summary": "ok", "scores": {}, "findings": []},
    )
    monkeypatch.setattr(
        "thirdeye.eval.runner._invoke_agent",
        lambda a, p, c: AgentInvocation(canned, "", 0, 10),
    )
    monkeypatch.setattr("thirdeye.eval.runner._read_timeline", lambda *a, **k: [])
    monkeypatch.setattr("thirdeye.eval.runner._list_session_ids_on_platform", lambda h, p: set())

    result = run_eval(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
        run_id=run_id,
    )
    assert result.id == run_id


def test_run_eval_background_writes_stub_and_returns_job_id(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Background launch writes a job stub immediately and returns the id."""
    recorded: dict = {}

    class FakeProc:
        pid = 99999

    def fake_popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    job_id = run_eval_background(
        thirdeye_home=home,
        platform="claude",
        session_id="abc",
        definition_name="test",
        agent_name="claude",
        thirdeye_bin="/usr/bin/thirdeye",
    )
    assert len(job_id) == 26
    sd = session_dir(home, "claude", "abc")
    job = EvalStore(sd).read_job(job_id)
    assert job["status"] == "running"
    assert job["agent"] == "claude"
    assert job["pid"] == 99999
    assert recorded["cmd"][1:5] == ["eval", "_run-worker", job_id, "claude"]
    assert recorded["kwargs"].get("start_new_session") is True
