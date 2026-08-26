from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from thirdeye.commands.eval import eval_group
from thirdeye.eval.result import EvalResult
from thirdeye.eval.store import EvalStore
from thirdeye.paths import eval_def_path, session_dir


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    sd = session_dir(tmp_path, "claude", "abc123")
    sd.mkdir(parents=True)
    (sd / "meta.yaml").write_text(
        "schema_version: 2\nsession_id: abc123\nplatform: claude\n"
        "cwd: /x\nstarted_at: 2026-05-16T00:00:00Z\nended_at: null\n"
        "status: open\nevent_count: 1\nlast_seq: -1\nlast_ts: null\n"
    )
    # Custom definition
    cfg = eval_def_path(tmp_path, "test")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "name: test\ndescription: t\ndirective: |\n  evaluate\n"
        "default_agent: claude\noutput_schema: v1\n"
    )
    return tmp_path


def _seed_result(home: Path, sid: str, **overrides) -> EvalResult:
    base = dict(
        id="01J7XYZ",
        session_id=sid,
        definition="test",
        agent="claude",
        agent_model="",
        agent_session_id=None,
        started_at="2026-05-16T01:42:00Z",
        ended_at="2026-05-16T01:42:18Z",
        duration_ms=18000,
        verdict="warn",
        summary="seeded summary",
    )
    base.update(overrides)
    r = EvalResult(**base)
    EvalStore(session_dir(home, "claude", sid)).append(r)
    return r


def test_run_dispatch_invokes_runner(home: Path, monkeypatch):
    captured = {}

    def fake_run_eval(**kw):
        captured.update(kw)
        return EvalResult(
            id="X",
            session_id="abc123",
            definition="test",
            agent="claude",
            agent_model="",
            agent_session_id=None,
            started_at="t",
            ended_at="t",
            duration_ms=10,
            verdict="pass",
            summary="great",
        )

    monkeypatch.setattr("thirdeye.commands.eval.run_eval", fake_run_eval)
    result = CliRunner().invoke(
        eval_group, ["run", "abc", "--agent", "claude", "--using", "test"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "VERDICT: pass" in result.output
    assert captured["definition_name"] == "test"


def test_run_rejects_unknown_agent(home: Path):
    result = CliRunner().invoke(eval_group, ["run", "abc", "--agent", "not-real"])
    assert result.exit_code != 0
    assert "unknown agent" in result.output


def test_run_background_prints_job_id(home: Path, monkeypatch):
    monkeypatch.setattr("thirdeye.commands.eval.run_eval_background", lambda **kw: "01J7BG")
    result = CliRunner().invoke(
        eval_group,
        ["run", "abc", "--agent", "claude", "--using", "test", "--background"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "01J7BG" in result.output


def test_show_latest(home: Path):
    _seed_result(home, "abc123", summary="latest one")
    result = CliRunner().invoke(eval_group, ["show", "abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "latest one" in result.output


def test_show_by_id(home: Path):
    _seed_result(home, "abc123", id="ID-A", summary="first")
    _seed_result(home, "abc123", id="ID-B", summary="second")
    result = CliRunner().invoke(eval_group, ["show", "abc", "--id", "ID-A"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "first" in result.output
    assert "second" not in result.output


def test_show_no_results_errors(home: Path):
    result = CliRunner().invoke(eval_group, ["show", "abc"])
    assert result.exit_code != 0
    assert "no eval results" in result.output


def test_list_filters(home: Path):
    _seed_result(home, "abc123", id="A", verdict="pass", definition="test")
    _seed_result(home, "abc123", id="B", verdict="fail", definition="test")
    _seed_result(home, "abc123", id="C", verdict="pass", definition="other")
    result = CliRunner().invoke(
        eval_group,
        ["list", "--verdict", "pass", "--using", "test", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    ids = [json.loads(line)["id"] for line in result.output.splitlines() if line.strip()]
    assert ids == ["A"]


def test_status_no_jobs(home: Path):
    result = CliRunner().invoke(eval_group, ["status", "abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No background eval jobs" in result.output


def test_status_orphan_detection(home: Path):
    sd = session_dir(home, "claude", "abc123")
    EvalStore(sd).write_job(
        "J1",
        {
            "job_id": "J1",
            "session_id": "abc123",
            "using": "test",
            "agent": "claude",
            "status": "running",
            "started_at": "t",
            "pid": 99999999,  # almost certainly not running
        },
    )
    result = CliRunner().invoke(eval_group, ["status", "abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "orphaned" in result.output


def test_def_list_shows_shipped(home: Path):
    result = CliRunner().invoke(eval_group, ["def", "list"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "default" in result.output
    assert "(shipped)" in result.output


def test_def_show_default(home: Path):
    result = CliRunner().invoke(eval_group, ["def", "show", "default"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "name: default" in result.output


def test_def_create_with_directive(home: Path):
    result = CliRunner().invoke(
        eval_group,
        ["def", "create", "my-eval", "--directive", "evaluate X", "--description", "custom"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "Created" in result.output
    # Verify it loads
    result2 = CliRunner().invoke(eval_group, ["def", "show", "my-eval"], catch_exceptions=False)
    assert "evaluate X" in result2.output


def test_def_create_rejects_no_source(home: Path):
    result = CliRunner().invoke(eval_group, ["def", "create", "bad"])
    assert result.exit_code != 0
    assert "exactly one of" in result.output


def test_def_create_rejects_multiple_sources(home: Path):
    result = CliRunner().invoke(
        eval_group,
        ["def", "create", "bad", "--directive", "a", "--from", "default"],
    )
    assert result.exit_code != 0


def test_def_rm_shipped_allows_restore(home: Path):
    # Materialize shipped 'default'
    CliRunner().invoke(eval_group, ["def", "show", "default"], catch_exceptions=False)
    result = CliRunner().invoke(eval_group, ["def", "rm", "default"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "shipped version restored" in result.output


def test_def_rm_missing_errors(home: Path):
    result = CliRunner().invoke(eval_group, ["def", "rm", "nonexistent"])
    assert result.exit_code != 0


# --- _run-worker background entry point ---


def test_run_worker_success_removes_stub_and_appends_result(home: Path, monkeypatch):
    """On successful eval, the worker removes the job stub and the result lives
    in evals.jsonl (this happens because run_eval(save=True) appends)."""
    sd = session_dir(home, "claude", "abc123")
    EvalStore(sd).write_job(
        "JOB1",
        {
            "job_id": "JOB1",
            "session_id": "abc123",
            "using": "test",
            "agent": "claude",
            "platform": "claude",
            "status": "running",
            "started_at": "t",
            "pid": 12345,
        },
    )

    def fake_run_eval(**kw):
        # Mimic what run_eval normally does on save=True: append a result.
        EvalStore(session_dir(kw["thirdeye_home"], kw["platform"], kw["session_id"])).append(
            EvalResult(
                id="01J",
                session_id=kw["session_id"],
                definition=kw["definition_name"],
                agent=kw["agent_name"],
                agent_model="",
                agent_session_id=None,
                started_at="t",
                ended_at="t",
                duration_ms=10,
                verdict="pass",
                summary="ok",
            )
        )
        return None

    monkeypatch.setattr("thirdeye.commands.eval.run_eval", fake_run_eval)

    result = CliRunner().invoke(
        eval_group,
        ["_run-worker", "JOB1", "claude", "abc123", "test", "claude"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # Stub gone, result persisted
    assert EvalStore(sd).read_job("JOB1") is None
    persisted = list(EvalStore(sd).iter_results())
    assert len(persisted) == 1 and persisted[0].verdict == "pass"


def test_run_worker_failure_updates_stub_to_failed(home: Path, monkeypatch):
    sd = session_dir(home, "claude", "abc123")
    EvalStore(sd).write_job(
        "JOB2",
        {
            "job_id": "JOB2",
            "session_id": "abc123",
            "using": "test",
            "agent": "claude",
            "platform": "claude",
            "status": "running",
            "started_at": "t",
            "pid": 12345,
        },
    )

    def fake_run_eval(**kw):
        raise RuntimeError("agent crashed")

    monkeypatch.setattr("thirdeye.commands.eval.run_eval", fake_run_eval)

    result = CliRunner().invoke(
        eval_group,
        ["_run-worker", "JOB2", "claude", "abc123", "test", "claude"],
    )
    # Worker re-raises after writing failed stub — non-zero exit
    assert result.exit_code != 0
    stub = EvalStore(sd).read_job("JOB2")
    assert stub is not None
    assert stub["status"] == "failed"
    assert "RuntimeError" in stub["error"]
    assert "agent crashed" in stub["error"]


# --- run --json + RuntimeError wrapping ---


def test_run_json_output(home: Path, monkeypatch):
    def fake_run_eval(**kw):
        return EvalResult(
            id="X",
            session_id="abc123",
            definition="test",
            agent="claude",
            agent_model="m",
            agent_session_id=None,
            started_at="t",
            ended_at="t",
            duration_ms=42,
            verdict="warn",
            summary="meh",
            scores={"overall": 6.0},
        )

    monkeypatch.setattr("thirdeye.commands.eval.run_eval", fake_run_eval)
    result = CliRunner().invoke(
        eval_group,
        ["run", "abc", "--agent", "claude", "--using", "test", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    line = json.loads(result.output.strip())
    assert line["verdict"] == "warn"
    assert line["scores"] == {"overall": 6.0}


def test_run_runtime_error_becomes_click_exception(home: Path, monkeypatch):
    def boom(**kw):
        raise RuntimeError("agent 'claude' exited 1: stderr tail")

    monkeypatch.setattr("thirdeye.commands.eval.run_eval", boom)
    result = CliRunner().invoke(
        eval_group,
        ["run", "abc", "--agent", "claude", "--using", "test"],
    )
    assert result.exit_code != 0
    # Clean ClickException output, no traceback
    assert "Traceback" not in result.output
    assert "exited 1" in result.output


# --- def edit (materialization side effect) ---


def test_def_edit_materializes_shipped_then_invokes_editor(home: Path, monkeypatch):
    """`def edit default` should ensure the user copy exists before opening EDITOR."""
    user_path = eval_def_path(home, "default")
    assert not user_path.exists()

    edited = {}

    def fake_edit(*, filename):
        edited["filename"] = filename

    monkeypatch.setattr("click.edit", fake_edit)

    result = CliRunner().invoke(eval_group, ["def", "edit", "default"], catch_exceptions=False)
    assert result.exit_code == 0
    assert user_path.exists()
    assert edited["filename"] == str(user_path)


def test_def_edit_unknown_errors(home: Path):
    result = CliRunner().invoke(eval_group, ["def", "edit", "nope"])
    assert result.exit_code != 0
    assert "no eval definition" in result.output


# --- def create: --directive-file, --from, --force ---


def test_def_create_from_directive_file(home: Path, tmp_path: Path):
    df = tmp_path / "rubric.md"
    df.write_text("custom rubric body from file")
    result = CliRunner().invoke(
        eval_group,
        ["def", "create", "from-file", "--directive-file", str(df)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    shown = CliRunner().invoke(eval_group, ["def", "show", "from-file"], catch_exceptions=False)
    assert "custom rubric body from file" in shown.output


def test_def_create_from_existing_def(home: Path):
    """--from copies the directive of an existing definition."""
    # Materialize 'default' so we can copy from it
    CliRunner().invoke(eval_group, ["def", "show", "default"], catch_exceptions=False)
    result = CliRunner().invoke(
        eval_group,
        ["def", "create", "default-copy", "--from", "default"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    shown = CliRunner().invoke(eval_group, ["def", "show", "default-copy"], catch_exceptions=False)
    # The copied directive should contain unique text from the shipped default
    assert "Token efficiency" in shown.output or "evaluate" in shown.output.lower()


def test_def_create_from_unknown_errors(home: Path):
    result = CliRunner().invoke(
        eval_group,
        ["def", "create", "bad", "--from", "nonexistent"],
    )
    assert result.exit_code != 0


def test_def_create_existing_without_force_errors(home: Path):
    CliRunner().invoke(
        eval_group, ["def", "create", "dup", "--directive", "first"], catch_exceptions=False
    )
    result = CliRunner().invoke(eval_group, ["def", "create", "dup", "--directive", "second"])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_def_create_force_overwrites(home: Path):
    CliRunner().invoke(
        eval_group,
        ["def", "create", "dup2", "--directive", "first-version"],
        catch_exceptions=False,
    )
    result = CliRunner().invoke(
        eval_group,
        ["def", "create", "dup2", "--directive", "second-version", "--force"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    shown = CliRunner().invoke(eval_group, ["def", "show", "dup2"], catch_exceptions=False)
    assert "second-version" in shown.output
    assert "first-version" not in shown.output


def test_def_show_user_created(home: Path):
    """def show should display a user-created definition (not just shipped)."""
    CliRunner().invoke(
        eval_group, ["def", "create", "mine", "--directive", "my body"], catch_exceptions=False
    )
    result = CliRunner().invoke(eval_group, ["def", "show", "mine"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "name: mine" in result.output
    assert "my body" in result.output


# --- list: time-window + agent filters + global scope ---


def test_list_no_prefix_iterates_all_sessions(home: Path):
    _seed_result(home, "abc123", id="A")
    # Create a second session with its own meta + result
    sd2 = session_dir(home, "claude", "def456")
    sd2.mkdir(parents=True)
    (sd2 / "meta.yaml").write_text(
        "schema_version: 2\nsession_id: def456\nplatform: claude\n"
        "cwd: /y\nstarted_at: 2026-05-15T00:00:00Z\nended_at: null\n"
        "status: open\nevent_count: 1\nlast_seq: -1\nlast_ts: null\n"
    )
    EvalStore(sd2).append(
        EvalResult(
            id="B",
            session_id="def456",
            definition="test",
            agent="claude",
            agent_model="",
            agent_session_id=None,
            started_at="2026-05-15T00:00:00Z",
            ended_at="t",
            duration_ms=0,
            verdict="pass",
            summary="",
        )
    )
    result = CliRunner().invoke(eval_group, ["list", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    ids = {json.loads(l)["id"] for l in result.output.splitlines() if l.strip()}
    assert ids == {"A", "B"}


def test_list_filter_by_agent(home: Path):
    _seed_result(home, "abc123", id="A", agent="claude")
    _seed_result(home, "abc123", id="B", agent="codex")
    result = CliRunner().invoke(
        eval_group,
        ["list", "abc", "--agent", "codex", "--json"],
        catch_exceptions=False,
    )
    ids = [json.loads(l)["id"] for l in result.output.splitlines() if l.strip()]
    assert ids == ["B"]


def test_list_filter_by_time_window(home: Path):
    _seed_result(home, "abc123", id="A", started_at="2026-05-01T00:00:00Z")
    _seed_result(home, "abc123", id="B", started_at="2026-05-10T00:00:00Z")
    _seed_result(home, "abc123", id="C", started_at="2026-05-20T00:00:00Z")
    result = CliRunner().invoke(
        eval_group,
        ["list", "abc", "--since", "2026-05-05", "--until", "2026-05-15", "--json"],
        catch_exceptions=False,
    )
    ids = {json.loads(l)["id"] for l in result.output.splitlines() if l.strip()}
    assert ids == {"B"}


def test_list_empty_returns_clean_message(home: Path):
    result = CliRunner().invoke(eval_group, ["list", "abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No eval results" in result.output


def test_list_default_table_includes_overall(home: Path):
    _seed_result(home, "abc123", scores={"overall": 7.5})
    result = CliRunner().invoke(eval_group, ["list", "abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "7.5/10" in result.output


# --- show --using ---


def test_show_filtered_by_using_picks_latest_of_definition(home: Path):
    _seed_result(home, "abc123", id="A", definition="default", summary="default-1")
    _seed_result(home, "abc123", id="B", definition="token-efficiency", summary="token-1")
    _seed_result(home, "abc123", id="C", definition="default", summary="default-2")
    result = CliRunner().invoke(
        eval_group,
        ["show", "abc", "--using", "default"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # Latest of 'default' is the third seeded → summary "default-2"
    assert "default-2" in result.output
    assert "token-1" not in result.output


def test_show_unknown_id_errors(home: Path):
    _seed_result(home, "abc123")
    result = CliRunner().invoke(eval_group, ["show", "abc", "--id", "NOPE"])
    assert result.exit_code != 0
    assert "no eval result" in result.output


# --- status global + failed status ---


def test_status_no_prefix_iterates_sessions(home: Path):
    sd = session_dir(home, "claude", "abc123")
    EvalStore(sd).write_job(
        "J1",
        {
            "job_id": "J1",
            "session_id": "abc123",
            "using": "test",
            "agent": "claude",
            "status": "running",
            "started_at": "t",
            "pid": 99999999,
        },
    )
    result = CliRunner().invoke(eval_group, ["status"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "J1" in result.output


def test_status_failed_job_rendered(home: Path):
    sd = session_dir(home, "claude", "abc123")
    EvalStore(sd).write_job(
        "J-FAIL",
        {
            "job_id": "J-FAIL",
            "session_id": "abc123",
            "using": "test",
            "agent": "claude",
            "status": "failed",
            "started_at": "t",
            "pid": 1,
            "error": "RuntimeError: agent exited 1",
        },
    )
    result = CliRunner().invoke(eval_group, ["status", "abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "failed" in result.output
    assert "J-FAIL" in result.output


def test_status_json_output(home: Path):
    sd = session_dir(home, "claude", "abc123")
    EvalStore(sd).write_job(
        "JJ",
        {
            "job_id": "JJ",
            "session_id": "abc123",
            "using": "test",
            "agent": "claude",
            "status": "running",
            "started_at": "t",
            "pid": 99999999,
        },
    )
    result = CliRunner().invoke(eval_group, ["status", "abc", "--json"], catch_exceptions=False)
    lines = [json.loads(l) for l in result.output.splitlines() if l.strip()]
    assert len(lines) == 1
    # pid-not-alive upgrades status to orphaned
    assert lines[0]["status"] == "orphaned"


# --- batch ---


def _make_session(home: Path, sid: str) -> None:
    sd = session_dir(home, "claude", sid)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "meta.yaml").write_text(
        "schema_version: 2\n"
        f"session_id: {sid}\n"
        "platform: claude\n"
        "cwd: /x\nstarted_at: 2026-05-16T00:00:00Z\nended_at: null\n"
        "status: open\nevent_count: 1\nlast_seq: -1\nlast_ts: null\n"
    )


def test_batch_dispatches_valid_and_skips_invalid(home: Path, monkeypatch):
    _make_session(home, "sess0001")
    _make_session(home, "sess0002")

    calls: list[dict] = []

    def fake_run_eval_background(**kw):
        calls.append(kw)
        return "01J7" + kw["session_id"][:4].upper()

    monkeypatch.setattr("thirdeye.commands.eval.run_eval_background", fake_run_eval_background)
    result = CliRunner().invoke(
        eval_group,
        ["batch", "test", "sess0001", "missing", "sess0002", "--agent", "claude"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Two dispatches happened (one per resolvable prefix)
    assert len(calls) == 2
    dispatched_lines = [line for line in result.output.splitlines() if "dispatched" in line]
    assert len(dispatched_lines) == 2
    assert "sess0001\t01J7SESS\tdispatched" in result.output
    assert "sess0002\t01J7SESS\tdispatched" in result.output
    # Invalid prefix surfaces a skip message on stderr (mixed into output here)
    assert "skip missing" in result.output


def test_batch_caps_at_25_sessions(home: Path, monkeypatch):
    monkeypatch.setattr("thirdeye.commands.eval.run_eval_background", lambda **kw: "RUN")
    prefixes = [f"s{i:04d}" for i in range(26)]
    result = CliRunner().invoke(
        eval_group,
        ["batch", "test", *prefixes, "--agent", "claude"],
    )
    assert result.exit_code != 0
    assert "capped at 25" in result.output


# --- results ---


def test_results_table_lists_runs_desc(home: Path):
    _make_session(home, "sess0001")
    _make_session(home, "sess0002")
    _seed_result(
        home,
        "sess0001",
        id="01JOLDXXX",
        definition="test",
        agent="claude",
        verdict="pass",
        started_at="2026-05-15T01:00:00Z",
    )
    _seed_result(
        home,
        "sess0002",
        id="01JNEWXXX",
        definition="test",
        agent="claude",
        verdict="warn",
        started_at="2026-05-20T01:00:00Z",
    )
    # A result on a different definition that must not appear
    _seed_result(
        home,
        "sess0001",
        id="01JOTHER",
        definition="other",
        agent="claude",
        verdict="fail",
        started_at="2026-05-25T01:00:00Z",
    )
    result = CliRunner().invoke(eval_group, ["results", "--def", "test"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 2
    # DESC by started_at — sess0002 (newer) first
    assert lines[0].startswith("sess0002\t")
    assert lines[1].startswith("sess0001\t")
    # Tab-separated, run_id truncated to 8 chars
    parts = lines[0].split("\t")
    assert parts == ["sess0002", "01JNEWXX", "claude", "warn", "2026-05-20T01:00:00Z"]


def test_results_as_json(home: Path):
    _make_session(home, "sess0001")
    _seed_result(
        home,
        "sess0001",
        id="01JNEWXXX",
        definition="test",
        agent="claude",
        verdict="pass",
        started_at="2026-05-20T01:00:00Z",
        duration_ms=4242,
    )
    result = CliRunner().invoke(
        eval_group,
        ["results", "--def", "test", "--as-json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    lines = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0] == {
        "session_id": "sess0001",
        "platform": "claude",
        "run_id": "01JNEWXXX",
        "agent": "claude",
        "verdict": "pass",
        "started_at": "2026-05-20T01:00:00Z",
        "duration_ms": 4242,
    }
