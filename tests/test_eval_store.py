from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.eval.result import EvalResult, Finding
from thirdeye.eval.store import EvalStore, iter_results_by_definition
from thirdeye.paths import session_dir
from thirdeye.store import Store


def _result(**overrides) -> EvalResult:
    base = dict(
        id="01J7XYZ",
        session_id="abc",
        definition="default",
        agent="claude",
        agent_model="claude-sonnet-4-6",
        agent_session_id="eval-sid",
        started_at="2026-05-16T01:42:00Z",
        ended_at="2026-05-16T01:42:18Z",
        duration_ms=18432,
        verdict="warn",
        summary="ok",
    )
    base.update(overrides)
    return EvalResult(**base)


@pytest.fixture
def session(tmp_path: Path) -> Path:
    sd = tmp_path / "traces" / "claude" / "abc"
    sd.mkdir(parents=True)
    return sd


# --- evals.jsonl ---


def test_append_and_iter(session: Path):
    store = EvalStore(session)
    store.append(_result(id="a"))
    store.append(_result(id="b"))
    ids = [r.id for r in store.iter_results()]
    assert ids == ["a", "b"]


def test_iter_handles_missing_file(session: Path):
    assert list(EvalStore(session).iter_results()) == []


def test_iter_skips_malformed_lines(session: Path):
    store = EvalStore(session)
    store.append(_result(id="a"))
    # Append a malformed line manually
    with store.jsonl_path.open("a") as f:
        f.write("this is not json\n")
    store.append(_result(id="b"))
    ids = [r.id for r in store.iter_results()]
    assert ids == ["a", "b"]


def test_latest_returns_most_recent(session: Path):
    store = EvalStore(session)
    store.append(_result(id="a"))
    store.append(_result(id="b"))
    assert store.latest().id == "b"


def test_latest_filtered_by_definition(session: Path):
    store = EvalStore(session)
    store.append(_result(id="a", definition="default"))
    store.append(_result(id="b", definition="token-efficiency"))
    store.append(_result(id="c", definition="default"))
    assert store.latest(definition="default").id == "c"
    assert store.latest(definition="token-efficiency").id == "b"


def test_latest_returns_none_when_empty(session: Path):
    assert EvalStore(session).latest() is None


def test_find_by_id(session: Path):
    store = EvalStore(session)
    store.append(_result(id="a"))
    store.append(_result(id="b"))
    assert store.find_by_id("a").id == "a"
    assert store.find_by_id("missing") is None


def test_append_creates_session_dir(tmp_path: Path):
    sd = tmp_path / "fresh" / "session"
    assert not sd.exists()
    EvalStore(sd).append(_result())
    assert sd.is_dir()


def test_append_preserves_findings(session: Path):
    store = EvalStore(session)
    f = Finding(seq=42, severity="warn", note="x")
    store.append(_result(findings=[f]))
    loaded = next(store.iter_results())
    assert loaded.findings == [f]


# --- job stubs ---


def test_write_read_job(session: Path):
    store = EvalStore(session)
    store.write_job("J1", {"status": "running", "pid": 123})
    assert store.read_job("J1") == {"status": "running", "pid": 123}


def test_read_missing_job_returns_none(session: Path):
    assert EvalStore(session).read_job("missing") is None


def test_remove_existing_job(session: Path):
    store = EvalStore(session)
    store.write_job("J1", {"status": "running"})
    assert store.remove_job("J1") is True
    assert store.read_job("J1") is None


def test_remove_missing_job_returns_false(session: Path):
    assert EvalStore(session).remove_job("ghost") is False


def test_iter_jobs(session: Path):
    store = EvalStore(session)
    store.write_job("J1", {"status": "running", "job_id": "J1"})
    store.write_job("J2", {"status": "failed", "job_id": "J2"})
    jobs = list(store.iter_jobs())
    assert {j["job_id"] for j in jobs} == {"J1", "J2"}


def test_iter_jobs_handles_missing_dir(session: Path):
    assert list(EvalStore(session).iter_jobs()) == []


def test_write_job_is_atomic(session: Path):
    store = EvalStore(session)
    store.write_job("J1", {"status": "running"})
    # No tmp leftover
    assert not (store.jobs_dir / "J1.json.tmp").exists()


def test_iter_jobs_skips_malformed(session: Path):
    store = EvalStore(session)
    store.write_job("J1", {"job_id": "J1"})
    # Corrupt one of the stubs
    (store.jobs_dir / "bad.json").write_text("{not json")
    jobs = list(store.iter_jobs())
    assert len(jobs) == 1 and jobs[0]["job_id"] == "J1"


# --- iter_results_by_definition ---


def _make_session(config: Config, platform: str, sid: str) -> Path:
    store = Store(config)
    with store.open_session(sid, platform=platform, cwd=f"/proj/{sid}") as w:
        w.append("user_message", "hi")
    return session_dir(config.root, platform, sid)


def test_iter_results_by_definition_across_sessions(tmp_path: Path):
    config = Config(root=tmp_path)
    sd_a = _make_session(config, "claude", "01J9G7XK4P")
    sd_b = _make_session(config, "cursor", "02ABCDEF12")

    EvalStore(sd_a).append(
        _result(id="a1", definition="default", started_at="2026-05-16T01:00:00Z")
    )
    EvalStore(sd_a).append(
        _result(id="a2", definition="token-efficiency", started_at="2026-05-16T02:00:00Z")
    )
    EvalStore(sd_b).append(
        _result(id="b1", definition="default", started_at="2026-05-16T03:00:00Z")
    )

    pairs = list(iter_results_by_definition(config, "default"))
    assert [r.id for _, r in pairs] == ["b1", "a1"]
    assert {m.session_id for m, _ in pairs} == {"01J9G7XK4P", "02ABCDEF12"}


def test_iter_results_by_definition_empty_when_no_match(tmp_path: Path):
    config = Config(root=tmp_path)
    sd = _make_session(config, "claude", "01J9G7XK4P")
    EvalStore(sd).append(_result(id="a", definition="default"))

    assert list(iter_results_by_definition(config, "nonexistent")) == []


def test_iter_results_by_definition_is_case_sensitive(tmp_path: Path):
    config = Config(root=tmp_path)
    sd = _make_session(config, "claude", "01J9G7XK4P")
    EvalStore(sd).append(_result(id="a", definition="Default"))
    EvalStore(sd).append(_result(id="b", definition="default"))

    pairs = list(iter_results_by_definition(config, "default"))
    assert [r.id for _, r in pairs] == ["b"]


def test_iter_results_by_definition_ordered_desc(tmp_path: Path):
    config = Config(root=tmp_path)
    sd = _make_session(config, "claude", "01J9G7XK4P")
    EvalStore(sd).append(_result(id="old", definition="default", started_at="2026-05-01T00:00:00Z"))
    EvalStore(sd).append(_result(id="new", definition="default", started_at="2026-05-20T00:00:00Z"))
    EvalStore(sd).append(_result(id="mid", definition="default", started_at="2026-05-10T00:00:00Z"))

    pairs = list(iter_results_by_definition(config, "default"))
    assert [r.id for _, r in pairs] == ["new", "mid", "old"]
