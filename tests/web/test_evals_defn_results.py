from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def _make_result(*, eval_id: str, sid: str, started_at: str, definition: str = "myeval"):
    from thirdeye.eval.result import EvalResult

    return EvalResult(
        id=eval_id,
        session_id=sid,
        definition=definition,
        agent="claude",
        agent_model="",
        agent_session_id=None,
        started_at=started_at,
        ended_at=started_at,
        duration_ms=1000,
        verdict="pass",
        summary="ok",
        scores={},
        findings=[],
    )


def test_defn_results_empty_returns_200(client):
    r = client.get("/evals/defs/never-run/results")
    assert r.status_code == 200
    assert b"No runs" in r.content


def test_defn_results_missing_definition_returns_200_empty(client):
    r = client.get("/evals/defs/zz-does-not-exist/results")
    assert r.status_code == 200
    assert b"No runs" in r.content


def test_defn_results_lists_runs_across_sessions(client, web_store, web_config):
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid_a = "01J9DEFNA01"
    sid_b = "01J9DEFNB01"
    for sid in (sid_a, sid_b):
        with web_store.open_session(sid, platform="claude", cwd="/p") as w:
            w.append("user_message", {"prompt": "x"})
        web_store.close_session(sid, platform="claude")

    sdir_a = session_dir(web_config.root, "claude", sid_a)
    sdir_b = session_dir(web_config.root, "claude", sid_b)
    EvalStore(sdir_a).append(
        _make_result(eval_id="01RUN001A", sid=sid_a, started_at="2026-05-25T01:00:00Z")
    )
    EvalStore(sdir_a).append(
        _make_result(eval_id="01RUN002A", sid=sid_a, started_at="2026-05-25T02:00:00Z")
    )
    EvalStore(sdir_b).append(
        _make_result(eval_id="01RUN001B", sid=sid_b, started_at="2026-05-25T03:00:00Z")
    )

    r = client.get("/evals/defs/myeval/results")
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    for run_id in ("01RUN001A", "01RUN002A", "01RUN001B"):
        assert run_id in body
    pos_a1 = body.find("01RUN001A")
    pos_a2 = body.find("01RUN002A")
    pos_b1 = body.find("01RUN001B")
    assert pos_a1 >= 0 and pos_a2 >= 0 and pos_b1 >= 0
    assert pos_b1 < pos_a2 < pos_a1


def test_run_show_renders_result(client, web_store, web_config):
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid = "01J9RUNSHOW1"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")
    sdir = session_dir(web_config.root, "claude", sid)
    EvalStore(sdir).append(
        _make_result(eval_id="01RUNSHOWID", sid=sid, started_at="2026-05-25T04:00:00Z")
    )

    r = client.get(f"/sessions/{sid}/evals/runs/01RUNSHOWID")
    assert r.status_code == 200
    assert b"01RUNSHOW" in r.content
    assert b"pass" in r.content


def test_run_show_unknown_returns_404(client, web_store):
    sid = "01J9RUNSHOW2"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    r = client.get(f"/sessions/{sid}/evals/runs/01NONEXISTENT")
    assert r.status_code == 404


def test_run_show_does_not_serve_status_fragment(client, web_store, web_config):
    from thirdeye.paths import session_dir

    sid = "01J9RUNSHOW3"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    sdir = session_dir(web_config.root, "claude", sid)
    jobs_dir = sdir / "evals.jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "01JOBONLY.json").write_text('{"status": "running"}')

    r = client.get(f"/sessions/{sid}/evals/runs/01JOBONLY")
    assert r.status_code == 404


def test_run_status_poll_returns_fragment_while_running(client, web_store, web_config):
    from thirdeye.paths import session_dir

    sid = "01J9RUNSTAT1"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    sdir = session_dir(web_config.root, "claude", sid)
    jobs_dir = sdir / "evals.jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "01RUNSTATX.json").write_text('{"status": "running"}')

    r = client.get(f"/sessions/{sid}/evals/runs/01RUNSTATX/status")
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    assert "<!DOCTYPE" not in body
    assert "<html" not in body
    assert "run-status" in body
    assert "running" in body
    assert f"/sessions/{sid}/evals/runs/01RUNSTATX/status" in body


def test_run_status_poll_shows_verdict_when_complete(client, web_store, web_config):
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid = "01J9RUNSTAT2"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    sdir = session_dir(web_config.root, "claude", sid)
    jobs_dir = sdir / "evals.jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "01RUNDONEXX.json").write_text('{"status": "done"}')
    EvalStore(sdir).append(
        _make_result(eval_id="01RUNDONEXX", sid=sid, started_at="2026-05-25T05:00:00Z")
    )

    r = client.get(f"/sessions/{sid}/evals/runs/01RUNDONEXX/status")
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    assert "<!DOCTYPE" not in body
    assert "run-status" in body
    assert "verdict-pass" in body
