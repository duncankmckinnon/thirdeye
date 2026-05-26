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
        assert run_id[:8] in body
    pos_latest = body.find("01RUN001"[:8])  # nb: only checks ids appear; ordering asserted below
    assert pos_latest >= 0
    pos_a1 = body.find("01RUN001A"[:8])
    pos_a2 = body.find("01RUN002A"[:8])
    pos_b1 = body.find("01RUN001B"[:8])
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
