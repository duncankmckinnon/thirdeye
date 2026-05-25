from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_defs_list_shows_saved_defs(client, web_config):
    (web_config.root / "evals" / "defs" / "foo.yaml").write_text(
        "name: foo\n"
        "description: test def\n"
        "default_agent: claude\n"
        "directive: |\n"
        "  do the thing\n"
    )
    r = client.get("/evals/defs")
    assert r.status_code == 200
    assert b"foo" in r.content


def test_def_show_renders_directive(client, web_config):
    (web_config.root / "evals" / "defs" / "foo.yaml").write_text(
        "name: foo\n"
        "default_agent: claude\n"
        "description: ''\n"
        "directive: |\n"
        "  do the thing\n"
    )
    r = client.get("/evals/defs/foo")
    assert r.status_code == 200
    assert b"do the thing" in r.content


def test_def_show_missing_returns_404(client):
    r = client.get("/evals/defs/zz-does-not-exist")
    assert r.status_code == 404


def test_session_evals_lists_result(client, web_store, web_config):
    from thirdeye.eval.result import EvalResult
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid = "01J9RESULT01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    sdir = session_dir(web_config.root, "claude", sid)
    EvalStore(sdir).append(
        EvalResult(
            id="01EVALID01",
            session_id=sid,
            definition="default",
            agent="claude",
            agent_model="",
            agent_session_id=None,
            started_at="2026-05-25T00:00:00Z",
            ended_at="2026-05-25T00:00:10Z",
            duration_ms=10000,
            verdict="pass",
            summary="ok",
            scores={"overall": 8.0},
            findings=[],
        )
    )

    r = client.get(f"/sessions/{sid}/evals")
    assert r.status_code == 200
    assert b"pass" in r.content


def test_session_evals_missing_session_returns_404(client):
    r = client.get("/sessions/01NOPE/evals")
    assert r.status_code == 404
