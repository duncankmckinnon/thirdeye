from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_batch_empty_returns_400(client):
    r = client.post("/evals/runs/batch", data={"agent": "claude", "using": "default"})
    assert r.status_code == 400


def test_batch_over_limit_returns_400(client, web_store):
    sids = []
    for i in range(26):
        sid = f"01J9BATCH{i:03d}"
        with web_store.open_session(sid, platform="claude", cwd="/p") as w:
            w.append("user_message", {"prompt": "x"})
        web_store.close_session(sid, platform="claude")
        sids.append(sid)

    r = client.post(
        "/evals/runs/batch",
        data={"session_id": sids, "agent": "claude", "using": "default"},
    )
    assert r.status_code == 400
    assert b"capped" in r.content


def test_batch_missing_agent_returns_400(client, web_store):
    sid = "01J9BATCHX"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    r = client.post(
        "/evals/runs/batch",
        data={"session_id": sid, "using": "default", "agent": ""},
    )
    assert r.status_code == 400


def test_batch_dispatches_per_session(monkeypatch, client, web_store):
    sids = []
    for i in range(3):
        sid = f"01J9BATCHOK{i:02d}"
        with web_store.open_session(sid, platform="claude", cwd="/p") as w:
            w.append("user_message", {"prompt": "x"})
        web_store.close_session(sid, platform="claude")
        sids.append(sid)

    canned = iter(["01RUNA0001", "01RUNA0002", "01RUNA0003"])
    captured: list[dict] = []

    def fake_run_eval_background(
        *,
        thirdeye_home,
        platform,
        session_id,
        definition_name="default",
        agent_name,
        cwd=None,
        thirdeye_bin=None,
    ):
        captured.append(
            {
                "platform": platform,
                "session_id": session_id,
                "definition_name": definition_name,
                "agent_name": agent_name,
            }
        )
        return next(canned)

    monkeypatch.setattr(
        "thirdeye.web.routes.evals.run_eval_background",
        fake_run_eval_background,
    )

    r = client.post(
        "/evals/runs/batch",
        data={"session_id": sids, "agent": "claude", "using": "default"},
    )
    assert r.status_code == 200
    assert len(captured) == 3
    body = r.content.decode("utf-8")
    for run_id in ("01RUNA0001", "01RUNA0002", "01RUNA0003"):
        assert run_id in body
    for sid, run_id in zip(sids, ("01RUNA0001", "01RUNA0002", "01RUNA0003"), strict=True):
        assert f"/sessions/{sid}/evals/runs/{run_id}/status" in body


def test_batch_uses_def_field_from_sessions_page(monkeypatch, client, web_store):
    sid = "01J9BATCHDEF1"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    captured: list[dict] = []

    def fake_run_eval_background(
        *,
        thirdeye_home,
        platform,
        session_id,
        definition_name="default",
        agent_name,
        cwd=None,
        thirdeye_bin=None,
    ):
        captured.append({"definition_name": definition_name})
        return "01RUNDEF0001"

    monkeypatch.setattr(
        "thirdeye.web.routes.evals.run_eval_background",
        fake_run_eval_background,
    )

    r = client.post(
        "/evals/runs/batch",
        data={"session_id": [sid], "agent": "claude", "def": "myeval"},
    )
    assert r.status_code == 200
    assert len(captured) == 1
    assert captured[0]["definition_name"] == "myeval"


def test_batch_poll_returns_status_fragment(client, web_store, web_config):
    from thirdeye.paths import session_dir

    sid = "01J9BATCHPOLL"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    sdir = session_dir(web_config.root, "claude", sid)
    jobs_dir = sdir / "evals.jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "01RUNBATCHPOLL.json").write_text('{"status": "running"}')

    r = client.get(f"/sessions/{sid}/evals/runs/01RUNBATCHPOLL/status")
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    assert "<!DOCTYPE" not in body
    assert "<html" not in body
    assert "run-status" in body
    assert "running" in body
