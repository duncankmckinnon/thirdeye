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
        data=[("session_id", s) for s in sids]
        + [("agent", "claude"), ("using", "default")],
    )
    assert r.status_code == 400


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
        data=[("session_id", s) for s in sids]
        + [("agent", "claude"), ("using", "default")],
    )
    assert r.status_code == 200
    assert len(captured) == 3
    for run_id in ("01RUNA0001", "01RUNA0002", "01RUNA0003"):
        assert run_id.encode() in r.content
