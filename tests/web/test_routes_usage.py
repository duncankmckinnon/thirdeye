from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_global_usage_200(client):
    r = client.get("/usage")
    assert r.status_code == 200


def test_session_usage_200(client, web_store):
    sid = "01J9USG01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    r = client.get(f"/sessions/{sid}/usage")
    assert r.status_code == 200


def test_session_usage_404(client):
    r = client.get("/sessions/bogus/usage")
    assert r.status_code == 404
