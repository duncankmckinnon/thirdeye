from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_event_detail_renders(client, web_store):
    sid = "01J9EVT01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "first"})
        w.append("tool_call", {"tool_name": "Read"})
        w.append("assistant_message", {"text": "done"})
    web_store.close_session(sid, platform="claude")

    r = client.get(f"/sessions/{sid}/events/1")
    assert r.status_code == 200
    assert b"tool_call" in r.content


def test_event_detail_seq_not_found(client, web_store):
    sid = "01J9EVT02"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    r = client.get(f"/sessions/{sid}/events/9999")
    assert r.status_code == 404


def test_event_detail_session_not_found(client):
    r = client.get("/sessions/bogus/events/0")
    assert r.status_code == 404
