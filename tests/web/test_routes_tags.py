import pytest

pytest.importorskip("starlette")


def test_tag_add_then_remove(client, web_store):
    sid = "01J9TAG01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    r = client.post(f"/sessions/{sid}/events/0/tags", data={"tag": "review"})
    assert r.status_code == 200
    assert b"review" in r.content

    r = client.request("DELETE", f"/sessions/{sid}/events/0/tags/review")
    assert r.status_code == 200
    assert b"review" not in r.content


def test_tag_invalid_returns_400(client, web_store):
    sid = "01J9TAG02"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    r = client.post(f"/sessions/{sid}/events/0/tags", data={"tag": ""})
    assert r.status_code == 400


def test_tag_session_404(client):
    r = client.post("/sessions/bogus/events/0/tags", data={"tag": "review"})
    assert r.status_code == 404
