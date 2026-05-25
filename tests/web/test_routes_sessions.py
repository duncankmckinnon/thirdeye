from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from thirdeye.store import Store


@pytest.fixture()
def fixture_session(web_store: Store) -> tuple[str, str]:
    """Build one closed session and return (platform, sid)."""
    sid = "01J9G7XK4P"
    with web_store.open_session(sid, platform="claude", cwd="/tmp") as w:
        w.append("user_message", {"prompt": "hi"})
        w.append(
            "tool_call",
            {"tool_use_id": "tu1", "tool_name": "Read", "tool_input": {"file_path": "/x"}},
        )
        w.append("tool_result", {"tool_use_id": "tu1", "content": "ok"})
        w.append("assistant_message", {"text": "done"})
    web_store.close_session(sid, platform="claude")
    return ("claude", sid)


def test_view_renders_meta(client, fixture_session):
    _, sid = fixture_session
    r = client.get(f"/sessions/{sid}")
    assert r.status_code == 200
    assert sid.encode() in r.content
    assert b"claude" in r.content


def test_view_resolves_prefix(client, fixture_session):
    _, sid = fixture_session
    prefix = sid[:6]
    r = client.get(f"/sessions/{prefix}")
    assert r.status_code == 200
    assert sid.encode() in r.content


def test_tree_renders_paired_events(client, fixture_session):
    _, sid = fixture_session
    r = client.get(f"/sessions/{sid}/tree")
    assert r.status_code == 200
    body = r.content
    assert b"evt-user_message" in body
    assert b"evt-tool_call" in body
    assert b"evt-tool_result" in body
    assert b"evt-assistant_message" in body
    # tool_result is nested under tool_call (children ol)
    assert b'<ol class="children">' in body
    # tool_result should appear inside the children list, not as a top-level row
    call_idx = body.find(b"evt-tool_call")
    children_idx = body.find(b'<ol class="children">')
    result_idx = body.find(b"evt-tool_result")
    assert call_idx < children_idx < result_idx


def test_view_404(client):
    r = client.get("/sessions/does-not-exist")
    assert r.status_code == 404
    assert b"No unique session matches" in r.content


def test_tree_for_empty_session(client, web_store):
    sid = "01J0EMPTY00"
    with web_store.open_session(sid, platform="claude", cwd="/tmp"):
        pass
    web_store.close_session(sid, platform="claude")
    r = client.get(f"/sessions/{sid}/tree")
    assert r.status_code == 200
    assert b"No events" in r.content
