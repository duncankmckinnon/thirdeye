from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_search_empty_query(client):
    r = client.get("/search")
    assert r.status_code == 200


def test_search_includes_ask_panel(client, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/dummy")
    r = client.get("/search")
    assert r.status_code == 200
    body = r.text
    assert 'name="nl"' in body
    assert 'name="agent"' in body
    for agent in ("claude", "codex", "gemini"):
        assert f'value="{agent}"' in body


def test_search_finds_hello(client, web_store):
    sid = "01J9SRCH01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "hello world"})
    web_store.close_session(sid, platform="claude")

    r = client.get("/search?q=hello")
    assert r.status_code == 200
    assert sid.encode() in r.content


def test_search_platform_filter_pass_through(client, web_store):
    sid_a = "01J9SRCH02"
    with web_store.open_session(sid_a, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "needle"})
    web_store.close_session(sid_a, platform="claude")

    sid_b = "01J9SRCH03"
    with web_store.open_session(sid_b, platform="cursor", cwd="/p") as w:
        w.append("user_message", {"prompt": "needle"})
    web_store.close_session(sid_b, platform="cursor")

    r = client.get("/search?q=needle&platform=claude")
    assert r.status_code == 200
    assert sid_a.encode() in r.content
    assert sid_b.encode() not in r.content
