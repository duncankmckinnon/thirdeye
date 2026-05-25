from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("starlette")


def test_stream_yields_events_then_closes(web_store, client):
    sid = "01J9STREAM01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "hi"})
        w.append("assistant_message", {"text": "ok"})
    web_store.close_session(sid, platform="claude")

    with client.stream("GET", f"/sessions/{sid}/stream?last_seq=-1") as r:
        assert r.status_code == 200
        body = r.read().decode("utf-8", errors="replace")
    assert "data:" in body
    assert "user_message" in body
    assert "assistant_message" in body
    assert "event: closed" in body


def test_stream_open_session_streams_then_closes(web_store, client):
    """An open session streams appended events, then terminates when closed."""
    sid = "01J9STREAM04"
    writer_cm = web_store.open_session(sid, platform="claude", cwd="/p")
    writer = writer_cm.__enter__()
    writer.append("user_message", {"prompt": "first"})

    def append_then_close():
        # Give the stream a moment to start tailing, then append and close.
        time.sleep(0.05)
        writer.append("assistant_message", {"text": "second"})
        time.sleep(0.05)
        writer_cm.__exit__(None, None, None)
        web_store.close_session(sid, platform="claude")

    t = threading.Thread(target=append_then_close)
    t.start()
    try:
        with client.stream("GET", f"/sessions/{sid}/stream?last_seq=-1") as r:
            assert r.status_code == 200
            body = r.read().decode("utf-8", errors="replace")
    finally:
        t.join(timeout=5.0)

    assert "first" in body
    assert "second" in body
    assert "event: closed" in body


def test_stream_unknown_session_returns_404(client):
    r = client.get("/sessions/01JNOPE/stream")
    assert r.status_code == 404


def test_stream_last_seq_skips_earlier_events(web_store, client):
    """?last_seq=0 must return only events with seq > 0."""
    sid = "01J9STREAM02"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "first"})  # seq 0
        w.append("assistant_message", {"text": "second"})  # seq 1
    web_store.close_session(sid, platform="claude")

    with client.stream("GET", f"/sessions/{sid}/stream?last_seq=0") as r:
        assert r.status_code == 200
        body = r.read().decode("utf-8", errors="replace")
    assert "second" in body
    assert "first" not in body
    assert "event: closed" in body


def test_stream_resolves_session_prefix(web_store, client):
    """The stream URL must accept a unique session_id prefix."""
    sid = "01J9STREAM03ABCDEF"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "hello"})
    web_store.close_session(sid, platform="claude")

    with client.stream("GET", "/sessions/01J9STREAM03/stream?last_seq=-1") as r:
        assert r.status_code == 200
        body = r.read().decode("utf-8", errors="replace")
    assert "hello" in body
    assert "event: closed" in body
