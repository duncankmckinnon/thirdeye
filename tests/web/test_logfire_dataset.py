from __future__ import annotations

from thirdeye.config import LogfireSettings


def test_export_requires_dataset_name(client):
    response = client.post("/sessions/logfire-dataset", data={})
    assert response.status_code == 400
    assert "dataset name" in response.text


def test_export_requires_api_key(client):
    response = client.post("/sessions/logfire-dataset", data={"dataset_name": "sessions"})
    assert response.status_code == 400
    assert "Settings" in response.text


def test_export_uses_current_filters(client, app, web_store, monkeypatch):
    app.state.config = app.state.config.write_logfire_settings(
        LogfireSettings(api_key="dataset-key")
    )
    app.state.store = web_store
    for sid, platform in (("claude-one", "claude"), ("codex-one", "codex")):
        with web_store.open_session(sid, platform=platform, cwd="/project") as writer:
            writer.append("user_message", "hello")
    captured = {}

    def fake_export_sessions(**kwargs):
        captured.update(kwargs)
        return len(kwargs["sessions"])

    monkeypatch.setattr("thirdeye.web.routes.sessions.export_sessions", fake_export_sessions)
    response = client.post(
        "/sessions/logfire-dataset",
        data={"dataset_name": "claude-sessions", "platform": "claude", "since": "2020-01-01"},
    )

    assert response.status_code == 200
    assert "Sent 1 session" in response.text
    assert captured["name"] == "claude-sessions"
    assert captured["api_key"] == "dataset-key"
    assert captured["scope"] == "session"
    assert captured["turn_id"] is None
    assert [m.session_id for m in captured["sessions"]] == ["claude-one"]


def test_export_passes_turn_scope_and_exact_turn(client, app, web_store, monkeypatch):
    app.state.config = app.state.config.write_logfire_settings(
        LogfireSettings(api_key="dataset-key")
    )
    app.state.store = web_store
    with web_store.open_session("claude-one", platform="claude", cwd="/project") as writer:
        writer.append("user_message", {"prompt": "hello"})
        writer.append("assistant_message", {"text": "done"})
    captured = {}

    def fake_export_sessions(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr("thirdeye.web.routes.sessions.export_sessions", fake_export_sessions)
    response = client.post(
        "/sessions/logfire-dataset",
        data={
            "dataset_name": "one-turn",
            "dataset_scope": "turn",
            "turn": "0",
            "since": "2020-01-01",
        },
    )

    assert response.status_code == 200
    assert "Sent 1 turn" in response.text
    assert captured["scope"] == "turn"
    assert captured["turn_id"] == "0"


def test_filter_form_has_logfire_dataset_action(client):
    response = client.get("/")
    assert 'name="dataset_name"' in response.text
    assert 'name="dataset_scope"' in response.text
    assert 'name="turn"' in response.text
    assert 'hx-post="/sessions/logfire-dataset"' in response.text
