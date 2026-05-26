from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from thirdeye.web.agentic import ProposedFilters  # noqa: E402


def _stub_proposed_sessions() -> ProposedFilters:
    return ProposedFilters(
        q=None,
        platform="claude",
        cwd="/Users/me/api",
        tags=["refactor"],
        since=None,
        until=None,
        status="open",
        order="longest",
        rationale="Open refactor sessions.",
    )


def test_sessions_agentic_empty_agent_returns_400(client):
    r = client.post("/sessions/agentic", data={"nl": "x", "agent": ""})
    assert r.status_code == 400
    assert b"Pick an agent" in r.content


def test_sessions_agentic_success_renders_preview(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        assert surface == "sessions"
        return _stub_proposed_sessions()

    monkeypatch.setattr("thirdeye.web.routes.sessions.propose_filters", fake_propose)
    r = client.post(
        "/sessions/agentic",
        data={"nl": "open refactor work longest", "agent": "claude"},
    )
    assert r.status_code == 200
    body = r.content.decode()
    assert "status=open" in body
    assert "order=longest" in body
    assert "tag=refactor" in body
    assert "platform=claude" in body
    assert "/?" in body


def test_sessions_agentic_runtime_error_returns_400(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        raise RuntimeError("agent exited 2: broken")

    monkeypatch.setattr("thirdeye.web.routes.sessions.propose_filters", fake_propose)
    r = client.post("/sessions/agentic", data={"nl": "x", "agent": "claude"})
    assert r.status_code == 400
    assert b"agent exited 2" in r.content


def test_proposed_filters_query_string_sessions_surface():
    proposed = _stub_proposed_sessions()
    qs = proposed.query_string()
    assert qs.startswith("?")
    assert "platform=claude" in qs
    assert "status=open" in qs
    assert "order=longest" in qs
    assert "tag=refactor" in qs
    assert "q=" not in qs
