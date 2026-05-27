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
    monkeypatch.setattr(
        "thirdeye.web.routes.sessions.inventory_tags",
        lambda cfg: ["refactor", "wip"],
    )
    r = client.post(
        "/sessions/agentic",
        data={"nl": "open refactor work longest", "agent": "claude"},
    )
    assert r.status_code == 200
    body = r.content.decode()
    assert 'id="sessions-filter-form"' in body
    assert 'value="claude"' in body
    assert '<option value="open" selected' in body
    assert '<option value="longest" selected' in body
    assert '<option value="refactor"' in body and "selected" in body
    assert "<button" in body and "Filter" in body


def test_sessions_agentic_populates_cwd(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        return _stub_proposed_sessions()

    monkeypatch.setattr("thirdeye.web.routes.sessions.propose_filters", fake_propose)
    monkeypatch.setattr(
        "thirdeye.web.routes.sessions.inventory_tags",
        lambda cfg: ["refactor"],
    )
    r = client.post(
        "/sessions/agentic",
        data={"nl": "x", "agent": "claude"},
    )
    body = r.content.decode()
    assert 'value="/Users/me/api"' in body


def test_sessions_agentic_no_legacy_chips(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        return _stub_proposed_sessions()

    monkeypatch.setattr("thirdeye.web.routes.sessions.propose_filters", fake_propose)
    monkeypatch.setattr(
        "thirdeye.web.routes.sessions.inventory_tags",
        lambda cfg: ["refactor"],
    )
    r = client.post("/sessions/agentic", data={"nl": "x", "agent": "claude"})
    body = r.content.decode()
    assert 'class="proposed-filters"' not in body
    assert 'class="filter-chips"' not in body


def test_sessions_agentic_multiple_tags_all_selected(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        return ProposedFilters(
            q=None, platform=None, cwd=None,
            tags=["alpha", "gamma"],
            since=None, until=None, status=None, order=None, rationale=None,
        )

    monkeypatch.setattr("thirdeye.web.routes.sessions.propose_filters", fake_propose)
    monkeypatch.setattr(
        "thirdeye.web.routes.sessions.inventory_tags",
        lambda cfg: ["alpha", "beta", "gamma"],
    )
    r = client.post("/sessions/agentic", data={"nl": "x", "agent": "claude"})
    body = r.content.decode()
    assert '<option value="alpha"' in body
    assert '<option value="beta"' in body
    assert '<option value="gamma"' in body
    assert body.count("selected") >= 2
    import re
    assert re.search(r'<option value="alpha"[^>]*selected', body)
    assert re.search(r'<option value="gamma"[^>]*selected', body)
    assert not re.search(r'<option value="beta"[^>]*selected', body)


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
