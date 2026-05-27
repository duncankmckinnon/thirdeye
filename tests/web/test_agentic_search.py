from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from thirdeye.web import agentic as agentic_module  # noqa: E402
from thirdeye.web.agentic import ProposedFilters  # noqa: E402


def _stub_proposed_search() -> ProposedFilters:
    return ProposedFilters(
        q="needle",
        platform="claude",
        cwd=None,
        tags=["bug"],
        since="7d",
        until=None,
        status=None,
        order=None,
        rationale="Filtered to claude sessions tagged bug.",
    )


def test_search_agentic_empty_agent_returns_400(client):
    r = client.post("/search/agentic", data={"nl": "find things", "agent": ""})
    assert r.status_code == 400
    assert b"Pick an agent" in r.content


def test_search_agentic_missing_agent_returns_400(client):
    r = client.post("/search/agentic", data={"nl": "find things"})
    assert r.status_code == 400
    assert b"Pick an agent" in r.content


def test_search_agentic_success_renders_preview(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        assert surface == "search"
        assert agent_name == "claude"
        return _stub_proposed_search()

    monkeypatch.setattr("thirdeye.web.routes.search.propose_filters", fake_propose)
    monkeypatch.setattr("thirdeye.web.routes.search.inventory_tags", lambda cfg: ["bug", "wip"])
    r = client.post("/search/agentic", data={"nl": "claude bug from last week", "agent": "claude"})
    assert r.status_code == 200
    body = r.content.decode()
    assert 'id="search-filter-form"' in body
    assert 'value="needle"' in body
    assert 'value="claude"' in body
    assert 'value="7d"' in body
    assert '<option value="bug"' in body and "selected" in body
    assert "<button" in body and "Search" in body


def test_search_agentic_no_legacy_chips(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        return _stub_proposed_search()

    monkeypatch.setattr("thirdeye.web.routes.search.propose_filters", fake_propose)
    monkeypatch.setattr("thirdeye.web.routes.search.inventory_tags", lambda cfg: ["bug"])
    r = client.post("/search/agentic", data={"nl": "x", "agent": "claude"})
    body = r.content.decode()
    assert 'class="proposed-filters"' not in body
    assert 'class="filter-chips"' not in body
    assert ">Run<" not in body


def test_search_agentic_form_action_is_search(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        return _stub_proposed_search()

    monkeypatch.setattr("thirdeye.web.routes.search.propose_filters", fake_propose)
    monkeypatch.setattr("thirdeye.web.routes.search.inventory_tags", lambda cfg: ["bug"])
    r = client.post("/search/agentic", data={"nl": "x", "agent": "claude"})
    body = r.content.decode()
    assert 'action="/search"' in body
    assert 'method="get"' in body


def test_search_agentic_all_empty_renders_blank_form(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        return ProposedFilters(
            q=None,
            platform=None,
            cwd=None,
            tags=[],
            since=None,
            until=None,
            status=None,
            order=None,
            rationale=None,
        )

    monkeypatch.setattr("thirdeye.web.routes.search.propose_filters", fake_propose)
    monkeypatch.setattr("thirdeye.web.routes.search.inventory_tags", lambda cfg: ["one"])
    r = client.post("/search/agentic", data={"nl": "", "agent": "claude"})
    assert r.status_code == 200
    body = r.content.decode()
    assert 'id="search-filter-form"' in body
    assert 'name="q"' in body
    assert '<option value="one"' in body
    assert "selected" not in body or body.count("selected") <= 1


def test_search_inventory_tags_are_html_escaped(client, monkeypatch):
    monkeypatch.setattr(
        "thirdeye.web.routes.search.inventory_tags",
        lambda cfg: ["<script>", "a&b"],
    )
    r = client.get("/search")
    body = r.content.decode()
    assert '<option value="<script>"' not in body
    assert '<option value="&lt;script&gt;"' in body
    assert '<option value="a&amp;b"' in body


def test_search_agentic_agent_failure_returns_400(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        raise FileNotFoundError("`claude` not found on PATH")

    monkeypatch.setattr("thirdeye.web.routes.search.propose_filters", fake_propose)
    r = client.post("/search/agentic", data={"nl": "x", "agent": "claude"})
    assert r.status_code == 400
    assert b"not found on PATH" in r.content


def test_search_agentic_value_error_returns_400(client, monkeypatch):
    def fake_propose(config, *, nl, agent_name, surface):
        raise ValueError("unknown agent: 'nope'")

    monkeypatch.setattr("thirdeye.web.routes.search.propose_filters", fake_propose)
    r = client.post("/search/agentic", data={"nl": "x", "agent": "nope"})
    assert r.status_code == 400
    assert b"unknown agent" in r.content


def test_proposed_filters_query_string_search_surface():
    proposed = _stub_proposed_search()
    qs = proposed.query_string()
    assert qs.startswith("?")
    assert "q=needle" in qs
    assert "platform=claude" in qs
    assert "tag=bug" in qs
    assert "since=7d" in qs


def test_proposed_filters_empty_query_string():
    p = ProposedFilters(
        q=None,
        platform=None,
        cwd=None,
        tags=[],
        since=None,
        until=None,
        status=None,
        order=None,
        rationale=None,
    )
    assert p.query_string() == ""


def test_agentic_module_parse_envelope_strips_fences():
    raw = '```json\n{"q": "x"}\n```'
    env = agentic_module._parse_envelope(raw)
    assert env == {"q": "x"}


def test_agentic_module_parse_envelope_preserves_inner_backticks():
    raw = '```\n{"q": "`raw`"}\n```'
    env = agentic_module._parse_envelope(raw)
    assert env == {"q": "`raw`"}


def test_installed_agent_names_filters_by_path(web_config, monkeypatch):
    monkeypatch.setattr(
        "thirdeye.web.agentic.shutil.which",
        lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
    )
    assert agentic_module.installed_agent_names(web_config) == ["claude"]


def test_installed_agent_names_empty_when_none(web_config, monkeypatch):
    monkeypatch.setattr("thirdeye.web.agentic.shutil.which", lambda cmd: None)
    assert agentic_module.installed_agent_names(web_config) == []


def test_search_page_passes_only_installed_agents(client, monkeypatch):
    monkeypatch.setattr(
        "thirdeye.web.agentic.shutil.which",
        lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
    )
    called: dict = {}
    real = agentic_module.installed_agent_names

    def spy(config):
        result = real(config)
        called["agents"] = result
        return result

    monkeypatch.setattr("thirdeye.web.routes.search.installed_agent_names", spy)
    r = client.get("/search")
    assert r.status_code == 200
    assert called["agents"] == ["claude"]
