from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from starlette.applications import Starlette
from starlette.testclient import TestClient

from thirdeye.web.routes import agents as agents_module


def _make_app(config):
    app = Starlette()
    app.state.config = config
    agents_module.register(app)
    return app


def test_agents_list_returns_only_installed_binaries(web_config, monkeypatch):
    def fake_which(cmd):
        return "/usr/bin/claude" if cmd == "claude" else None

    monkeypatch.setattr(
        "thirdeye.web.routes.agents.shutil.which", fake_which
    )
    app = _make_app(web_config)
    with TestClient(app) as client:
        r = client.get("/agents")
    assert r.status_code == 200
    data = r.json()
    assert data == {"agents": ["claude"]}


def test_agents_list_empty_when_none_installed(web_config, monkeypatch):
    monkeypatch.setattr(
        "thirdeye.web.routes.agents.shutil.which", lambda cmd: None
    )
    app = _make_app(web_config)
    with TestClient(app) as client:
        r = client.get("/agents")
    assert r.status_code == 200
    assert r.json() == {"agents": []}


def test_agents_list_all_when_all_installed(web_config, monkeypatch):
    monkeypatch.setattr(
        "thirdeye.web.routes.agents.shutil.which", lambda cmd: f"/usr/bin/{cmd}"
    )
    app = _make_app(web_config)
    with TestClient(app) as client:
        r = client.get("/agents")
    assert r.status_code == 200
    names = r.json()["agents"]
    assert "claude" in names
    assert "codex" in names
    assert "gemini" in names
