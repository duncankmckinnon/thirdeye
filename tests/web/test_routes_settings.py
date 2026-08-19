from __future__ import annotations

import pytest
import yaml

from thirdeye.config import Config


@pytest.fixture(autouse=True)
def _stub_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("thirdeye.otel_export.is_available", lambda: True)


def _on_disk_logfire(config: Config) -> dict:
    return yaml.safe_load(config.config_file.read_text())["logfire"]


def test_settings_page_renders(client):
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert "Pydantic Logfire" in resp.text
    assert "disabled" in resp.text


def test_enable_persists_and_updates_panel(client, web_config: Config):
    resp = client.post(
        "/settings/logfire", data={"token": "pylf_v1_us_abcd1234", "project": "myproj"}
    )
    assert resp.status_code == 200
    assert "enabled" in resp.text
    assert "...1234" in resp.text  # masked suffix shown, not the full token
    assert "abcd1234" not in resp.text
    on_disk = _on_disk_logfire(web_config)
    assert on_disk["enabled"] is True
    assert on_disk["token"] == "pylf_v1_us_abcd1234"
    assert on_disk["project"] == "myproj"


def test_enable_requires_token(client):
    resp = client.post("/settings/logfire", data={"project": "myproj"})
    assert resp.status_code == 400


def test_disable_flips_flag_but_keeps_token(client, web_config: Config):
    client.post("/settings/logfire", data={"token": "tok", "project": "p"})
    resp = client.post("/settings/logfire/disable")
    assert resp.status_code == 200
    assert "disabled" in resp.text
    on_disk = _on_disk_logfire(web_config)
    assert on_disk["enabled"] is False
    assert on_disk["token"] == "tok"
