from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_index_empty_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"thirdeye" in r.content


def test_index_filter_params_accepted(client):
    r = client.get("/?platform=claude&since=2026-01-01")
    assert r.status_code == 200
