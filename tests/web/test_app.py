from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_app_factory_returns_starlette(app):
    from starlette.applications import Starlette

    assert isinstance(app, Starlette)


def test_static_mount(client):
    r = client.get("/static/htmx.min.js")
    assert r.status_code == 200
    # Sanity-check that real htmx content was vendored (not a stub) — htmx
    # always defines the global `htmx` JS object and the `hx-get` attribute.
    assert b"hx-get" in r.content or b"htmx" in r.content


def test_app_css_served(client):
    r = client.get("/static/app.css")
    assert r.status_code == 200
