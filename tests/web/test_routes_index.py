from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir

pytest.importorskip("starlette")


def _set_window(store, platform, sid, started, last):
    mp = meta_path(session_dir(store.config.root, platform, sid))
    m = read_meta(mp)
    assert m is not None
    m.started_at = started
    m.last_ts = last
    write_meta(mp, m)


def _populate(store):
    now = datetime.now(UTC)
    recent = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    old = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    sid_recent = "01RECENT0001"
    with store.open_session(sid_recent, platform="claude", cwd="/proj/a") as w:
        w.append("user_message", "hi")
    store.close_session(sid_recent, platform="claude")
    _set_window(store, "claude", sid_recent, recent, recent)

    sid_old = "01OLDOLD0001"
    with store.open_session(sid_old, platform="claude", cwd="/proj/b") as w:
        w.append("user_message", "hi")
    store.close_session(sid_old, platform="claude")
    _set_window(store, "claude", sid_old, old, old)

    return sid_recent, sid_old


def test_index_empty_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"thirdeye" in r.content


def test_index_filter_params_accepted(client):
    r = client.get("/?platform=claude&since=2026-01-01")
    assert r.status_code == 200


def test_index_has_ask_panel(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b'name="nl"' in r.content


def test_index_has_saved_views_sidebar_with_empty_state(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b'class="saved-views"' in r.content
    assert b"No saved views yet." in r.content


def test_index_default_since_7d_filters_old_sessions(client, web_store):
    sid_recent, sid_old = _populate(web_store)
    r = client.get("/")
    assert r.status_code == 200
    assert sid_recent.encode() in r.content
    assert sid_old.encode() not in r.content
    checkboxes = r.content.count(b'name="session_id"')
    assert checkboxes == 1


def test_index_explicit_since_wins_over_default(client, web_store):
    sid_recent, sid_old = _populate(web_store)
    r = client.get("/?since=2020-01-01")
    assert r.status_code == 200
    assert sid_recent.encode() in r.content
    assert sid_old.encode() in r.content


def test_index_default_since_in_form(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b'value="7d"' in r.content


def test_index_has_sessions_filter_form(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b'id="sessions-filter-form"' in r.content
    assert b'<select name="tag" multiple' in r.content


def test_index_tag_multiselect_lists_inventory(client, monkeypatch):
    monkeypatch.setattr(
        "thirdeye.web.routes.index.inventory_tags",
        lambda cfg: ["alpha", "beta"],
    )
    r = client.get("/")
    body = r.content.decode()
    assert '<option value="alpha"' in body
    assert '<option value="beta"' in body


def test_index_tag_round_trip_selects_options(client, monkeypatch):
    monkeypatch.setattr(
        "thirdeye.web.routes.index.inventory_tags",
        lambda cfg: ["foo", "bar"],
    )
    r = client.get("/?tag=foo&tag=bar")
    body = r.content.decode()
    assert '<option value="foo"' in body and "selected" in body
    assert '<option value="bar"' in body


def test_index_ask_panel_uses_outer_html_swap(client):
    r = client.get("/")
    body = r.content.decode()
    assert 'hx-swap="outerHTML"' in body
    assert 'hx-target="#sessions-filter-form"' in body


def test_index_status_and_order_round_trip(client):
    r = client.get("/?status=closed&order=oldest")
    body = r.content.decode()
    assert '<option value="closed" selected' in body
    assert '<option value="oldest" selected' in body


def test_index_no_legacy_ask_target_div(client):
    r = client.get("/")
    body = r.content.decode()
    assert 'id="ask-sessions-target"' not in body


def test_proposed_filters_templates_removed():
    import tempfile
    from pathlib import Path

    from thirdeye.config import Config
    from thirdeye.web.app import create_app

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "traces").mkdir()
        (root / "evals" / "defs").mkdir(parents=True)
        app = create_app(Config(root=root))
        env = app.state.templates.env
        from jinja2 import TemplateNotFound

        for name in ("search/_proposed_filters.html", "sessions/_proposed_filters.html"):
            try:
                env.get_template(name)
            except TemplateNotFound:
                continue
            raise AssertionError(f"{name} should be deleted")
