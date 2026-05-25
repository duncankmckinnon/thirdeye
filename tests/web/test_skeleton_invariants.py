"""Additional coverage for web skeleton invariants from the plan.

Verifies design and testing invariants that the three baseline web test files
don't directly exercise:

* The stub route modules really are no-ops (only ``/`` and ``/static/*`` exist).
* The session list renders rows for populated sessions and honors filters
  (platform, cwd, tag, since/until) — exercised against the real Store, not
  a mock, using the project-style ``Config(root=tmp_path)`` fixture.
* htmx is vendored as real bytes-on-disk with the canonical
  ``hx-*`` attribute names referenced by feature templates downstream.
* The ``ui`` CLI command exposes the documented ``--host``, ``--port``,
  ``--no-browser`` flags.
* The stub route modules don't touch ``app.routes`` (they exist purely to
  give Wave 2 a stable import surface).
* ``cli.py`` registers the new ``ui`` command on the top-level group.
* No module outside ``src/thirdeye/web/`` and ``src/thirdeye/commands/ui.py``
  imports starlette / uvicorn / jinja2 — i.e. the ``ui`` extra stays opt-in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("starlette")

from starlette.applications import Starlette  # noqa: E402

from thirdeye.config import Config  # noqa: E402
from thirdeye.store import Store  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def populated_web(tmp_path: Path):
    """Mirror project's populated_store fixture but build the web app on top."""
    (tmp_path / "traces").mkdir()
    (tmp_path / "evals" / "defs").mkdir(parents=True)
    config = Config(root=tmp_path)
    store = Store(config)
    with store.open_session("01J9G7XK4P", platform="claude", cwd="/proj/a") as w:
        w.append("user_message", "hi from claude")
        w.append("assistant_message", "hello user")
    with store.open_session("02ABCDEF12", platform="cursor", cwd="/proj/b") as w:
        w.append("user_message", "hi from cursor")

    from starlette.testclient import TestClient

    from thirdeye.web.app import create_app

    app = create_app(config)
    return app, TestClient(app), config


# --------------------------------------------------------------------------- #
# App / route wiring
# --------------------------------------------------------------------------- #


def test_only_root_and_static_routes_registered(app):
    """Wave-1 invariant: stub modules MUST be no-ops. Only `/` + `/static` exist."""
    paths = sorted({getattr(r, "path", None) for r in app.routes if hasattr(r, "path")})
    paths = [p for p in paths if p]
    assert paths == ["/", "/static"], (
        f"Stub modules must not register routes in Wave 1. Found extra: {paths}"
    )


def test_stub_modules_register_function_is_noop():
    """Every stub module exposes a register(app) that returns None and adds nothing."""
    from thirdeye.web.routes import evals, events, search, sessions, stream, tags, usage

    app = Starlette()
    before = len(app.routes)
    for module in (sessions, events, search, usage, evals, tags, stream):
        result = module.register(app)
        assert result is None, f"{module.__name__}.register must return None"
    assert len(app.routes) == before, "stub register() must not append routes"


def test_app_state_exposes_config_store_templates(app):
    assert isinstance(app.state.config, Config)
    assert isinstance(app.state.store, Store)
    # Templates is a Jinja2Templates instance with the templates dir mounted.
    from starlette.templating import Jinja2Templates

    assert isinstance(app.state.templates, Jinja2Templates)


# --------------------------------------------------------------------------- #
# Static assets: vendored htmx + CSS placeholders
# --------------------------------------------------------------------------- #


def test_htmx_vendored_with_version_comment(client):
    r = client.get("/static/htmx.min.js")
    assert r.status_code == 200
    body = r.content
    # The plan requires the first line to be a vendor comment naming v2.0.4.
    first_line = body.splitlines()[0]
    assert b"htmx" in first_line and b"v2.0.4" in first_line, (
        f"missing vendor comment header, got: {first_line!r}"
    )
    # File must be more than just a stub: still has to define the htmx global
    # / canonical attribute string referenced by future templates.
    assert b"htmx" in body
    assert len(body) > 200, "htmx asset suspiciously small"


def test_all_css_placeholders_served(client):
    for name in ("app.css", "sessions.css", "events.css", "search.css", "usage.css", "evals.css"):
        r = client.get(f"/static/{name}")
        assert r.status_code == 200, f"missing /static/{name}"


def test_tree_js_served_as_placeholder(client):
    r = client.get("/static/tree.js")
    assert r.status_code == 200
    # Wave 1: tree.js must remain empty; Wave 2 fills it in.
    assert r.content == b"", "tree.js should be empty in Wave 1"


def test_per_feature_css_files_empty_in_wave_1():
    """`Don't pre-populate per-feature CSS files` — they MUST stay empty in Wave 1."""
    root = Path(__file__).resolve().parents[2] / "src" / "thirdeye" / "web" / "static"
    for name in ("sessions.css", "events.css", "search.css", "usage.css", "evals.css"):
        size = (root / name).stat().st_size
        assert size == 0, f"{name} should be empty in Wave 1 (was {size} bytes)"


def test_missing_static_returns_404(client):
    r = client.get("/static/does-not-exist.css")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Index route: filters, rendering, pagination basics
# --------------------------------------------------------------------------- #


def test_index_renders_sessions(populated_web):
    _, client, _ = populated_web
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    # Short session IDs (first 8 chars) appear as links to /sessions/<full>.
    assert "01J9G7XK" in body
    assert "02ABCDEF" in body
    # Status column + platform names show through.
    assert "claude" in body
    assert "cursor" in body


def test_index_filters_by_platform(populated_web):
    _, client, _ = populated_web
    r = client.get("/?platform=claude")
    assert r.status_code == 200
    assert "01J9G7XK" in r.text
    assert "02ABCDEF" not in r.text  # cursor session filtered out


def test_index_filters_by_cwd(populated_web):
    _, client, _ = populated_web
    r = client.get("/?cwd=/proj/b")
    assert r.status_code == 200
    assert "02ABCDEF" in r.text
    assert "01J9G7XK" not in r.text


def test_index_empty_filter_strings_treated_as_none(populated_web):
    """Empty query params (e.g. `platform=`) must not crash list_sessions."""
    _, client, _ = populated_web
    r = client.get("/?platform=&cwd=&since=&until=")
    assert r.status_code == 200
    # Both sessions still render — empty filters were coerced to None.
    assert "01J9G7XK" in r.text
    assert "02ABCDEF" in r.text


def test_index_invalid_since_returns_500_not_200(populated_web):
    """`since` is parsed via parse_when; garbage values surface as a 500.

    parse_when raises ValueError on bad input; the route doesn't catch this
    in Wave 1, so a malformed `since` produces a 500. We just verify the
    server doesn't silently return 200 with junk — the next wave can decide
    how to surface this to the user with a friendlier error page.
    """
    from starlette.testclient import TestClient

    app, _, _ = populated_web
    # `raise_server_exceptions=False` lets us see the 500 response instead of
    # the raw Python exception bubbling out of the test client.
    bare_client = TestClient(app, raise_server_exceptions=False)
    r = bare_client.get("/?since=this-is-not-a-date")
    # A 200 here would be wrong (silent ignore). Wave 1 surfaces it as 500.
    assert r.status_code in (400, 500), f"unexpected: {r.status_code}"


def test_index_accepts_multiple_tag_params(populated_web):
    """`tag` is repeatable — multiple tag query params become a set."""
    _, client, _ = populated_web
    # No matching sessions tagged — should still return 200 with empty list.
    r = client.get("/?tag=alpha&tag=beta")
    assert r.status_code == 200


def test_index_empty_message_when_no_sessions(client):
    """Empty store: page renders the "No sessions yet." empty state."""
    r = client.get("/")
    assert r.status_code == 200
    assert "No sessions yet" in r.text


def test_index_includes_topbar_nav(client):
    """Every page extends base.html which carries nav links to /evals/defs & /usage."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'href="/evals/defs"' in r.text
    assert 'href="/usage"' in r.text
    assert 'href="/"' in r.text  # brand link


# --------------------------------------------------------------------------- #
# CLI command surface
# --------------------------------------------------------------------------- #


def test_ui_command_has_documented_options():
    from click.testing import CliRunner

    from thirdeye.commands.ui import ui

    result = CliRunner().invoke(ui, ["--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "--host" in out
    assert "--port" in out
    assert "--no-browser" in out
    assert "127.0.0.1" in out  # default host shown


def test_cli_main_registers_ui_command():
    from click.testing import CliRunner

    from thirdeye.cli import main

    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    # The "ui" subcommand should appear in the top-level command list.
    assert re.search(r"^\s*ui\b", result.output, re.MULTILINE), result.output


# --------------------------------------------------------------------------- #
# Dependency isolation: starlette/uvicorn/jinja2 ONLY in web/ and commands/ui.py
# --------------------------------------------------------------------------- #


def test_no_starlette_import_outside_web_layer():
    """`ui` extra MUST stay opt-in — only web/ and commands/ui.py may import it."""
    src = Path(__file__).resolve().parents[2] / "src" / "thirdeye"
    forbidden = ("starlette", "uvicorn", "jinja2")
    allowed_prefixes = (
        src / "web",
        src / "commands" / "ui.py",
    )

    offenders: list[tuple[Path, str]] = []
    for py in src.rglob("*.py"):
        if any(
            str(py).startswith(str(allowed)) or py == allowed
            for allowed in allowed_prefixes
        ):
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for mod in forbidden:
            # Match `import starlette` or `from starlette...` at module scope.
            if re.search(rf"^\s*(?:import|from)\s+{mod}\b", text, re.MULTILINE):
                offenders.append((py, mod))

    assert not offenders, (
        "These files import a UI-only dependency outside the allowed surface "
        f"(src/thirdeye/web/ and src/thirdeye/commands/ui.py): {offenders}"
    )


# --------------------------------------------------------------------------- #
# Templates contain expected partials & extends
# --------------------------------------------------------------------------- #


def test_index_template_extends_base_and_links_to_session_view(populated_web):
    """Session rows must link to /sessions/<full_sid> per the partial."""
    _, client, _ = populated_web
    r = client.get("/")
    assert r.status_code == 200
    # Each session row is an anchor to /sessions/<full_sid>.
    assert 'href="/sessions/01J9G7XK4P"' in r.text
    assert 'href="/sessions/02ABCDEF12"' in r.text


def test_404_template_exists():
    """_404.html is shipped with the package even though no route hits it yet."""
    tmpl = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "thirdeye"
        / "web"
        / "templates"
        / "_404.html"
    )
    assert tmpl.exists()
    body = tmpl.read_text()
    assert "{% extends" in body
    assert "suggestions" in body  # uses suggestions context var


def test_error_partial_exists():
    tmpl = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "thirdeye"
        / "web"
        / "templates"
        / "_error.html"
    )
    assert tmpl.exists()
    assert 'class="error"' in tmpl.read_text()
