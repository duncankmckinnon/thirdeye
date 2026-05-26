from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette  # noqa: E402
from starlette.templating import Jinja2Templates  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from thirdeye.config import Config  # noqa: E402
from thirdeye.web.routes import views as views_module  # noqa: E402

PKG = Path(__file__).resolve().parents[2] / "src/thirdeye/web"


def make_app(config: Config) -> Starlette:
    app = Starlette()
    app.state.config = config
    app.state.templates = Jinja2Templates(directory=PKG / "templates")
    views_module.register(app)
    return app


@pytest.fixture()
def views_client(tmp_path: Path) -> TestClient:
    (tmp_path / "traces").mkdir()
    config = Config(root=tmp_path)
    return TestClient(make_app(config))


def test_list_empty(views_client: TestClient) -> None:
    r = views_client.get("/views/sessions")
    assert r.status_code == 200
    assert "No saved views yet." in r.text


def test_save_then_list(views_client: TestClient) -> None:
    r = views_client.post(
        "/views/sessions",
        data={"name": "foo", "path": "?since=7d"},
    )
    assert r.status_code == 200
    assert "foo" in r.text


def test_save_invalid_name(views_client: TestClient) -> None:
    r = views_client.post(
        "/views/sessions",
        data={"name": "bad/name", "path": "?since=7d"},
    )
    assert r.status_code == 400


def test_save_invalid_path(views_client: TestClient) -> None:
    r = views_client.post(
        "/views/sessions",
        data={"name": "ok", "path": "no-question-mark"},
    )
    assert r.status_code == 400


def test_delete_removes(views_client: TestClient) -> None:
    views_client.post(
        "/views/sessions",
        data={"name": "foo", "path": "?since=7d"},
    )
    r = views_client.delete("/views/sessions/foo")
    assert r.status_code == 200
    assert "foo" not in r.text
    assert "No saved views yet." in r.text


def test_unknown_page_get(views_client: TestClient) -> None:
    r = views_client.get("/views/unknown")
    assert r.status_code == 404


def test_unknown_page_post(views_client: TestClient) -> None:
    r = views_client.post(
        "/views/unknown",
        data={"name": "foo", "path": "?since=7d"},
    )
    assert r.status_code == 404


def test_unknown_page_delete(views_client: TestClient) -> None:
    r = views_client.delete("/views/unknown/foo")
    assert r.status_code == 404
