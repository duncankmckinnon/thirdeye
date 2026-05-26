from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.store import Store

pytest.importorskip("starlette")
pytest.importorskip("httpx")  # required by starlette.testclient


@pytest.fixture()
def web_config(tmp_path: Path) -> Config:
    """Config rooted at tmp_path; mirrors project's tmp_store fixture."""
    (tmp_path / "traces").mkdir()
    (tmp_path / "evals" / "defs").mkdir(parents=True)
    return Config(root=tmp_path)


@pytest.fixture()
def web_store(web_config: Config) -> Store:
    return Store(web_config)


@pytest.fixture()
def app(web_config: Config):
    from thirdeye.web.app import create_app

    return create_app(web_config)


@pytest.fixture()
def client(app):
    from starlette.testclient import TestClient

    return TestClient(app)
