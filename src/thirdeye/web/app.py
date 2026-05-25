from __future__ import annotations

import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from thirdeye.config import Config
from thirdeye.store import Store
from thirdeye.web.routes import (
    evals,
    events,
    index,
    search,
    sessions,
    stream,
    tags,
    usage,
)

_PACKAGE_ROOT = Path(__file__).parent


def create_app(config: Config | None = None) -> Starlette:
    config = config or Config.load()
    store = Store(config)
    templates = Jinja2Templates(directory=_PACKAGE_ROOT / "templates")
    templates.env.filters["basename"] = lambda p: os.path.basename(p) if p else ""

    app = Starlette(debug=False)
    app.state.config = config
    app.state.store = store
    app.state.templates = templates

    app.mount(
        "/static",
        StaticFiles(directory=_PACKAGE_ROOT / "static"),
        name="static",
    )

    for module in (
        index,
        sessions,
        events,
        search,
        usage,
        evals,
        tags,
        stream,
    ):
        module.register(app)

    return app
