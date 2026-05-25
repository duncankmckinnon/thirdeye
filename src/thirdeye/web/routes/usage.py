from __future__ import annotations

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.paths import session_dir
from thirdeye.usage.store import UsageStore


async def _session_usage(request: Request) -> HTMLResponse:
    prefix = request.path_params["sid"]
    store = request.app.state.store
    config = request.app.state.config
    try:
        platform, sid = store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    sdir = session_dir(config.root, platform, sid)
    rows = list(UsageStore(sdir).iter_rows())
    aggregate = store.stats(session_id=sid)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "usage/session.html",
        {"rows": rows, "aggregate": aggregate, "sid": sid, "platform": platform},
    )


async def _global_usage(request: Request) -> HTMLResponse:
    store = request.app.state.store
    aggregate = store.stats()
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "usage/global.html",
        {"aggregate": aggregate},
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/sessions/{sid}/usage", _session_usage, methods=["GET"]))
    app.routes.append(Route("/usage", _global_usage, methods=["GET"]))
