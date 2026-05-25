from __future__ import annotations

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.eval.definition import list_definitions, load_definition
from thirdeye.eval.store import EvalStore
from thirdeye.paths import session_dir


async def _defs_list(request: Request) -> HTMLResponse:
    config = request.app.state.config
    names = list_definitions(config.root)
    defs = []
    for name in names:
        try:
            defs.append(load_definition(config.root, name))
        except Exception:
            continue
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/defs_list.html",
        {"defs": defs},
    )


async def _def_show(request: Request) -> HTMLResponse:
    config = request.app.state.config
    name = request.path_params["name"]
    try:
        defn = load_definition(config.root, name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/def_show.html",
        {"defn": defn},
    )


async def _session_evals(request: Request) -> HTMLResponse:
    config = request.app.state.config
    store = request.app.state.store
    prefix = request.path_params["sid"]
    try:
        platform, sid = store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    sdir = session_dir(config.root, platform, sid)
    results = sorted(
        EvalStore(sdir).iter_results(),
        key=lambda r: r.started_at,
        reverse=True,
    )
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/results.html",
        {"results": results, "sid": sid, "platform": platform},
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/evals/defs", _defs_list, methods=["GET"]))
    app.routes.append(Route("/evals/defs/{name}", _def_show, methods=["GET"]))
    app.routes.append(Route("/sessions/{sid}/evals", _session_evals, methods=["GET"]))
