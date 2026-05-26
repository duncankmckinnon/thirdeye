from __future__ import annotations

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.web.views_store import ALLOWED_PAGES, SavedView, ViewStore, now_iso


def _render_sidebar(request: Request, page: str) -> HTMLResponse:
    config = request.app.state.config
    saved = ViewStore(config.root, page).list()
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "_saved_views.html",
        {"page": page, "saved": saved},
    )


async def _list(request: Request) -> HTMLResponse:
    page = request.path_params["page"]
    if page not in ALLOWED_PAGES:
        raise HTTPException(404, detail=f"unknown page: {page!r}")
    return _render_sidebar(request, page)


async def _save(request: Request) -> HTMLResponse:
    page = request.path_params["page"]
    if page not in ALLOWED_PAGES:
        raise HTTPException(404, detail=f"unknown page: {page!r}")
    form = await request.form()
    name = (form.get("name") or "").strip()
    path = form.get("path") or ""
    try:
        view = SavedView(name=name, path=path, created_at=now_iso())
        ViewStore(request.app.state.config.root, page).save(view)
    except ValueError as e:
        raise HTTPException(400, detail=str(e)) from e
    return _render_sidebar(request, page)


async def _delete(request: Request) -> HTMLResponse:
    page = request.path_params["page"]
    name = request.path_params["name"]
    if page not in ALLOWED_PAGES:
        raise HTTPException(404, detail=f"unknown page: {page!r}")
    ViewStore(request.app.state.config.root, page).delete(name)
    return _render_sidebar(request, page)


def register(app: Starlette) -> None:
    app.routes.append(Route("/views/{page}", _list, methods=["GET"]))
    app.routes.append(Route("/views/{page}", _save, methods=["POST"]))
    app.routes.append(Route("/views/{page}/{name}", _delete, methods=["DELETE"]))
