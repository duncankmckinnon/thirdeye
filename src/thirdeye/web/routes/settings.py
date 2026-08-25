from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.config import LogfireSettings
from thirdeye.otel_export import status
from thirdeye.store import Store


def _render(request: Request, config) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "settings/_logfire_panel.html",
        {"logfire": status(config), "config": config},
    )


async def _page(request: Request) -> HTMLResponse:
    config = request.app.state.config
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "settings/index.html",
        {"logfire": status(config), "config": config},
    )


async def _enable(request: Request) -> HTMLResponse:
    form = await request.form()
    token = (form.get("token") or "").strip()
    if not token:
        return HTMLResponse("a Logfire gateway key is required", status_code=400)
    config = request.app.state.config
    config = config.write_logfire_settings(LogfireSettings(enabled=True, token=token))
    request.app.state.config = config
    request.app.state.store = Store(config)
    return _render(request, config)


async def _disable(request: Request) -> HTMLResponse:
    config = request.app.state.config
    settings = config.logfire
    config = config.write_logfire_settings(LogfireSettings(enabled=False, token=settings.token))
    request.app.state.config = config
    request.app.state.store = Store(config)
    return _render(request, config)


def register(app: Starlette) -> None:
    app.routes.append(Route("/settings", _page, methods=["GET"]))
    app.routes.append(Route("/settings/logfire", _enable, methods=["POST"]))
    app.routes.append(Route("/settings/logfire/disable", _disable, methods=["POST"]))
