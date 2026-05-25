from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.timeparse import parse_when


async def _index(request: Request) -> HTMLResponse:
    store = request.app.state.store
    templates = request.app.state.templates
    params = request.query_params
    platform = params.get("platform") or None
    cwd = params.get("cwd") or None
    since_str = params.get("since") or None
    until_str = params.get("until") or None
    tag_list = params.getlist("tag") or []
    tags = set(tag_list) if tag_list else None

    sessions = list(
        store.list_sessions(
            platform=platform,
            cwd=cwd,
            tags=tags,
            since=parse_when(since_str),
            until=parse_when(until_str),
        )
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sessions": sessions,
            "filters": {
                "platform": platform,
                "cwd": cwd,
                "since": since_str,
                "until": until_str,
                "tag": tag_list,
            },
        },
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/", _index, methods=["GET"]))
