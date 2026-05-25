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
    status = params.get("status") or None
    since_str = params.get("since") or None
    until_str = params.get("until") or None
    tag_list = [t for t in params.getlist("tag") if t]
    tags = set(tag_list) if tag_list else None

    sessions = sorted(
        store.list_sessions(
            platform=platform,
            cwd=cwd,
            status=status,
            tags=tags,
            since=parse_when(since_str),
            until=parse_when(until_str),
        ),
        key=lambda s: s.started_at or "",
        reverse=True,
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sessions": sessions,
            "filters": {
                "platform": platform,
                "cwd": cwd,
                "status": status,
                "since": since_str,
                "until": until_str,
                "tag": tag_list,
            },
        },
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/", _index, methods=["GET"]))
