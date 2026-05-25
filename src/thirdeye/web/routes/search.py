from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.search import search
from thirdeye.timeparse import parse_when


async def _search(request: Request) -> HTMLResponse:
    params = request.query_params
    q = params.get("q") or ""
    platform = params.get("platform") or None
    cwd = params.get("cwd") or None
    tag_list = [t for t in params.getlist("tag") if t]
    since_str = params.get("since") or None
    until_str = params.get("until") or None
    filters = {
        "platform": platform,
        "cwd": cwd,
        "tag": tag_list,
        "since": since_str,
        "until": until_str,
    }
    hits: list = []
    if q:
        hits = list(
            search(
                request.app.state.store,
                q,
                platform=platform,
                cwd=cwd,
                tags=set(tag_list) if tag_list else None,
                since=parse_when(since_str),
                until=parse_when(until_str),
            )
        )
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "search.html",
        {"query": q, "filters": filters, "hits": hits},
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/search", _search, methods=["GET"]))
