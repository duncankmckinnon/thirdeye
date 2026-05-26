from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.eval.agents import list_agent_names
from thirdeye.search import search
from thirdeye.timeparse import parse_when
from thirdeye.web.agentic import propose_filters


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
    config = request.app.state.config
    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "query": q,
            "filters": filters,
            "hits": hits,
            "agents": list_agent_names(config.root),
        },
    )


async def _search_agentic(request: Request) -> HTMLResponse:
    form = await request.form()
    nl = (form.get("nl") or "").strip()
    agent = (form.get("agent") or "").strip()
    templates = request.app.state.templates
    if not agent:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": "Pick an agent."},
            status_code=400,
        )
    config = request.app.state.config
    try:
        proposed = await run_in_threadpool(
            propose_filters, config, nl=nl, agent_name=agent, surface="search"
        )
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError, ValueError) as e:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": str(e)},
            status_code=400,
        )
    return templates.TemplateResponse(
        request,
        "search/_proposed_filters.html",
        {"proposed": proposed, "nl": nl},
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/search", _search, methods=["GET"]))
    app.routes.append(Route("/search/agentic", _search_agentic, methods=["POST"]))
