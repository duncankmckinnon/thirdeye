from __future__ import annotations

import shutil

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.eval.agents import BUILTIN_ADAPTERS, list_agent_names
from thirdeye.eval.definition import list_definitions
from thirdeye.timeparse import parse_when
from thirdeye.web.views_store import ViewStore


def _installed_agents(root) -> list[str]:
    out: list[str] = []
    for name in list_agent_names(root):
        cmd = BUILTIN_ADAPTERS[name]().config.command if name in BUILTIN_ADAPTERS else name
        if shutil.which(cmd) is not None:
            out.append(name)
    return out


async def _index(request: Request) -> HTMLResponse:
    store = request.app.state.store
    config = request.app.state.config
    templates = request.app.state.templates
    params = request.query_params
    platform = params.get("platform") or None
    cwd = params.get("cwd") or None
    status = params.get("status") or None
    since_str = params.get("since") or None
    until_str = params.get("until") or None
    order = params.get("order") or None
    tag_list = [t for t in params.getlist("tag") if t]
    tags = set(tag_list) if tag_list else None

    defaults_applied = since_str is None and order is None
    if since_str is None:
        since_str = "7d"
    if order is None:
        order = "newest"

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
        reverse=(order != "oldest"),
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
                "order": order,
                "tag": tag_list,
            },
            "defaults_applied": defaults_applied,
            "agents": _installed_agents(config.root),
            "eval_defs": list(list_definitions(config.root)),
            "saved_views": ViewStore(config.root, "sessions").list(),
        },
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/", _index, methods=["GET"]))
