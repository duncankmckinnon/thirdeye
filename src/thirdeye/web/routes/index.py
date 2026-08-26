from __future__ import annotations

import shutil

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.eval.agents import BUILTIN_ADAPTERS, list_agent_names
from thirdeye.eval.definition import list_definitions
from thirdeye.timeparse import parse_when
from thirdeye.turns import filter_turns
from thirdeye.web.views_store import ViewStore
from thirdeye.web.vocabulary import inventory_tags


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
    turn = params.get("turn") or None
    tag_list = [t for t in params.getlist("tag") if t]
    tags = set(tag_list) if tag_list else None

    defaults_applied = since_str is None and order is None
    if since_str is None:
        since_str = "7d"
    if order is None:
        order = "newest"

    matching_sessions = list(
        store.list_sessions(
            platform=platform,
            cwd=cwd,
            status=status,
            tags=tags,
            since=parse_when(since_str),
            until=parse_when(until_str),
        )
    )
    if turn:
        matching_sessions = [
            meta for meta in matching_sessions if filter_turns([meta], store, turn)
        ]
    sessions = sorted(
        matching_sessions,
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
                "turn": turn,
                "tag": tag_list,
            },
            "defaults_applied": defaults_applied,
            "agents": _installed_agents(config.root),
            "eval_defs": list(list_definitions(config.root)),
            "saved_views": ViewStore(config.root, "sessions").list(),
            "all_tags": inventory_tags(config),
        },
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/", _index, methods=["GET"]))
