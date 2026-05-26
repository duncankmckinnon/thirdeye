from __future__ import annotations

import shutil

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from thirdeye.eval.agents import BUILTIN_ADAPTERS, list_agent_names


async def _list_agents(request: Request) -> JSONResponse:
    config = request.app.state.config
    names = list_agent_names(config.root)
    installed = []
    for name in names:
        adapter_cls = BUILTIN_ADAPTERS.get(name)
        cmd = adapter_cls().config.command if adapter_cls else name
        if shutil.which(cmd) is not None:
            installed.append(name)
    return JSONResponse({"agents": installed})


def register(app: Starlette) -> None:
    app.routes.append(Route("/agents", _list_agents, methods=["GET"]))
