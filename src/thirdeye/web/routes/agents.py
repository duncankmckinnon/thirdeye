from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from thirdeye.web.agentic import installed_agent_names


async def _list_agents(request: Request) -> JSONResponse:
    config = request.app.state.config
    return JSONResponse({"agents": installed_agent_names(config)})


def register(app: Starlette) -> None:
    app.routes.append(Route("/agents", _list_agents, methods=["GET"]))
