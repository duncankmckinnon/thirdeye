from __future__ import annotations

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route


async def _event_detail(request: Request) -> HTMLResponse:
    prefix = request.path_params["sid"]
    seq = int(request.path_params["seq"])
    store = request.app.state.store
    try:
        platform, sid = store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        event = store.reader(sid).get_event(seq)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"seq {seq} not found")
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "events/detail.html",
        {"event": event, "sid": sid, "platform": platform},
    )


def register(app: Starlette) -> None:
    app.routes.append(
        Route("/sessions/{sid}/events/{seq:int}", _event_detail, methods=["GET"]),
    )
