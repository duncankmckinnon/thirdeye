from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from thirdeye.web.watcher import tail_events


async def _stream(request: Request):
    store = request.app.state.store
    prefix = request.path_params["sid"]
    try:
        platform, sid = store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    last_seq = int(request.query_params.get("last_seq") or "-1")
    reader = store.reader(sid)

    async def gen():
        try:
            async for ev in tail_events(reader, start_seq=last_seq):
                if await request.is_disconnected():
                    break
                payload = json.dumps(ev)
                yield f"data: {payload}\n\n"
                meta = store.get_meta(sid)
                if getattr(meta, "status", None) == "closed":
                    yield "event: closed\ndata: {}\n\n"
                    break
        except Exception as e:  # degraded fallback notification
            yield f"event: degraded\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def register(app: Starlette) -> None:
    app.routes.append(Route("/sessions/{sid}/stream", _stream, methods=["GET"]))
