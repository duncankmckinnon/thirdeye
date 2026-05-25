from __future__ import annotations

import asyncio
import json

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

POLL_INTERVAL = 0.5


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
            last_yielded = last_seq
            while True:
                if await request.is_disconnected():
                    break
                try:
                    events = list(reader.iter_events())
                except FileNotFoundError:
                    events = []
                for ev in events:
                    seq = ev.get("seq", -1)
                    if seq > last_yielded:
                        yield f"data: {json.dumps(ev)}\n\n"
                        last_yielded = seq
                # Re-check status AFTER draining so an already-closed
                # session emits all buffered events before the close
                # sentinel — and so a mid-stream close still terminates
                # even when no further events arrive.
                meta = store.get_meta(sid)
                if getattr(meta, "status", None) == "closed":
                    yield "event: closed\ndata: {}\n\n"
                    break
                await asyncio.sleep(POLL_INTERVAL)
        except Exception as e:  # degraded fallback notification
            yield f"event: degraded\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def register(app: Starlette) -> None:
    app.routes.append(Route("/sessions/{sid}/stream", _stream, methods=["GET"]))
