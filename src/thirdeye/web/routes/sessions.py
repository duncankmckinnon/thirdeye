from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route


class _NotFound(Exception):
    """Signals resolve failed; caller returns the rendered 404 response."""

    def __init__(self, response: HTMLResponse) -> None:
        self.response = response


def _resolve_or_404(request: Request, prefix: str) -> tuple[str, str]:
    store = request.app.state.store
    try:
        return store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        suggestions: list[dict[str, str]] = []
        try:
            for s in store.list_sessions():
                if s.session_id.startswith(prefix):
                    suggestions.append(
                        {"session_id": s.session_id, "platform": s.platform}
                    )
                    if len(suggestions) >= 10:
                        break
        except Exception:
            pass
        templates = request.app.state.templates
        response = templates.TemplateResponse(
            request,
            "_404.html",
            {
                "message": f"No unique session matches '{prefix}': {e}",
                "suggestions": suggestions,
            },
            status_code=404,
        )
        raise _NotFound(response) from e


async def _view(request: Request) -> HTMLResponse:
    prefix = request.path_params["sid"]
    try:
        platform, sid = _resolve_or_404(request, prefix)
    except _NotFound as nf:
        return nf.response
    store = request.app.state.store
    meta = store.get_meta(sid)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "sessions/view.html",
        {"meta": meta, "platform": platform, "sid": sid},
    )


def _pair_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each tool_call with its matching tool_result.

    Pairs primarily by ``data.tool_use_id``; falls back to the nearest prior
    unpaired ``tool_call`` when an id is absent. Paired results become the
    ``child`` of their call and are removed from the top-level list.
    """
    by_id: dict[str, dict[str, Any]] = {}
    open_calls: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for ev in events:
        ev["child"] = None
        t = ev.get("t")
        data = ev.get("data") or {}
        if t == "tool_call":
            tu_id = data.get("tool_use_id") if isinstance(data, dict) else None
            if tu_id:
                by_id[tu_id] = ev
            open_calls.append(ev)
            out.append(ev)
        elif t == "tool_result":
            tu_id = data.get("tool_use_id") if isinstance(data, dict) else None
            parent: dict[str, Any] | None = None
            if tu_id and tu_id in by_id:
                parent = by_id.pop(tu_id)
                if parent in open_calls:
                    open_calls.remove(parent)
            elif open_calls:
                parent = open_calls.pop()
            if parent is not None:
                parent["child"] = ev
            else:
                out.append(ev)
        else:
            out.append(ev)
    return out


async def _tree(request: Request) -> HTMLResponse:
    prefix = request.path_params["sid"]
    try:
        platform, sid = _resolve_or_404(request, prefix)
    except _NotFound as nf:
        return nf.response
    store = request.app.state.store
    reader = store.reader(sid)
    events = list(reader.iter_events())
    paired = _pair_events(events)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "sessions/_tree.html",
        {"events": paired, "sid": sid, "platform": platform},
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/sessions/{sid}", _view, methods=["GET"]))
    app.routes.append(Route("/sessions/{sid}/tree", _tree, methods=["GET"]))
