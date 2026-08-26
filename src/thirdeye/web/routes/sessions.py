from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.logfire_dataset import DatasetExportError, export_sessions
from thirdeye.timeparse import parse_when
from thirdeye.web.agentic import propose_filters
from thirdeye.web.vocabulary import inventory_tags


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
                    suggestions.append({"session_id": s.session_id, "platform": s.platform})
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


async def _sessions_agentic(request: Request) -> HTMLResponse:
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
            propose_filters, config, nl=nl, agent_name=agent, surface="sessions"
        )
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError, ValueError) as e:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": str(e)},
            status_code=400,
        )
    filters = {
        "platform": proposed.platform,
        "cwd": proposed.cwd,
        "status": proposed.status,
        "since": proposed.since,
        "until": proposed.until,
        "order": proposed.order,
        "turn": proposed.turn,
        "turn_query": proposed.turn_query,
        "tag": list(proposed.tags),
    }
    return templates.TemplateResponse(
        request,
        "sessions/_filter_form.html",
        {"filters": filters, "all_tags": inventory_tags(config)},
    )


async def _export_logfire_dataset(request: Request) -> HTMLResponse:
    form = await request.form()
    name = (form.get("dataset_name") or "").strip()
    templates = request.app.state.templates
    if not name:
        return templates.TemplateResponse(
            request, "_error.html", {"message": "Enter a dataset name."}, status_code=400
        )

    config = request.app.state.config
    api_key = config.logfire.api_key
    if not api_key:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": "Save a Logfire dataset API key in Settings first."},
            status_code=400,
        )

    store = request.app.state.store
    scope = (form.get("dataset_scope") or "session").strip()
    if scope not in {"session", "turn"}:
        return templates.TemplateResponse(
            request, "_error.html", {"message": "Invalid dataset scope."}, status_code=400
        )
    turn_id = (form.get("turn") or "").strip() or None
    turn_query = (form.get("turn_query") or "").strip() or None
    tag_list = [str(t) for t in form.getlist("tag") if t]
    sessions = list(
        store.list_sessions(
            platform=(form.get("platform") or "").strip() or None,
            cwd=(form.get("cwd") or "").strip() or None,
            status=(form.get("status") or "").strip() or None,
            tags=set(tag_list) if tag_list else None,
            since=parse_when((form.get("since") or "").strip() or None),
            until=parse_when((form.get("until") or "").strip() or None),
        )
    )
    if not sessions:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": "No sessions match these filters; no dataset was created."},
            status_code=400,
        )
    try:
        count = await run_in_threadpool(
            export_sessions,
            api_key=api_key,
            name=name,
            sessions=sessions,
            store=store,
            scope=scope,
            turn_id=turn_id,
            turn_query=turn_query,
        )
    except DatasetExportError as exc:
        return templates.TemplateResponse(
            request, "_error.html", {"message": str(exc)}, status_code=400
        )
    return templates.TemplateResponse(
        request,
        "sessions/_dataset_status.html",
        {"name": name, "count": count, "unit": scope},
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/sessions/agentic", _sessions_agentic, methods=["POST"]))
    app.routes.append(Route("/sessions/logfire-dataset", _export_logfire_dataset, methods=["POST"]))
    app.routes.append(Route("/sessions/{sid}", _view, methods=["GET"]))
    app.routes.append(Route("/sessions/{sid}/tree", _tree, methods=["GET"]))
