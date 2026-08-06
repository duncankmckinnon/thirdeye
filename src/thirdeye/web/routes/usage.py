from __future__ import annotations

import json as _json
from datetime import UTC, datetime, timedelta

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.paths import session_dir
from thirdeye.timeparse import parse_when
from thirdeye.usage.aggregate import aggregate_by_day
from thirdeye.usage.read import iter_calls


async def _session_usage(request: Request) -> HTMLResponse:
    prefix = request.path_params["sid"]
    store = request.app.state.store
    config = request.app.state.config
    try:
        platform, sid = store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    sdir = session_dir(config.root, platform, sid)
    rows = list(iter_calls(sdir))
    aggregate = store.stats(session_id=sid)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "usage/session.html",
        {"rows": rows, "aggregate": aggregate, "sid": sid, "platform": platform},
    )


async def _global_usage(request: Request) -> HTMLResponse:
    config = request.app.state.config
    params = request.query_params
    platform = params.get("platform") or None
    since_str = params.get("since") or None
    until_str = params.get("until") or None
    try:
        since = parse_when(since_str)
    except ValueError:
        since = None
    try:
        until = parse_when(until_str)
    except ValueError:
        until = None
    if since is None and until is None:
        until = datetime.now(UTC)
        since = until - timedelta(days=30)
    report = aggregate_by_day(config, platform=platform, since=since, until=until)
    chart_data = {
        "days": [b.day for b in report.buckets],
        "input": [b.input_tokens for b in report.buckets],
        "output": [b.output_tokens for b in report.buckets],
        "sessions": [b.sessions for b in report.buckets],
    }
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "usage/global.html",
        {
            "report": report,
            "filters": {
                "platform": platform,
                "since": since_str,
                "until": until_str,
            },
            "chart_data_json": _json.dumps(chart_data),
        },
    )


def register(app: Starlette) -> None:
    app.routes.append(Route("/sessions/{sid}/usage", _session_usage, methods=["GET"]))
    app.routes.append(Route("/usage", _global_usage, methods=["GET"]))
