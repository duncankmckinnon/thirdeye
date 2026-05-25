from __future__ import annotations

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.paths import session_dir
from thirdeye.tags import TagStore, validate_tag


def _resolve(request: Request) -> tuple[str, str, int]:
    store = request.app.state.store
    prefix = request.path_params["sid"]
    seq = int(request.path_params["seq"])
    try:
        platform, sid = store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return platform, sid, seq


def _render_chips(request: Request, platform: str, sid: str, seq: int) -> HTMLResponse:
    config = request.app.state.config
    sdir = session_dir(config.root, platform, sid)
    tags = sorted(TagStore(sdir).tags_for(seq))
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "_tag_chips.html",
        {"tags": tags, "sid": sid, "seq": seq},
    )


async def _add(request: Request) -> HTMLResponse:
    platform, sid, seq = _resolve(request)
    form = await request.form()
    raw = (form.get("tag") or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="tag is required")
    try:
        tag = validate_tag(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    config = request.app.state.config
    sdir = session_dir(config.root, platform, sid)
    TagStore(sdir).add(seq, tag, source="ui")
    return _render_chips(request, platform, sid, seq)


async def _remove(request: Request) -> HTMLResponse:
    platform, sid, seq = _resolve(request)
    tag = request.path_params["tag"]
    config = request.app.state.config
    sdir = session_dir(config.root, platform, sid)
    TagStore(sdir).remove(seq, tag, source="ui")
    return _render_chips(request, platform, sid, seq)


def register(app: Starlette) -> None:
    app.routes.append(
        Route(
            "/sessions/{sid}/events/{seq:int}/tags",
            _add,
            methods=["POST"],
        )
    )
    app.routes.append(
        Route(
            "/sessions/{sid}/events/{seq:int}/tags/{tag}",
            _remove,
            methods=["DELETE"],
        )
    )
