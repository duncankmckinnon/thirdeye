from __future__ import annotations

import yaml
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from thirdeye.eval.definition import (
    EvalDefinition,
    delete_definition,
    list_definitions,
    load_definition,
    save_definition,
    validate_directive_yaml,
)
from thirdeye.eval.runner import run_eval_background
from thirdeye.eval.store import EvalStore
from thirdeye.paths import session_dir


async def _defs_list(request: Request) -> HTMLResponse:
    config = request.app.state.config
    names = list_definitions(config.root)
    defs = []
    for name in names:
        try:
            defs.append(load_definition(config.root, name))
        except Exception:
            continue
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/defs_list.html",
        {"defs": defs},
    )


async def _def_new(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/def_editor.html",
        {"defn": None, "mode": "create", "errors": None},
    )


async def _def_show(request: Request) -> HTMLResponse:
    config = request.app.state.config
    name = request.path_params["name"]
    try:
        defn = load_definition(config.root, name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/def_show.html",
        {"defn": defn},
    )


async def _def_edit(request: Request) -> HTMLResponse:
    config = request.app.state.config
    name = request.path_params["name"]
    try:
        defn = load_definition(config.root, name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no eval def named '{name}'")
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/def_editor.html",
        {"defn": defn, "mode": "edit", "errors": None},
    )


async def _def_validate(request: Request) -> HTMLResponse:
    form = await request.form()
    text = form.get("yaml") or ""
    ok, err = validate_directive_yaml(text)
    templates = request.app.state.templates
    if ok:
        return HTMLResponse("<div class='ok'>YAML is valid.</div>")
    return templates.TemplateResponse(
        request, "_error.html", {"message": err}, status_code=400
    )


async def _def_create(request: Request) -> HTMLResponse:
    config = request.app.state.config
    form = await request.form()
    yaml_text = form.get("yaml") or ""
    ok, err = validate_directive_yaml(yaml_text)
    templates = request.app.state.templates
    if not ok:
        return templates.TemplateResponse(
            request, "_error.html", {"message": err}, status_code=400
        )
    data = yaml.safe_load(yaml_text)
    defn = EvalDefinition.from_dict(data)
    try:
        save_definition(config.root, defn, force=False)
    except FileExistsError:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": f"definition '{defn.name}' already exists"},
            status_code=409,
        )
    return HTMLResponse(
        f'<meta http-equiv="refresh" content="0; url=/evals/defs/{defn.name}">',
    )


async def _def_update(request: Request) -> HTMLResponse:
    config = request.app.state.config
    name = request.path_params["name"]
    form = await request.form()
    yaml_text = form.get("yaml") or ""
    ok, err = validate_directive_yaml(yaml_text)
    templates = request.app.state.templates
    if not ok:
        return templates.TemplateResponse(
            request, "_error.html", {"message": err}, status_code=400
        )
    data = yaml.safe_load(yaml_text)
    data["name"] = name  # preserve URL name
    defn = EvalDefinition.from_dict(data)
    save_definition(config.root, defn, force=True)
    return HTMLResponse(
        f'<meta http-equiv="refresh" content="0; url=/evals/defs/{name}">',
    )


async def _def_delete(request: Request) -> HTMLResponse:
    config = request.app.state.config
    name = request.path_params["name"]
    delete_definition(config.root, name)
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/evals/defs">')


async def _session_evals(request: Request) -> HTMLResponse:
    config = request.app.state.config
    store = request.app.state.store
    prefix = request.path_params["sid"]
    try:
        platform, sid = store.resolve_session_id(prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    sdir = session_dir(config.root, platform, sid)
    results = sorted(
        EvalStore(sdir).iter_results(),
        key=lambda r: r.started_at,
        reverse=True,
    )
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/results.html",
        {"results": results, "sid": sid, "platform": platform},
    )


async def _run_dispatch(request: Request) -> HTMLResponse:
    config = request.app.state.config
    store = request.app.state.store
    form = await request.form()
    sid_prefix = form.get("session_id") or ""
    using = form.get("using") or "default"
    agent = form.get("agent") or ""
    try:
        platform, sid = store.resolve_session_id(sid_prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    job_id = run_eval_background(
        thirdeye_home=config.root,
        platform=platform,
        session_id=sid,
        definition_name=using,
        agent_name=agent,
    )
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/_run_status.html",
        {
            "sid": sid,
            "platform": platform,
            "job_id": job_id,
            "status": "running",
            "verdict": None,
        },
    )


async def _run_status(request: Request) -> HTMLResponse:
    config = request.app.state.config
    store = request.app.state.store
    sid_prefix = request.path_params["sid"]
    job_id = request.path_params["job_id"]
    try:
        platform, sid = store.resolve_session_id(sid_prefix)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    sdir = session_dir(config.root, platform, sid)
    es = EvalStore(sdir)
    job = es.read_job(job_id)
    status = (job or {}).get("status", "unknown")
    verdict = None
    if status not in ("running", "queued"):
        for r in es.iter_results():
            if getattr(r, "job_id", None) == job_id:
                verdict = getattr(r, "verdict", None)
                break
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "evals/_run_status.html",
        {
            "sid": sid,
            "platform": platform,
            "job_id": job_id,
            "status": status,
            "verdict": verdict,
        },
    )


def register(app: Starlette) -> None:
    # reads
    app.routes.append(Route("/evals/defs", _defs_list, methods=["GET"]))
    app.routes.append(Route("/evals/defs/_new", _def_new, methods=["GET"]))
    app.routes.append(Route("/evals/defs/_validate", _def_validate, methods=["POST"]))
    app.routes.append(Route("/evals/defs/{name}/edit", _def_edit, methods=["GET"]))
    app.routes.append(Route("/evals/defs/{name}", _def_show, methods=["GET"]))
    app.routes.append(Route("/sessions/{sid}/evals", _session_evals, methods=["GET"]))
    # writes (definitions)
    app.routes.append(Route("/evals/defs", _def_create, methods=["POST"]))
    app.routes.append(Route("/evals/defs/{name}", _def_update, methods=["PUT"]))
    app.routes.append(Route("/evals/defs/{name}", _def_delete, methods=["DELETE"]))
    # runs
    app.routes.append(Route("/evals/runs", _run_dispatch, methods=["POST"]))
    app.routes.append(
        Route(
            "/sessions/{sid}/evals/runs/{job_id}",
            _run_status,
            methods=["GET"],
        )
    )
