from __future__ import annotations

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")


def _make_result(*, eval_id, sid, definition, started_at, scores=None, findings=None):
    from thirdeye.eval.result import EvalResult

    return EvalResult(
        id=eval_id,
        session_id=sid,
        definition=definition,
        agent="claude",
        agent_model="",
        agent_session_id=None,
        started_at=started_at,
        ended_at=started_at,
        duration_ms=1000,
        verdict="pass",
        summary="ok",
        scores=scores or {},
        findings=findings or [],
    )


def _seed(web_store, sid):
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")


def test_panel_renders_directive_for_shipped_default(client, web_store, web_config):
    from thirdeye.eval.definition import load_definition
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid = "01J9PANEL001"
    _seed(web_store, sid)
    sd = session_dir(web_config.root, "claude", sid)
    EvalStore(sd).append(
        _make_result(
            eval_id="01RUNDEF01",
            sid=sid,
            definition="default",
            started_at="2026-05-26T00:00:00Z",
        )
    )

    defn = load_definition(web_config.root, "default")
    snippet = defn.directive.splitlines()[0].strip().encode()

    r = client.get(f"/sessions/{sid}/evals/default")
    assert r.status_code == 200
    assert b"<pre" in r.content
    assert snippet in r.content


def test_panel_filters_to_definition_and_sorts_desc(client, web_store, web_config):
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid = "01J9PANEL002"
    _seed(web_store, sid)
    sd = session_dir(web_config.root, "claude", sid)
    es = EvalStore(sd)
    es.append(
        _make_result(
            eval_id="01PARITYA0XX",
            sid=sid,
            definition="parity",
            started_at="2026-05-26T01:00:00Z",
        )
    )
    es.append(
        _make_result(
            eval_id="01PARITYB1YY",
            sid=sid,
            definition="parity",
            started_at="2026-05-26T02:00:00Z",
        )
    )
    es.append(
        _make_result(
            eval_id="01DEFAULTZZZ",
            sid=sid,
            definition="default",
            started_at="2026-05-26T03:00:00Z",
        )
    )

    (web_config.root / "evals" / "defs" / "parity.yaml").write_text(
        "name: parity\n"
        "description: ''\n"
        "default_agent: claude\n"
        "directive: |\n"
        "  parity directive\n"
    )

    r = client.get(f"/sessions/{sid}/evals/parity")
    assert r.status_code == 200
    body = r.content.decode()
    assert "01PARITYA0XX" in body
    assert "01PARITYB1YY" in body
    assert "01DEFAULTZZZ" not in body
    assert body.index("01PARITYB1YY") < body.index("01PARITYA0XX")


def test_panel_score_columns_union_with_dash_for_missing(client, web_store, web_config):
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid = "01J9PANEL003"
    _seed(web_store, sid)
    sd = session_dir(web_config.root, "claude", sid)
    es = EvalStore(sd)
    es.append(
        _make_result(
            eval_id="01SCORE001",
            sid=sid,
            definition="default",
            started_at="2026-05-26T01:00:00Z",
            scores={"correctness": 0.8},
        )
    )
    es.append(
        _make_result(
            eval_id="01SCORE002",
            sid=sid,
            definition="default",
            started_at="2026-05-26T02:00:00Z",
            scores={"correctness": 0.4, "tone": 0.9},
        )
    )

    r = client.get(f"/sessions/{sid}/evals/default")
    assert r.status_code == 200
    body = r.content.decode()
    assert "correctness" in body
    assert "tone" in body
    assert "0.800" in body
    assert "0.400" in body
    assert "0.900" in body
    assert "—" in body


def test_panel_unknown_session_returns_404(client):
    r = client.get("/sessions/01NOPENOPE/evals/default")
    assert r.status_code == 404


def test_panel_unknown_definition_returns_404(client, web_store):
    sid = "01J9PANEL004"
    _seed(web_store, sid)
    r = client.get(f"/sessions/{sid}/evals/zz-no-such-def")
    assert r.status_code == 404


def test_results_page_shows_def_chip_links(client, web_store, web_config):
    from thirdeye.eval.store import EvalStore
    from thirdeye.paths import session_dir

    sid = "01J9PANEL005"
    _seed(web_store, sid)
    sd = session_dir(web_config.root, "claude", sid)
    es = EvalStore(sd)
    es.append(
        _make_result(
            eval_id="01CHIP0001",
            sid=sid,
            definition="default",
            started_at="2026-05-26T01:00:00Z",
        )
    )
    es.append(
        _make_result(
            eval_id="01CHIP0002",
            sid=sid,
            definition="parity",
            started_at="2026-05-26T02:00:00Z",
        )
    )

    r = client.get(f"/sessions/{sid}/evals")
    assert r.status_code == 200
    body = r.content.decode()
    assert "def-chip" in body
    assert f'href="/sessions/{sid}/evals/default"' in body
    assert f'href="/sessions/{sid}/evals/parity"' in body
