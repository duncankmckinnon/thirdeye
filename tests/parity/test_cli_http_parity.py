from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from thirdeye.commands.eval import eval_group  # noqa: E402
from thirdeye.commands.views import views_group  # noqa: E402
from thirdeye.config import Config  # noqa: E402
from thirdeye.eval.result import EvalResult  # noqa: E402
from thirdeye.eval.store import EvalStore  # noqa: E402
from thirdeye.paths import session_dir  # noqa: E402
from thirdeye.store import Store  # noqa: E402
from thirdeye.web.app import create_app  # noqa: E402


def _make_seed_session(store: Store, sid: str) -> None:
    with store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    store.close_session(sid, platform="claude")


def _make_result(*, eval_id: str, sid: str, definition: str, started_at: str) -> EvalResult:
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
        scores={},
        findings=[],
    )


def test_views_save_cli_matches_http(tmp_path: Path, monkeypatch) -> None:
    cli_root = tmp_path / "cli"
    http_root = tmp_path / "http"
    (cli_root / "traces").mkdir(parents=True)
    (http_root / "traces").mkdir(parents=True)

    monkeypatch.setenv("THIRDEYE_HOME", str(cli_root))
    runner = CliRunner()
    r = runner.invoke(
        views_group,
        ["save", "--page", "sessions", "--name", "foo", "--path", "?since=7d"],
    )
    assert r.exit_code == 0, r.output

    http_config = Config(root=http_root)
    app = create_app(http_config)
    with TestClient(app) as client:
        resp = client.post(
            "/views/sessions",
            data={"name": "foo", "path": "?since=7d"},
        )
        assert resp.status_code == 200

    cli_file = cli_root / "views" / "sessions.json"
    http_file = http_root / "views" / "sessions.json"
    assert cli_file.exists()
    assert http_file.exists()

    cli_data = json.loads(cli_file.read_text())
    http_data = json.loads(http_file.read_text())
    assert len(cli_data["views"]) == 1
    assert len(http_data["views"]) == 1
    cli_view = cli_data["views"][0]
    http_view = http_data["views"][0]
    assert cli_view["name"] == http_view["name"] == "foo"
    assert cli_view["path"] == http_view["path"] == "?since=7d"
    assert "created_at" in cli_view and "created_at" in http_view


def test_eval_batch_cli_matches_http(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "shared"
    (root / "traces").mkdir(parents=True)
    config = Config(root=root)
    store = Store(config)

    sid = "01J9PARITYBATCH1"
    _make_seed_session(store, sid)

    captured: list[dict] = []

    def fake_run_eval_background(
        *,
        thirdeye_home,
        platform,
        session_id,
        definition_name="default",
        agent_name,
        cwd=None,
        thirdeye_bin=None,
    ):
        captured.append(
            {
                "thirdeye_home": Path(thirdeye_home),
                "platform": platform,
                "session_id": session_id,
                "definition_name": definition_name,
                "agent_name": agent_name,
            }
        )
        return "01RUNPARITY01"

    monkeypatch.setattr(
        "thirdeye.commands.eval.run_eval_background",
        fake_run_eval_background,
    )
    monkeypatch.setattr(
        "thirdeye.web.routes.evals.run_eval_background",
        fake_run_eval_background,
    )

    monkeypatch.setenv("THIRDEYE_HOME", str(root))
    runner = CliRunner()
    r = runner.invoke(
        eval_group,
        ["batch", "default", sid, "--agent", "claude"],
    )
    assert r.exit_code == 0, r.output

    app = create_app(config)
    with TestClient(app) as client:
        resp = client.post(
            "/evals/runs/batch",
            data={"session_id": sid, "agent": "claude", "using": "default"},
        )
        assert resp.status_code == 200

    assert len(captured) == 2
    cli_call, http_call = captured
    for key in ("platform", "session_id", "definition_name", "agent_name"):
        assert cli_call[key] == http_call[key]
    assert cli_call["thirdeye_home"] == http_call["thirdeye_home"] == root


def test_eval_results_cli_matches_http(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "shared"
    (root / "traces").mkdir(parents=True)
    config = Config(root=root)
    store = Store(config)

    sid_a = "01J9PARITYR0A"
    sid_b = "01J9PARITYR0B"
    for sid in (sid_a, sid_b):
        _make_seed_session(store, sid)

    EvalStore(session_dir(root, "claude", sid_a)).append(
        _make_result(
            eval_id="01RUNRES0001",
            sid=sid_a,
            definition="parity",
            started_at="2026-05-25T01:00:00Z",
        )
    )
    EvalStore(session_dir(root, "claude", sid_b)).append(
        _make_result(
            eval_id="01RUNRES0002",
            sid=sid_b,
            definition="parity",
            started_at="2026-05-25T02:00:00Z",
        )
    )

    monkeypatch.setenv("THIRDEYE_HOME", str(root))
    runner = CliRunner()
    r = runner.invoke(
        eval_group,
        ["results", "--def", "parity", "--as-json"],
    )
    assert r.exit_code == 0, r.output
    cli_rows = [json.loads(ln) for ln in r.output.splitlines() if ln.strip()]
    cli_run_ids = [row["run_id"] for row in cli_rows]

    app = create_app(config)
    with TestClient(app) as client:
        resp = client.get("/evals/defs/parity/results")
        assert resp.status_code == 200
        body = resp.content.decode("utf-8")

    for run_id in cli_run_ids:
        assert run_id in body

    assert set(cli_run_ids) == {"01RUNRES0001", "01RUNRES0002"}
    assert cli_run_ids[0] == "01RUNRES0002"
    assert cli_run_ids[1] == "01RUNRES0001"
    pos_first = body.find(cli_run_ids[0])
    pos_second = body.find(cli_run_ids[1])
    assert 0 <= pos_first < pos_second
