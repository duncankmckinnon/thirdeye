from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_def_create_valid_yaml_writes_file(client, web_config):
    yaml_body = (
        "name: foo\n"
        "description: test def\n"
        "default_agent: claude\n"
        "directive: |\n"
        "  do the thing\n"
    )
    r = client.post("/evals/defs", data={"yaml": yaml_body})
    assert r.status_code == 200
    assert (web_config.root / "evals" / "defs" / "foo.yaml").exists()


def test_def_create_invalid_yaml_returns_400(client):
    r = client.post("/evals/defs", data={"yaml": "name: ::: bad"})
    assert r.status_code == 400


def test_def_create_duplicate_name_returns_409(client, web_config):
    yaml_body = "name: dup\n" "description: x\n" "default_agent: claude\n" "directive: |\n" "  d\n"
    client.post("/evals/defs", data={"yaml": yaml_body})
    r = client.post("/evals/defs", data={"yaml": yaml_body})
    assert r.status_code == 409


def test_def_delete_removes_file(client, web_config):
    (web_config.root / "evals" / "defs" / "tokill.yaml").write_text(
        "name: tokill\ndefault_agent: claude\ndirective: |\n  x\n"
    )
    r = client.request("DELETE", "/evals/defs/tokill")
    assert r.status_code == 200
    assert not (web_config.root / "evals" / "defs" / "tokill.yaml").exists()


def test_run_dispatch_mocks_runner(monkeypatch, client, web_store):
    sid = "01J9EVAL01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    captured = {}

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
        captured["thirdeye_home"] = thirdeye_home
        captured["platform"] = platform
        captured["session_id"] = session_id
        captured["definition_name"] = definition_name
        captured["agent_name"] = agent_name
        return "01JOBID01"

    monkeypatch.setattr(
        "thirdeye.web.routes.evals.run_eval_background",
        fake_run_eval_background,
    )

    r = client.post(
        "/evals/runs",
        data={"session_id": sid, "using": "default", "agent": "claude"},
    )
    assert r.status_code == 200
    assert b"01JOBID01" in r.content
    assert captured["session_id"] == sid
    assert captured["agent_name"] == "claude"
    assert captured["thirdeye_home"] == web_store.config.root


def test_run_status_reads_job_payload(client, web_store, web_config):
    from thirdeye.paths import session_dir

    sid = "01J9EVAL02"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    sdir = session_dir(web_config.root, "claude", sid)
    jobs_dir = sdir / "evals.jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "01JOBID02.json").write_text('{"status": "running"}')

    r = client.get(f"/sessions/{sid}/evals/runs/01JOBID02")
    assert r.status_code == 200
    assert b"running" in r.content
