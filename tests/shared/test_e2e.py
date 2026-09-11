import json
import os
import subprocess
import sys
from pathlib import Path


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["THIRDEYE_HOME"] = str(tmp_path)
    return env


def _run(args: list[str], env: dict[str, str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "thirdeye"] + args, env=env, **kw)


def _output(args: list[str], env: dict[str, str]) -> str:
    return subprocess.check_output([sys.executable, "-m", "thirdeye"] + args, env=env, text=True)


def _ingest(env: dict[str, str], platform: str, sid: str, cwd: str, events: list[dict]) -> None:
    payload = "".join(json.dumps(e) + "\n" for e in events)
    _run(
        ["ingest", "--platform", platform, "--session-id", sid, "--cwd", cwd],
        env=env,
        input=payload.encode(),
        check=True,
    )


def test_full_round_trip(tmp_path: Path):
    env = _env(tmp_path)

    _ingest(env, "claude", "01J9G7XK4P", "/p", [{"t": "user_message", "data": "hello world"}])
    _ingest(env, "cursor", "02ABCDEF12", "/q", [{"t": "user_message", "data": "different msg"}])

    out = _output(["list"], env)
    assert "01J9G7XK4P" in out
    assert "02ABCDEF12" in out

    out = _output(["events", "01J9"], env)
    assert "0 user_message hello world" in out

    out = _output(["event", "01J9", "0"], env)
    assert "hello world" in out

    out = _output(["search", "different"], env)
    assert "02ABCDEF12" in out

    out = _output(["stats"], env)
    obj = json.loads(out.strip())
    assert obj["session_count"] == 2
    assert obj["event_count"] == 2


def test_multi_event_ingest(tmp_path: Path):
    env = _env(tmp_path)
    events = [
        {"t": "start", "data": "begin"},
        {"t": "tool_call", "data": {"name": "grep", "args": ["foo"]}},
        {"t": "tool_result", "data": {"output": "found"}},
        {"t": "end", "data": "done"},
    ]
    _ingest(env, "claude", "01MULTIEVT1", "/w", events)

    out = _output(["events", "01MULTI"], env)
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 4
    assert "0 start begin" in lines[0]
    assert "3 end done" in lines[3]

    obj = json.loads(_output(["stats"], env).strip())
    assert obj["event_count"] == 4


def test_show_command(tmp_path: Path):
    env = _env(tmp_path)
    _ingest(env, "claude", "01SHOWTEST1", "/p", [{"t": "msg", "data": "show me"}])

    out = _output(["show", "01SHOW"], env)
    assert "0 msg show me" in out


def test_tail_command(tmp_path: Path):
    env = _env(tmp_path)
    events = [{"t": f"ev{i}", "data": i} for i in range(10)]
    _ingest(env, "claude", "01TAILTEST1", "/p", events)

    out = _output(["tail", "01TAIL", "-n", "3"], env)
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 3
    assert "7 ev7" in lines[0]
    assert "9 ev9" in lines[2]


def test_events_json_mode(tmp_path: Path):
    env = _env(tmp_path)
    _ingest(
        env,
        "claude",
        "01JSONTEST1",
        "/p",
        [
            {"t": "msg", "data": "json check"},
            {"t": "other", "data": 42},
        ],
    )

    out = _output(["events", "01JSON", "--json"], env)
    lines = out.strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "t" in parsed
        assert "ts" in parsed
        assert "seq" in parsed


def test_event_field_extraction(tmp_path: Path):
    env = _env(tmp_path)
    _ingest(
        env,
        "claude",
        "01FIELDTST1",
        "/p",
        [
            {"t": "tool", "data": {"name": "read", "path": "/foo/bar"}},
        ],
    )

    out = _output(["event", "01FIELD", "0", "--field", "name"], env)
    assert out.strip() == "read"

    out = _output(["event", "01FIELD", "0", "--field", "path"], env)
    assert out.strip() == "/foo/bar"


def test_list_platform_filter(tmp_path: Path):
    env = _env(tmp_path)
    _ingest(env, "claude", "01PLATCLD01", "/p", [{"t": "a", "data": 1}])
    _ingest(env, "cursor", "02PLATCUR01", "/q", [{"t": "b", "data": 2}])

    out = _output(["list", "--platform", "claude"], env)
    assert "01PLATCLD01" in out
    assert "02PLATCUR01" not in out

    out = _output(["list", "--platform", "cursor"], env)
    assert "02PLATCUR01" in out
    assert "01PLATCLD01" not in out


def test_stats_single_session(tmp_path: Path):
    env = _env(tmp_path)
    _ingest(env, "claude", "01STATSONE1", "/p", [{"t": "a"}, {"t": "b"}, {"t": "c"}])

    out = _output(["stats", "01STATS"], env)
    obj = json.loads(out.strip())
    assert obj["session_id"] == "01STATSONE1"
    assert obj["event_count"] == 3
    assert obj["platform"] == "claude"
    assert obj["bytes_compressed"] > 0


def test_search_no_match(tmp_path: Path):
    env = _env(tmp_path)
    _ingest(env, "claude", "01NOMATCH01", "/p", [{"t": "msg", "data": "apple"}])

    out = _output(["search", "zzzznotfound"], env)
    assert out.strip() == ""


def test_search_platform_filter(tmp_path: Path):
    env = _env(tmp_path)
    _ingest(env, "claude", "01SRCHCLD01", "/p", [{"t": "msg", "data": "findme"}])
    _ingest(env, "cursor", "02SRCHCUR01", "/q", [{"t": "msg", "data": "findme"}])

    out = _output(["search", "findme", "--platform", "cursor"], env)
    assert "02SRCHCUR01" in out
    assert "01SRCHCLD01" not in out


def test_unknown_session_prefix(tmp_path: Path):
    env = _env(tmp_path)
    result = _run(["events", "ZZNOEXIST"], env, capture_output=True)
    assert result.returncode != 0


def test_empty_ingest(tmp_path: Path):
    env = _env(tmp_path)
    _run(
        ["ingest", "--platform", "claude", "--session-id", "01EMPTYSES1", "--cwd", "/p"],
        env=env,
        input=b"",
        check=True,
    )

    out = _output(["list"], env)
    assert "01EMPTYSES1" in out

    out = _output(["events", "01EMPTY"], env)
    assert out.strip() == ""

    obj = json.loads(_output(["stats"], env).strip())
    assert obj["event_count"] == 0


def test_ingest_generates_session_id(tmp_path: Path):
    env = _env(tmp_path)
    result = _run(
        ["ingest", "--platform", "claude", "--cwd", "/p"],
        env=env,
        input=(json.dumps({"t": "msg", "data": "auto id"}) + "\n").encode(),
        capture_output=True,
    )
    assert result.returncode == 0
    sid = result.stdout.decode().strip()
    assert len(sid) > 0

    out = _output(["list"], env)
    assert sid in out
