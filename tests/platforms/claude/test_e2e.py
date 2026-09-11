from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["THIRDEYE_HOME"] = str(tmp_path / "thirdeye")
    return env


def _thirdeye(
    *args: str, env: dict[str, str], stdin: bytes | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "thirdeye", *args],
        env=env,
        input=stdin,
        check=False,
        capture_output=True,
    )


def _hook(name: str, payload: dict, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [name],
        env=env,
        input=json.dumps(payload).encode(),
        check=False,
        capture_output=True,
    )


@pytest.fixture(autouse=True)
def _require_hook_entrypoints():
    if shutil.which("thirdeye-claude-session-start") is None:
        pytest.skip("hook entry points not installed (run uv sync)")


# -- full lifecycle ------------------------------------------------------------


def test_full_claude_session_lifecycle(tmp_path: Path):
    env = _env(tmp_path)
    settings_file = tmp_path / ".claude" / "settings.json"

    from thirdeye.platforms.claude.install import ClaudePlatform

    ClaudePlatform(settings_file=settings_file).install()
    assert settings_file.exists()

    sid = "e2e-session-001"

    r = _hook(
        "thirdeye-claude-session-start",
        {"session_id": sid, "cwd": "/proj/x", "source": "cli"},
        env,
    )
    assert r.returncode == 0, r.stderr.decode()

    r = _hook(
        "thirdeye-claude-user-prompt-submit",
        {"session_id": sid, "cwd": "/proj/x", "prompt": "explain this codebase"},
        env,
    )
    assert r.returncode == 0, r.stderr.decode()

    _hook(
        "thirdeye-claude-pre-tool-use",
        {
            "session_id": sid,
            "cwd": "/proj/x",
            "tool_name": "Read",
            "tool_use_id": "tu_1",
            "tool_input": {"file_path": "src/main.py"},
        },
        env,
    )
    _hook(
        "thirdeye-claude-post-tool-use",
        {
            "session_id": sid,
            "cwd": "/proj/x",
            "tool_name": "Read",
            "tool_use_id": "tu_1",
            "tool_response": "<contents of main.py>",
        },
        env,
    )

    r = _hook("thirdeye-claude-session-end", {"session_id": sid, "cwd": "/proj/x"}, env)
    assert r.returncode == 0, r.stderr.decode()

    r = _thirdeye("list", env=env)
    assert r.returncode == 0
    meta = json.loads(r.stdout.decode().strip().splitlines()[0])
    assert meta["session_id"] == sid
    assert meta["platform"] == "claude"
    assert meta["status"] == "closed"
    assert meta["event_count"] == 5

    r = _thirdeye("events", sid, env=env)
    assert r.returncode == 0
    lines = r.stdout.decode().strip().splitlines()
    types = [line.split()[1] for line in lines]
    assert types == [
        "session_start",
        "user_message",
        "tool_call",
        "tool_result",
        "session_end",
    ]

    r = _thirdeye("event", sid, "1", env=env)
    assert r.returncode == 0
    expanded = json.loads(r.stdout.decode())
    assert expanded["data"]["prompt"] == "explain this codebase"
    assert "session_id" not in expanded["data"]


# -- uninstall -----------------------------------------------------------------


def test_uninstall_removes_hooks(tmp_path: Path):
    settings_file = tmp_path / "settings.json"
    from thirdeye.platforms.claude.install import ClaudePlatform

    p = ClaudePlatform(settings_file=settings_file)
    p.install()
    p.uninstall()
    data = json.loads(settings_file.read_text())
    assert "hooks" not in data


# -- silent noop edge cases (subprocess) ---------------------------------------


def test_hook_no_session_id_exits_zero_creates_nothing(tmp_path: Path):
    env = _env(tmp_path)
    r = _hook(
        "thirdeye-claude-session-start",
        {"cwd": "/proj/x"},
        env,
    )
    assert r.returncode == 0
    r = _thirdeye("list", env=env)
    assert r.returncode == 0
    assert r.stdout.decode().strip() == ""


def test_hook_empty_stdin_exits_zero(tmp_path: Path):
    env = _env(tmp_path)
    r = subprocess.run(
        ["thirdeye-claude-session-start"],
        env=env,
        input=b"",
        check=False,
        capture_output=True,
    )
    assert r.returncode == 0
    r = _thirdeye("list", env=env)
    assert r.stdout.decode().strip() == ""


def test_hook_invalid_json_exits_zero(tmp_path: Path):
    env = _env(tmp_path)
    r = subprocess.run(
        ["thirdeye-claude-user-prompt-submit"],
        env=env,
        input=b"not valid json {{{",
        check=False,
        capture_output=True,
    )
    assert r.returncode == 0


def test_hook_empty_session_id_is_noop(tmp_path: Path):
    env = _env(tmp_path)
    r = _hook(
        "thirdeye-claude-stop",
        {"session_id": "", "cwd": "/proj/x"},
        env,
    )
    assert r.returncode == 0
    r = _thirdeye("list", env=env)
    assert r.stdout.decode().strip() == ""


# -- hooks produce no stdout ---------------------------------------------------


def test_hooks_produce_no_stdout(tmp_path: Path):
    env = _env(tmp_path)
    sid = "stdout-check"

    for name, payload in [
        ("thirdeye-claude-session-start", {"session_id": sid, "cwd": "/p"}),
        ("thirdeye-claude-user-prompt-submit", {"session_id": sid, "cwd": "/p", "prompt": "hi"}),
        ("thirdeye-claude-pre-tool-use", {"session_id": sid, "cwd": "/p", "tool_name": "X"}),
        ("thirdeye-claude-post-tool-use", {"session_id": sid, "cwd": "/p", "tool_name": "X"}),
        ("thirdeye-claude-stop", {"session_id": sid, "cwd": "/p"}),
        ("thirdeye-claude-subagent-stop", {"session_id": sid, "cwd": "/p"}),
        ("thirdeye-claude-stop-failure", {"session_id": sid, "cwd": "/p", "error": "e"}),
        ("thirdeye-claude-notification", {"session_id": sid, "cwd": "/p", "msg": "n"}),
        ("thirdeye-claude-permission-request", {"session_id": sid, "cwd": "/p"}),
        ("thirdeye-claude-session-end", {"session_id": sid, "cwd": "/p"}),
    ]:
        r = _hook(name, payload, env)
        assert r.stdout == b"", f"{name} wrote to stdout: {r.stdout!r}"


# -- all 10 hook types fire correctly as subprocesses --------------------------


def test_all_ten_hook_types_as_subprocess(tmp_path: Path):
    env = _env(tmp_path)
    sid = "all-hooks-001"

    hooks_in_order = [
        ("thirdeye-claude-session-start", "session_start"),
        ("thirdeye-claude-user-prompt-submit", "user_message"),
        ("thirdeye-claude-pre-tool-use", "tool_call"),
        ("thirdeye-claude-post-tool-use", "tool_result"),
        ("thirdeye-claude-stop", "assistant_message"),
        ("thirdeye-claude-subagent-stop", "subagent_message"),
        ("thirdeye-claude-stop-failure", "error"),
        ("thirdeye-claude-notification", "notification"),
        ("thirdeye-claude-permission-request", "permission_request"),
        ("thirdeye-claude-session-end", "session_end"),
    ]

    for script_name, _ in hooks_in_order:
        r = _hook(script_name, {"session_id": sid, "cwd": "/proj"}, env)
        assert r.returncode == 0, f"{script_name} failed: {r.stderr.decode()}"

    r = _thirdeye("events", sid, env=env)
    assert r.returncode == 0
    lines = r.stdout.decode().strip().splitlines()
    assert len(lines) == 10
    types = [line.split()[1] for line in lines]
    assert types == [t for _, t in hooks_in_order]


# -- multiple sessions via hooks -----------------------------------------------


def test_multiple_sessions_independent(tmp_path: Path):
    env = _env(tmp_path)
    sid_a = "multi-sess-aaa"
    sid_b = "multi-sess-bbb"

    _hook("thirdeye-claude-session-start", {"session_id": sid_a, "cwd": "/a"}, env)
    _hook(
        "thirdeye-claude-user-prompt-submit",
        {"session_id": sid_a, "cwd": "/a", "prompt": "alpha"},
        env,
    )
    _hook("thirdeye-claude-session-start", {"session_id": sid_b, "cwd": "/b"}, env)
    _hook("thirdeye-claude-session-end", {"session_id": sid_a, "cwd": "/a"}, env)

    r = _thirdeye("list", env=env)
    assert r.returncode == 0
    metas = [json.loads(line) for line in r.stdout.decode().strip().splitlines()]
    by_sid = {m["session_id"]: m for m in metas}
    assert by_sid[sid_a]["status"] == "closed"
    assert by_sid[sid_a]["event_count"] == 3
    assert by_sid[sid_b]["status"] == "open"
    assert by_sid[sid_b]["event_count"] == 1


# -- CLI queries on hook-created data ------------------------------------------


def test_search_on_hook_created_data(tmp_path: Path):
    env = _env(tmp_path)
    sid = "search-hook-001"

    _hook("thirdeye-claude-session-start", {"session_id": sid, "cwd": "/proj"}, env)
    _hook(
        "thirdeye-claude-user-prompt-submit",
        {
            "session_id": sid,
            "cwd": "/proj",
            "prompt": "xylophone_unique_word",
        },
        env,
    )
    _hook("thirdeye-claude-session-end", {"session_id": sid, "cwd": "/proj"}, env)

    r = _thirdeye("search", "xylophone_unique_word", env=env)
    assert r.returncode == 0
    assert sid in r.stdout.decode()


def test_stats_on_hook_created_data(tmp_path: Path):
    env = _env(tmp_path)
    sid = "stats-hook-001"

    _hook("thirdeye-claude-session-start", {"session_id": sid, "cwd": "/proj"}, env)
    _hook(
        "thirdeye-claude-user-prompt-submit",
        {"session_id": sid, "cwd": "/proj", "prompt": "hi"},
        env,
    )
    _hook("thirdeye-claude-session-end", {"session_id": sid, "cwd": "/proj"}, env)

    r = _thirdeye("stats", env=env)
    assert r.returncode == 0
    obj = json.loads(r.stdout.decode().strip())
    assert obj["session_count"] == 1
    assert obj["event_count"] == 3

    r = _thirdeye("stats", sid, env=env)
    assert r.returncode == 0
    obj = json.loads(r.stdout.decode().strip())
    assert obj["session_id"] == sid
    assert obj["platform"] == "claude"
    assert obj["event_count"] == 3


def test_event_field_extraction_on_hook_data(tmp_path: Path):
    env = _env(tmp_path)
    sid = "field-ext-001"

    _hook("thirdeye-claude-session-start", {"session_id": sid, "cwd": "/proj"}, env)
    _hook(
        "thirdeye-claude-pre-tool-use",
        {
            "session_id": sid,
            "cwd": "/proj",
            "tool_name": "Bash",
            "tool_use_id": "tu_99",
            "tool_input": {"command": "ls -la"},
        },
        env,
    )

    r = _thirdeye("event", sid, "1", "--field", "tool_name", env=env)
    assert r.returncode == 0
    assert r.stdout.decode().strip() == "Bash"

    r = _thirdeye("event", sid, "1", "--field", "tool_use_id", env=env)
    assert r.returncode == 0
    assert r.stdout.decode().strip() == "tu_99"


def test_events_json_mode_on_hook_data(tmp_path: Path):
    env = _env(tmp_path)
    sid = "json-mode-001"

    _hook("thirdeye-claude-session-start", {"session_id": sid, "cwd": "/proj"}, env)
    _hook(
        "thirdeye-claude-user-prompt-submit",
        {"session_id": sid, "cwd": "/proj", "prompt": "hi"},
        env,
    )

    r = _thirdeye("events", sid, "--json", env=env)
    assert r.returncode == 0
    lines = r.stdout.decode().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "t" in parsed
        assert "ts" in parsed
        assert "seq" in parsed


# -- payload preservation via subprocess ---------------------------------------


def test_nested_payload_preserved_via_subprocess(tmp_path: Path):
    env = _env(tmp_path)
    sid = "nested-payload-001"

    _hook("thirdeye-claude-session-start", {"session_id": sid, "cwd": "/p"}, env)
    _hook(
        "thirdeye-claude-pre-tool-use",
        {
            "session_id": sid,
            "cwd": "/p",
            "tool_name": "Read",
            "tool_input": {"nested": {"deep": True, "list": [1, 2, 3]}},
        },
        env,
    )

    r = _thirdeye("event", sid, "1", env=env)
    assert r.returncode == 0
    expanded = json.loads(r.stdout.decode())
    assert expanded["data"]["tool_input"] == {"nested": {"deep": True, "list": [1, 2, 3]}}
    assert "session_id" not in expanded["data"]


def test_large_payload_via_subprocess(tmp_path: Path):
    env = _env(tmp_path)
    sid = "large-payload-001"
    big_content = "x" * 10000

    _hook(
        "thirdeye-claude-stop",
        {
            "session_id": sid,
            "cwd": "/p",
            "content": big_content,
        },
        env,
    )

    r = _thirdeye("event", sid, "0", env=env)
    assert r.returncode == 0
    expanded = json.loads(r.stdout.decode())
    assert len(expanded["data"]["content"]) == 10000


# -- install verifies settings content ----------------------------------------


def test_install_writes_all_hooks_to_settings(tmp_path: Path):
    settings_file = tmp_path / ".claude" / "settings.json"
    from thirdeye.platforms.claude.constants import HOOK_EVENTS
    from thirdeye.platforms.claude.install import ClaudePlatform

    ClaudePlatform(settings_file=settings_file).install()
    data = json.loads(settings_file.read_text())
    assert set(data["hooks"].keys()) == set(HOOK_EVENTS.keys())
    for event_name, entries in data["hooks"].items():
        cmds = [h["command"] for entry in entries for h in entry["hooks"]]
        assert any("thirdeye-claude" in c for c in cmds), f"no thirdeye command for {event_name}"


def test_install_idempotent_via_e2e(tmp_path: Path):
    settings_file = tmp_path / "settings.json"
    from thirdeye.platforms.claude.install import ClaudePlatform

    p = ClaudePlatform(settings_file=settings_file)
    p.install()
    first = settings_file.read_text()
    p.install()
    second = settings_file.read_text()
    assert first == second


def test_uninstall_preserves_other_settings(tmp_path: Path):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"theme": "dark"}))
    from thirdeye.platforms.claude.install import ClaudePlatform

    p = ClaudePlatform(settings_file=settings_file)
    p.install()
    p.uninstall()
    data = json.loads(settings_file.read_text())
    assert data["theme"] == "dark"
    assert "hooks" not in data


# -- list with platform filter -------------------------------------------------


def test_list_platform_filter_on_hook_data(tmp_path: Path):
    env = _env(tmp_path)
    sid = "plat-filter-001"

    _hook("thirdeye-claude-session-start", {"session_id": sid, "cwd": "/proj"}, env)

    r = _thirdeye("list", "--platform", "claude", env=env)
    assert r.returncode == 0
    assert sid in r.stdout.decode()

    r = _thirdeye("list", "--platform", "cursor", env=env)
    assert r.returncode == 0
    assert sid not in r.stdout.decode()
