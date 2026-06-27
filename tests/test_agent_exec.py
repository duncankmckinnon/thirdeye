from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thirdeye.agent.exec import run_agent_streaming
from thirdeye.agent.harness import AgentHarness
from thirdeye.eval.agents.claude import ClaudeAdapter


def _make_harness(mode: str = "review") -> AgentHarness:
    return AgentHarness(ClaudeAdapter(), mode)


def _make_mock_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    proc = MagicMock()
    proc.stdout = io.StringIO(stdout)
    proc.stderr = io.StringIO(stderr)
    proc.returncode = returncode
    proc.wait = MagicMock(return_value=returncode)
    return proc


def test_raises_file_not_found_when_binary_missing():
    harness = _make_harness()
    with patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="claude"):
            run_agent_streaming(harness, "x", Path("/tmp"))


def test_streams_stdout_to_output_callback():
    captured: list[str] = []
    mock_proc = _make_mock_proc(stdout="line one\nline two\n")

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        run_agent_streaming(_make_harness(), "task", Path("/tmp"), output=captured.append)

    assert captured == ["line one\n", "line two\n"]


def test_returns_returncode_zero_on_success():
    mock_proc = _make_mock_proc(stdout="ok\n", returncode=0)

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        rc, duration = run_agent_streaming(
            _make_harness(), "task", Path("/tmp"), output=lambda _: None
        )

    assert rc == 0


def test_returns_nonzero_returncode_on_failure():
    mock_proc = _make_mock_proc(stdout="", stderr="something broke\n", returncode=1)

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        rc, _ = run_agent_streaming(_make_harness(), "task", Path("/tmp"), output=lambda _: None)

    assert rc == 1


def test_returns_duration_ms_as_int():
    mock_proc = _make_mock_proc(stdout="x\n")

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        _, duration = run_agent_streaming(
            _make_harness(), "task", Path("/tmp"), output=lambda _: None
        )

    assert isinstance(duration, int)
    assert duration >= 0


def test_empty_stdout_produces_no_callback_calls():
    captured: list[str] = []
    mock_proc = _make_mock_proc(stdout="")

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        run_agent_streaming(_make_harness(), "task", Path("/tmp"), output=captured.append)

    assert captured == []


def test_default_output_uses_click_echo(capsys):
    mock_proc = _make_mock_proc(stdout="hello\n")

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        run_agent_streaming(_make_harness(), "task", Path("/tmp"))

    out = capsys.readouterr().out
    assert "hello" in out


def test_build_command_receives_composed_prompt():
    built_cmd: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        built_cmd.append(cmd)
        return _make_mock_proc(stdout="ok\n")

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", side_effect=_fake_popen),
    ):
        run_agent_streaming(
            _make_harness(),
            "my composed prompt",
            Path("/tmp"),
            output=lambda _: None,
        )

    assert len(built_cmd) == 1
    assert "my composed prompt" in built_cmd[0]


# --- tagging: new sessions are tagged 'thirdeye-agent' ---


def test_no_thirdeye_home_skips_tagging():
    """Without thirdeye_home, no session snapshot or tag write occurs."""
    mock_proc = _make_mock_proc(stdout="ok\n")
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("thirdeye.agent.exec._list_sessions") as mock_list,
        patch("thirdeye.agent.exec._tag_sessions") as mock_tag,
    ):
        run_agent_streaming(_make_harness(), "x", Path("/tmp"), output=lambda _: None)

    mock_list.assert_not_called()
    mock_tag.assert_not_called()


def test_thirdeye_home_triggers_pre_and_post_snapshot(tmp_path):
    """When thirdeye_home is given, sessions are snapshotted before and after."""
    mock_proc = _make_mock_proc(stdout="ok\n")
    call_log: list[str] = []

    def _fake_list(home, platform):
        call_log.append(platform)
        return set()

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("thirdeye.agent.exec._list_sessions", side_effect=_fake_list),
        patch("thirdeye.agent.exec._tag_sessions"),
    ):
        run_agent_streaming(
            _make_harness(),
            "x",
            Path("/tmp"),
            output=lambda _: None,
            thirdeye_home=tmp_path,
        )

    # called twice: pre-run and post-run, both for the harness platform ("claude")
    assert call_log == ["claude", "claude"]


def test_new_sessions_are_tagged(tmp_path):
    """Session IDs that appear after the run receive the 'thirdeye-agent' tag."""
    mock_proc = _make_mock_proc(stdout="ok\n")
    tagged: list[tuple] = []
    list_calls = [0]

    def _fake_list(home, platform):
        # First call (pre-run) returns existing session; second (post-run) adds a new one
        list_calls[0] += 1
        return {"existing-sid"} if list_calls[0] == 1 else {"existing-sid", "new-sid"}

    def _fake_tag(home, platform, sids):
        tagged.append((platform, frozenset(sids)))

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("thirdeye.agent.exec._list_sessions", side_effect=_fake_list),
        patch("thirdeye.agent.exec._tag_sessions", side_effect=_fake_tag),
    ):
        run_agent_streaming(
            _make_harness(),
            "x",
            Path("/tmp"),
            output=lambda _: None,
            thirdeye_home=tmp_path,
        )

    assert len(tagged) == 1
    platform, sids = tagged[0]
    assert platform == "claude"
    assert sids == frozenset({"new-sid"})


def test_no_new_sessions_means_no_tagging(tmp_path):
    """If no new sessions appeared, _tag_sessions is called with an empty set."""
    mock_proc = _make_mock_proc(stdout="ok\n")
    tagged: list = []

    def _fake_tag(home, platform, sids):
        tagged.append((platform, frozenset(sids)))

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("thirdeye.agent.exec._list_sessions", return_value={"sid-a"}),
        patch("thirdeye.agent.exec._tag_sessions", side_effect=_fake_tag),
    ):
        run_agent_streaming(
            _make_harness(),
            "x",
            Path("/tmp"),
            output=lambda _: None,
            thirdeye_home=tmp_path,
        )

    # _tag_sessions receives empty set → it should short-circuit without writing
    assert tagged[0] == ("claude", frozenset())


def test_integration_list_and_tag_sessions(tmp_path):
    """Integration test that uses the real Store and TagStore to test session listing and tagging."""
    from thirdeye.config import Config
    from thirdeye.paths import session_dir
    from thirdeye.store import Store
    from thirdeye.tags import TagStore

    config = Config(root=tmp_path)
    store = Store(config)

    # 1. Create an existing session
    store.append_event(
        session_id="session-existing",
        platform="claude",
        cwd="/proj/foo",
        t="input",
        data="hello",
    )

    # 2. Setup mock Popen that adds a new session during run
    mock_proc = _make_mock_proc(stdout="run completed\n")

    def _fake_popen(cmd, **kwargs):
        # Simulator agent creating a new session
        store.append_event(
            session_id="session-new",
            platform="claude",
            cwd="/proj/foo",
            t="input",
            data="hello from agent",
        )
        return mock_proc

    harness = _make_harness()
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.Popen", side_effect=_fake_popen),
    ):
        rc, _ = run_agent_streaming(
            harness,
            "task text",
            cwd=Path("/proj/foo"),
            output=lambda _: None,
            thirdeye_home=tmp_path,
        )

    assert rc == 0

    # 3. Check that both sessions are in the Store
    sessions = {s.session_id for s in store.list_sessions(platform="claude")}
    assert "session-existing" in sessions
    assert "session-new" in sessions

    # 4. Check that only the new session has the 'thirdeye-agent' tag on sequence 0
    existing_sd = session_dir(tmp_path, "claude", "session-existing")
    new_sd = session_dir(tmp_path, "claude", "session-new")

    existing_ts = TagStore(existing_sd)
    new_ts = TagStore(new_sd)

    assert "thirdeye-agent" not in existing_ts.tags_for(0)
    assert "thirdeye-agent" in new_ts.tags_for(0)
