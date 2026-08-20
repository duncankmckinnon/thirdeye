"""Tests for platforms/codex/hooks_json.py — Codex's newer, stdin-delivered,
Claude-hooks-shaped hooks.json mechanism. Distinct from test_codex_hooks.py,
which covers the older argv/thread-id-keyed notify + session_start.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir, tags_path
from thirdeye.platforms.codex import hooks_json
from thirdeye.store import Store


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    return tmp_path


def _stdin(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


def _tags_lines(env: Path, sid: str) -> list[dict]:
    path = tags_path(session_dir(env, "codex", sid))
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- session_start ---------------------------------------------------------


class TestSessionStart:
    def test_creates_session_tagged_codex(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc-123", "cwd": "/proj/x", "source": "cli"})
        hooks_json.session_start()
        store = Store(Config.load())
        metas = list(store.list_sessions())
        assert len(metas) == 1
        m = metas[0]
        assert m.session_id == "abc-123"
        assert m.platform == "codex"
        assert m.cwd == "/proj/x"

    def test_event_type_is_session_start(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["t"] == "session_start"

    def test_missing_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks_json.session_start()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_env_capture_becomes_auto_tags(self, monkeypatch, env: Path):
        monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
        monkeypatch.setenv("WB_PLAN", "p")
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        tags = {line["tag"] for line in _tags_lines(env, "s1")}
        assert "plan-p" in tags


# -- user_prompt_submit ------------------------------------------------------


class TestUserPromptSubmit:
    def test_appends_user_message_tagged_codex(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "hello"})
        hooks_json.user_prompt_submit()
        store = Store(Config.load())
        events = list(store.reader("s1").iter_events())
        assert events[1]["t"] == "user_message"
        assert events[1]["data"]["prompt"] == "hello"
        assert "session_id" not in events[1].get("data", {})
        assert store.list_sessions().__next__().platform == "codex"

    def test_hashtags_become_auto_tags(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "fix the #bug please"})
        hooks_json.user_prompt_submit()
        tags = {line["tag"] for line in _tags_lines(env, "s1")}
        assert "bug" in tags

    def test_missing_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p", "prompt": "hi"})
        hooks_json.user_prompt_submit()
        assert list(Store(Config.load()).list_sessions()) == []


# -- subagent_start / subagent_stop -----------------------------------------


class TestSubagentStartStop:
    def test_subagent_start_event_type(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "agent": "explore"})
        hooks_json.subagent_start()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "subagent_start"
        assert events[1]["data"]["agent"] == "explore"

    def test_subagent_stop_uses_subagent_message_type(self, monkeypatch, env: Path):
        # Matches claude/hooks.py's naming for the same event concept, not a
        # historical-data constraint of codex's own — see hooks_json.py.
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "agent": "explore"})
        hooks_json.subagent_stop()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "subagent_message"


# -- permission_request / pre_compact / post_compact -------------------------


class TestSimpleEmitters:
    def test_permission_request(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "tool_name": "Bash"})
        hooks_json.permission_request()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "permission_request"

    def test_pre_compact_maps_to_compact_start(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.pre_compact()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "compact_start"

    def test_post_compact_maps_to_compact_end(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.post_compact()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "compact_end"


# -- session_end --------------------------------------------------------------


class TestSessionEnd:
    def test_closes_session(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks_json.session_start()
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks_json.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.status == "closed"
        assert m.platform == "codex"
        events = list(store.reader("abc").iter_events())
        assert events[-1]["t"] == "session_end"

    def test_missing_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks_json.session_end()
        assert list(Store(Config.load()).list_sessions()) == []
