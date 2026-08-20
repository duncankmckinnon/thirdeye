from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.store import Store
from thirdeye.usage.store import UsageStore

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.jsonl"
FIXTURE_SID = "019fb579-cdda-7a03-86df-65c87b6c4ae2"


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    # Sandbox the rollout sessions root so notify() never scans the developer's
    # real ~/.codex/sessions. Empty by default -> resolve_rollout returns None.
    sessions_root = tmp_path / "codex_sessions"
    sessions_root.mkdir()
    monkeypatch.setattr("thirdeye.platforms.codex.rollout.CODEX_SESSIONS_ROOT", sessions_root)
    return tmp_path


def _sessions_root(env: Path) -> Path:
    return env / "codex_sessions"


def _plant_rollout(env: Path, sid: str = FIXTURE_SID) -> Path:
    """Copy the real rollout fixture into the sandboxed sessions tree."""
    nested = _sessions_root(env) / "2026" / "07" / "30"
    nested.mkdir(parents=True, exist_ok=True)
    dest = nested / f"rollout-2026-07-30T17-01-26-{sid}.jsonl"
    dest.write_bytes(FIXTURE.read_bytes())
    return dest


def _usage_rows(env: Path, sid: str) -> list:
    return list(UsageStore(session_dir(env, "codex", sid)).iter_rows())


def _argv(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.argv", ["notify", json.dumps(payload)])


# -- _read_argv ----------------------------------------------------------------


class TestReadArgv:
    def test_valid_json_returns_dict(self, monkeypatch):
        from thirdeye.platforms.codex import hooks

        _argv(monkeypatch, {"thread-id": "abc", "cwd": "/p"})
        result = hooks._read_argv()
        assert result == {"thread-id": "abc", "cwd": "/p"}

    def test_missing_argv_returns_empty_dict(self, monkeypatch):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setattr("sys.argv", ["notify"])
        assert hooks._read_argv() == {}

    def test_invalid_json_returns_empty_dict(self, monkeypatch):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setattr("sys.argv", ["notify", "not json"])
        assert hooks._read_argv() == {}

    def test_empty_string_argv_returns_empty_dict(self, monkeypatch):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setattr("sys.argv", ["notify", ""])
        assert hooks._read_argv() == {}


# -- _strip_payload ------------------------------------------------------------


class TestStripPayload:
    def test_strips_thread_id_kebab(self):
        from thirdeye.platforms.codex import hooks

        result = hooks._strip_payload({"thread-id": "abc", "extra": 1})
        assert "thread-id" not in result

    def test_strips_thread_id_snake(self):
        from thirdeye.platforms.codex import hooks

        result = hooks._strip_payload({"thread_id": "abc", "extra": 1})
        assert "thread_id" not in result

    def test_strips_threadId_camel(self):
        from thirdeye.platforms.codex import hooks

        result = hooks._strip_payload({"threadId": "abc", "extra": 1})
        assert "threadId" not in result

    def test_strips_cwd(self):
        from thirdeye.platforms.codex import hooks

        result = hooks._strip_payload({"cwd": "/p", "extra": 1})
        assert "cwd" not in result

    def test_strips_working_directory_kebab(self):
        from thirdeye.platforms.codex import hooks

        result = hooks._strip_payload({"working-directory": "/p", "extra": 1})
        assert "working-directory" not in result

    def test_strips_working_directory_snake(self):
        from thirdeye.platforms.codex import hooks

        result = hooks._strip_payload({"working_directory": "/p", "extra": 1})
        assert "working_directory" not in result

    def test_preserves_other_keys(self):
        from thirdeye.platforms.codex import hooks

        payload = {
            "thread-id": "abc",
            "cwd": "/p",
            "type": "agent-turn-complete",
            "token_usage": {"input": 100},
        }
        result = hooks._strip_payload(payload)
        assert result == {"type": "agent-turn-complete", "token_usage": {"input": 100}}


# -- _emit ---------------------------------------------------------------------


class TestEmit:
    def test_routes_thread_id_kebab_as_session_id(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        result = hooks._emit("agent_turn", {"thread-id": "abc-123", "cwd": "/p"})
        assert result is not None
        metas = list(Store(Config.load()).list_sessions())
        assert len(metas) == 1
        assert metas[0].session_id == "abc-123"

    def test_routes_thread_id_snake_as_session_id(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        result = hooks._emit("agent_turn", {"thread_id": "snake-123", "cwd": "/p"})
        assert result is not None
        metas = list(Store(Config.load()).list_sessions())
        assert len(metas) == 1
        assert metas[0].session_id == "snake-123"

    def test_routes_threadId_camel_as_session_id(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        result = hooks._emit("agent_turn", {"threadId": "camel-123", "cwd": "/p"})
        assert result is not None
        metas = list(Store(Config.load()).list_sessions())
        assert len(metas) == 1
        assert metas[0].session_id == "camel-123"

    def test_uses_cwd_from_payload(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        hooks._emit("agent_turn", {"thread-id": "s1", "cwd": "/my/project"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == "/my/project"

    def test_uses_working_directory_if_cwd_absent(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        hooks._emit("agent_turn", {"thread-id": "s1", "working-directory": "/alt/path"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == "/alt/path"

    def test_falls_back_to_os_getcwd(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        monkeypatch.chdir(env)
        hooks._emit("agent_turn", {"thread-id": "s1"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == str(env)

    def test_returns_none_when_no_thread_id(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        result = hooks._emit("agent_turn", {"cwd": "/p"})
        assert result is None
        assert list(Store(Config.load()).list_sessions()) == []

    def test_stores_platform_as_codex(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        hooks._emit("agent_turn", {"thread-id": "s1", "cwd": "/p"})
        m = next(Store(Config.load()).list_sessions())
        assert m.platform == "codex"

    def test_strips_routing_keys_from_data(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        hooks._emit("agent_turn", {"thread-id": "s1", "cwd": "/p", "extra": 42})
        events = list(Store(Config.load()).reader("s1").iter_events())
        data = events[0].get("data", {})
        assert "thread-id" not in data
        assert "cwd" not in data
        assert data["extra"] == 42


# -- notify --------------------------------------------------------------------


class TestNotify:
    def test_agent_turn_complete_creates_event(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        payload = {
            "type": "agent-turn-complete",
            "thread-id": "t-001",
            "cwd": "/project",
            "token_usage": {"input": 100, "output": 50},
        }
        _argv(monkeypatch, payload)
        hooks.notify()
        store = Store(Config.load())
        metas = list(store.list_sessions())
        assert len(metas) == 1
        assert metas[0].session_id == "t-001"
        assert metas[0].platform == "codex"
        events = list(store.reader("t-001").iter_events())
        assert len(events) == 1
        assert events[0]["t"] == "agent_turn"
        data = events[0]["data"]
        assert "thread-id" not in data
        assert "cwd" not in data
        assert data["token_usage"] == {"input": 100, "output": 50}

    def test_wrong_type_is_silent_noop(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        payload = {"type": "something-else", "thread-id": "t-001", "cwd": "/p"}
        _argv(monkeypatch, payload)
        hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_no_payload_is_silent_noop(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setattr("sys.argv", ["notify"])
        hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_bad_json_in_argv_is_silent_noop(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setattr("sys.argv", ["notify", "not valid json"])
        hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_preserves_nested_fields_in_data(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        payload = {
            "type": "agent-turn-complete",
            "thread-id": "t-002",
            "cwd": "/proj",
            "last-assistant-message": "Here is the answer.",
            "input-messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "token_usage": {"input": 200, "output": 100, "total": 300},
            "tool_calls": [
                {"name": "Read", "input": {"file": "x.py"}},
                {"name": "Bash", "input": {"command": "ls"}},
            ],
        }
        _argv(monkeypatch, payload)
        hooks.notify()
        events = list(Store(Config.load()).reader("t-002").iter_events())
        data = events[0]["data"]
        assert data["last-assistant-message"] == "Here is the answer."
        assert data["input-messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert data["token_usage"] == {"input": 200, "output": 100, "total": 300}
        assert data["tool_calls"] == [
            {"name": "Read", "input": {"file": "x.py"}},
            {"name": "Bash", "input": {"command": "ls"}},
        ]

    def test_does_not_print_to_stdout(self, monkeypatch, env: Path, capsys):
        from thirdeye.platforms.codex import hooks

        payload = {
            "type": "agent-turn-complete",
            "thread-id": "t-003",
            "cwd": "/p",
        }
        _argv(monkeypatch, payload)
        hooks.notify()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_noop_does_not_print_to_stdout(self, monkeypatch, env: Path, capsys):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setattr("sys.argv", ["notify"])
        hooks.notify()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_missing_thread_id_is_silent_noop(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        payload = {"type": "agent-turn-complete", "cwd": "/p"}
        _argv(monkeypatch, payload)
        hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []


# -- platform constant ---------------------------------------------------------


class TestPlatformConstant:
    def test_platform_is_codex(self):
        from thirdeye.platforms.codex import hooks

        assert hooks._PLATFORM == "codex"


# -- _flex_get -----------------------------------------------------------------


class TestFlexGet:
    def test_returns_first_matching_key(self):
        from thirdeye.platforms.codex import hooks

        d = {"thread-id": "abc"}
        assert hooks._flex_get(d, "thread-id", "thread_id", "threadId") == "abc"

    def test_returns_second_key_if_first_missing(self):
        from thirdeye.platforms.codex import hooks

        d = {"thread_id": "def"}
        assert hooks._flex_get(d, "thread-id", "thread_id", "threadId") == "def"

    def test_returns_third_key_if_others_missing(self):
        from thirdeye.platforms.codex import hooks

        d = {"threadId": "ghi"}
        assert hooks._flex_get(d, "thread-id", "thread_id", "threadId") == "ghi"

    def test_returns_default_when_no_keys_match(self):
        from thirdeye.platforms.codex import hooks

        d = {"other": "val"}
        assert hooks._flex_get(d, "thread-id", "thread_id", "threadId") is None

    def test_returns_custom_default(self):
        from thirdeye.platforms.codex import hooks

        d = {"other": "val"}
        assert hooks._flex_get(d, "thread-id", default="fallback") == "fallback"

    def test_skips_empty_string_values(self):
        from thirdeye.platforms.codex import hooks

        d = {"thread-id": "", "thread_id": "notempty"}
        assert hooks._flex_get(d, "thread-id", "thread_id") == "notempty"

    def test_skips_none_values(self):
        from thirdeye.platforms.codex import hooks

        d = {"thread-id": None, "threadId": "found"}
        assert hooks._flex_get(d, "thread-id", "threadId") == "found"


# -- session_start env capture -> auto tags -----------------------------------


class TestSessionStartEnvTags:
    def _tags_lines(self, env: Path, sid: str) -> list[dict]:
        from thirdeye.paths import session_dir, tags_path

        path = tags_path(session_dir(env, "codex", sid))
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_no_patterns_set_writes_no_tags(self, monkeypatch, env: Path):
        from thirdeye.paths import session_dir, tags_path
        from thirdeye.platforms.codex import hooks

        monkeypatch.delenv("THIRDEYE_CAPTURE_ENV", raising=False)
        monkeypatch.setenv("WB_PLAN", "p")
        monkeypatch.setenv("WB_STEP", "test#1")
        _argv(monkeypatch, {"thread-id": "s1", "cwd": "/p"})
        hooks.session_start()
        assert not tags_path(session_dir(env, "codex", "s1")).exists()

    def test_matching_env_vars_become_auto_tags(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
        monkeypatch.setenv("WB_PLAN", "p")
        monkeypatch.setenv("WB_STEP", "test#1")
        _argv(monkeypatch, {"thread-id": "s1", "cwd": "/p"})
        hooks.session_start()

        events = list(Store(Config.load()).reader("s1").iter_events())
        start_seq = events[0]["seq"]

        lines = self._tags_lines(env, "s1")
        tags = {line["tag"] for line in lines}
        assert "plan-p" in tags
        assert "step-test#1" in tags
        for line in lines:
            assert line["op"] == "add"
            assert line["source"] == "auto"
            assert line["seq"] == start_seq

    def test_invalid_tag_value_is_skipped(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
        monkeypatch.setenv("WB_PLAN", "p")
        monkeypatch.setenv("WB_X", "===")
        _argv(monkeypatch, {"thread-id": "s1", "cwd": "/p"})
        hooks.session_start()

        lines = self._tags_lines(env, "s1")
        tags = {line["tag"] for line in lines}
        assert "plan-p" in tags
        assert not any(t.startswith("x-") for t in tags)

    def test_missing_session_id_writes_no_tags(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
        monkeypatch.setenv("WB_PLAN", "p")
        _argv(monkeypatch, {"cwd": "/p"})
        hooks.session_start()
        assert list(Store(Config.load()).list_sessions()) == []


# -- notify: rollout event + usage wiring --------------------------------------


class TestNotifyRolloutWiring:
    def _payload(self, sid: str = FIXTURE_SID, key: str = "thread-id") -> dict:
        return {
            "type": "agent-turn-complete",
            key: sid,
            "cwd": "/proj/codex",
            "input-messages": ["fix the bug"],
            "last-assistant-message": "Fixed it.",
        }

    def test_type_not_agent_turn_complete_produces_nothing(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        _plant_rollout(env)
        _argv(monkeypatch, {"type": "session-configured", "thread-id": FIXTURE_SID})
        hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []
        assert _usage_rows(env, FIXTURE_SID) == []

    def test_no_thread_id_produces_nothing_and_does_not_raise(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        _plant_rollout(env)
        _argv(monkeypatch, {"type": "agent-turn-complete", "cwd": "/p"})
        hooks.notify()  # must not raise
        assert list(Store(Config.load()).list_sessions()) == []

    def test_resolvable_rollout_appends_events_and_usage(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        _plant_rollout(env)
        _argv(monkeypatch, self._payload())
        hooks.notify()

        events = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        types = {e["t"] for e in events}
        # The agent_turn from the notify payload, PLUS rollout-derived events.
        assert "agent_turn" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "user_message" in types
        # Exactly one agent_turn; the rest come from the rollout.
        assert sum(1 for e in events if e["t"] == "agent_turn") == 1
        assert len(events) > 1
        # Usage rows were extracted from the rollout's token_count frames.
        assert len(_usage_rows(env, FIXTURE_SID)) > 0

    def test_agent_turn_data_carries_no_usage_attributes(self, monkeypatch, env: Path):
        """Codex usage stays in the local sqlite sidecar only for now — it
        isn't exported to Logfire at all (unlike Claude, whose individual LLM
        calls export via extract_new_calls_claude / export_llm_calls). A
        rollout's ~80 token_count entries routinely correlate with only a
        handful of visible content frames, so there's no clean per-call
        content to attach the way there is for Claude's transcript.
        """
        from thirdeye.platforms.codex import hooks

        _plant_rollout(env)
        _argv(monkeypatch, self._payload())
        hooks.notify()

        events = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        agent_turn = next(e for e in events if e["t"] == "agent_turn")
        assert not any(k.startswith("gen_ai.") for k in agent_turn["data"])

    def test_thread_id_kebab_snake_camel_all_accepted(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        for key, sid in (
            ("thread-id", "kebab-sid"),
            ("thread_id", "snake-sid"),
            ("threadId", "camel-sid"),
        ):
            _argv(monkeypatch, self._payload(sid=sid, key=key))
            hooks.notify()
            events = list(Store(Config.load()).reader(sid).iter_events())
            assert len(events) == 1
            assert events[0]["t"] == "agent_turn"

    def test_agent_turn_data_strips_routing_keeps_message(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        _argv(monkeypatch, self._payload())
        hooks.notify()

        events = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        agent_turn = next(e for e in events if e["t"] == "agent_turn")
        data = agent_turn["data"]
        assert "thread-id" not in data
        assert "cwd" not in data
        assert data["last-assistant-message"] == "Fixed it."

    def test_unresolvable_rollout_still_records_agent_turn_and_logs(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        # No rollout planted -> resolve_rollout returns None.
        _argv(monkeypatch, self._payload())
        hooks.notify()

        events = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        assert len(events) == 1
        assert events[0]["t"] == "agent_turn"
        assert _usage_rows(env, FIXTURE_SID) == []

        errlog = env / "logs" / "usage-errors.jsonl"
        assert errlog.exists()
        entries = [json.loads(line) for line in errlog.read_text().splitlines() if line.strip()]
        assert any(e["phase"] == "open_source" for e in entries)

    def test_two_notifications_unchanged_rollout_add_no_new_rollout_events(
        self, monkeypatch, env: Path
    ):
        from thirdeye.platforms.codex import hooks

        _plant_rollout(env)
        _argv(monkeypatch, self._payload())
        hooks.notify()

        events1 = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        rollout_events1 = sum(1 for e in events1 if e["t"] != "agent_turn")
        usage1 = len(_usage_rows(env, FIXTURE_SID))
        assert rollout_events1 > 0
        assert usage1 > 0

        _argv(monkeypatch, self._payload())
        hooks.notify()

        events2 = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        # agent_turn appended a second time...
        assert sum(1 for e in events2 if e["t"] == "agent_turn") == 2
        # ...but no new rollout-derived events and no new usage rows.
        assert sum(1 for e in events2 if e["t"] != "agent_turn") == rollout_events1
        assert len(_usage_rows(env, FIXTURE_SID)) == usage1

    def test_rollout_path_and_offset_persisted_in_state(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        rollout = _plant_rollout(env)
        _argv(monkeypatch, self._payload())
        hooks.notify()

        state = UsageStore(session_dir(env, "codex", FIXTURE_SID)).read_state()
        assert state.get("rollout_path") == str(rollout)
        assert state.get("rollout_offset") == rollout.stat().st_size

    def test_exception_from_events_capture_does_not_propagate(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        _plant_rollout(env)

        def _boom(**kwargs):
            raise RuntimeError("events capture blew up")

        monkeypatch.setattr("thirdeye.platforms.codex.events.capture_events_codex", _boom)
        _argv(monkeypatch, self._payload())
        hooks.notify()  # must not raise

        # agent_turn is appended before the capture calls, so it survives.
        events = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        assert any(e["t"] == "agent_turn" for e in events)
        # The bookmark must NOT advance when a capture step fails: the final
        # write_state is never reached, so the range replays next run rather than
        # being silently skipped ("do not advance the offset before both capture
        # calls complete").
        state = UsageStore(session_dir(env, "codex", FIXTURE_SID)).read_state()
        assert state.get("rollout_offset", 0) == 0

    def test_exception_from_usage_capture_does_not_propagate(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        _plant_rollout(env)

        def _boom(**kwargs):
            raise RuntimeError("usage capture blew up")

        monkeypatch.setattr("thirdeye.platforms.codex.usage.capture_usage_codex", _boom)
        _argv(monkeypatch, self._payload())
        hooks.notify()  # must not raise

        events = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        assert any(e["t"] == "agent_turn" for e in events)
