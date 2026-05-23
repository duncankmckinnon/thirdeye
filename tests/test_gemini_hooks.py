from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.platforms.gemini import hooks
from thirdeye.store import Store


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    return tmp_path


def _stdin(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


# -- _read_stdin ---------------------------------------------------------------


class TestReadStdin:
    def test_valid_json(self, monkeypatch):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        result = hooks._read_stdin()
        assert result == {"session_id": "abc", "cwd": "/p"}

    def test_empty_stdin_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert hooks._read_stdin() == {}

    def test_invalid_json_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        assert hooks._read_stdin() == {}

    def test_io_error_returns_empty_dict(self, monkeypatch):
        class BrokenStdin:
            def read(self):
                raise OSError("broken pipe")

        monkeypatch.setattr("sys.stdin", BrokenStdin())
        assert hooks._read_stdin() == {}

    def test_nested_payload(self, monkeypatch):
        payload = {"session_id": "s", "nested": {"a": [1, 2, 3]}}
        _stdin(monkeypatch, payload)
        assert hooks._read_stdin() == payload


# -- _flex_get -----------------------------------------------------------------


class TestFlexGet:
    def test_returns_first_matching_key(self):
        d = {"sessionId": "abc"}
        assert hooks._flex_get(d, "session_id", "sessionId") == "abc"

    def test_prefers_first_key_listed(self):
        d = {"session_id": "first", "sessionId": "second"}
        assert hooks._flex_get(d, "session_id", "sessionId") == "first"

    def test_skips_none_values(self):
        d = {"session_id": None, "sessionId": "fallback"}
        assert hooks._flex_get(d, "session_id", "sessionId") == "fallback"

    def test_skips_empty_string_values(self):
        d = {"session_id": "", "sessionId": "fallback"}
        assert hooks._flex_get(d, "session_id", "sessionId") == "fallback"

    def test_returns_default_when_no_match(self):
        d = {"other_key": "val"}
        assert hooks._flex_get(d, "session_id", "sessionId") is None

    def test_returns_custom_default(self):
        d = {}
        assert hooks._flex_get(d, "session_id", default="fallback") == "fallback"


# -- _strip_payload ------------------------------------------------------------


class TestStripPayload:
    def test_removes_routing_keys(self):
        result = hooks._strip_payload({"session_id": "abc", "cwd": "/p", "prompt": "hi"})
        assert "session_id" not in result
        assert "cwd" not in result
        assert result == {"prompt": "hi"}

    def test_removes_camel_case_variants(self):
        result = hooks._strip_payload({"sessionId": "abc", "workingDir": "/p", "prompt": "hi"})
        assert "sessionId" not in result
        assert "workingDir" not in result
        assert result == {"prompt": "hi"}

    def test_removes_working_dir_snake_case(self):
        result = hooks._strip_payload({"working_dir": "/p", "data": "val"})
        assert "working_dir" not in result
        assert result == {"data": "val"}

    def test_preserves_other_keys(self):
        payload = {"session_id": "abc", "tool_name": "Read", "tool_input": {"x": 1}}
        result = hooks._strip_payload(payload)
        assert result == {"tool_name": "Read", "tool_input": {"x": 1}}

    def test_empty_dict(self):
        assert hooks._strip_payload({}) == {}

    def test_only_strip_keys(self):
        payload = {
            "session_id": "abc",
            "sessionId": "abc",
            "cwd": "/p",
            "workingDir": "/p",
            "working_dir": "/p",
        }
        assert hooks._strip_payload(payload) == {}


# -- _emit ---------------------------------------------------------------------


class TestEmit:
    def test_returns_seq_on_success(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        payload = hooks._read_stdin()
        seq = hooks._emit("test_event", payload)
        assert isinstance(seq, int)
        assert seq >= 0

    def test_returns_none_without_session_id(self, monkeypatch, env: Path):
        assert hooks._emit("test_event", {"cwd": "/p"}) is None

    def test_returns_none_with_empty_session_id(self, monkeypatch, env: Path):
        assert hooks._emit("test_event", {"session_id": "", "cwd": "/p"}) is None

    def test_returns_none_with_none_session_id(self, monkeypatch, env: Path):
        assert hooks._emit("test_event", {"session_id": None, "cwd": "/p"}) is None

    def test_stores_event_with_correct_type(self, monkeypatch, env: Path):
        hooks._emit("my_type", {"session_id": "s1", "cwd": "/p", "key": "val"})
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert len(events) == 1
        assert events[0]["t"] == "my_type"

    def test_strips_routing_keys_from_data(self, monkeypatch, env: Path):
        hooks._emit("x", {"session_id": "s1", "cwd": "/p", "extra": 42})
        events = list(Store(Config.load()).reader("s1").iter_events())
        data = events[0].get("data", {})
        assert "session_id" not in data
        assert "cwd" not in data
        assert data["extra"] == 42

    def test_uses_cwd_from_payload(self, monkeypatch, env: Path):
        hooks._emit("x", {"session_id": "s1", "cwd": "/my/project"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == "/my/project"

    def test_falls_back_to_os_cwd_when_no_cwd(self, monkeypatch, env: Path):
        monkeypatch.chdir(env)
        hooks._emit("x", {"session_id": "s1"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == str(env)

    def test_accepts_camel_case_session_id(self, monkeypatch, env: Path):
        hooks._emit("x", {"sessionId": "camel1", "cwd": "/p"})
        m = next(Store(Config.load()).list_sessions())
        assert m.session_id == "camel1"

    def test_accepts_camel_case_working_dir(self, monkeypatch, env: Path):
        hooks._emit("x", {"session_id": "s1", "workingDir": "/camel/dir"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == "/camel/dir"

    def test_platform_is_gemini(self, monkeypatch, env: Path):
        hooks._emit("x", {"session_id": "s1", "cwd": "/p"})
        m = next(Store(Config.load()).list_sessions())
        assert m.platform == "gemini"


# -- session_start -------------------------------------------------------------


class TestSessionStart:
    def test_creates_session(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc-123", "cwd": "/proj/x", "source": "cli"})
        hooks.session_start()
        store = Store(Config.load())
        metas = list(store.list_sessions())
        assert len(metas) == 1
        m = metas[0]
        assert m.session_id == "abc-123"
        assert m.platform == "gemini"
        assert m.cwd == "/proj/x"
        assert m.event_count == 1
        assert m.status == "open"

    def test_event_type_is_session_start(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["t"] == "session_start"

    def test_stores_payload_fields_in_data(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "source": "cli"})
        hooks.session_start()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["data"]["source"] == "cli"


# -- session_start env capture -> auto tags -----------------------------------


class TestSessionStartEnvTags:
    def _tags_lines(self, env: Path, sid: str) -> list[dict]:
        from thirdeye.paths import session_dir as _sd
        from thirdeye.paths import tags_path

        path = tags_path(_sd(env, "gemini", sid))
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_no_patterns_set_writes_no_tags(self, monkeypatch, env: Path):
        from thirdeye.paths import session_dir as _sd
        from thirdeye.paths import tags_path

        monkeypatch.delenv("THIRDEYE_CAPTURE_ENV", raising=False)
        monkeypatch.setenv("WB_PLAN", "p")
        monkeypatch.setenv("WB_STEP", "test#1")
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        assert not tags_path(_sd(env, "gemini", "s1")).exists()

    def test_matching_env_vars_become_auto_tags(self, monkeypatch, env: Path):
        monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
        monkeypatch.setenv("WB_PLAN", "p")
        monkeypatch.setenv("WB_STEP", "test#1")
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
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
        monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
        monkeypatch.setenv("WB_PLAN", "p")
        monkeypatch.setenv("WB_X", "===")
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()

        lines = self._tags_lines(env, "s1")
        tags = {line["tag"] for line in lines}
        assert "plan-p" in tags
        assert not any(t.startswith("x-") for t in tags)

    def test_missing_session_id_writes_no_tags(self, monkeypatch, env: Path):
        monkeypatch.setenv("THIRDEYE_CAPTURE_ENV", "WB_*")
        monkeypatch.setenv("WB_PLAN", "p")
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.session_start()
        assert list(Store(Config.load()).list_sessions()) == []


# -- session_end ---------------------------------------------------------------


class TestSessionEnd:
    def test_closes_session(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.status == "closed"
        assert m.ended_at is not None
        events = list(store.reader("abc").iter_events())
        assert events[-1]["t"] == "session_end"

    def test_appends_event_before_closing(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.event_count == 2

    def test_session_end_no_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.session_end()
        assert list(Store(Config.load()).list_sessions()) == []


# -- before_agent --------------------------------------------------------------


class TestBeforeAgent:
    def test_appends_user_message(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p", "prompt": "hello"})
        hooks.before_agent()
        store = Store(Config.load())
        events = list(store.reader("abc").iter_events())
        assert events[0]["t"] == "session_start"
        assert events[1]["t"] == "user_message"
        assert events[1]["data"]["prompt"] == "hello"
        assert "session_id" not in events[1].get("data", {})


# -- before_agent hashtag extraction -------------------------------------------


class TestBeforeAgentHashtagExtract:
    def test_extracts_hashtag_from_prompt(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"sessionId": "g1", "prompt": "let's #ship"})
        hooks.before_agent()
        store = Store(Config.load())
        events = list(store.reader("g1").iter_events())
        assert len(events) == 1
        assert events[0]["t"] == "user_message"

        from thirdeye.paths import session_dir as _sd
        from thirdeye.paths import tags_path
        from thirdeye.tags import TagStore

        sd = _sd(Config.load().root, "gemini", "g1")
        tag_file = tags_path(sd)
        assert tag_file.exists()
        lines = [json.loads(line) for line in tag_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        assert lines[0]["tag"] == "ship"
        assert lines[0]["op"] == "add"
        assert lines[0]["source"] == "auto"

        ts = TagStore(sd)
        assert ts.unique_tags() == {"ship"}

        m = next(store.list_sessions())
        assert m.tag_count == 1

        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_no_prompt_no_tags(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"sessionId": "g2", "cwd": "/p"})
        hooks.before_agent()

        from thirdeye.paths import session_dir as _sd
        from thirdeye.paths import tags_path

        sd = _sd(Config.load().root, "gemini", "g2")
        assert not tags_path(sd).exists()

        m = next(Store(Config.load()).list_sessions())
        assert m.tag_count == 0

        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_malformed_payload_no_crash(self, monkeypatch, env: Path, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        hooks.before_agent()
        assert list(Store(Config.load()).list_sessions()) == []
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_extracts_from_alternate_field_name(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"sessionId": "g3", "input": "ship it #fast"})
        hooks.before_agent()

        from thirdeye.paths import session_dir as _sd
        from thirdeye.tags import TagStore

        sd = _sd(Config.load().root, "gemini", "g3")
        ts = TagStore(sd)
        assert ts.unique_tags() == {"fast"}

        m = next(Store(Config.load()).list_sessions())
        assert m.tag_count == 1

        captured = capsys.readouterr()
        assert captured.out == "{}\n"


# -- after_agent ---------------------------------------------------------------


class TestAfterAgent:
    def test_appends_assistant_message(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "response": "done"})
        hooks.after_agent()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "assistant_message"
        assert events[1]["data"]["response"] == "done"


# -- before_model --------------------------------------------------------------


class TestBeforeModel:
    def test_appends_model_request(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "model": "gemini-2.5"})
        hooks.before_model()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "model_request"
        assert events[1]["data"]["model"] == "gemini-2.5"


# -- after_model ---------------------------------------------------------------


class TestAfterModel:
    def test_appends_model_response(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "tokens": 150})
        hooks.after_model()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "model_response"
        assert events[1]["data"]["tokens"] == 150


# -- before_tool ---------------------------------------------------------------


class TestBeforeTool:
    def test_appends_tool_call(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": "abc",
                "cwd": "/p",
                "tool_name": "Read",
                "tool_input": {"file_path": "x.py"},
            },
        )
        hooks.before_tool()
        events = list(Store(Config.load()).reader("abc").iter_events())
        assert events[1]["t"] == "tool_call"
        assert events[1]["data"]["tool_name"] == "Read"
        assert events[1]["data"]["tool_input"] == {"file_path": "x.py"}


# -- after_tool ----------------------------------------------------------------


class TestAfterTool:
    def test_appends_tool_result(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": "abc",
                "cwd": "/p",
                "tool_name": "Read",
                "tool_response": "<file contents>",
            },
        )
        hooks.after_tool()
        events = list(Store(Config.load()).reader("abc").iter_events())
        assert events[1]["t"] == "tool_result"
        assert events[1]["data"]["tool_response"] == "<file contents>"


# -- camelCase sessionId -------------------------------------------------------


class TestCamelCaseSessionId:
    def test_session_start_with_camel_case_session_id(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"sessionId": "camel-1", "cwd": "/p"})
        hooks.session_start()
        store = Store(Config.load())
        metas = list(store.list_sessions())
        assert len(metas) == 1
        assert metas[0].session_id == "camel-1"

    def test_session_end_with_camel_case_session_id(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"sessionId": "camel-2", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"sessionId": "camel-2", "cwd": "/p"})
        hooks.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.status == "closed"
        assert m.ended_at is not None

    def test_before_agent_with_camel_case_session_id(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"sessionId": "camel-3", "cwd": "/p", "prompt": "hi"})
        hooks.before_agent()
        events = list(Store(Config.load()).reader("camel-3").iter_events())
        assert events[0]["t"] == "user_message"

    def test_working_dir_camel_case_accepted(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"sessionId": "camel-4", "workingDir": "/camel/path"})
        hooks.session_start()
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == "/camel/path"

    def test_strips_camel_case_keys_from_data(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"sessionId": "camel-5", "workingDir": "/p", "extra": 1})
        hooks.session_start()
        events = list(Store(Config.load()).reader("camel-5").iter_events())
        data = events[0].get("data", {})
        assert "sessionId" not in data
        assert "workingDir" not in data
        assert data["extra"] == 1


# -- stdout contract -----------------------------------------------------------


class TestStdoutContract:
    """Gemini hooks MUST print {} to stdout. Absence of stdout breaks Gemini."""

    def test_session_start_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_session_end_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_end()
        captured = capsys.readouterr()
        # session_start prints {}\n, session_end prints {}\n
        assert captured.out == "{}\n{}\n"

    def test_before_agent_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.before_agent()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_after_agent_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.after_agent()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_before_model_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.before_model()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_after_model_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.after_model()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_before_tool_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.before_tool()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_after_tool_prints_empty_json(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.after_tool()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_noop_still_prints_empty_json(self, monkeypatch, env: Path, capsys):
        """Even when session_id is missing, stdout must have {}."""
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.session_start()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_broken_stdin_still_prints_empty_json(self, monkeypatch, env: Path, capsys):
        class BrokenStdin:
            def read(self):
                raise OSError("broken pipe")

        monkeypatch.setattr("sys.stdin", BrokenStdin())
        hooks.before_agent()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"

    def test_invalid_json_still_prints_empty_json(self, monkeypatch, env: Path, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        hooks.after_model()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"


# -- silent noop edge cases ----------------------------------------------------


class TestSilentNoop:
    def test_missing_session_id_is_silent_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.before_agent()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_invalid_json_is_silent_noop(self, monkeypatch, env: Path):
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        hooks.session_start()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_empty_stdin_is_silent_noop(self, monkeypatch, env: Path):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        hooks.session_start()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_broken_stdin_is_silent_noop(self, monkeypatch, env: Path):
        class BrokenStdin:
            def read(self):
                raise OSError("broken pipe")

        monkeypatch.setattr("sys.stdin", BrokenStdin())
        hooks.before_tool()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_empty_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "", "cwd": "/p"})
        hooks.after_agent()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_null_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": None, "cwd": "/p"})
        hooks.before_model()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_session_end_without_session_does_not_crash(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.session_end()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_noop_still_prints_stdout(self, monkeypatch, env: Path, capsys):
        """Even on noop, Gemini requires {} on stdout."""
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.session_start()
        captured = capsys.readouterr()
        assert captured.out == "{}\n"


# -- all event types route correctly -------------------------------------------


class TestAllEventTypesRoute:
    def test_all_event_types_route_correctly(self, monkeypatch, env: Path):
        expected = [
            (hooks.session_start, "session_start"),
            (hooks.session_end, "session_end"),
            (hooks.before_agent, "user_message"),
            (hooks.after_agent, "assistant_message"),
            (hooks.before_model, "model_request"),
            (hooks.after_model, "model_response"),
            (hooks.before_tool, "tool_call"),
            (hooks.after_tool, "tool_result"),
        ]
        for fn, t in expected:
            _stdin(monkeypatch, {"session_id": "s", "cwd": "/p"})
            fn()
        events = list(Store(Config.load()).reader("s").iter_events())
        assert [e["t"] for e in events] == [t for _, t in expected]

    def test_exactly_eight_hooks(self):
        hook_fns = [
            hooks.session_start,
            hooks.session_end,
            hooks.before_agent,
            hooks.after_agent,
            hooks.before_model,
            hooks.after_model,
            hooks.before_tool,
            hooks.after_tool,
        ]
        assert len(hook_fns) == 8
        # all are callable
        for fn in hook_fns:
            assert callable(fn)


# -- session_end closes session ------------------------------------------------


class TestSessionEndClosesSession:
    def test_status_is_closed(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.status == "closed"

    def test_ended_at_is_set(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.ended_at is not None

    def test_does_not_close_without_emit(self, monkeypatch, env: Path):
        """session_end should NOT call close_session if _emit returned False."""
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.session_end()
        # no sessions should exist at all
        assert list(Store(Config.load()).list_sessions()) == []

    def test_close_uses_gemini_platform(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.platform == "gemini"

    def test_close_with_camel_case_session_id(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"sessionId": "camel-close", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"sessionId": "camel-close", "cwd": "/p"})
        hooks.session_end()
        store = Store(Config.load())
        m = next(store.list_sessions())
        assert m.status == "closed"
        assert m.ended_at is not None


# -- platform constant ---------------------------------------------------------


class TestPlatformConstant:
    def test_platform_is_gemini(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        m = next(Store(Config.load()).list_sessions())
        assert m.platform == "gemini"

    def test_platform_constant_value(self):
        assert hooks._PLATFORM == "gemini"


# -- multiple sessions ---------------------------------------------------------


class TestMultipleSessions:
    def test_different_sessions_are_independent(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/a"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s2", "cwd": "/b"})
        hooks.session_start()
        store = Store(Config.load())
        metas = sorted(store.list_sessions(), key=lambda m: m.session_id)
        assert len(metas) == 2
        assert metas[0].session_id == "s1"
        assert metas[0].cwd == "/a"
        assert metas[1].session_id == "s2"
        assert metas[1].cwd == "/b"

    def test_closing_one_session_does_not_affect_other(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/a"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s2", "cwd": "/b"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/a"})
        hooks.session_end()
        store = Store(Config.load())
        metas = {m.session_id: m for m in store.list_sessions()}
        assert metas["s1"].status == "closed"
        assert metas["s2"].status == "open"


# -- complex payload preservation ----------------------------------------------


class TestPayloadPreservation:
    def test_nested_dict_preserved(self, monkeypatch, env: Path):
        payload = {
            "session_id": "s1",
            "cwd": "/p",
            "tool_input": {"nested": {"deep": True, "list": [1, 2, 3]}},
        }
        _stdin(monkeypatch, payload)
        hooks.before_tool()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["data"]["tool_input"] == {"nested": {"deep": True, "list": [1, 2, 3]}}

    def test_large_payload(self, monkeypatch, env: Path):
        big_data = {"session_id": "s1", "cwd": "/p", "content": "x" * 10000}
        _stdin(monkeypatch, big_data)
        hooks.after_agent()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert len(events[0]["data"]["content"]) == 10000

    def test_payload_with_special_chars(self, monkeypatch, env: Path):
        payload = {
            "session_id": "s1",
            "cwd": "/p",
            "text": "line1\nline2\ttab\r\nwindows",
        }
        _stdin(monkeypatch, payload)
        hooks.before_agent()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["data"]["text"] == "line1\nline2\ttab\r\nwindows"

    def test_unicode_payload(self, monkeypatch, env: Path):
        payload = {"session_id": "s1", "cwd": "/p", "text": "hello world"}
        _stdin(monkeypatch, payload)
        hooks.after_model()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["data"]["text"] == "hello world"
