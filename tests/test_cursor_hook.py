from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir, tags_path
from thirdeye.platforms.cursor import hook
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
        _stdin(monkeypatch, {"conversation_id": "abc", "cwd": "/p"})
        assert hook._read_stdin() == {"conversation_id": "abc", "cwd": "/p"}

    def test_empty_stdin_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert hook._read_stdin() == {}

    def test_invalid_json_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        assert hook._read_stdin() == {}

    def test_io_error_returns_empty_dict(self, monkeypatch):
        class BrokenStdin:
            def read(self):
                raise OSError("broken pipe")

        monkeypatch.setattr("sys.stdin", BrokenStdin())
        assert hook._read_stdin() == {}


# -- payload helpers -----------------------------------------------------------


class TestPayloadHelpers:
    def test_event_name_reads_snake_case(self):
        assert hook._event_name({"hook_event_name": "sessionStart"}) == "sessionStart"

    def test_event_name_reads_camel_case(self):
        assert hook._event_name({"hookEventName": "sessionStart"}) == "sessionStart"

    def test_event_name_prefers_snake_when_both_present(self):
        assert hook._event_name({"hook_event_name": "snake", "hookEventName": "camel"}) == "snake"

    def test_session_id_reads_both_cases(self):
        assert hook._session_id({"conversation_id": "s1"}) == "s1"
        assert hook._session_id({"conversationId": "s2"}) == "s2"

    def test_cwd_fallback_chain(self, monkeypatch, tmp_path: Path):
        assert hook._cwd({"cwd": "/explicit"}) == "/explicit"
        assert hook._cwd({"workspace_roots": ["/root/a", "/root/b"]}) == "/root/a"
        monkeypatch.chdir(tmp_path)
        assert hook._cwd({}) == str(tmp_path)

    def test_strip_removes_routing_and_pii_keys(self):
        payload = {
            "conversation_id": "abc",
            "conversationId": "abc",
            "workspace_roots": ["/r"],
            "transcript_path": "/x",
            "user_email": "u@example.com",
            "hook_event_name": "sessionStart",
            "hookEventName": "sessionStart",
            "prompt": "hi",
        }
        assert hook._strip(payload) == {"prompt": "hi"}


# -- _print_permissive ---------------------------------------------------------


class TestPrintPermissive:
    def test_before_events_get_permission_allow(self, capfd):
        hook._print_permissive("beforeSubmitPrompt")
        out = capfd.readouterr().out
        assert '"permission": "allow"' in out

    def test_other_events_get_continue_true(self, capfd):
        hook._print_permissive("afterAgentResponse")
        out = capfd.readouterr().out
        assert '"continue": true' in out

    def test_main_prints_permissive_even_when_handler_raises(self, monkeypatch, env: Path, capfd):
        def boom(payload):
            raise RuntimeError("nope")

        monkeypatch.setitem(hook._HANDLERS, "sessionStart", boom)
        _stdin(
            monkeypatch,
            {"hook_event_name": "sessionStart", "conversation_id": "s1"},
        )
        rc = hook.main()
        assert rc == 0
        out = capfd.readouterr().out
        # sessionStart doesn't start with "before" → continue
        assert '"continue": true' in out


# -- dispatch ------------------------------------------------------------------


class TestDispatch:
    def _events(self, sid: str) -> list[dict]:
        return list(Store(Config.load()).reader(sid).iter_events())

    def test_session_start(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "sessionStart",
                "conversation_id": "s1",
                "cwd": "/p",
                "model": "composer-1",
            },
        )
        hook.main()
        events = self._events("s1")
        assert len(events) == 1
        assert events[0]["t"] == "session_start"
        assert events[0]["data"]["model"] == "composer-1"
        assert "conversation_id" not in events[0]["data"]
        assert "hook_event_name" not in events[0]["data"]

    def test_before_submit_prompt(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": "s1",
                "cwd": "/p",
                "prompt": "hello",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "user_message"
        assert events[0]["data"]["prompt"] == "hello"

    def test_after_agent_response(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": "s1",
                "cwd": "/p",
                "response": "done",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "assistant_message"
        assert events[0]["data"]["response"] == "done"

    def test_before_shell(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "beforeShellExecution",
                "conversation_id": "s1",
                "cwd": "/p",
                "command": "ls",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_call"
        assert events[0]["data"]["tool_name"] == "shell"
        assert events[0]["data"]["command"] == "ls"

    def test_after_shell(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "afterShellExecution",
                "conversation_id": "s1",
                "cwd": "/p",
                "exit_code": 0,
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_result"
        assert events[0]["data"]["tool_name"] == "shell"

    def test_before_mcp(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "beforeMCPExecution",
                "conversation_id": "s1",
                "cwd": "/p",
                "tool_name": "my_mcp_tool",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_call"
        assert events[0]["data"]["tool_name"] == "my_mcp_tool"

    def test_before_mcp_default_name(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "beforeMCPExecution",
                "conversation_id": "s1",
                "cwd": "/p",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["data"]["tool_name"] == "mcp"

    def test_after_mcp(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "afterMCPExecution",
                "conversation_id": "s1",
                "cwd": "/p",
                "toolName": "some_tool",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_result"
        assert events[0]["data"]["tool_name"] == "some_tool"

    def test_before_read_file(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "beforeReadFile",
                "conversation_id": "s1",
                "cwd": "/p",
                "file_path": "x.py",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_call"
        assert events[0]["data"]["tool_name"] == "read_file"

    def test_after_file_edit(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "afterFileEdit",
                "conversation_id": "s1",
                "cwd": "/p",
                "file_path": "x.py",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_result"
        assert events[0]["data"]["tool_name"] == "edit_file"

    def test_before_tab_read(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "beforeTabFileRead",
                "conversation_id": "s1",
                "cwd": "/p",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_call"
        assert events[0]["data"]["tool_name"] == "tab_read_file"

    def test_after_tab_edit(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "afterTabFileEdit",
                "conversation_id": "s1",
                "cwd": "/p",
            },
        )
        hook.main()
        events = self._events("s1")
        assert events[0]["t"] == "tool_result"
        assert events[0]["data"]["tool_name"] == "tab_edit_file"


# -- missing session id --------------------------------------------------------


class TestMissingSessionId:
    @pytest.mark.parametrize(
        "event",
        [
            "sessionStart",
            "sessionEnd",
            "beforeSubmitPrompt",
            "afterAgentResponse",
            "beforeShellExecution",
            "afterShellExecution",
            "beforeMCPExecution",
            "afterMCPExecution",
            "beforeReadFile",
            "afterFileEdit",
            "beforeTabFileRead",
            "afterTabFileEdit",
            "postToolUse",
            "stop",
        ],
    )
    def test_no_event_appended(self, monkeypatch, env: Path, event: str):
        _stdin(monkeypatch, {"hook_event_name": event, "cwd": "/p"})
        rc = hook.main()
        assert rc == 0
        assert list(Store(Config.load()).list_sessions()) == []


# -- beforeSubmitPrompt autotag ------------------------------------------------


class TestBeforeSubmitAutotag:
    def _tags_lines(self, env: Path, sid: str) -> list[dict]:
        path = tags_path(session_dir(env, "cursor", sid))
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_hashtags_in_prompt_become_tags(self, monkeypatch, env: Path):
        # Open the session first so meta exists.
        _stdin(
            monkeypatch,
            {"hook_event_name": "sessionStart", "conversation_id": "s1", "cwd": "/p"},
        )
        hook.main()

        _stdin(
            monkeypatch,
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": "s1",
                "cwd": "/p",
                "prompt": "fix #bug in #parser",
            },
        )
        hook.main()

        lines = self._tags_lines(env, "s1")
        tags = {line["tag"] for line in lines}
        assert tags == {"bug", "parser"}
        for line in lines:
            assert line["op"] == "add"
            assert line["source"] == "auto"

        m = Store(Config.load()).get_meta("s1")
        assert m.tag_count == 1


# -- postToolUse dedup ---------------------------------------------------------


class TestPostToolUseDedup:
    def _events(self, sid: str) -> list[dict]:
        return list(Store(Config.load()).reader(sid).iter_events())

    def test_skips_dedicated_tool_names(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "s1",
                "cwd": "/p",
                "tool_name": "shell",
            },
        )
        hook.main()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_lowercases_for_comparison(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "s1",
                "cwd": "/p",
                "tool_name": "Shell",
            },
        )
        hook.main()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_non_dedicated_tool_writes_event(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "s1",
                "cwd": "/p",
                "tool_name": "my_custom_tool",
            },
        )
        hook.main()
        events = self._events("s1")
        assert len(events) == 1
        assert events[0]["t"] == "tool_result"
        assert events[0]["data"]["tool_name"] == "my_custom_tool"

    def test_unknown_tool_name_falls_back_to_unknown(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "s1",
                "cwd": "/p",
            },
        )
        hook.main()
        events = self._events("s1")
        assert len(events) == 1
        assert events[0]["t"] == "tool_result"
        assert events[0]["data"]["tool_name"] == "unknown"


# -- stop ----------------------------------------------------------------------


class TestStop:
    def test_stop_does_not_append_event(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "stop",
                "conversation_id": "s1",
                "cwd": "/p",
                "model": "composer-1",
                "input_tokens": 10,
                "output_tokens": 5,
            },
        )
        hook.main()
        # No event was appended to the store.
        assert list(Store(Config.load()).list_sessions()) == []

    def test_stop_calls_capture_usage_cursor(self, monkeypatch, env: Path):
        calls: list[dict] = []

        def fake(*, thirdeye_home, session_id, payload, triggering_seq):
            calls.append(
                {
                    "thirdeye_home": thirdeye_home,
                    "session_id": session_id,
                    "payload": payload,
                    "triggering_seq": triggering_seq,
                }
            )
            return 1

        monkeypatch.setattr(hook, "capture_usage_cursor", fake)
        payload = {
            "hook_event_name": "stop",
            "conversation_id": "s1",
            "cwd": "/p",
            "model": "composer-1",
            "input_tokens": 10,
            "output_tokens": 5,
        }
        _stdin(monkeypatch, payload)
        hook.main()
        assert len(calls) == 1
        assert calls[0]["session_id"] == "s1"
        assert calls[0]["payload"]["model"] == "composer-1"
        assert "triggering_seq" in calls[0]

    def test_stop_with_missing_conversation_id_is_noop(self, monkeypatch, env: Path):
        calls: list = []
        monkeypatch.setattr(
            hook,
            "capture_usage_cursor",
            lambda **kw: calls.append(kw) or 0,
        )
        _stdin(monkeypatch, {"hook_event_name": "stop", "cwd": "/p"})
        hook.main()
        assert calls == []


# -- sessionEnd ----------------------------------------------------------------


class TestSessionEnd:
    def test_appends_session_end_event(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {"hook_event_name": "sessionStart", "conversation_id": "s1", "cwd": "/p"},
        )
        hook.main()
        _stdin(
            monkeypatch,
            {"hook_event_name": "sessionEnd", "conversation_id": "s1", "cwd": "/p"},
        )
        hook.main()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[-1]["t"] == "session_end"

    def test_closes_session(self, monkeypatch, env: Path):
        _stdin(
            monkeypatch,
            {"hook_event_name": "sessionStart", "conversation_id": "s1", "cwd": "/p"},
        )
        hook.main()
        _stdin(
            monkeypatch,
            {"hook_event_name": "sessionEnd", "conversation_id": "s1", "cwd": "/p"},
        )
        hook.main()
        m = next(Store(Config.load()).list_sessions())
        assert m.status == "closed"
        assert m.ended_at is not None

    def test_invokes_usage_capture_as_fallback(self, monkeypatch, env: Path):
        calls: list = []
        monkeypatch.setattr(
            hook,
            "capture_usage_cursor",
            lambda **kw: calls.append(kw) or 0,
        )
        _stdin(
            monkeypatch,
            {"hook_event_name": "sessionStart", "conversation_id": "s1", "cwd": "/p"},
        )
        hook.main()
        _stdin(
            monkeypatch,
            {
                "hook_event_name": "sessionEnd",
                "conversation_id": "s1",
                "cwd": "/p",
                "model": "composer-1",
                "input_tokens": 10,
                "output_tokens": 5,
            },
        )
        hook.main()
        assert len(calls) == 1
        assert calls[0]["session_id"] == "s1"
        # triggering_seq comes from the session_end seq which is >= 0.
        assert calls[0]["triggering_seq"] >= 0


# -- unknown event -------------------------------------------------------------


class TestUnknownEvent:
    def test_unknown_event_name_is_noop(self, monkeypatch, env: Path, capfd):
        _stdin(
            monkeypatch,
            {"hook_event_name": "preCompact", "conversation_id": "s1", "cwd": "/p"},
        )
        rc = hook.main()
        assert rc == 0
        assert list(Store(Config.load()).list_sessions()) == []
        out = capfd.readouterr().out
        assert '"continue": true' in out
