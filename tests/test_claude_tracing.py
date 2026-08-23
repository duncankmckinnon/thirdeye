from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.claude import hooks, tracing
from thirdeye.platforms.claude.usage import extract_calls_from_transcript


@pytest.fixture
def env(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    return tmp_path


def _stdin(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


def _user_frame(content) -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}})


def _assistant_frame(
    msg_id: str,
    content: list,
    *,
    model: str = "claude-sonnet-5",
    input_tokens: int = 10,
    output_tokens: int = 5,
    ts: str = "2026-01-01T00:00:00.000Z",
    extra_usage: dict | None = None,
) -> str:
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if extra_usage:
        usage.update(extra_usage)
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": ts,
            "message": {"id": msg_id, "model": model, "content": content, "usage": usage},
        }
    )


# -- extract_calls_from_transcript ---------------------------------------------


class TestExtractCallsFromTranscript:
    def test_no_transcript_path_yields_nothing(self):
        parsed = extract_calls_from_transcript(None, 0)
        assert parsed.calls == []
        assert parsed.offset == 0

    def test_missing_file_yields_nothing(self):
        parsed = extract_calls_from_transcript("/nonexistent/path.jsonl", 0)
        assert parsed.calls == []
        assert parsed.offset == 0

    def test_simple_call_has_llm_call_span_shape(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("hello")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "text", "text": "hi there"}])
            + "\n"
        )
        parsed = extract_calls_from_transcript(str(transcript), 0)
        calls, new_offset = parsed.calls, parsed.offset
        assert len(calls) == 1
        call = calls[0]
        assert call["call_id"] == "msg_1"
        assert call["provider"] == "anthropic"
        assert call["model"] == "claude-sonnet-5"
        assert call["start_ts"] == "2026-01-01T00:00:00.000Z"
        assert call["end_ts"] == "2026-01-01T00:00:00.000Z"
        assert call["input_messages"] == [
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
        ]
        assert call["output_messages"] == [
            {"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]}
        ]
        assert call["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert call["tool_calls"] == []
        assert new_offset == len(transcript.read_bytes())

    def test_usage_includes_cache_fields_when_present(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _assistant_frame(
                "msg_1",
                [{"type": "text", "text": "hi"}],
                extra_usage={"cache_read_input_tokens": 7, "cache_creation_input_tokens": 3},
            )
            + "\n"
        )
        calls = extract_calls_from_transcript(str(transcript), 0).calls
        usage = calls[0]["usage"]
        assert usage["cache_read_input_tokens"] == 7
        assert usage["cache_creation_input_tokens"] == 3
        # cache reads/creation fold into input_tokens, matching OTel GenAI convention.
        assert usage["input_tokens"] == 10 + 7 + 3

    def test_tool_use_produces_empty_tool_calls_but_tool_call_part(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("run ls")
            + "\n"
            + _assistant_frame(
                "msg_1",
                [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}}],
            )
            + "\n"
        )
        calls = extract_calls_from_transcript(str(transcript), 0).calls
        assert len(calls) == 1
        # tool_calls is always empty here: timing/results live in the local event
        # store, correlated separately by build_turn.
        assert calls[0]["tool_calls"] == []
        parts = calls[0]["output_messages"][0]["parts"]
        assert parts == [
            {"type": "tool_call", "id": "tu_1", "name": "Bash", "arguments": {"command": "ls"}}
        ]

    def test_tool_result_becomes_input_to_the_next_call(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("run ls")
            + "\n"
            + _assistant_frame(
                "msg_1",
                [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}}],
            )
            + "\n"
            + _user_frame([{"type": "tool_result", "tool_use_id": "tu_1", "content": "file1"}])
            + "\n"
            + _assistant_frame("msg_2", [{"type": "text", "text": "found file1"}])
            + "\n"
        )
        calls = extract_calls_from_transcript(str(transcript), 0).calls
        assert len(calls) == 2
        first, second = calls
        assert first["call_id"] == "msg_1"
        assert second["call_id"] == "msg_2"
        assert second["input_messages"] == [
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "tu_1", "response": "file1"}],
            }
        ]

    def test_consecutive_calls_with_no_intervening_user_frame_has_no_input(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("go")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "thinking", "thinking": "first"}])
            + "\n"
            + _assistant_frame("msg_2", [{"type": "text", "text": "second"}])
            + "\n"
        )
        calls = extract_calls_from_transcript(str(transcript), 0).calls
        assert len(calls) == 2
        assert calls[0]["input_messages"] != []
        assert calls[1]["input_messages"] == []

    def test_frames_sharing_one_message_id_merge_into_one_call(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("go")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "thinking", "thinking": "planning"}])
            + "\n"
            + _assistant_frame(
                "msg_1", [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {}}]
            )
            + "\n"
        )
        calls = extract_calls_from_transcript(str(transcript), 0).calls
        assert len(calls) == 1
        parts = calls[0]["output_messages"][0]["parts"]
        assert [p["type"] for p in parts] == ["reasoning", "tool_call"]

    def test_synthetic_model_is_skipped(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _assistant_frame("msg_1", [{"type": "text", "text": "hi"}], model="<synthetic>") + "\n"
        )
        calls = extract_calls_from_transcript(str(transcript), 0).calls
        assert calls == []

    def test_offset_makes_a_second_call_incremental(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("hello")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "text", "text": "hi"}])
            + "\n"
        )
        first = extract_calls_from_transcript(str(transcript), 0)
        offset_after = first.offset
        assert len(first.calls) == 1

        with transcript.open("a") as f:
            f.write(_assistant_frame("msg_2", [{"type": "text", "text": "again"}]) + "\n")
        second_calls = extract_calls_from_transcript(str(transcript), offset_after).calls
        assert len(second_calls) == 1
        assert second_calls[0]["call_id"] == "msg_2"

    def test_no_output_content_yields_empty_output_messages(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _assistant_frame("msg_1", [{"type": "redacted_thinking", "data": "..."}]) + "\n"
        )
        calls = extract_calls_from_transcript(str(transcript), 0).calls
        assert len(calls) == 1
        assert calls[0]["output_messages"] == []


# -- build_turn ------------------------------------------------------------------


class TestBuildTurn:
    def test_returns_none_without_open_turn_marker(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        sd = session_dir(env, "claude", "s1")
        turn = tracing.build_turn(
            config=Config.load(),
            session_dir_=sd,
            session_id="s1",
            cwd="/p",
            stop_seq=999,
            stop_ts="2026-01-01T00:00:01.000Z",
            transcript_path=None,
            final_response="done",
        )
        assert turn is None

    def test_full_turn_assembly(self, monkeypatch, env: Path, tmp_path: Path):
        sid = "s1"
        main_transcript = tmp_path / "main.jsonl"
        sub_transcript = tmp_path / "sub.jsonl"
        sub_transcript.write_text(
            _assistant_frame(
                "sub_msg_1", [{"type": "text", "text": "sub result"}], model="claude-fable-5"
            )
            + "\n"
        )

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()

        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "prompt": "do the thing",
                "transcript_path": str(main_transcript),
            },
        )
        hooks.user_prompt_submit()

        # Written only now, after the prompt was submitted: the marker records
        # the transcript's size at submit time, so frames the turn itself
        # produces have to land after it or they fall behind the cursor.
        main_transcript.write_text(
            _user_frame("do the thing")
            + "\n"
            + _assistant_frame(
                "msg_1",
                [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}}],
                ts="2026-01-01T00:00:01.000Z",
            )
            + "\n"
            + _user_frame([{"type": "tool_result", "tool_use_id": "tu_1", "content": "out"}])
            + "\n"
            + _assistant_frame(
                "msg_2", [{"type": "text", "text": "done"}], ts="2026-01-01T00:00:02.000Z"
            )
            + "\n"
        )

        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "tool_name": "Bash",
                "tool_use_id": "tu_1",
                "tool_input": {"command": "ls"},
            },
        )
        hooks.pre_tool_use()

        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "tool_name": "Bash",
                "tool_use_id": "tu_1",
                "tool_response": "out",
            },
        )
        hooks.post_tool_use()

        _stdin(
            monkeypatch,
            {"session_id": sid, "cwd": "/p", "tool_name": "Write", "command": "echo hi"},
        )
        hooks.permission_request()

        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "tool_name": "Task",
                "tool_use_id": "tu_task1",
                "tool_input": {"description": "explore", "prompt": "explore code"},
            },
        )
        hooks.pre_tool_use()

        _stdin(
            monkeypatch,
            {"session_id": sid, "cwd": "/p", "agent_id": "agent_1", "agent_type": "explore"},
        )
        hooks.subagent_start()

        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "agent_id": "agent_1",
                "agent_transcript_path": str(sub_transcript),
                "last_assistant_message": "found stuff",
            },
        )
        hooks.subagent_stop()

        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "tool_name": "Task",
                "tool_use_id": "tu_task1",
                "tool_response": "found stuff",
            },
        )
        hooks.post_tool_use()

        sd = session_dir(env, "claude", sid)
        from thirdeye.reader import SessionReader

        events = list(SessionReader(sd).iter_events())
        turn_seq = next(e["seq"] for e in events if e["t"] == "user_message")
        stop_seq = events[-1]["seq"] + 1

        turn = tracing.build_turn(
            config=Config.load(),
            session_dir_=sd,
            session_id=sid,
            cwd="/p",
            stop_seq=stop_seq,
            stop_ts="2026-01-01T00:00:10.000Z",
            transcript_path=str(main_transcript),
            final_response="all done",
        )

        assert turn is not None
        assert turn["turn_id"] == str(turn_seq)
        assert turn["input_message"] == "do the thing"
        assert turn["output_message"] == "all done"
        assert turn["status"] == "completed"
        assert turn["end_ts"] == "2026-01-01T00:00:10.000Z"

        assert len(turn["llm_calls"]) == 2
        first, second = turn["llm_calls"]
        assert first["call_id"] == "msg_1"
        assert len(first["tool_calls"]) == 1
        assert first["tool_calls"][0]["tool_call_id"] == "tu_1"
        assert first["tool_calls"][0]["name"] == "Bash"
        assert second["call_id"] == "msg_2"
        assert second["tool_calls"] == []

        assert len(turn["permission_requests"]) == 1
        assert turn["permission_requests"][0]["tool_name"] == "Write"

        assert len(turn["subagents"]) == 1
        sub = turn["subagents"][0]
        assert sub["input_message"] == "explore code"
        assert sub["output_message"] == "found stuff"
        assert sub["status"] == "completed"
        assert sub["subagents"] == []
        assert len(sub["llm_calls"]) == 1
        assert sub["llm_calls"][0]["call_id"] == "sub_msg_1"
        assert sub["llm_calls"][0]["model"] == "claude-fable-5"

    def test_tool_call_still_open_is_omitted(self, monkeypatch, env: Path, tmp_path: Path):
        """A tool_call with no matching tool_result yet (still running, or the
        turn was interrupted before it finished) must not appear anywhere in
        the assembled turn.
        """
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "go"})
        hooks.user_prompt_submit()
        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "tool_name": "Bash",
                "tool_use_id": "tu_1",
                "tool_input": {},
            },
        )
        hooks.pre_tool_use()

        sd = session_dir(env, "claude", sid)
        from thirdeye.reader import SessionReader

        events = list(SessionReader(sd).iter_events())
        stop_seq = events[-1]["seq"] + 1

        turn = tracing.build_turn(
            config=Config.load(),
            session_dir_=sd,
            session_id=sid,
            cwd="/p",
            stop_seq=stop_seq,
            stop_ts="2026-01-01T00:00:05.000Z",
            transcript_path=None,
            final_response="",
        )
        assert turn is not None
        assert turn["llm_calls"] == []
        assert turn["permission_requests"] == []
        assert turn["subagents"] == []


# -- interruption handling (_close_stale_turn_if_open) ---------------------------


class TestInterruptionHandling:
    def test_different_prompt_id_closes_interrupted_turn_on_any_hook(self, monkeypatch, env: Path):
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda config, sd, sid, platform, cwd, turn: exported.append(turn),
        )
        sid = "s1"
        _stdin(
            monkeypatch,
            {"session_id": sid, "cwd": "/p", "prompt": "first", "prompt_id": "p1"},
        )
        hooks.user_prompt_submit()

        _stdin(
            monkeypatch,
            {"session_id": sid, "cwd": "/p", "message": "next", "prompt_id": "p2"},
        )
        hooks.notification()

        assert len(exported) == 1
        assert exported[0]["status"] == "interrupted"
        assert not hooks._open_turn_path(session_dir(env, "claude", sid)).exists()

    def test_session_end_closes_stale_turn_as_interrupted(self, monkeypatch, env: Path):
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda config, sd, sid, platform, cwd, turn: exported.append(turn),
        )
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "first prompt"})
        hooks.user_prompt_submit()

        sd = session_dir(env, "claude", sid)
        assert hooks._open_turn_path(sd).exists()

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_end()

        assert len(exported) == 1
        assert exported[0]["status"] == "interrupted"
        assert exported[0]["input_message"] == "first prompt"
        assert not hooks._open_turn_path(sd).exists()

    def test_second_user_prompt_submit_closes_previous_turn_as_interrupted(
        self, monkeypatch, env: Path
    ):
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda config, sd, sid, platform, cwd, turn: exported.append(turn),
        )
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "first prompt"})
        hooks.user_prompt_submit()

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "second prompt"})
        hooks.user_prompt_submit()

        assert len(exported) == 1
        assert exported[0]["status"] == "interrupted"
        assert exported[0]["input_message"] == "first prompt"

        # The second prompt opened its own fresh marker for itself.
        sd = session_dir(env, "claude", sid)
        marker = json.loads(hooks._open_turn_path(sd).read_text())
        assert marker["prompt"] == "second prompt"

    def test_no_marker_is_a_silent_noop(self, monkeypatch, env: Path):
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda *a: exported.append(a[-1]),
        )
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_end()
        assert exported == []

    def test_notification_mid_turn_does_not_close_the_open_turn(self, monkeypatch, env: Path):
        """Claude Code fires the Notification hook *during* an active,
        non-interrupted turn (e.g. "permission needed", "idle 60s+") — this
        must not be treated as proof the turn was abandoned. If it wrongly
        closes/exports the marker here, the turn's real `stop()` call later
        finds no marker, returns None from build_turn, and the turn's actual
        completion (LLM calls, tool calls, final response) is silently never
        exported at all.
        """
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda config, sd, sid, platform, cwd, turn: exported.append(turn),
        )
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "first prompt"})
        hooks.user_prompt_submit()

        sd = session_dir(env, "claude", sid)
        assert hooks._open_turn_path(sd).exists()

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "message": "needs permission"})
        hooks.notification()

        assert exported == [], "notification() must not export the still-running turn"
        assert hooks._open_turn_path(
            sd
        ).exists(), "notification() must not delete the marker of a still-running turn"

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "response": "done"})
        hooks.stop()

        assert len(exported) == 1, "the turn's real completion must still be exported by stop()"
        assert exported[0]["status"] == "completed"

    def test_pre_compact_mid_turn_does_not_close_the_open_turn(self, monkeypatch, env: Path):
        """PreCompact/PostCompact fire mid-turn for automatic context
        compaction (triggered when the context window fills up during
        generation), not only for the standalone /compact command between
        turns — so, like Notification, it must not be treated as proof of
        turn abandonment.
        """
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda config, sd, sid, platform, cwd, turn: exported.append(turn),
        )
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "first prompt"})
        hooks.user_prompt_submit()

        sd = session_dir(env, "claude", sid)

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.pre_compact()

        assert exported == [], "pre_compact() must not export the still-running turn"
        assert hooks._open_turn_path(sd).exists()

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "response": "done"})
        hooks.stop()

        assert len(exported) == 1
        assert exported[0]["status"] == "completed"

    @pytest.mark.parametrize(
        "hook_name,payload",
        [
            ("pre_tool_use", {"tool_name": "Bash", "tool_use_id": "tu_1", "tool_input": {}}),
            ("post_tool_use", {"tool_name": "Bash", "tool_use_id": "tu_1", "tool_response": "ok"}),
            ("permission_request", {"tool_name": "Write"}),
            ("permission_denied", {"tool_name": "Write"}),
            ("subagent_start", {"agent_id": "agent_1", "agent_type": "explore"}),
            ("subagent_stop", {"agent_id": "agent_1", "last_assistant_message": "done"}),
            ("post_compact", {}),
        ],
    )
    def test_every_mid_turn_event_type_leaves_the_open_turn_untouched(
        self, monkeypatch, env: Path, hook_name: str, payload: dict
    ):
        """Regression coverage for the whole `_MID_TURN_EVENT_TYPES` set, not
        just notification/pre_compact (the two that were actually caught
        exporting early): every event type in that set must be exempt from
        `_close_stale_turn_if_open`, or a still-running turn gets truncated
        and exported early the moment that event type fires.
        """
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda config, sd, sid, platform, cwd, turn: exported.append(turn),
        )
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "first prompt"})
        hooks.user_prompt_submit()

        sd = session_dir(env, "claude", sid)
        assert hooks._open_turn_path(sd).exists()

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", **payload})
        getattr(hooks, hook_name)()

        assert exported == [], f"{hook_name}() must not export the still-running turn"
        assert hooks._open_turn_path(
            sd
        ).exists(), f"{hook_name}() must not delete the marker of a still-running turn"


# -- stop() normal completion path -----------------------------------------------


class TestStopNormalCompletion:
    def test_stop_exports_completed_turn_and_clears_marker(
        self, monkeypatch, env: Path, tmp_path: Path
    ):
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda config, sd, sid, platform, cwd, turn: exported.append(turn),
        )
        sid = "s1"
        transcript = tmp_path / "main.jsonl"
        transcript.write_text(_assistant_frame("msg_1", [{"type": "text", "text": "hi"}]) + "\n")

        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "prompt": "hello",
                "transcript_path": str(transcript),
            },
        )
        hooks.user_prompt_submit()

        sd = session_dir(env, "claude", sid)
        assert hooks._open_turn_path(sd).exists()

        _stdin(
            monkeypatch,
            {
                "session_id": sid,
                "cwd": "/p",
                "transcript_path": str(transcript),
                "response": "done",
            },
        )
        hooks.stop()

        assert len(exported) == 1
        turn = exported[0]
        assert turn["status"] == "completed"
        assert turn["input_message"] == "hello"
        assert turn["output_message"] == "done"
        assert not hooks._open_turn_path(sd).exists()

    def test_stop_without_open_marker_does_not_export(self, monkeypatch, env: Path):
        exported = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda *a: exported.append(a[-1]),
        )
        sid = "s1"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
        hooks.session_start()
        # No user_prompt_submit call, so no marker was ever written.
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "response": "done"})
        hooks.stop()
        assert exported == []
