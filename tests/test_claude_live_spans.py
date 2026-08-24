from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import otel_jobs_dir, session_dir
from thirdeye.platforms.claude import hooks, live_spans
from thirdeye.reader import SessionReader
from thirdeye.span_ids import chat_span_id, tool_span_id
from thirdeye.store import Store


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Config:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    configured = Config(root=tmp_path).write_logfire_settings(
        LogfireSettings(enabled=True, token="test-token")
    )
    monkeypatch.setattr("thirdeye.otel_export.subprocess.Popen", lambda *args, **kwargs: None)
    return configured


def _stdin(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


def _start_turn(
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    transcript: Path,
    *,
    session_id: str = "live-session",
) -> Path:
    transcript.write_text("")
    _stdin(
        monkeypatch,
        {
            "session_id": session_id,
            "cwd": "/project",
            "prompt": "inspect the project",
            "transcript_path": str(transcript),
        },
    )
    hooks.user_prompt_submit()
    return session_dir(config.root, "claude", session_id)


def _assistant_frame(call_id: str, timestamp: str, *tool_ids: str) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "id": call_id,
            "model": "claude-sonnet-5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Read",
                    "input": {"file_path": f"{tool_id}.py"},
                }
                for tool_id in tool_ids
            ],
        },
    }


def _user_tool_results(timestamp: str, *tool_ids: str) -> dict:
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "done"}
                for tool_id in tool_ids
            ]
        },
    }


def _append_frames(transcript: Path, *frames: dict) -> None:
    with transcript.open("a", encoding="utf-8") as file:
        for frame in frames:
            file.write(json.dumps(frame) + "\n")


def _append_tool_pair(config: Config, session_id: str, tool_use_id: str) -> tuple[str, str]:
    store = Store(config)
    call_seq = store.append_event(
        session_id=session_id,
        platform="claude",
        cwd="/project",
        t="tool_call",
        data={
            "tool_name": "Read",
            "tool_use_id": tool_use_id,
            "tool_input": {"file_path": f"{tool_use_id}.py"},
        },
    )
    result_seq = store.append_event(
        session_id=session_id,
        platform="claude",
        cwd="/project",
        t="tool_result",
        data={"tool_use_id": tool_use_id, "tool_response": "done"},
    )
    reader = Store(config).reader(session_id)
    return reader.get_event(call_seq)["ts"], reader.get_event(result_seq)["ts"]


def _jobs(config: Config) -> list[dict]:
    jobs_dir = otel_jobs_dir(config.root)
    if not jobs_dir.exists():
        return []
    return [json.loads(path.read_text()) for path in sorted(jobs_dir.glob("*.json"))]


class TestLiveSpanJob:
    def test_tool_parents_to_chat_and_uses_event_timing(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        session_id = "parent-and-timing"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        _append_frames(
            transcript,
            _assistant_frame("msg-request", "2026-08-22T10:00:01.000Z", "tool-1"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-1"),
        )
        event_start, event_end = _append_tool_pair(config, session_id, "tool-1")

        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-1")

        job = _jobs(config)[0]
        chat, tool = job["spans"]
        assert chat["span_id"] == str(chat_span_id(session_id, "msg-request"))
        assert tool["span_id"] == str(tool_span_id(session_id, "tool-1"))
        assert tool["parent_span_id"] == chat["span_id"]
        assert (tool["start_ts"], tool["end_ts"]) == (event_start, event_end)
        assert (tool["start_ts"], tool["end_ts"]) != (
            "2026-08-22T10:00:01.000Z",
            "2026-08-22T10:00:02.000Z",
        )

    def test_each_chat_is_emitted_once_across_two_growing_transcript_hooks(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        session_id = "incremental"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)

        for index in (1, 2):
            _append_frames(
                transcript,
                _assistant_frame(
                    f"msg-{index}", f"2026-08-22T10:00:0{index}.000Z", f"tool-{index}"
                ),
                _user_tool_results(f"2026-08-22T10:00:1{index}.000Z", f"tool-{index}"),
            )
            _append_tool_pair(config, session_id, f"tool-{index}")
            live_spans.emit_live_spans(
                config, session_dir_, session_id, "/project", f"tool-{index}"
            )

        jobs = _jobs(config)
        chat_ids = [
            span["span_id"]
            for job in jobs
            for span in job["spans"]
            if span["name"].startswith("chat")
        ]
        assert sorted(chat_ids) == sorted(
            [
                str(chat_span_id(session_id, "msg-1")),
                str(chat_span_id(session_id, "msg-2")),
            ]
        )
        marker = hooks._read_open_turn(session_dir_)
        assert marker is not None
        assert marker["transcript_offset"] == transcript.stat().st_size

    def test_parent_chat_already_emitted_is_still_found_in_transcript(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        session_id = "prior-parent"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        _append_frames(
            transcript,
            _assistant_frame("msg-parallel", "2026-08-22T10:00:01.000Z", "tool-1", "tool-2"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-1", "tool-2"),
        )
        _append_tool_pair(config, session_id, "tool-1")
        _append_tool_pair(config, session_id, "tool-2")

        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-1")
        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-2")

        tool_2_span_id = str(tool_span_id(session_id, "tool-2"))
        second_job = next(
            job
            for job in _jobs(config)
            if any(span["span_id"] == tool_2_span_id for span in job["spans"])
        )
        assert [span["name"] for span in second_job["spans"]] == ["tool: Read"]
        assert second_job["spans"][0]["parent_span_id"] == str(
            chat_span_id(session_id, "msg-parallel")
        )

    def test_failed_job_write_does_not_advance_cursor(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        session_id = "failed-export"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        before = hooks._read_open_turn(session_dir_)
        assert before is not None
        _append_frames(
            transcript,
            _assistant_frame("msg-fail", "2026-08-22T10:00:01.000Z", "tool-fail"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-fail"),
        )
        _append_tool_pair(config, session_id, "tool-fail")
        monkeypatch.setattr(live_spans, "export_spans", lambda *args, **kwargs: False)

        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-fail")

        after = hooks._read_open_turn(session_dir_)
        assert after is not None
        assert after["transcript_offset"] == before["transcript_offset"]
        assert after["last_frame_ts"] == before["last_frame_ts"]

    def test_raising_job_write_does_not_advance_cursor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: Config,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_id = "raising-export-cursor"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        before = hooks._read_open_turn(session_dir_)
        assert before is not None
        _append_frames(
            transcript,
            _assistant_frame("msg-raise", "2026-08-22T10:00:01.000Z", "tool-raise"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-raise"),
        )
        _append_tool_pair(config, session_id, "tool-raise")
        monkeypatch.setattr(
            live_spans,
            "export_spans",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("job write failed")),
        )

        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-raise")

        after = hooks._read_open_turn(session_dir_)
        captured = capsys.readouterr()
        assert after is not None
        assert after["transcript_offset"] == before["transcript_offset"]
        assert after["last_frame_ts"] == before["last_frame_ts"]
        assert captured.out == ""
        assert captured.err == ""

    def test_lock_failure_does_not_raise_or_write_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: Config,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def raise_lock_error(*args, **kwargs):
            raise OSError("lock unavailable")

        monkeypatch.setattr(live_spans, "_locked_open_turn", raise_lock_error)

        live_spans.emit_live_spans(config, tmp_path, "lock-failure", "/project", "tool-1")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_stale_turn_rejection_is_quiet_and_preserves_marker(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        session_id = "stale"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        before = hooks._read_open_turn(session_dir_)
        assert before is not None
        _append_frames(
            transcript,
            _assistant_frame("msg-stale", "2026-08-22T10:00:01.000Z", "tool-stale"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-stale"),
        )
        _append_tool_pair(config, session_id, "tool-stale")
        monkeypatch.setattr(live_spans, "_advance_turn_cursor", lambda *args, **kwargs: False)

        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-stale")

        assert hooks._read_open_turn(session_dir_) == before
        assert len(_jobs(config)) == 1

    def test_stop_does_not_reexport_a_call_already_committed_live(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        """The trailing fragment of a `message.id` split across a `user` frame
        can arrive after the last PostToolUse, leaving Stop to parse it. Its
        chat span was already exported live, so Stop must not export it again.
        """
        session_id = "stop-committed"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        _append_frames(
            transcript,
            _assistant_frame("msg-split", "2026-08-22T10:00:01.000Z", "tool-a"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-a"),
        )
        _append_tool_pair(config, session_id, "tool-a")
        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-a")

        # Same message.id reopening after the user frame, with no further
        # PostToolUse to consume it.
        _append_frames(
            transcript,
            {
                "type": "assistant",
                "timestamp": "2026-08-22T10:00:03.000Z",
                "message": {
                    "id": "msg-split",
                    "model": "claude-sonnet-5",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [{"type": "text", "text": "done"}],
                },
            },
        )
        _stdin(
            monkeypatch,
            {
                "session_id": session_id,
                "cwd": "/project",
                "transcript_path": str(transcript),
                "last_assistant_message": "done",
            },
        )

        hooks.stop()

        turn_job = next(job for job in _jobs(config) if job["kind"] == "turn")
        assert [call["call_id"] for call in turn_job["turn"]["llm_calls"]] == []

    def test_stop_exports_only_remaining_final_call_and_completes_turn(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        session_id = "stop-completion"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        _append_frames(
            transcript,
            _assistant_frame("msg-tool", "2026-08-22T10:00:01.000Z", "tool-1"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-1"),
        )
        _append_tool_pair(config, session_id, "tool-1")
        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-1")
        _append_frames(
            transcript,
            {
                "type": "assistant",
                "timestamp": "2026-08-22T10:00:03.000Z",
                "message": {
                    "id": "msg-final",
                    "model": "claude-sonnet-5",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                    "content": [{"type": "text", "text": "complete"}],
                },
            },
        )
        _stdin(
            monkeypatch,
            {
                "session_id": session_id,
                "cwd": "/project",
                "transcript_path": str(transcript),
                "last_assistant_message": "complete",
            },
        )

        hooks.stop()

        jobs = _jobs(config)
        turn_job = next(job for job in jobs if job["kind"] == "turn")
        assert [call["call_id"] for call in turn_job["turn"]["llm_calls"]] == ["msg-final"]
        assert turn_job["turn"]["status"] == "completed"
        assert hooks._read_open_turn(session_dir_) is None
        assert [event["t"] for event in SessionReader(session_dir_).iter_events()][-1] == (
            "assistant_message"
        )

    def test_message_split_by_tool_result_emits_one_chat_span(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        """Claude writes a parallel tool batch as separate assistant frames and
        interleaves each tool_result between them, so one `message.id` arrives
        either side of a `user` frame. Both fragments describe the same LLM
        call, so only one chat span may be emitted for it."""
        session_id = "split-group"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)

        _append_frames(
            transcript,
            _assistant_frame("msg-split", "2026-08-22T10:00:01.000Z", "tool-a"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-a"),
        )
        _append_tool_pair(config, session_id, "tool-a")
        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-a")

        _append_frames(
            transcript,
            _assistant_frame("msg-split", "2026-08-22T10:00:03.000Z", "tool-b"),
            _user_tool_results("2026-08-22T10:00:04.000Z", "tool-b"),
        )
        _append_tool_pair(config, session_id, "tool-b")
        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-b")

        chat_ids = [
            span["span_id"]
            for job in _jobs(config)
            for span in job["spans"]
            if span["name"].startswith("chat")
        ]
        assert chat_ids == [str(chat_span_id(session_id, "msg-split"))]

    def test_live_spans_carry_session_and_turn_identity(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
    ) -> None:
        """A live span's `agent-turn` parent does not exist until Stop, so the
        span must name its session and turn itself to be attributable."""
        session_id = "identity"
        transcript = tmp_path / "transcript.jsonl"
        session_dir_ = _start_turn(monkeypatch, config, transcript, session_id=session_id)
        _append_frames(
            transcript,
            _assistant_frame("msg-ident", "2026-08-22T10:00:01.000Z", "tool-ident"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-ident"),
        )
        _append_tool_pair(config, session_id, "tool-ident")

        live_spans.emit_live_spans(config, session_dir_, session_id, "/project", "tool-ident")

        marker = hooks._read_open_turn(session_dir_)
        assert marker is not None
        job = _jobs(config)[0]
        assert [span["name"] for span in job["spans"]] == ["chat claude-sonnet-5", "tool: Read"]
        for span in job["spans"]:
            assert span["turn_seq"] == marker["turn_seq"]
            assert span["turn_span_id"] == marker["turn_span_id"]


class TestPostToolUseIsolation:
    def test_disabled_export_still_records_result_silently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
        _stdin(
            monkeypatch,
            {
                "session_id": "disabled",
                "cwd": "/project",
                "tool_use_id": "tool-disabled",
                "tool_response": "done",
            },
        )

        hooks.post_tool_use()

        events = list(Store(Config.load()).reader("disabled").iter_events())
        captured = capsys.readouterr()
        assert events[-1]["t"] == "tool_result"
        assert captured.out == ""
        assert captured.err == ""

    def test_raising_export_still_records_result_silently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: Config,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_id = "raising-export"
        transcript = tmp_path / "transcript.jsonl"
        _start_turn(monkeypatch, config, transcript, session_id=session_id)
        _append_frames(
            transcript,
            _assistant_frame("msg-raise", "2026-08-22T10:00:01.000Z", "tool-raise"),
            _user_tool_results("2026-08-22T10:00:02.000Z", "tool-raise"),
        )
        store = Store(config)
        store.append_event(
            session_id=session_id,
            platform="claude",
            cwd="/project",
            t="tool_call",
            data={
                "tool_name": "Read",
                "tool_use_id": "tool-raise",
                "tool_input": {"file_path": "raise.py"},
            },
        )
        monkeypatch.setattr(
            live_spans,
            "export_spans",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("job write failed")),
        )
        _stdin(
            monkeypatch,
            {
                "session_id": session_id,
                "cwd": "/project",
                "tool_use_id": "tool-raise",
                "tool_response": "done",
            },
        )

        hooks.post_tool_use()

        events = list(Store(config).reader(session_id).iter_events())
        captured = capsys.readouterr()
        assert events[-1]["t"] == "tool_result"
        assert captured.out == ""
        assert captured.err == ""
