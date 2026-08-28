from __future__ import annotations

import fcntl
import io
import json
import threading
import time
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir, tags_path
from thirdeye.platforms.claude import hooks
from thirdeye.platforms.provenance import foreign_payload_reason
from thirdeye.reader import SessionReader
from thirdeye.span_ids import turn_span_id
from thirdeye.store import Store
from thirdeye.usage.store import UsageStore


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


# -- _strip_payload ------------------------------------------------------------


class TestStripPayload:
    def test_removes_routing_keys(self):
        result = hooks._strip_payload({"session_id": "abc", "cwd": "/p", "prompt": "hi"})
        assert "session_id" not in result
        assert "cwd" not in result
        assert result == {"prompt": "hi"}

    def test_removes_transcript_paths(self):
        payload = {
            "session_id": "abc",
            "transcript_path": "/long/path/to/transcript.jsonl",
            "agent_transcript_path": "/long/path/to/agent.jsonl",
            "prompt": "hi",
        }
        assert hooks._strip_payload(payload) == {"prompt": "hi"}

    def test_preserves_other_keys(self):
        payload = {"session_id": "abc", "tool_name": "Read", "tool_input": {"x": 1}}
        result = hooks._strip_payload(payload)
        assert result == {"tool_name": "Read", "tool_input": {"x": 1}}

    def test_empty_dict(self):
        assert hooks._strip_payload({}) == {}

    def test_only_strip_keys(self):
        payload = {
            "session_id": "abc",
            "cwd": "/p",
            "transcript_path": "/x",
            "agent_transcript_path": "/y",
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

    def test_strips_transcript_paths_from_data(self, monkeypatch, env: Path):
        hooks._emit(
            "x",
            {
                "session_id": "s1",
                "cwd": "/p",
                "transcript_path": "/long/path.jsonl",
                "agent_transcript_path": "/long/agent.jsonl",
                "prompt": "hi",
            },
        )
        events = list(Store(Config.load()).reader("s1").iter_events())
        data = events[0].get("data", {})
        assert "transcript_path" not in data
        assert "agent_transcript_path" not in data
        assert data == {"prompt": "hi"}

    def test_uses_cwd_from_payload(self, monkeypatch, env: Path):
        hooks._emit("x", {"session_id": "s1", "cwd": "/my/project"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == "/my/project"

    def test_falls_back_to_os_cwd_when_no_cwd(self, monkeypatch, env: Path):
        monkeypatch.chdir(env)
        hooks._emit("x", {"session_id": "s1"})
        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == str(env)


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
        assert m.platform == "claude"
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
        path = tags_path(session_dir(env, "claude", sid))
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_no_patterns_set_writes_no_tags(self, monkeypatch, env: Path):
        monkeypatch.delenv("THIRDEYE_CAPTURE_ENV", raising=False)
        monkeypatch.setenv("WB_PLAN", "p")
        monkeypatch.setenv("WB_STEP", "test#1")
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        assert not tags_path(session_dir(env, "claude", "s1")).exists()

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


# -- user_prompt_submit --------------------------------------------------------


class TestUserPromptSubmit:
    def test_appends_user_message(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc-123", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "abc-123", "cwd": "/p", "prompt": "hello"})
        hooks.user_prompt_submit()
        store = Store(Config.load())
        events = list(store.reader("abc-123").iter_events())
        assert events[0]["t"] == "session_start"
        assert events[1]["t"] == "user_message"
        assert events[1]["data"]["prompt"] == "hello"
        assert "session_id" not in events[1].get("data", {})


# -- user_prompt_submit hashtag extraction -------------------------------------


class TestUserPromptHashtagExtract:
    def _tags_lines(self, env: Path, sid: str) -> list[dict]:
        path = tags_path(session_dir(env, "claude", sid))
        if not path.exists():
            return []
        out: list[dict] = []
        for line in path.read_text().splitlines():
            if line:
                out.append(json.loads(line))
        return out

    def test_extracts_hashtags_into_tag_store(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "prompt": "let's #fix the #login-bug today",
            },
        )
        hooks.user_prompt_submit()

        store = Store(Config.load())
        events = list(store.reader("s1").iter_events())
        assert events[1]["t"] == "user_message"
        user_seq = events[1]["seq"]

        lines = self._tags_lines(env, "s1")
        assert len(lines) == 2
        tags = {line["tag"] for line in lines}
        assert tags == {"fix", "login-bug"}
        for line in lines:
            assert line["op"] == "add"
            assert line["source"] == "auto"
            assert line["seq"] == user_seq

        m = store.get_meta("s1")
        assert m.tag_count == 1

    def test_no_hashtags_does_not_create_tags_file(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "just a plain prompt"})
        hooks.user_prompt_submit()
        path = tags_path(session_dir(env, "claude", "s1"))
        assert not path.exists() or path.read_text() == ""
        m = Store(Config.load()).get_meta("s1")
        assert m.tag_count == 0

    def test_missing_prompt_field_no_crash(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.user_prompt_submit()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "user_message"
        assert self._tags_lines(env, "s1") == []

    def test_null_prompt_no_crash(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": None})
        hooks.user_prompt_submit()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "user_message"
        assert self._tags_lines(env, "s1") == []

    def test_malformed_stdin_silent_noop(self, monkeypatch, env: Path):
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        hooks.user_prompt_submit()
        assert list(Store(Config.load()).list_sessions()) == []


# -- user_prompt_submit open-turn marker offset --------------------------------


class TestOpenTurnCursor:
    def test_user_prompt_writes_platform_scoped_turn_span_id(self, monkeypatch, env: Path):
        sid = "marker-round-trip"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "hello"})
        hooks.user_prompt_submit()

        sd = session_dir(env, "claude", sid)
        marker = hooks._read_open_turn(sd)
        assert marker is not None
        event = SessionReader(sd).get_event(marker["turn_seq"])

        assert marker["turn_span_id"] == str(turn_span_id("claude", sid, marker["turn_seq"]))
        assert marker["last_frame_ts"] == event["ts"]
        assert marker["start_ts"] == event["ts"]

    def test_advance_updates_cursor_fields(self, monkeypatch, env: Path):
        sid = "advance"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "hello"})
        hooks.user_prompt_submit()
        sd = session_dir(env, "claude", sid)
        original = hooks._read_open_turn(sd)
        assert original is not None

        advanced = hooks._advance_turn_cursor(
            sd,
            expected_turn_seq=original["turn_seq"],
            offset=9182,
            last_frame_ts="2026-08-22T12:34:56.789Z",
        )

        marker = hooks._read_open_turn(sd)
        assert advanced is True
        assert marker is not None
        assert marker["transcript_offset"] == 9182
        assert marker["last_frame_ts"] == "2026-08-22T12:34:56.789Z"
        assert marker["turn_span_id"] == original["turn_span_id"]

    def test_advance_rejects_stale_turn_without_mutating_marker(self, monkeypatch, env: Path):
        sid = "stale-writer"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "hello"})
        hooks.user_prompt_submit()
        sd = session_dir(env, "claude", sid)
        before = hooks._read_open_turn(sd)
        assert before is not None

        advanced = hooks._advance_turn_cursor(
            sd,
            expected_turn_seq=before["turn_seq"] - 1,
            offset=999999,
            last_frame_ts="2099-01-01T00:00:00Z",
        )

        assert advanced is False
        assert hooks._read_open_turn(sd) == before

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("turn_seq", -1),
            ("turn_span_id", "0"),
            ("turn_span_id", str(2**64)),
            ("transcript_offset", -1),
            ("last_frame_ts", 123),
        ],
    )
    def test_reader_rejects_semantically_invalid_marker(
        self, monkeypatch, env: Path, field: str, value: object
    ):
        sid = "invalid-marker"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "hello"})
        hooks.user_prompt_submit()
        sd = session_dir(env, "claude", sid)
        marker = json.loads(hooks._open_turn_path(sd).read_text())
        marker[field] = value
        hooks._open_turn_path(sd).write_text(json.dumps(marker))

        assert hooks._read_open_turn(sd) is None

    def test_compare_and_delete_preserves_newer_marker(self, monkeypatch, env: Path):
        sid = "compare-delete"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "hello"})
        hooks.user_prompt_submit()
        sd = session_dir(env, "claude", sid)
        marker = hooks._read_open_turn(sd)
        assert marker is not None
        stale_seq = marker["turn_seq"]
        marker["turn_seq"] += 1
        marker["turn_span_id"] = str(turn_span_id("claude", sid, marker["turn_seq"]))
        hooks._write_open_turn(sd, marker)

        assert hooks._delete_open_turn(sd, expected_turn_seq=stale_seq) is False
        assert hooks._read_open_turn(sd) == marker

    def test_delete_rejects_marker_that_fails_shared_validation(self, monkeypatch, env: Path):
        sid = "invalid-delete"
        _stdin(monkeypatch, {"session_id": sid, "cwd": "/p", "prompt": "hello"})
        hooks.user_prompt_submit()
        sd = session_dir(env, "claude", sid)
        marker_path = hooks._open_turn_path(sd)
        marker = json.loads(marker_path.read_text())
        marker["last_frame_ts"] = 123
        marker_path.write_text(json.dumps(marker))

        assert hooks._delete_open_turn(sd, expected_turn_seq=marker["turn_seq"]) is False
        assert marker_path.exists()


class TestLockedOpenTurnBoundedRetry:
    """A background subagent's dispatching PostToolUse and its own
    SubagentStart can fire concurrently, both wanting this lock. Blocking
    indefinitely risks the harness's hook timeout killing the process with
    zero signal; every caller already tolerates a raised error (see
    TestHookInvocationBreadcrumbs and the `except OSError`/`except Exception`
    around every call site) and falls back to Stop-time reconstruction, so
    the lock must give up after a bounded wait instead of blocking forever.
    """

    def test_uncontended_acquisition_still_works(self, tmp_path: Path):
        entered = False
        with hooks._locked_open_turn(tmp_path, fcntl.LOCK_EX):
            entered = True
        assert entered

    def test_reentrant_acquisition_still_works(self, tmp_path: Path):
        depths = []
        with hooks._locked_open_turn(tmp_path, fcntl.LOCK_EX):
            depths.append(1)
            with hooks._locked_open_turn(tmp_path, fcntl.LOCK_EX):
                depths.append(2)
        assert depths == [1, 2]

    def test_gives_up_with_timeout_error_instead_of_blocking_forever(self, tmp_path: Path):
        lock_path = hooks._open_turn_lock_path(tmp_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # A separate open file description on the same path -- flock locks
        # are scoped to the open file description, not the process, so this
        # genuinely contends with a fresh `_locked_open_turn` call the same
        # way a different hook process holding the lock would.
        holder = lock_path.open("a+")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                with hooks._locked_open_turn(tmp_path, fcntl.LOCK_EX):
                    pass
            elapsed = time.monotonic() - start
            assert elapsed < 2.0, "must give up well before a realistic hook timeout, not hang"
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

    def test_succeeds_once_contention_clears_within_budget(self, tmp_path: Path):
        lock_path = hooks._open_turn_lock_path(tmp_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        holder = lock_path.open("a+")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)

        def release_shortly() -> None:
            time.sleep(0.05)
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

        releaser = threading.Thread(target=release_shortly)
        releaser.start()
        try:
            entered = False
            with hooks._locked_open_turn(tmp_path, fcntl.LOCK_EX):
                entered = True
            assert entered, (
                "must retry and succeed once the other holder releases, not give up early"
            )
        finally:
            releaser.join()

    def test_shared_lock_also_gives_up_under_contention(self, tmp_path: Path):
        lock_path = hooks._open_turn_lock_path(tmp_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        holder = lock_path.open("a+")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        try:
            with pytest.raises(TimeoutError):
                with hooks._locked_open_turn(tmp_path, fcntl.LOCK_SH):
                    pass
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()


class TestUserPromptSubmitTranscriptOffset:
    """The marker's starting `transcript_offset` is measured fresh from the
    transcript file, not inherited from the usage bookmark.

    The usage bookmark only advances at `Stop`, so an interrupted turn left it
    pointing at the *previous* turn's start — and the next turn's marker
    inherited that staleness, making `build_turn` re-parse and re-emit the
    previous turn's chat calls as duplicates.
    """

    def _marker(self, env: Path, sid: str) -> dict:
        return json.loads((session_dir(env, "claude", sid) / "claude-open-turn.json").read_text())

    def _append_frames(self, transcript: Path, n: int) -> None:
        with transcript.open("a", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"type": "assistant", "message": {"id": f"msg_{i}"}}) + "\n")

    def test_interrupted_turn_does_not_replay_previous_frames(
        self, monkeypatch, env: Path, tmp_path: Path
    ):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("")

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()

        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "prompt": "turn one",
                "transcript_path": str(transcript),
            },
        )
        hooks.user_prompt_submit()
        assert self._marker(env, "s1")["transcript_offset"] == 0

        # Turn 1 produces frames, then is interrupted — `stop()` never runs, so
        # the usage bookmark never advances.
        self._append_frames(transcript, 3)
        size_after_turn_1 = transcript.stat().st_size

        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "prompt": "turn two",
                "transcript_path": str(transcript),
            },
        )
        hooks.user_prompt_submit()

        assert self._marker(env, "s1")["transcript_offset"] == size_after_turn_1

    def test_completed_turn_starts_next_marker_after_its_frames(
        self, monkeypatch, env: Path, tmp_path: Path
    ):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("")

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()

        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "prompt": "turn one",
                "transcript_path": str(transcript),
            },
        )
        hooks.user_prompt_submit()

        self._append_frames(transcript, 2)
        size_after_turn_1 = transcript.stat().st_size

        _stdin(
            monkeypatch,
            {"session_id": "s1", "cwd": "/p", "transcript_path": str(transcript)},
        )
        hooks.stop()

        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "prompt": "turn two",
                "transcript_path": str(transcript),
            },
        )
        hooks.user_prompt_submit()

        assert self._marker(env, "s1")["transcript_offset"] == size_after_turn_1

    def test_missing_transcript_path_yields_zero(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "hi"})
        hooks.user_prompt_submit()
        assert self._marker(env, "s1")["transcript_offset"] == 0

    def test_nonexistent_transcript_yields_zero(self, monkeypatch, env: Path, tmp_path: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "prompt": "hi",
                "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
            },
        )
        hooks.user_prompt_submit()
        assert self._marker(env, "s1")["transcript_offset"] == 0

    def test_directory_as_transcript_path_yields_zero(self, monkeypatch, env: Path, tmp_path: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {"session_id": "s1", "cwd": "/p", "prompt": "hi", "transcript_path": str(tmp_path)},
        )
        hooks.user_prompt_submit()
        assert self._marker(env, "s1")["transcript_offset"] == 0

    def test_usage_bookmark_is_left_untouched(self, monkeypatch, env: Path, tmp_path: Path):
        transcript = tmp_path / "transcript.jsonl"
        self._append_frames(transcript, 4)

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()

        usage = UsageStore(session_dir(env, "claude", "s1"))
        usage.write_state(transcript_offset=17, last_seq=3)

        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "prompt": "hi",
                "transcript_path": str(transcript),
            },
        )
        hooks.user_prompt_submit()

        assert usage.read_state() == {"transcript_offset": 17, "last_seq": 3}
        assert self._marker(env, "s1")["transcript_offset"] == transcript.stat().st_size


# -- pre_tool_use --------------------------------------------------------------


class TestPreToolUse:
    def test_appends_tool_call(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": "abc",
                "cwd": "/p",
                "tool_name": "Read",
                "tool_use_id": "tu_1",
                "tool_input": {"file_path": "x.py"},
            },
        )
        hooks.pre_tool_use()
        events = list(Store(Config.load()).reader("abc").iter_events())
        assert events[1]["t"] == "tool_call"
        assert events[1]["data"]["tool_name"] == "Read"
        assert events[1]["data"]["tool_use_id"] == "tu_1"
        assert events[1]["data"]["tool_input"] == {"file_path": "x.py"}


# -- post_tool_use -------------------------------------------------------------


class TestPostToolUse:
    def test_appends_tool_result(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "abc", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": "abc",
                "cwd": "/p",
                "tool_name": "Read",
                "tool_use_id": "tu_1",
                "tool_response": "<file contents>",
            },
        )
        hooks.post_tool_use()
        events = list(Store(Config.load()).reader("abc").iter_events())
        assert events[1]["t"] == "tool_result"
        assert events[1]["data"]["tool_response"] == "<file contents>"


# -- stop ----------------------------------------------------------------------


class TestStop:
    def test_appends_assistant_message(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "response": "done"})
        hooks.stop()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "assistant_message"
        assert events[1]["data"]["response"] == "done"

    def test_assistant_message_carries_no_usage_attributes(self, monkeypatch, env: Path):
        """Usage now rides on each LLM call's own span, not merged onto
        assistant_message's attributes — a span can't be amended once built,
        so there's no way to attach a turn-wide total to assistant_message
        without either delaying its export (which this codebase deliberately
        never does — see otel_export.py's module docstring) or losing
        per-call granularity, which the reasoning/content capture depends on.
        """
        transcript = Path(__file__).parent / "fixtures" / "usage" / "claude_transcript.jsonl"
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {"session_id": "s1", "cwd": "/p", "transcript_path": str(transcript)},
        )
        hooks.stop()

        events = list(Store(Config.load()).reader("s1").iter_events())
        assistant_message = next(e for e in events if e["t"] == "assistant_message")
        assert not any(k.startswith("gen_ai.") for k in assistant_message["data"])


# -- subagent_stop -------------------------------------------------------------


class TestSubagentStop:
    def test_appends_subagent_message(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "agent": "explore"})
        hooks.subagent_stop()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "subagent_message"
        assert events[1]["data"]["agent"] == "explore"


# -- hook invocation breadcrumbs ------------------------------------------------


def _error_log_entries(home: Path) -> list[dict]:
    log = home / "logs" / "usage-errors.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


# -- foreign payload provenance ------------------------------------------------


_FOREIGN_CURSOR_HOOKS = [
    ("pre_tool_use", "beforeShellExecution"),
    ("user_prompt_submit", "beforeSubmitPrompt"),
    ("stop", "stop"),
    ("subagent_stop", "subagentStop"),
    ("stop_failure", "afterAgentResponse"),
]


@pytest.mark.parametrize("handler_name,hook_event_name", _FOREIGN_CURSOR_HOOKS)
def test_foreign_cursor_payload_writes_no_claude_event(
    monkeypatch,
    env: Path,
    handler_name: str,
    hook_event_name: str,
):
    sid = f"foreign-{handler_name}"
    _stdin(monkeypatch, {"session_id": sid, "cwd": "/claude/project"})
    hooks.session_start()
    before = list(Store(Config.load()).reader(f"claude:{sid}").iter_events())

    _stdin(
        monkeypatch,
        {
            "session_id": sid,
            "cwd": "/cursor/project",
            "hook_event_name": hook_event_name,
            "cursor_version": "1.7.0",
            "prompt": "must not be stored",
            "response": "must not be stored",
            "error": "must not be stored",
        },
    )
    getattr(hooks, handler_name)()

    after = list(Store(Config.load()).reader(f"claude:{sid}").iter_events())
    assert after == before


def test_foreign_payload_does_not_change_open_turn_marker(monkeypatch, env: Path):
    sid = "foreign-marker"
    _stdin(
        monkeypatch,
        {
            "session_id": sid,
            "cwd": "/claude/project",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "genuine Claude turn",
        },
    )
    hooks.user_prompt_submit()
    marker_path = hooks._open_turn_path(session_dir(env, "claude", sid))
    marker_before = marker_path.read_bytes()

    _stdin(
        monkeypatch,
        {
            "session_id": sid,
            "cwd": "/cursor/project",
            "hook_event_name": "stop",
            "cursor_version": "1.7.0",
            "response": "foreign completion",
        },
    )
    hooks.stop()

    assert marker_path.exists(), "foreign Stop must not close the genuine Claude turn"
    assert marker_path.read_bytes() == marker_before
    events = list(Store(Config.load()).reader(f"claude:{sid}").iter_events())
    assert [event["t"] for event in events] == ["user_message"]


def test_foreign_payload_logs_one_warning_with_reason_and_session(monkeypatch, env: Path):
    payload = {
        "session_id": "foreign-warning-session",
        "cwd": "/cursor/project",
        "hook_event_name": "beforeShellExecution",
    }
    expected_reason = foreign_payload_reason(payload, expected="claude")
    assert expected_reason is not None
    _stdin(monkeypatch, payload)

    hooks.pre_tool_use()

    entries = _error_log_entries(env)
    assert len(entries) == 1
    assert entries[0]["level"] == "warn"
    assert entries[0]["phase"] == "foreign_payload"
    assert entries[0]["platform"] == "claude"
    assert entries[0]["session_id"] == "foreign-warning-session"
    assert expected_reason in entries[0]["message"]


def test_foreign_payload_emits_no_hook_invoked_info_breadcrumb(monkeypatch, env: Path):
    _stdin(
        monkeypatch,
        {
            "session_id": "foreign-no-info",
            "cwd": "/cursor/project",
            "composer_mode": "agent",
        },
    )

    hooks.pre_tool_use()

    entries = _error_log_entries(env)
    assert len(entries) == 1
    assert entries[0]["phase"] == "foreign_payload"
    assert not any(entry["phase"] == "hook_invoked" for entry in entries)
    assert not any(entry["level"] == "info" for entry in entries)


def test_genuine_claude_payload_records_unchanged(monkeypatch, env: Path):
    payload = {
        "session_id": "genuine-claude",
        "cwd": "/claude/project",
        "hook_event_name": "PreToolUse",
        "tool_name": "Read",
        "tool_use_id": "tool-1",
        "tool_input": {"file_path": "README.md"},
    }
    _stdin(monkeypatch, payload)

    hooks.pre_tool_use()

    events = list(Store(Config.load()).reader("claude:genuine-claude").iter_events())
    assert len(events) == 1
    assert events[0]["t"] == "tool_call"
    assert events[0]["data"] == {
        key: value for key, value in payload.items() if key not in {"session_id", "cwd"}
    }
    assert not any(entry["phase"] == "foreign_payload" for entry in _error_log_entries(env))


def test_missing_hook_event_name_fails_open_and_records(monkeypatch, env: Path):
    _stdin(
        monkeypatch,
        {
            "session_id": "missing-event-name",
            "cwd": "/claude/project",
            "tool_name": "Read",
            "tool_use_id": "tool-without-event-name",
        },
    )

    hooks.pre_tool_use()

    events = list(Store(Config.load()).reader("claude:missing-event-name").iter_events())
    assert len(events) == 1
    assert events[0]["t"] == "tool_call"
    assert events[0]["data"]["tool_use_id"] == "tool-without-event-name"
    assert not any(entry["phase"] == "foreign_payload" for entry in _error_log_entries(env))


def test_rejection_writes_nothing_to_stdout_or_stderr(monkeypatch, env: Path, capsys):
    _stdin(
        monkeypatch,
        {
            "session_id": "foreign-silent",
            "cwd": "/cursor/project",
            "hook_event_name": "beforeSubmitPrompt",
        },
    )

    hooks.user_prompt_submit()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_foreign_payload_skips_lifecycle_and_export_side_effects(monkeypatch, env: Path):
    def unexpected(*args, **kwargs):
        pytest.fail("foreign payload reached a Claude lifecycle or export side effect")

    from thirdeye import otel_export
    from thirdeye.platforms.claude import live_spans, usage

    monkeypatch.setattr(hooks, "capture_env", unexpected)
    monkeypatch.setattr(hooks, "_close_stale_turn_if_open", unexpected)
    monkeypatch.setattr(Store, "close_session", unexpected)
    monkeypatch.setattr(usage, "capture_usage_claude", unexpected)
    monkeypatch.setattr(live_spans, "emit_live_spans", unexpected)
    monkeypatch.setattr(otel_export, "export_turn", unexpected)
    monkeypatch.setattr(otel_export, "export_subagent_turn", unexpected)

    handlers = (
        hooks.session_start,
        hooks.post_tool_use,
        hooks.stop,
        hooks.subagent_stop,
        hooks.stop_failure,
        hooks.session_end,
    )
    for handler in handlers:
        _stdin(
            monkeypatch,
            {
                "session_id": f"foreign-side-effect-{handler.__name__}",
                "cwd": "/cursor/project",
                "hook_event_name": "stop",
                "cursor_version": "1.7.0",
                "tool_use_id": "foreign-tool",
            },
        )
        handler()


def test_foreign_payload_stays_rejected_when_warning_write_fails(monkeypatch, env: Path):
    def broken_warning(**kwargs):
        raise OSError("read-only log directory")

    sid = "foreign-warning-failure"
    _stdin(monkeypatch, {"session_id": sid, "cwd": "/claude/project"})
    hooks.session_start()
    before = list(Store(Config.load()).reader(f"claude:{sid}").iter_events())

    monkeypatch.setattr(hooks, "log_capture_error", broken_warning)
    _stdin(
        monkeypatch,
        {
            "session_id": sid,
            "cwd": "/cursor/project",
            "hook_event_name": "beforeShellExecution",
        },
    )

    hooks.pre_tool_use()

    after = list(Store(Config.load()).reader(f"claude:{sid}").iter_events())
    assert after == before


def test_provenance_classifier_failure_fails_open_and_records(monkeypatch, env: Path):
    def broken_classifier(payload, expected):
        raise ValueError("unexpected payload shape")

    monkeypatch.setattr(hooks, "foreign_payload_reason", broken_classifier)
    _stdin(
        monkeypatch,
        {
            "session_id": "classifier-failure",
            "cwd": "/claude/project",
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
        },
    )

    hooks.pre_tool_use()

    events = list(Store(Config.load()).reader("claude:classifier-failure").iter_events())
    assert [event["t"] for event in events] == ["tool_call"]


class TestHookInvocationBreadcrumbs:
    def test_post_tool_use_logs_breadcrumb_even_without_session_id(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p", "tool_use_id": "tu_missing_session"})

        hooks.post_tool_use()

        entries = _error_log_entries(env)
        assert any(
            e["phase"] == "hook_invoked" and "tu_missing_session" in e["message"] for e in entries
        )

    def test_subagent_start_logs_breadcrumb_with_agent_id(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "agent_id": "agent-123"})

        hooks.subagent_start()

        entries = _error_log_entries(env)
        assert any(
            e["phase"] == "hook_invoked" and e["session_id"] == "s1" and "agent-123" in e["message"]
            for e in entries
        )

    def test_subagent_stop_logs_breadcrumb_even_without_session_id(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p", "agent_id": "agent-456"})

        hooks.subagent_stop()

        entries = _error_log_entries(env)
        assert any(e["phase"] == "hook_invoked" and "agent-456" in e["message"] for e in entries)


# -- stop_failure --------------------------------------------------------------


class TestStopFailure:
    def test_appends_error(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "error": "timeout"})
        hooks.stop_failure()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "error"
        assert events[1]["data"]["error"] == "timeout"


# -- notification --------------------------------------------------------------


class TestNotification:
    def test_appends_notification(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "message": "task done"})
        hooks.notification()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "notification"
        assert events[1]["data"]["message"] == "task done"


# -- permission_request --------------------------------------------------------


class TestPermissionRequest:
    def test_appends_permission_request(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "tool_name": "Bash",
                "command": "rm -rf /",
            },
        )
        hooks.permission_request()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[1]["t"] == "permission_request"
        assert events[1]["data"]["tool_name"] == "Bash"


# -- new hook events (Claude Code 2.1.195) -------------------------------------


# (handler, mapped event type)
_NEW_HANDLERS = [
    ("post_tool_use_failure", "tool_result"),
    ("subagent_start", "subagent_start"),
    ("user_prompt_expansion", "user_prompt_expansion"),
    ("pre_compact", "compact_start"),
    ("post_compact", "compact_end"),
    ("permission_denied", "permission_denied"),
]


class TestNewHooks:
    @pytest.mark.parametrize("handler_name,event_type", _NEW_HANDLERS)
    def test_appends_one_event_of_mapped_type(
        self, monkeypatch, env: Path, handler_name: str, event_type: str
    ):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "extra": 42})
        getattr(hooks, handler_name)()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert len(events) == 1
        assert events[0]["t"] == event_type
        assert events[0]["data"]["extra"] == 42

    @pytest.mark.parametrize("handler_name,event_type", _NEW_HANDLERS)
    def test_strips_routing_keys(self, monkeypatch, env: Path, handler_name: str, event_type: str):
        _stdin(
            monkeypatch,
            {
                "session_id": "s1",
                "cwd": "/p",
                "transcript_path": "/long/path.jsonl",
                "agent_transcript_path": "/long/agent.jsonl",
                "kept": "yes",
            },
        )
        getattr(hooks, handler_name)()
        data = list(Store(Config.load()).reader("s1").iter_events())[0].get("data", {})
        assert "session_id" not in data
        assert "cwd" not in data
        assert "transcript_path" not in data
        assert "agent_transcript_path" not in data
        assert data == {"kept": "yes"}

    @pytest.mark.parametrize("handler_name,event_type", _NEW_HANDLERS)
    def test_missing_session_id_is_noop(
        self, monkeypatch, env: Path, handler_name: str, event_type: str
    ):
        _stdin(monkeypatch, {"cwd": "/p"})
        getattr(hooks, handler_name)()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_pre_compact_preserves_trigger(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "trigger": "auto"})
        hooks.pre_compact()
        data = list(Store(Config.load()).reader("s1").iter_events())[0].get("data", {})
        assert data["trigger"] == "auto"


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


# -- silent noop edge cases ----------------------------------------------------


class TestSilentNoop:
    def test_missing_session_id_is_silent_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.user_prompt_submit()
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
        hooks.pre_tool_use()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_empty_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "", "cwd": "/p"})
        hooks.stop()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_null_session_id_is_noop(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": None, "cwd": "/p"})
        hooks.notification()
        assert list(Store(Config.load()).list_sessions()) == []


# -- no stdout output ----------------------------------------------------------


class TestNoStdout:
    def test_hooks_do_not_print_to_stdout(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_noop_hooks_do_not_print_to_stdout(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"cwd": "/p"})
        hooks.user_prompt_submit()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_session_end_does_not_print_to_stdout(self, monkeypatch, env: Path, capsys):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_end()
        captured = capsys.readouterr()
        assert captured.out == ""


# -- all event types route correctly -------------------------------------------


class TestAllEventTypesRoute:
    def test_all_event_types_route_correctly(self, monkeypatch, env: Path):
        expected = [
            (hooks.session_start, "session_start"),
            (hooks.user_prompt_submit, "user_message"),
            (hooks.pre_tool_use, "tool_call"),
            (hooks.post_tool_use, "tool_result"),
            (hooks.stop, "assistant_message"),
            (hooks.subagent_stop, "subagent_message"),
            (hooks.stop_failure, "error"),
            (hooks.notification, "notification"),
            (hooks.permission_request, "permission_request"),
            (hooks.session_end, "session_end"),
        ]
        for fn, t in expected:
            _stdin(monkeypatch, {"session_id": "s", "cwd": "/p"})
            fn()
        events = list(Store(Config.load()).reader("s").iter_events())
        assert [e["t"] for e in events] == [t for _, t in expected]


# -- platform constant ---------------------------------------------------------


class TestPlatformConstant:
    def test_platform_is_claude(self, monkeypatch, env: Path):
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks.session_start()
        m = next(Store(Config.load()).list_sessions())
        assert m.platform == "claude"

    def test_platform_constant_value(self):
        assert hooks._PLATFORM == "claude"


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
        hooks.pre_tool_use()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["data"]["tool_input"] == {"nested": {"deep": True, "list": [1, 2, 3]}}

    def test_large_payload(self, monkeypatch, env: Path):
        big_data = {"session_id": "s1", "cwd": "/p", "content": "x" * 10000}
        _stdin(monkeypatch, big_data)
        hooks.stop()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert len(events[0]["data"]["content"]) == 10000

    def test_payload_with_special_chars(self, monkeypatch, env: Path):
        payload = {
            "session_id": "s1",
            "cwd": "/p",
            "text": "line1\nline2\ttab\r\nwindows",
        }
        _stdin(monkeypatch, payload)
        hooks.user_prompt_submit()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["data"]["text"] == "line1\nline2\ttab\r\nwindows"

    def test_unicode_payload(self, monkeypatch, env: Path):
        payload = {"session_id": "s1", "cwd": "/p", "text": "hello world"}
        _stdin(monkeypatch, payload)
        hooks.notification()
        events = list(Store(Config.load()).reader("s1").iter_events())
        assert events[0]["data"]["text"] == "hello world"
