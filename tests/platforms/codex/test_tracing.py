"""Tests for the Codex turn adapter: platforms/codex/tracing.py's `build_turn`
(assembling a full `TurnSpanDict` from an already-reshaped `extract_turn_codex`
result plus the local event store), `hooks.py`'s `notify()` wiring to
`thirdeye.otel_export.export_turn`, and the `interrupt_marker.py` fallback path
for turns that never get a `notify` call at all.

`extract_turn_codex`'s own reshaped-output coverage (usage key renaming,
`status` on a `turn_aborted` frame, per-call message/tool reconstruction)
lives in test_turn.py, which this file does not duplicate.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.codex.tracing import build_turn
from thirdeye.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "rollout.jsonl"
FIXTURE_SID = "019fb579-cdda-7a03-86df-65c87b6c4ae2"
FIXTURE_TURN_ID = "019fb57a-12ee-7870-a8d8-76808c75b368"


def _marker_lock_holder(path: Path) -> subprocess.Popen[str]:
    source_root = Path(__file__).parents[3] / "src"
    environment = os.environ | {"PYTHONPATH": str(source_root)}
    script = """
from pathlib import Path
import sys
from thirdeye._compat.locking import LockMode, locked_fd

with Path(sys.argv[1]).open("a+") as marker:
    with locked_fd(marker.fileno(), LockMode.EXCLUSIVE):
        print("locked", flush=True)
        sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"
    return process


def _release_marker_lock(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.close()
    process.wait(timeout=2)


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    sessions_root = tmp_path / "codex_sessions"
    sessions_root.mkdir()
    monkeypatch.setattr("thirdeye.platforms.codex.rollout.CODEX_SESSIONS_ROOT", sessions_root)
    return tmp_path


def _plant_rollout(env: Path, sid: str = FIXTURE_SID) -> Path:
    nested = env / "codex_sessions" / "2026" / "07" / "30"
    nested.mkdir(parents=True, exist_ok=True)
    dest = nested / f"rollout-2026-07-30T17-01-26-{sid}.jsonl"
    dest.write_bytes(FIXTURE.read_bytes())
    return dest


def _argv(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.argv", ["notify", json.dumps(payload)])


def _stdin(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


def _bare_turn(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        turn_id="t1",
        start_ts="2026-01-01T00:00:00.000Z",
        end_ts="2026-01-01T00:00:10.000Z",
        user_prompt="do it",
        assistant_output="done",
        status="completed",
        calls=[],
    )
    defaults.update(overrides)
    return defaults


def _set_next_timestamps(monkeypatch: pytest.MonkeyPatch, timestamps: list[str]) -> None:
    """Make the next events appended via ``Store.append_event`` carry these
    exact ``ts`` values in order.

    `SessionWriter.append` always stamps "now" via the module-private
    `_utc_iso_ms`, with no way to pass an explicit ts through
    `Store.append_event` — the only seam is monkeypatching the clock itself.
    A session's *first* `append_event` call also stamps `SessionMeta.started_at`
    from the same clock before the event itself, consuming one extra value up
    front — callers pass only the timestamps for the events they care about.
    """
    remaining = ["1970-01-01T00:00:00.000Z", *timestamps]

    def _next() -> str:
        return remaining.pop(0)

    monkeypatch.setattr("thirdeye.writer._utc_iso_ms", _next)


# -- build_turn: permission_requests / subagents by timestamp range -----------


class TestBuildTurnFromLocalStore:
    @pytest.mark.parametrize("hook_output", ["", "hook result"])
    @pytest.mark.parametrize("terminal", ["task_complete", "turn_aborted"])
    def test_subagent_backfills_child_rollout(self, monkeypatch, env, hook_output, terminal):
        child = env / "codex_sessions" / "rollout-child-A.jsonl"
        frames = [
            ("00", "session_meta", {"id": "A"}),
            ("02", "turn_context", {"turn_id": "child-turn", "model": "gpt-test"}),
            ("03", "event_msg", {"type": "user_message", "message": "child task"}),
            ("06", "event_msg", {"type": "agent_message", "message": "child result"}),
            ("07", "event_msg", {"type": terminal, "turn_id": "child-turn"}),
        ]
        child.write_text(
            "".join(
                json.dumps({"timestamp": f"2026-01-01T00:00:{s}Z", "type": t, "payload": p}) + "\n"
                for s, t, p in frames
            )
        )
        store = Store(Config(root=env))
        _set_next_timestamps(monkeypatch, ["2026-01-01T00:00:02Z", "2026-01-01T00:00:08Z"])
        for kind, data in [
            ("subagent_start", {"agent_id": "A"}),
            ("subagent_message", {"agent_id": "A", "message": hook_output}),
        ]:
            seq = store.append_event(session_id="s1", platform="codex", cwd="/p", t=kind, data=data)
        turn = build_turn(
            session_dir_=session_dir(env, "codex", "s1"),
            session_id="s1",
            seq=seq + 1,
            turn=_bare_turn(),
        )
        sub = turn["subagents"][0]
        assert sub["input_message"] == "child task"
        assert sub["output_message"] == (hook_output or "child result")
        assert sub["llm_calls"][0]["model"] == "gpt-test"
        assert sub["start_ts"] == "2026-01-01T00:00:02Z"
        assert sub["end_ts"] == "2026-01-01T00:00:08Z"
        assert sub["status"] == ("interrupted" if terminal == "turn_aborted" else "completed")

    def test_subagent_backfills_child_turn_that_starts_before_the_start_hook(
        self, monkeypatch, env
    ):
        """The child's own ``task_started.started_at`` is whole unix seconds and
        lands before the SubagentStart hook is recorded (about 2s apart in
        Logfire 2026-11-04). Requiring the child turn to sit wholly inside the
        hook window rejected every child, leaving the span with no output or
        calls."""
        child = env / "codex_sessions" / "rollout-child-A.jsonl"
        frames = [
            ("00.500", "session_meta", {"id": "A"}),
            ("01.900", "turn_context", {"turn_id": "child-turn", "model": "gpt-test"}),
            (
                "01.950",
                "event_msg",
                {"type": "task_started", "turn_id": "child-turn", "started_at": 1767225601},
            ),
            ("06.000", "event_msg", {"type": "agent_message", "message": "child result"}),
            (
                "07.500",
                "event_msg",
                {"type": "task_complete", "turn_id": "child-turn", "completed_at": 1767225607},
            ),
        ]
        child.write_text(
            "".join(
                json.dumps({"timestamp": f"2026-01-01T00:00:{s}Z", "type": t, "payload": p}) + "\n"
                for s, t, p in frames
            )
        )
        store = Store(Config(root=env))
        _set_next_timestamps(monkeypatch, ["2026-01-01T00:00:02.219Z", "2026-01-01T00:00:08.000Z"])
        for kind, data in [
            ("subagent_start", {"agent_id": "A"}),
            ("subagent_message", {"agent_id": "A"}),
        ]:
            seq = store.append_event(session_id="s1", platform="codex", cwd="/p", t=kind, data=data)
        turn = build_turn(
            session_dir_=session_dir(env, "codex", "s1"),
            session_id="s1",
            seq=seq + 1,
            turn=_bare_turn(),
        )
        sub = turn["subagents"][0]
        assert sub["output_message"] == "child result"
        assert sub["llm_calls"][0]["model"] == "gpt-test"

    @pytest.mark.parametrize("mode", ["missing", "malformed", "partial", "outside"])
    def test_subagent_rollout_fallback_is_conservative(self, monkeypatch, env, mode):
        from thirdeye.platforms.codex.tracing import _subagent_turn

        child = env / "codex_sessions" / "rollout-child-A.jsonl"
        if mode != "missing":
            frames = [
                {
                    "timestamp": "2026-01-02T00:00:00Z",
                    "type": "turn_context",
                    "payload": {"turn_id": "t1"},
                },
                {
                    "timestamp": "2026-01-02T00:00:01Z",
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "unrelated"},
                },
                {
                    "timestamp": "2026-01-02T00:00:02Z",
                    "type": "event_msg",
                    "payload": {"type": "task_complete", "turn_id": "t1"},
                },
            ]
            text = "\n".join(json.dumps(f) for f in frames) + "\n"
            child.write_text(
                "not json\n" if mode == "malformed" else text[:-5] if mode == "partial" else text
            )
        result = _subagent_turn(
            "s1",
            {
                "seq": 1,
                "ts": "2026-01-01T00:00:00Z",
                "data": {"agent_id": "A", "prompt": "hook task"},
            },
            {"ts": "2026-01-01T00:00:10Z", "data": {}},
        )
        assert result["input_message"] == "hook task"
        assert result["output_message"] == ""
        assert result["llm_calls"] == []

    def test_permission_request_inside_window_is_included(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(monkeypatch, ["2026-01-01T00:00:05.000Z"])
        seq = store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="permission_request",
            data={"tool_name": "Bash", "command": "ls"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(
            session_dir_=sd,
            session_id="s1",
            seq=seq + 1,
            turn=_bare_turn(),
        )
        assert len(turn["permission_requests"]) == 1
        pr = turn["permission_requests"][0]
        assert pr["ts"] == "2026-01-01T00:00:05.000Z"
        assert pr["tool_name"] == "Bash"
        assert pr["attributes"] == {"command": "ls"}

    def test_permission_request_outside_window_is_excluded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(
            monkeypatch,
            ["2026-01-01T00:00:05.000Z", "2026-01-01T23:59:59.000Z"],
        )
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="permission_request",
            data={"tool_name": "Bash", "command": "ls"},
        )
        seq = store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="permission_request",
            data={"tool_name": "Write", "command": "rm -rf /"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=seq + 1, turn=_bare_turn())
        assert [pr["tool_name"] for pr in turn["permission_requests"]] == ["Bash"]

    def test_subagent_pair_inside_window_becomes_nested_turn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(
            monkeypatch,
            ["2026-01-01T00:00:02.000Z", "2026-01-01T00:00:08.000Z"],
        )
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_start",
            data={"prompt": "investigate the bug"},
        )
        seq = store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_message",
            data={"message": "found the bug"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=seq + 1, turn=_bare_turn())
        assert len(turn["subagents"]) == 1
        sub = turn["subagents"][0]
        assert sub["input_message"] == "investigate the bug"
        assert sub["output_message"] == "found the bug"
        assert sub["status"] == "completed"
        # Codex hooks.json carries no transcript to reconstruct a subagent's
        # own LLM/tool calls from, unlike Claude's agent_transcript_path.
        assert sub["llm_calls"] == []
        assert sub["permission_requests"] == []
        assert sub["subagents"] == []

    def test_parallel_subagents_pair_by_agent_id_when_stops_are_not_lifo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(
            monkeypatch,
            [
                "2026-01-01T00:00:02.000Z",
                "2026-01-01T00:00:03.000Z",
                "2026-01-01T00:00:06.000Z",
                "2026-01-01T00:00:08.000Z",
            ],
        )
        for t, data in (
            ("subagent_start", {"agent_id": "A", "prompt": "task A"}),
            ("subagent_start", {"agent_id": "B", "prompt": "task B"}),
            ("subagent_message", {"agent_id": "A", "message": "result A"}),
            ("subagent_message", {"agent_id": "B", "message": "result B"}),
        ):
            seq = store.append_event(session_id="s1", platform="codex", cwd="/p", t=t, data=data)
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=seq + 1, turn=_bare_turn())

        assert [sub["input_message"] for sub in turn["subagents"]] == ["task A", "task B"]
        assert [sub["output_message"] for sub in turn["subagents"]] == ["result A", "result B"]
        assert [sub["attributes"]["agent_id"] for sub in turn["subagents"]] == ["A", "B"]

    def test_subagent_stop_last_assistant_message_is_the_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Codex's SubagentStop payload carries the child's reply as
        ``last_assistant_message`` (seen in Logfire 2026-11-04). Without reading
        it, the subagent span had no output messages and the reply leaked into
        its raw attributes instead."""
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(monkeypatch, ["2026-01-01T00:00:02.000Z", "2026-01-01T00:00:06.000Z"])
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_start",
            data={"agent_id": "A", "hook_event_name": "SubagentStart"},
        )
        seq = store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_message",
            data={
                "agent_id": "A",
                "hook_event_name": "SubagentStop",
                "last_assistant_message": "Wrote result.json",
            },
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=seq + 1, turn=_bare_turn())

        (sub,) = turn["subagents"]
        assert sub["output_message"] == "Wrote result.json"
        assert "last_assistant_message" not in sub["attributes"]

    def test_subagent_pair_outside_window_is_excluded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(
            monkeypatch,
            ["2026-01-02T00:00:02.000Z", "2026-01-02T00:00:08.000Z"],
        )
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_start",
            data={"prompt": "off-window"},
        )
        seq = store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_message",
            data={"message": "off-window"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=seq + 1, turn=_bare_turn())
        assert turn["subagents"] == []

    def test_unmatched_subagent_start_is_excluded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(monkeypatch, ["2026-01-01T00:00:02.000Z"])
        seq = store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_start",
            data={"prompt": "unfinished"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=seq + 1, turn=_bare_turn())
        assert turn["subagents"] == []

    def test_malformed_event_timestamp_is_excluded_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(monkeypatch, ["not-a-timestamp"])
        seq = store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="permission_request",
            data={"tool_name": "Bash"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=seq + 1, turn=_bare_turn())
        assert turn["permission_requests"] == []

    def test_carries_through_calls_status_and_messages_unchanged(self, tmp_path: Path) -> None:
        sd = session_dir(tmp_path, "codex", "s1")
        turn = build_turn(
            session_dir_=sd,
            session_id="s1",
            seq=1,
            turn=_bare_turn(status="interrupted", calls=[{"call_id": "t1:0"}]),
        )
        assert turn["turn_id"] == "t1"
        assert turn["input_message"] == "do it"
        assert turn["output_message"] == "done"
        assert turn["status"] == "interrupted"
        assert turn["llm_calls"] == [{"call_id": "t1:0"}]
        assert turn["attributes"] == {}


# -- notify(): wiring to export_turn -------------------------------------------


class TestNotifyExportsTurn:
    def test_resolvable_turn_calls_export_turn(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn",
            lambda *a, **k: calls.append(a),
        )
        _plant_rollout(env)
        _argv(
            monkeypatch,
            {
                "type": "agent-turn-complete",
                "thread-id": FIXTURE_SID,
                "turn-id": FIXTURE_TURN_ID,
                "cwd": "/proj/codex",
            },
        )
        hooks.notify()

        assert len(calls) == 1
        config, sd, sid, platform, cwd, turn = calls[0]
        assert isinstance(config, Config)
        assert sd == session_dir(env, "codex", FIXTURE_SID)
        assert sid == FIXTURE_SID
        assert platform == "codex"
        assert cwd == "/proj/codex"
        assert turn["turn_id"] == FIXTURE_TURN_ID
        assert turn["status"] == "completed"
        assert isinstance(turn["llm_calls"], list)
        assert turn["permission_requests"] == []
        assert turn["subagents"] == []

    def test_missing_turn_id_never_calls_export_turn(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        _plant_rollout(env)
        _argv(
            monkeypatch,
            {"type": "agent-turn-complete", "thread-id": FIXTURE_SID, "cwd": "/proj/codex"},
        )
        hooks.notify()
        assert calls == []

    def test_unresolvable_rollout_never_calls_export_turn(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        # No rollout planted -> resolve_rollout returns None -> notify() returns
        # early, before extract_turn_codex/build_turn/export_turn are reached.
        _argv(
            monkeypatch,
            {
                "type": "agent-turn-complete",
                "thread-id": FIXTURE_SID,
                "turn-id": FIXTURE_TURN_ID,
                "cwd": "/proj/codex",
            },
        )
        hooks.notify()
        assert calls == []

    def test_spawned_subagent_thread_is_not_exported_as_its_own_session(
        self, monkeypatch, env: Path
    ):
        """A spawned subagent's own thread gets a notify too. The parent's
        export already nests that execution under its SubagentStop span
        (backfilled from this same rollout), so exporting it again added a
        duplicate root session in Logfire (2026-11-04). It is still captured
        locally."""
        from thirdeye.platforms.codex import hooks
        from thirdeye.reader import SessionReader

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        rollout = _plant_rollout(env)
        lines = rollout.read_text().splitlines(keepends=True)
        meta = json.loads(lines[0])
        meta["payload"]["source"] = {
            "subagent": {"thread_spawn": {"parent_thread_id": "parent-thread", "depth": 1}}
        }
        lines[0] = json.dumps(meta) + "\n"
        rollout.write_text("".join(lines))
        _argv(
            monkeypatch,
            {
                "type": "agent-turn-complete",
                "thread-id": FIXTURE_SID,
                "turn-id": FIXTURE_TURN_ID,
                "cwd": "/proj/codex",
            },
        )
        hooks.notify()

        assert calls == []
        sd = session_dir(env, "codex", FIXTURE_SID)
        assert len(list(SessionReader(sd).iter_events(types={"agent_turn"}))) == 1


# -- interrupt_marker: fallback for a turn that never gets a notify call ------


class TestInterruptMarker:
    def test_mark_then_clear_removes_marker(self, tmp_path: Path):
        from thirdeye.platforms.codex.interrupt_marker import (
            clear_marker_not_after,
            has_open_marker,
            mark_turn_open,
        )

        sd = tmp_path / "s1"
        sd.mkdir()
        mark_turn_open(sd, prompt="hello")
        assert has_open_marker(sd)
        clear_marker_not_after(sd, not_after_ts="2999-01-01T00:00:00.000Z")
        assert not has_open_marker(sd)

    def test_clear_when_no_marker_is_a_noop(self, tmp_path: Path):
        from thirdeye.platforms.codex.interrupt_marker import clear_marker_not_after

        sd = tmp_path / "s1"
        sd.mkdir()
        clear_marker_not_after(sd, not_after_ts="2999-01-01T00:00:00.000Z")  # must not raise

    def test_clear_marker_not_after_leaves_newer_marker_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A delayed ``notify`` for an earlier turn must not clobber a newer
        marker a later ``UserPromptSubmit`` already opened while it was still
        in flight -- see the module docstring's race-safety argument.
        """
        from thirdeye.platforms.codex.interrupt_marker import (
            clear_marker_not_after,
            has_open_marker,
            mark_turn_open,
        )

        monkeypatch.setattr(
            "thirdeye.platforms.codex.interrupt_marker.utc_iso_ms",
            lambda: "2026-01-02T00:00:00.000Z",
        )
        sd = tmp_path / "s1"
        sd.mkdir()
        mark_turn_open(sd, prompt="newer turn")

        clear_marker_not_after(sd, not_after_ts="2026-01-01T00:00:00.000Z")

        assert has_open_marker(sd)

    def test_close_stale_turn_with_no_marker_exports_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import close_stale_turn_if_open

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        sd = tmp_path / "s1"
        sd.mkdir()
        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")
        assert calls == []

    def test_close_stale_turn_with_open_marker_exports_interrupted_and_clears(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import (
            close_stale_turn_if_open,
            has_open_marker,
            mark_turn_open,
        )

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        sd = tmp_path / "s1"
        sd.mkdir()
        mark_turn_open(sd, prompt="unfinished business")

        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")

        assert len(calls) == 1
        config, sd_arg, sid, platform, cwd, turn = calls[0]
        assert sd_arg == sd
        assert sid == "s1"
        assert platform == "codex"
        assert cwd == "/p"
        assert turn["status"] == "interrupted"
        assert turn["input_message"] == "unfinished business"
        assert turn["output_message"] == ""
        assert not has_open_marker(sd)

    def test_close_stale_turn_swallows_export_errors_and_still_clears_marker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """``close_stale_turn_if_open`` runs inside a hook subprocess and must
        never raise, however ``export_turn`` behaves -- otherwise a Logfire
        export failure would break a totally unrelated hook invocation
        (e.g. the next ``UserPromptSubmit``).
        """
        from thirdeye.platforms.codex.interrupt_marker import (
            close_stale_turn_if_open,
            has_open_marker,
            mark_turn_open,
        )

        def _boom(*a, **k):
            raise RuntimeError("export blew up")

        monkeypatch.setattr("thirdeye.otel_export.export_turn", _boom)
        sd = tmp_path / "s1"
        sd.mkdir()
        mark_turn_open(sd, prompt="x")

        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")  # must not raise
        assert not has_open_marker(sd)

    def test_marker_missing_turn_id_is_removed_without_exporting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import (
            _marker_path,
            close_stale_turn_if_open,
            has_open_marker,
        )

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        sd = tmp_path / "s1"
        sd.mkdir()
        _marker_path(sd).write_text(json.dumps({"start_ts": "2026-01-01T00:00:00.000Z"}))

        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")
        assert calls == []
        assert not has_open_marker(sd)

    def test_corrupt_marker_is_quarantined_without_exporting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import (
            _marker_path,
            close_stale_turn_if_open,
            has_open_marker,
        )

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        sd = tmp_path / "s1"
        sd.mkdir()
        _marker_path(sd).write_text("not json")

        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")
        assert calls == []
        # Corrupt content can't be trusted as a real open turn, so it's
        # truncated in place rather than left to wedge every future hook
        # call that reads this marker.
        assert not has_open_marker(sd)


class TestLockedMarkerBoundedRetry:
    """Concurrently dispatched Codex subagent hook processes (SubagentStart /
    SubagentStop firing alongside the parent's own hooks) can contend for
    codex-open-turn.json's lock at the same time. Every caller already
    tolerates a raised OSError (see TestInterruptMarker: each wraps its
    `_locked_marker` use in `except OSError`), so the lock must give up
    after a bounded wait instead of blocking forever.
    """

    def test_uncontended_acquisition_still_works(self, tmp_path: Path):
        from thirdeye.platforms.codex.interrupt_marker import _locked_marker

        sd = tmp_path / "s1"
        entered = False
        with _locked_marker(sd):
            entered = True
        assert entered

    def test_gives_up_with_timeout_error_instead_of_blocking_forever(self, tmp_path: Path):
        from thirdeye.platforms.codex.interrupt_marker import _locked_marker, _marker_path

        sd = tmp_path / "s1"
        sd.mkdir()
        holder = _marker_lock_holder(_marker_path(sd))
        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                with _locked_marker(sd):
                    pass
            elapsed = time.monotonic() - start
            assert elapsed < 2.0, "must give up well before a realistic hook timeout, not hang"
        finally:
            _release_marker_lock(holder)

    def test_succeeds_once_contention_clears_within_budget(self, tmp_path: Path):
        from thirdeye.platforms.codex.interrupt_marker import _locked_marker, _marker_path

        sd = tmp_path / "s1"
        sd.mkdir()
        holder = _marker_lock_holder(_marker_path(sd))
        assert holder.stdin is not None
        holder.stdin.write("release\n")
        holder.stdin.flush()
        try:
            entered = False
            with _locked_marker(sd):
                entered = True
            assert entered
        finally:
            holder.stdin.close()
            holder.wait(timeout=2)


# -- hooks_json.py wiring: user_prompt_submit / session_end -------------------


class TestHooksJsonMarkerWiring:
    def test_repeated_prompt_id_does_not_interrupt_or_replace_its_marker(self, monkeypatch, env):
        from thirdeye.platforms.codex import hooks_json
        from thirdeye.platforms.codex.interrupt_marker import _marker_path

        calls = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        payload = {"session_id": "s1", "cwd": "/p", "prompt": "go", "prompt_id": "p1"}
        _stdin(monkeypatch, payload)
        hooks_json.user_prompt_submit()
        path = _marker_path(session_dir(env, "codex", "s1"))
        original = path.read_text()
        _stdin(monkeypatch, payload)
        hooks_json.user_prompt_submit()
        assert calls == []
        assert path.read_text() == original

    def test_prompt_id_distinguishes_active_from_interrupted_turn(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks_json

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        _stdin(
            monkeypatch,
            {"session_id": "s1", "cwd": "/p", "prompt": "go", "prompt_id": "p1"},
        )
        hooks_json.user_prompt_submit()

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt_id": "p1"})
        hooks_json.subagent_start()
        assert calls == []

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt_id": "p2"})
        hooks_json.subagent_start()
        assert len(calls) == 1
        assert calls[0][-1]["status"] == "interrupted"

    def test_late_prompt_hook_cannot_overwrite_newer_marker(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex.interrupt_marker import _marker_path, replace_open_turn

        config = Config.load()
        sd = session_dir(env, "codex", "s1")
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: None)
        replace_open_turn(config, sd, "s1", "/p", prompt="new", prompt_id="p2", turn_seq=2)
        replace_open_turn(config, sd, "s1", "/p", prompt="old", prompt_id="p1", turn_seq=1)

        marker = json.loads(_marker_path(sd).read_text())
        assert marker["prompt_id"] == "p2"
        assert marker["turn_seq"] == 2

    def test_user_prompt_submit_opens_a_marker(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks_json
        from thirdeye.platforms.codex.interrupt_marker import _marker_path

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "hello"})
        hooks_json.user_prompt_submit()
        sd = session_dir(env, "codex", "s1")
        assert _marker_path(sd).exists()
        marker = json.loads(_marker_path(sd).read_text())
        assert marker["input_message"] == "hello"

    def test_second_user_prompt_submit_closes_the_first_marker_as_interrupted(
        self, monkeypatch, env: Path
    ):
        from thirdeye.platforms.codex import hooks_json

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "first"})
        hooks_json.user_prompt_submit()
        assert calls == []  # nothing stale on the very first prompt

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "second"})
        hooks_json.user_prompt_submit()

        assert len(calls) == 1
        turn = calls[0][5]
        assert turn["status"] == "interrupted"
        assert turn["input_message"] == "first"

    def test_session_end_closes_a_stale_marker(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks_json

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "only"})
        hooks_json.user_prompt_submit()
        assert calls == []

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_end()

        assert len(calls) == 1
        assert calls[0][5]["status"] == "interrupted"

    def test_prompt_for_the_same_codex_turn_keeps_the_open_marker(self, monkeypatch, env: Path):
        """Codex sends its own ``turn_id`` on UserPromptSubmit. Another prompt
        hook for that same turn -- a follow-up typed mid-turn, or one event run
        by two installed hook commands -- is not a new turn, so it must not
        close the running turn as interrupted. In Logfire (2026-11-04) each
        such prompt added an input-only "interrupted" turn beside the real one.
        """
        from thirdeye.platforms.codex import hooks_json
        from thirdeye.platforms.codex.interrupt_marker import _marker_path

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "go", "turn_id": "T1"})
        hooks_json.user_prompt_submit()
        path = _marker_path(session_dir(env, "codex", "s1"))
        original = path.read_text()

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "also", "turn_id": "T1"})
        hooks_json.user_prompt_submit()

        assert calls == []
        assert path.read_text() == original

    def test_session_end_does_not_interrupt_a_turn_the_rollout_completed(
        self, monkeypatch, env: Path
    ):
        """``codex exec`` delivers SessionEnd before the notify that exports
        the completed turn, so the open marker must not also be exported as an
        interrupted turn when the rollout already records its completion."""
        from thirdeye.platforms.codex import hooks_json

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        rollout = env / "codex_sessions" / "rollout-2026-01-01T00-00-00-s1.jsonl"
        rollout.write_text(
            json.dumps(
                {
                    "timestamp": "2026-01-01T00:00:05.000Z",
                    "type": "event_msg",
                    "payload": {"type": "task_complete", "turn_id": "T1"},
                }
            )
            + "\n"
        )
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "go", "turn_id": "T1"})
        hooks_json.user_prompt_submit()

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_end()

        assert calls == []


class TestMidTurnHooksReapAbandonedMarker:
    """subagent_start/subagent_stop/permission_request/pre_compact/post_compact/
    session_start all legitimately fire *mid-turn*, so each must check for a
    stale marker too (a catch-all in case UserPromptSubmit/SessionEnd never
    come next) -- but must not mistake the current, still-open turn's own
    fresh marker for an abandoned one.
    """

    @staticmethod
    def _seed_abandoned_marker(sd: Path) -> None:
        from thirdeye.platforms.codex.interrupt_marker import _marker_path

        sd.mkdir(parents=True, exist_ok=True)
        _marker_path(sd).write_text(
            json.dumps(
                {
                    "turn_id": "old-turn",
                    "start_ts": "2000-01-01T00:00:00.000Z",
                    "input_message": "long abandoned",
                }
            )
        )

    def test_each_mid_turn_hook_reaps_an_abandoned_marker(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks_json

        hooks_by_name = {
            "subagent_start": hooks_json.subagent_start,
            "subagent_stop": hooks_json.subagent_stop,
            "permission_request": hooks_json.permission_request,
            "pre_compact": hooks_json.pre_compact,
            "post_compact": hooks_json.post_compact,
            "session_start": hooks_json.session_start,
        }
        for name, fn in hooks_by_name.items():
            sid = f"s-{name}"
            sd = session_dir(env, "codex", sid)
            self._seed_abandoned_marker(sd)

            calls: list[Any] = []
            monkeypatch.setattr(
                "thirdeye.otel_export.export_turn", lambda *a, calls=calls, **k: calls.append(a)
            )
            _stdin(monkeypatch, {"session_id": sid, "cwd": "/p"})
            fn()

            assert len(calls) == 1, f"{name} did not reap the abandoned marker"
            assert calls[0][5]["status"] == "interrupted"
            assert calls[0][5]["input_message"] == "long abandoned"

    def test_mid_turn_hook_does_not_reap_its_own_fresh_marker(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks_json
        from thirdeye.platforms.codex.interrupt_marker import has_open_marker, mark_turn_open

        sd = session_dir(env, "codex", "s1")
        mark_turn_open(sd, prompt="still running")

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a, **k: calls.append(a))
        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.permission_request()

        assert calls == []
        assert has_open_marker(sd)


class TestNotifyClearsMarker:
    def test_notify_clears_a_marker_left_open_by_the_prior_prompt(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks, hooks_json
        from thirdeye.platforms.codex.interrupt_marker import has_open_marker

        marker_calls: list[Any] = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn", lambda *a, **k: marker_calls.append(a)
        )
        # The fixture rollout's own timestamps are fixed at 2026-07-31; the
        # marker's real wall-clock start_ts must predate the turn's end_ts for
        # clear_marker_not_after to consider it the *same* turn's marker.
        monkeypatch.setattr(
            "thirdeye.platforms.codex.interrupt_marker.utc_iso_ms",
            lambda: "2026-07-31T00:01:00.000Z",
        )

        _stdin(monkeypatch, {"session_id": FIXTURE_SID, "cwd": "/proj/codex", "prompt": "go"})
        hooks_json.user_prompt_submit()
        sd = session_dir(env, "codex", FIXTURE_SID)
        assert has_open_marker(sd)

        # A rollout resolves and the turn is exported for real -- clear_marker_not_after
        # sees the just-completed turn's own end_ts, which is always >= the marker's
        # start_ts (the marker was opened by the UserPromptSubmit that started this
        # very turn), so it clears it.
        _plant_rollout(env, sid=FIXTURE_SID)
        _argv(
            monkeypatch,
            {
                "type": "agent-turn-complete",
                "thread-id": FIXTURE_SID,
                "turn-id": FIXTURE_TURN_ID,
                "cwd": "/proj/codex",
            },
        )
        hooks.notify()

        assert not has_open_marker(sd)
        assert len(marker_calls) == 1
        assert marker_calls[0][5]["status"] == "completed"

    def test_notify_with_unresolvable_rollout_leaves_marker_open(self, monkeypatch, env: Path):
        """Documents a real gap rather than asserting a should-be: if ``notify``
        fires (proving the turn genuinely completed) but the rollout can't be
        resolved, ``notify()`` returns before ever reaching
        ``clear_marker_not_after`` (`hooks.py`'s early ``return`` on unresolved
        rollout happens before ``extract_turn_codex``/the marker clear). The
        marker is left open and a later hook will report this turn as
        "interrupted" even though it actually completed -- ``notify()``
        returning early on unresolved rollout doesn't distinguish "genuinely
        interrupted" from "we just couldn't find the rollout file".
        """
        from thirdeye.platforms.codex import hooks, hooks_json
        from thirdeye.platforms.codex.interrupt_marker import has_open_marker

        marker_calls: list[Any] = []
        monkeypatch.setattr(
            "thirdeye.otel_export.export_turn", lambda *a, **k: marker_calls.append(a)
        )

        _stdin(monkeypatch, {"session_id": FIXTURE_SID, "cwd": "/proj/codex", "prompt": "go"})
        hooks_json.user_prompt_submit()
        sd = session_dir(env, "codex", FIXTURE_SID)
        assert has_open_marker(sd)

        # No rollout planted this time -> resolve_rollout(sid) returns None.
        _argv(
            monkeypatch,
            {"type": "agent-turn-complete", "thread-id": FIXTURE_SID, "cwd": "/proj/codex"},
        )
        hooks.notify()

        assert has_open_marker(sd)
        assert marker_calls == []
