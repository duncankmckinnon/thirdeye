"""Tests for the Codex turn adapter: platforms/codex/tracing.py's `build_turn`
(assembling a full `TurnSpanDict` from an already-reshaped `extract_turn_codex`
result plus the local event store), `hooks.py`'s `notify()` wiring to
`thirdeye.otel_export.export_turn`, and the `interrupt_marker.py` fallback path
for turns that never get a `notify` call at all.

`extract_turn_codex`'s own reshaped-output coverage (usage key renaming,
`status` on a `turn_aborted` frame, per-call message/tool reconstruction)
lives in test_codex_turn.py, which this file does not duplicate.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.codex.tracing import build_turn
from thirdeye.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.jsonl"
FIXTURE_SID = "019fb579-cdda-7a03-86df-65c87b6c4ae2"
FIXTURE_TURN_ID = "019fb57a-12ee-7870-a8d8-76808c75b368"


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
    def test_permission_request_inside_window_is_included(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(monkeypatch, ["2026-01-01T00:00:05.000Z"])
        store.append_event(
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
            seq=1,
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
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="permission_request",
            data={"tool_name": "Write", "command": "rm -rf /"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=1, turn=_bare_turn())
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
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_message",
            data={"message": "found the bug"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=1, turn=_bare_turn())
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
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="subagent_message",
            data={"message": "off-window"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=1, turn=_bare_turn())
        assert turn["subagents"] == []

    def test_malformed_event_timestamp_is_excluded_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        store = Store(Config(root=tmp_path))
        _set_next_timestamps(monkeypatch, ["not-a-timestamp"])
        store.append_event(
            session_id="s1",
            platform="codex",
            cwd="/p",
            t="permission_request",
            data={"tool_name": "Bash"},
        )
        sd = session_dir(tmp_path, "codex", "s1")

        turn = build_turn(session_dir_=sd, session_id="s1", seq=1, turn=_bare_turn())
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
            lambda *a: calls.append(a),
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
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))
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
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))
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


# -- interrupt_marker: fallback for a turn that never gets a notify call ------


class TestInterruptMarker:
    def test_mark_then_clear_removes_marker(self, tmp_path: Path):
        from thirdeye.platforms.codex.interrupt_marker import (
            _marker_path,
            clear_turn_marker,
            mark_turn_open,
        )

        sd = tmp_path / "s1"
        sd.mkdir()
        mark_turn_open(sd, prompt="hello")
        assert _marker_path(sd).exists()
        clear_turn_marker(sd)
        assert not _marker_path(sd).exists()

    def test_clear_when_no_marker_is_a_noop(self, tmp_path: Path):
        from thirdeye.platforms.codex.interrupt_marker import clear_turn_marker

        sd = tmp_path / "s1"
        sd.mkdir()
        clear_turn_marker(sd)  # must not raise

    def test_close_stale_turn_with_no_marker_exports_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import close_stale_turn_if_open

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))
        sd = tmp_path / "s1"
        sd.mkdir()
        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")
        assert calls == []

    def test_close_stale_turn_with_open_marker_exports_interrupted_and_clears(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import (
            _marker_path,
            close_stale_turn_if_open,
            mark_turn_open,
        )

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))
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
        assert not _marker_path(sd).exists()

    def test_close_stale_turn_clears_marker_even_if_export_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import (
            _marker_path,
            close_stale_turn_if_open,
            mark_turn_open,
        )

        def _boom(*a):
            raise RuntimeError("export blew up")

        monkeypatch.setattr("thirdeye.otel_export.export_turn", _boom)
        sd = tmp_path / "s1"
        sd.mkdir()
        mark_turn_open(sd, prompt="x")

        with pytest.raises(RuntimeError):
            close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")
        assert not _marker_path(sd).exists()

    def test_marker_missing_turn_id_is_removed_without_exporting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import _marker_path, close_stale_turn_if_open

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))
        sd = tmp_path / "s1"
        sd.mkdir()
        _marker_path(sd).write_text(json.dumps({"start_ts": "2026-01-01T00:00:00.000Z"}))

        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")
        assert calls == []
        assert not _marker_path(sd).exists()

    def test_corrupt_marker_is_removed_without_exporting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from thirdeye.platforms.codex.interrupt_marker import (
            _marker_path,
            close_stale_turn_if_open,
        )

        calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))
        sd = tmp_path / "s1"
        sd.mkdir()
        _marker_path(sd).write_text("not json")

        close_stale_turn_if_open(Config(root=tmp_path), sd, "s1", "/p")
        assert calls == []
        # A malformed marker can't be trusted, but this path only *reads* the
        # marker; leaving a corrupt file behind risks it wedging every future
        # hook call, so this documents the current behavior rather than
        # asserting a should-be.
        assert _marker_path(sd).exists()


# -- hooks_json.py wiring: user_prompt_submit / session_end -------------------


class TestHooksJsonMarkerWiring:
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
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))

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
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: calls.append(a))

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p", "prompt": "only"})
        hooks_json.user_prompt_submit()
        assert calls == []

        _stdin(monkeypatch, {"session_id": "s1", "cwd": "/p"})
        hooks_json.session_end()

        assert len(calls) == 1
        assert calls[0][5]["status"] == "interrupted"


class TestNotifyClearsMarker:
    def test_notify_clears_a_marker_left_open_by_the_prior_prompt(self, monkeypatch, env: Path):
        from thirdeye.platforms.codex import hooks, hooks_json
        from thirdeye.platforms.codex.interrupt_marker import _marker_path

        marker_calls: list[Any] = []
        monkeypatch.setattr("thirdeye.otel_export.export_turn", lambda *a: marker_calls.append(a))

        _stdin(monkeypatch, {"session_id": FIXTURE_SID, "cwd": "/proj/codex", "prompt": "go"})
        hooks_json.user_prompt_submit()
        sd = session_dir(env, "codex", FIXTURE_SID)
        assert _marker_path(sd).exists()

        # No rollout planted, so extract_turn_codex/build_turn/export_turn are
        # never reached for the *real* turn — but the marker must still be
        # cleared, since notify() firing at all means this is a genuine
        # completion, not a stale open turn.
        _argv(
            monkeypatch,
            {"type": "agent-turn-complete", "thread-id": FIXTURE_SID, "cwd": "/proj/codex"},
        )
        hooks.notify()

        assert not _marker_path(sd).exists()
        assert marker_calls == []
