from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from thirdeye import otel_export, otel_worker
from thirdeye.config import Config, LogfireSettings
from thirdeye.span_ids import interaction_span_id, trace_id_for_session, turn_span_id

pytest.importorskip("logfire")

from logfire.testing import TestExporter  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    otel_export._state["attempted"] = False
    otel_export._state["instance"] = None
    yield
    otel_export._state["attempted"] = False
    otel_export._state["instance"] = None


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def enabled(home: Path) -> None:
    Config.load().write_logfire_settings(LogfireSettings(enabled=True, token="fake-token"))


def _turn(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        turn_id="turn_1",
        start_ts="2026-01-01T00:00:00.000Z",
        end_ts="2026-01-01T00:00:05.000Z",
        input_message="hi",
        output_message="hello",
        status="completed",
        llm_calls=[],
        permission_requests=[],
        subagents=[],
        attributes={},
    )
    defaults.update(overrides)
    return defaults


def _error_log_entries(home: Path) -> list[dict]:
    log = home / "logs" / "usage-errors.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _write_job(home: Path, **fields) -> Path:
    job_path = home / "job.json"
    payload = {
        "kind": "turn",
        "session_dir": str(home / "traces" / "claude" / "s1"),
        "session_id": "s1",
        "platform": "claude",
        "cwd": "/proj",
        "turn": _turn(),
        **fields,
    }
    job_path.write_text(json.dumps(payload))
    return job_path


class TestMainArgHandling:
    def test_no_args_does_nothing(self):
        otel_worker.main([])  # must not raise

    def test_missing_job_file_does_nothing(self, tmp_path: Path):
        otel_worker.main([str(tmp_path / "missing.json")])  # must not raise

    def test_malformed_json_is_ignored(self, tmp_path: Path):
        job_path = tmp_path / "job.json"
        job_path.write_text("not json")
        otel_worker.main([str(job_path)])  # must not raise

    def test_job_file_deleted_after_processing(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        # `enabled` means _export_turn_inner reaches a real logfire.configure()
        # call; stub it so this test doesn't make a real network request against
        # a fake token (that also spawns a background thread whose eventual
        # warning would leak into whatever test happens to run when it fires).
        import logfire

        monkeypatch.setattr(logfire, "configure", lambda **kwargs: object())
        job_path = _write_job(home)
        otel_worker.main([str(job_path)])
        assert not job_path.exists()

    def test_job_file_deleted_even_on_malformed_content(self, tmp_path: Path):
        job_path = tmp_path / "job.json"
        job_path.write_text("not json")
        otel_worker.main([str(job_path)])
        assert not job_path.exists()

    def test_unknown_kind_does_nothing(self, home: Path, enabled: None):
        job_path = _write_job(home, kind="something_else")
        otel_worker.main([str(job_path)])  # must not raise

    def test_malformed_spans_job_is_ignored_and_deleted(self, home: Path, enabled: None):
        job_path = _write_job(home, kind="spans", turn=None)
        otel_worker.main([str(job_path)])  # missing batch fields must not raise
        assert not job_path.exists()


class TestSpanBatchDispatch:
    def test_spans_job_is_routed_with_deserialized_envelope(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        calls = []
        spans = [
            {
                "name": "chat claude-sonnet-5",
                "span_id": "22",
                "parent_span_id": "11",
                "start_ts": "2026-01-01T00:00:01.000Z",
                "end_ts": "2026-01-01T00:00:02.000Z",
                "attributes": {},
            }
        ]
        monkeypatch.setattr(
            otel_export,
            "_export_spans_batch",
            lambda **kwargs: calls.append(kwargs),
        )
        job_path = _write_job(
            home,
            kind="spans",
            trace_id="340282366920938463463374607431768211455",
            spans=spans,
        )

        otel_worker.main([str(job_path)])

        assert calls == [
            {
                "config": Config.load(),
                "session_dir_": home / "traces" / "claude" / "s1",
                "session_id": "s1",
                "platform": "claude",
                "cwd": "/proj",
                "trace_id": "340282366920938463463374607431768211455",
                "spans": spans,
            }
        ]
        assert not job_path.exists()

    def test_unknown_kind_calls_neither_export_handler(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        calls = []
        monkeypatch.setattr(
            otel_export, "_export_turn_inner", lambda **kwargs: calls.append("turn")
        )
        monkeypatch.setattr(
            otel_export, "_export_spans_batch", lambda **kwargs: calls.append("spans")
        )
        job_path = _write_job(home, kind="future-kind")

        otel_worker.main([str(job_path)])

        assert calls == []
        assert not job_path.exists()


class TestMainExportsThroughToLogfire:
    def test_reads_job_and_exports_the_turn(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        import logfire

        exporter = TestExporter()
        instance = logfire.configure(
            send_to_logfire=False,
            console=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
        )
        monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)

        job_path = _write_job(home)
        otel_worker.main([str(job_path)])

        spans = exporter.exported_spans_as_dict()
        # root (session) span plus the turn span.
        assert len(spans) == 2
        turn_span = spans[-1]
        assert turn_span["name"] == "invoke_agent"
        assert turn_span["attributes"]["gen_ai.conversation.id"] == "s1"

    def test_disabled_config_never_reaches_logfire(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # No `enabled` fixture here: config on disk has logfire off, so
        # _get_instance's own guard must short-circuit before any real
        # configure() call.
        import logfire

        calls = []
        monkeypatch.setattr(logfire, "configure", lambda **kwargs: calls.append(1) or object())

        job_path = _write_job(home)
        otel_worker.main([str(job_path)])

        assert calls == []


class TestWorkerFailureLogging:
    """`otel_worker` must never raise, by design -- see the module docstring.
    But swallowing a failure with zero record of it is exactly what made
    subagent A's missing tool span undiagnosable: a dropped export and a
    never-attempted one look identical. These confirm a failure inside the
    detached worker leaves a breadcrumb instead of vanishing silently.
    """

    def test_turn_export_failure_is_logged_and_does_not_raise(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        def _boom(**kwargs):
            raise RuntimeError("logfire flush failed")

        monkeypatch.setattr(otel_export, "_export_turn_inner", _boom)
        job_path = _write_job(home)

        otel_worker.main([str(job_path)])  # must not raise

        entries = _error_log_entries(home)
        matches = [e for e in entries if e["phase"] == "otel_worker_export_failed"]
        assert len(matches) == 1
        assert matches[0]["session_id"] == "s1"
        assert matches[0]["platform"] == "claude"
        assert matches[0]["error_class"] == "RuntimeError"
        assert "kind=turn" in matches[0]["message"]
        assert "logfire flush failed" in matches[0]["traceback"]

    def test_spans_export_failure_is_logged_and_does_not_raise(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        def _boom(**kwargs):
            raise ConnectionError("network unreachable")

        monkeypatch.setattr(otel_export, "_export_spans_batch", _boom)
        job_path = _write_job(
            home,
            kind="spans",
            trace_id="1",
            spans=[],
        )

        otel_worker.main([str(job_path)])  # must not raise

        entries = _error_log_entries(home)
        matches = [e for e in entries if e["phase"] == "otel_worker_export_failed"]
        assert len(matches) == 1
        assert matches[0]["error_class"] == "ConnectionError"
        assert "kind=spans" in matches[0]["message"]

    def test_subagent_turn_export_failure_is_logged_and_does_not_raise(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        def _boom(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(otel_export, "_export_subagent_turn_inner", _boom)
        job_path = _write_job(
            home,
            kind="subagent_turn",
            trace_id="1",
            parent_span_id="2",
        )

        otel_worker.main([str(job_path)])  # must not raise

        entries = _error_log_entries(home)
        matches = [e for e in entries if e["phase"] == "otel_worker_export_failed"]
        assert len(matches) == 1
        assert "kind=subagent_turn" in matches[0]["message"]

    def test_successful_export_logs_nothing(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(otel_export, "_export_turn_inner", lambda **kwargs: None)
        job_path = _write_job(home)

        otel_worker.main([str(job_path)])

        entries = _error_log_entries(home)
        assert [e for e in entries if e["phase"] == "otel_worker_export_failed"] == []

    def test_malformed_job_file_is_logged(self, home: Path):
        job_path = home / "job.json"
        job_path.write_text("not json")

        otel_worker.main([str(job_path)])  # must not raise

        entries = _error_log_entries(home)
        matches = [e for e in entries if e["phase"] == "otel_worker_export_failed"]
        assert len(matches) == 1
        assert "kind=job_read" in matches[0]["message"]


class TestDuplicateChildDeliveryFirstWins:
    """Script 6: two workers pick up the same completed Cursor subagent stop.
    Raw duplicate capture upstream is allowed, but only one
    `subagent:<turn_id>` worker claim may ever reach "sent", and the span tree
    is emitted once with deterministic ids -- no assertion depends on which
    aligned thread wins.
    """

    def test_duplicate_child_delivery_is_first_wins_at_worker(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        import threading

        import logfire

        from thirdeye.paths import session_dir
        from thirdeye.platforms.cursor import hook
        from thirdeye.platforms.cursor.subagents import cursor_subagent_generation_id
        from thirdeye.reader import SessionReader
        from thirdeye.span_ids import (
            chat_span_id,
            tool_span_id,
            turn_span_id,
        )
        from thirdeye.store import Store

        otel_export._state["id_generator"] = None
        exporter = TestExporter()
        instance = logfire.configure(
            send_to_logfire=False,
            console=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
            advanced=logfire.AdvancedOptions(id_generator=otel_export._id_generator()),
        )
        monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)

        sid = "s1"
        store = Store(Config.load())
        store.append_event(
            session_id=sid,
            platform="cursor",
            cwd="/proj",
            t="user_message",
            data={"generation_id": "parent-gen", "prompt": "dispatch"},
        )
        store.append_event(
            session_id=sid,
            platform="cursor",
            cwd="/proj",
            t="tool_call",
            data={
                "generation_id": "parent-gen",
                "tool_name": "Task",
                "tool_call_id": "call-A",
            },
        )
        start_seq = store.append_event(
            session_id=sid,
            platform="cursor",
            cwd="/proj",
            t="subagent_start",
            data={
                "generation_id": "parent-gen",
                "subagent_id": "child-A",
                "tool_call_id": "call-A",
                "task": "do the thing",
            },
        )
        child_generation = cursor_subagent_generation_id("call-A")
        store.append_event(
            session_id=sid,
            platform="cursor",
            cwd="/proj",
            t="tool_call",
            data={
                "generation_id": child_generation,
                "tool_name": "search_web",
                "tool_use_id": "child-tool",
            },
        )
        store.append_event(
            session_id=sid,
            platform="cursor",
            cwd="/proj",
            t="tool_result",
            data={
                "generation_id": child_generation,
                "tool_name": "search_web",
                "tool_use_id": "child-tool",
                "tool_output": "done",
            },
        )

        jobs: list[Path] = []
        monkeypatch.setattr(otel_export, "_spawn", jobs.append)
        stop_payload = {
            "conversation_id": sid,
            "cwd": "/proj",
            "subagent_id": "child-A",
            "status": "completed",
        }

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def deliver() -> None:
            try:
                barrier.wait(timeout=5)
                hook._subagent_stop(dict(stop_payload))
            except BaseException as exc:  # noqa: BLE001 -- surfaced via assert below
                errors.append(exc)

        threads = [threading.Thread(target=deliver) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert errors == []

        sd = session_dir(home, "cursor", sid)
        captured_stops = [
            event for event in SessionReader(sd).iter_events() if event["t"] == "subagent_message"
        ]
        assert len(captured_stops) == 2
        assert len({event["seq"] for event in captured_stops}) == 2
        assert all(event["data"]["subagent_id"] == "child-A" for event in captured_stops)
        loaded = [json.loads(path.read_text()) for path in jobs]
        task_jobs = [job for job in loaded if job["kind"] == "spans"]
        subagent_jobs = [job for job in loaded if job["kind"] == "subagent_turn"]
        assert len(task_jobs) == len(subagent_jobs) == 1
        [task_span] = task_jobs[0]["spans"]
        assert int(task_span["span_id"]) == tool_span_id("cursor", sid, "call-A")

        job = subagent_jobs[0]
        assert job["turn"]["turn_id"] == str(start_seq)
        assert job["turn"]["turn_span_id"] == str(turn_span_id("cursor", sid, start_seq))
        assert job["turn"]["llm_calls"][0]["call_id"] == child_generation
        assert [tool["tool_call_id"] for tool in job["turn"]["llm_calls"][0]["tool_calls"]] == [
            "child-tool"
        ]
        assert int(job["parent_span_id"]) == tool_span_id("cursor", sid, "call-A")

        kwargs: dict[str, Any] = dict(
            config=Config.load(),
            session_dir_=sd,
            session_id=sid,
            platform="cursor",
            cwd="/proj",
            trace_id=job["trace_id"],
            parent_span_id=job["parent_span_id"],
            turn=job["turn"],
        )
        barrier = threading.Barrier(2)
        errors = []

        def deliver_worker() -> None:
            try:
                barrier.wait(timeout=5)
                otel_export._export_subagent_turn_inner(**kwargs)
            except BaseException as exc:  # noqa: BLE001 -- surfaced via assert below
                errors.append(exc)

        worker_threads = [threading.Thread(target=deliver_worker) for _ in range(2)]
        for thread in worker_threads:
            thread.start()
        for thread in worker_threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert errors == []

        claim = otel_export._turn_claim_path(sd, f"subagent:{start_seq}")
        assert claim.read_text() == "sent"

        spans = exporter.exported_spans_as_dict()
        turn_spans = [s for s in spans if s["name"] == "invoke_agent"]
        chat_spans = [s for s in spans if s["name"] == "chat" or s["name"].startswith("chat ")]
        tool_spans = [s for s in spans if s["name"].startswith("tool:")]
        assert len(turn_spans) == 1
        assert len(chat_spans) == 1
        assert len(tool_spans) == 1

        assert turn_spans[0]["context"]["span_id"] == turn_span_id("cursor", sid, start_seq)
        assert chat_spans[0]["context"]["span_id"] == chat_span_id("cursor", sid, child_generation)
        assert tool_spans[0]["context"]["span_id"] == tool_span_id("cursor", sid, "child-tool")
        assert turn_spans[0]["parent"]["span_id"] == tool_span_id("cursor", sid, "call-A")
        assert chat_spans[0]["parent"]["span_id"] == turn_spans[0]["context"]["span_id"]
        assert tool_spans[0]["parent"]["span_id"] == chat_spans[0]["context"]["span_id"]


class TestInteractionSpanWorkerRoundTrip:
    """Interaction spans cross the detached worker boundary as JSON job files.
    Nested payloads must survive serialization and decode back into structured
    Logfire attributes on the far side.
    """

    def test_spans_job_round_trips_nested_interaction_payload(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        import logfire

        otel_export._state["id_generator"] = None
        exporter = TestExporter()
        instance = logfire.configure(
            send_to_logfire=False,
            console=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
            advanced=logfire.AdvancedOptions(id_generator=otel_export._id_generator()),
        )
        monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)

        session_id = "cursor-worker-session"
        trace_id = trace_id_for_session("cursor", session_id)
        span_id = interaction_span_id("cursor", session_id, "assistant-1")
        parent_span_id = turn_span_id("cursor", session_id, 1)
        nested_payload = {"text": "done", "parts": [{"type": "text", "content": "done"}]}
        spans = [
            {
                "name": "interaction: assistant_message",
                "span_id": str(span_id),
                "parent_span_id": str(parent_span_id),
                "start_ts": "2026-01-01T00:00:01.000Z",
                "end_ts": "2026-01-01T00:00:02.000Z",
                "turn_seq": 1,
                "turn_span_id": str(parent_span_id),
                "attributes": {
                    "thirdeye.interaction.kind": "assistant_message",
                    "thirdeye.interaction.payload": nested_payload,
                },
            }
        ]
        job_path = home / "interaction-spans-job.json"
        job_path.write_text(
            json.dumps(
                {
                    "kind": "spans",
                    "session_dir": str(home / "traces" / "cursor" / session_id),
                    "session_id": session_id,
                    "platform": "cursor",
                    "cwd": "/proj",
                    "trace_id": str(trace_id),
                    "spans": spans,
                }
            )
        )

        otel_worker.main([str(job_path)])

        assert not job_path.exists()
        [span] = exporter.exported_spans_as_dict()
        assert span["name"] == "interaction: assistant_message"
        assert span["context"]["span_id"] == span_id
        assert span["parent"]["span_id"] == parent_span_id
        assert span["attributes"]["gen_ai.conversation.id"] == session_id
        assert span["attributes"]["thirdeye.turn.id"] == "1"
        assert json.loads(span["attributes"]["thirdeye.interaction.payload"]) == nested_payload

    def test_turn_job_round_trips_interaction_recovery_records(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        import logfire

        otel_export._state["id_generator"] = None
        exporter = TestExporter()
        instance = logfire.configure(
            send_to_logfire=False,
            console=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
            advanced=logfire.AdvancedOptions(id_generator=otel_export._id_generator()),
        )
        monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)

        session_id = "cursor-turn-recovery"
        turn_id = turn_span_id("cursor", session_id, 1)
        user_id = interaction_span_id("cursor", session_id, "user-1")
        payload = {"prompt": "ship it", "flags": {"urgent": True}}
        turn = _turn(
            turn_span_id=str(turn_id),
            interactions=[
                {
                    "interaction_id": "user-1",
                    "kind": "user_message",
                    "span_id": str(user_id),
                    "parent_span_id": str(turn_id),
                    "start_ts": "2026-01-01T00:00:01.000Z",
                    "end_ts": "2026-01-01T00:00:01.000Z",
                    "attributes": {
                        "thirdeye.interaction.kind": "user_message",
                        "thirdeye.interaction.payload": payload,
                    },
                }
            ],
        )
        job_path = home / "interaction-turn-job.json"
        job_path.write_text(
            json.dumps(
                {
                    "kind": "turn",
                    "session_dir": str(home / "traces" / "cursor" / session_id),
                    "session_id": session_id,
                    "platform": "cursor",
                    "cwd": "/proj",
                    "turn": turn,
                }
            )
        )

        otel_worker.main([str(job_path)])

        assert not job_path.exists()
        spans = exporter.exported_spans_as_dict()
        user_span = next(span for span in spans if span["context"]["span_id"] == user_id)
        turn_span = next(span for span in spans if span["name"] == "invoke_agent")
        assert user_span["name"] == "interaction: user_message"
        assert user_span["parent"]["span_id"] == turn_span["context"]["span_id"]
        assert json.loads(user_span["attributes"]["thirdeye.interaction.payload"]) == payload
