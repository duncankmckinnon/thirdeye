from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from thirdeye import otel_export, otel_worker
from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import otel_jobs_dir
from thirdeye.span_ids import chat_span_id
from thirdeye.usage.types import UsageRow

pytest.importorskip("logfire")

from logfire.testing import TestExporter  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402

_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "platforms"
    / "copilot"
    / "fixtures"
    / "reconciliation-cases"
)
_TRANSPORT = json.loads((_FIXTURES / "transport.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _reset_state():
    otel_export._state["attempted"] = False
    otel_export._state["instance"] = None
    otel_export._state["id_generator"] = None
    yield
    otel_export._state["attempted"] = False
    otel_export._state["instance"] = None
    otel_export._state["id_generator"] = None


@pytest.fixture
def exporter():
    return TestExporter()


@pytest.fixture
def wired_instance(exporter, monkeypatch: pytest.MonkeyPatch):
    import logfire

    instance = logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
        advanced=logfire.AdvancedOptions(id_generator=otel_export._id_generator()),
    )
    monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)
    return instance


@pytest.fixture
def enabled_config(tmp_path: Path) -> Config:
    return Config(
        root=tmp_path,
        logfire=LogfireSettings(enabled=True, token="fake-token"),
    )


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


def _llm_call(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        call_id="call_1",
        provider="anthropic",
        model="claude-sonnet-5",
        start_ts="2026-01-01T00:00:01.000Z",
        end_ts="2026-01-01T00:00:02.000Z",
        input_messages=[{"role": "user", "parts": [{"type": "text", "content": "hi"}]}],
        output_messages=[{"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}],
        usage={"input_tokens": 100, "output_tokens": 50},
        tool_calls=[],
    )
    defaults.update(overrides)
    return defaults


def _usage_row(**overrides: Any) -> dict[str, Any]:
    fields = dict(
        session_id="s1",
        seq=0,
        call_id="usage-1",
        ts="2026-01-01T00:00:01.500Z",
        platform="copilot",
        provider_name="unknown",
        response_model="gpt-test",
        input_tokens=6452,
        output_tokens=107,
        cache_creation_input_tokens=6449,
    )
    fields.update(overrides)
    return UsageRow(**fields).to_dict()


def _accounting_call(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        accounting_id="acct-1",
        usage=_usage_row(),
        attribution_status="matched",
        agent_id=None,
        call_id="call_1",
        attributes={"accounting.destination": "chat-span"},
    )
    defaults.update(overrides)
    return defaults


def _error_log_entries(home: Path) -> list[dict]:
    log = home / "logs" / "usage-errors.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def _accounting_job_path(root: Path, job_id: str) -> Path:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return otel_jobs_dir(root) / f"accounting-{digest}.json"


def _write_queued_accounting_job(root: Path, **fields: Any) -> Path:
    job_id = str(fields["job_id"])
    job_path = _accounting_job_path(root, job_id)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": "queued", "attempt": 0, **fields}
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    return job_path


class TestAccountingAttributes:
    def test_projects_usage_and_metadata(self):
        accounting = _accounting_call(
            accounting_id="acct-42",
            attribution_status="pending",
            agent_id="agent-9",
            attributes={"copilot.logical_call_id": "logical-1"},
        )
        attrs = otel_export._accounting_attributes(accounting)

        assert attrs["thirdeye.accounting.id"] == "acct-42"
        assert attrs["thirdeye.accounting.attribution_status"] == "pending"
        assert attrs["thirdeye.accounting.agent_id"] == "agent-9"
        assert attrs["gen_ai.usage.input_tokens"] == 6452
        assert attrs["copilot.logical_call_id"] == "logical-1"
        assert json.loads(attrs["thirdeye.accounting.usage"])["call_id"] == "usage-1"

    def test_labels_copilot_native_billing_distinct_from_usd(self):
        accounting = _accounting_call(
            attributes={"copilot.billing.nano_aiu": 12, "operation.cost": 0.05},
        )
        attrs = otel_export._accounting_attributes(accounting)

        assert attrs["thirdeye.accounting.billing.kind"] == "copilot-native-unit"
        assert attrs["copilot.billing.nano_aiu"] == 12

    def test_usd_only_cost_does_not_set_native_billing_kind(self):
        accounting = _accounting_call(attributes={"operation.cost": 0.05})
        attrs = otel_export._accounting_attributes(accounting)

        assert attrs["operation.cost"] == 0.05
        assert "thirdeye.accounting.billing.kind" not in attrs


class TestOrdinaryTurnCompatibility:
    def test_turn_without_accounting_calls_is_unchanged(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        call = _llm_call()
        turn = _turn(llm_calls=[call])
        session_dir = tmp_path / "traces" / "claude" / "s1"

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=session_dir,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )

        spans = exporter.exported_spans_as_dict()
        chat_spans = [span for span in spans if span["name"].startswith("chat")]
        accounting_spans = [span for span in spans if span["name"] == "accounting"]

        assert len(chat_spans) == 1
        assert accounting_spans == []
        assert chat_spans[0]["attributes"]["gen_ai.usage.input_tokens"] == 100
        assert "thirdeye.accounting.id" not in chat_spans[0]["attributes"]


class TestNestedTurnAccounting:
    def test_matched_accounting_merges_onto_chat_span_only(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        session_id = "copilot-session"
        platform = "copilot"
        call_id = "copilot:call:matched"
        call = _llm_call(call_id=call_id, usage={})
        accounting = _accounting_call(
            call_id=call_id,
            usage=_usage_row(
                call_id="usage-matched",
                input_tokens=6452,
                output_tokens=107,
            ),
        )
        turn = _turn(llm_calls=[call], accounting_calls=[accounting])

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / platform / session_id,
            session_id=session_id,
            platform=platform,
            cwd="/proj",
            turn=turn,
        )

        spans = exporter.exported_spans_as_dict()
        chat_span = next(span for span in spans if span["name"].startswith("chat"))
        accounting_spans = [span for span in spans if span["name"] == "accounting"]

        assert accounting_spans == []
        assert chat_span["context"]["span_id"] == chat_span_id(platform, session_id, call_id)
        assert chat_span["attributes"]["gen_ai.usage.input_tokens"] == 6452
        assert chat_span["attributes"]["thirdeye.accounting.id"] == "acct-1"
        schema = json.loads(chat_span["attributes"]["logfire.json_schema"])
        assert "gen_ai.input.messages" in schema["properties"]
        assert "gen_ai.output.messages" in schema["properties"]

    def test_matched_copilot_billing_labels_chat_span(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        session_id = "copilot-session"
        platform = "copilot"
        call_id = "copilot:call:billed"
        call = _llm_call(call_id=call_id, usage={})
        accounting = _accounting_call(
            call_id=call_id,
            usage=_usage_row(call_id="usage-billed", input_tokens=6452, output_tokens=107),
            attributes={"copilot.billing.nano_aiu": 12},
        )
        turn = _turn(llm_calls=[call], accounting_calls=[accounting])

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / platform / session_id,
            session_id=session_id,
            platform=platform,
            cwd="/proj",
            turn=turn,
        )

        chat_span = next(
            span for span in exporter.exported_spans_as_dict() if span["name"].startswith("chat")
        )
        assert chat_span["attributes"]["thirdeye.accounting.billing.kind"] == "copilot-native-unit"
        assert chat_span["attributes"]["copilot.billing.nano_aiu"] == 12
        assert chat_span["attributes"]["gen_ai.usage.input_tokens"] == 6452

    def test_unknown_call_id_emits_turn_owned_span_not_chat(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        session_id = "copilot-session"
        platform = "copilot"
        turn_id = "turn-unknown-call"
        call = _llm_call(call_id="known-call")
        accounting = _accounting_call(
            accounting_id="acct-unknown-call",
            call_id="not-in-llm-calls",
            attribution_status="pending",
            usage=_usage_row(call_id="usage-unknown", input_tokens=10, output_tokens=2),
            attributes={"accounting.destination": "turn-accounting-span"},
        )
        turn = _turn(turn_id=turn_id, llm_calls=[call], accounting_calls=[accounting])

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / platform / session_id,
            session_id=session_id,
            platform=platform,
            cwd="/proj",
            turn=turn,
        )

        spans = exporter.exported_spans_as_dict()
        chat_spans = [span for span in spans if span["name"].startswith("chat")]
        accounting_spans = [span for span in spans if span["name"] == "accounting"]
        turn_span = next(span for span in spans if span["name"] == "invoke_agent")

        assert len(chat_spans) == 1
        assert chat_spans[0]["attributes"].get("thirdeye.accounting.id") is None
        assert len(accounting_spans) == 1
        assert accounting_spans[0]["parent"]["span_id"] == turn_span["context"]["span_id"]
        assert accounting_spans[0]["attributes"]["thirdeye.accounting.id"] == "acct-unknown-call"

    def test_unmatched_accounting_emits_turn_owned_span_with_deterministic_id(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        session_id = "copilot-session"
        platform = "copilot"
        turn_id = "turn-unmatched"
        call = _llm_call(call_id="known-call", usage={})
        unmatched = _accounting_call(
            accounting_id="acct-unmatched",
            call_id=None,
            attribution_status="pending",
            usage=_usage_row(call_id="usage-unmatched", input_tokens=6587, output_tokens=5),
            attributes={"accounting.destination": "turn-accounting-span"},
        )
        turn = _turn(turn_id=turn_id, llm_calls=[call], accounting_calls=[unmatched])

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / platform / session_id,
            session_id=session_id,
            platform=platform,
            cwd="/proj",
            turn=turn,
        )

        spans = exporter.exported_spans_as_dict()
        chat_span = next(span for span in spans if span["name"].startswith("chat"))
        accounting_span = next(span for span in spans if span["name"] == "accounting")
        turn_span = next(span for span in spans if span["name"] == "invoke_agent")

        assert chat_span["attributes"].get("thirdeye.accounting.id") is None
        assert accounting_span["parent"]["span_id"] == turn_span["context"]["span_id"]
        assert accounting_span["context"]["span_id"] == otel_export._accounting_span_id(
            platform, session_id, "acct-unmatched", turn_id
        )
        assert accounting_span["attributes"]["gen_ai.usage.input_tokens"] == 6587
        assert accounting_span["attributes"]["thirdeye.accounting.attribution_status"] == "pending"

    def test_fixture_turn_with_matched_and_unmatched_accounting(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        turn = dict(_TRANSPORT["turn_span_with_accounting_calls"])
        session_id = turn["accounting_calls"][0]["usage"]["session_id"]
        platform = "copilot"

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / platform / session_id,
            session_id=session_id,
            platform=platform,
            cwd="/proj",
            turn=turn,
        )

        spans = exporter.exported_spans_as_dict()
        chat_spans = [span for span in spans if span["name"].startswith("chat")]
        accounting_spans = [span for span in spans if span["name"] == "accounting"]

        assert len(chat_spans) == 1
        assert len(accounting_spans) == 1
        matched_id = turn["accounting_calls"][0]["accounting_id"]
        unmatched_id = turn["accounting_calls"][1]["accounting_id"]
        assert chat_spans[0]["attributes"]["thirdeye.accounting.id"] == matched_id
        assert accounting_spans[0]["attributes"]["thirdeye.accounting.id"] == unmatched_id


class TestSessionAccountingExport:
    def test_queues_deterministic_job_without_duplicates(
        self, enabled_config: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        spawned: list[Path] = []
        monkeypatch.setattr(otel_export, "_spawn", spawned.append)
        session_dir = tmp_path / "traces" / "copilot" / "s1"
        accounting = {
            "accounting_id": "acct-session",
            "usage": _usage_row(session_id="s1"),
            "attribution_status": "pending",
            "agent_id": None,
            "attributes": {"accounting.destination": "session-accounting-span"},
        }

        first = otel_export.export_session_accounting(
            enabled_config, session_dir, "s1", "copilot", "/proj", accounting
        )
        second = otel_export.export_session_accounting(
            enabled_config, session_dir, "s1", "copilot", "/proj", accounting
        )

        assert first is True
        assert second is True
        jobs = list(otel_jobs_dir(enabled_config.root).glob("accounting-*.json"))
        assert len(jobs) == 1
        payload = json.loads(jobs[0].read_text(encoding="utf-8"))
        assert payload["kind"] == "session_accounting"
        assert payload["destination"] == "session-accounting-span"
        assert payload["state"] == "queued"
        assert payload["job_id"] == "accounting:s1:acct-session"
        assert payload["span_id"] == payload["job_id"]
        assert len(spawned) == 2

    def test_disabled_config_does_not_queue(self, tmp_path: Path):
        config = Config(root=tmp_path, logfire=LogfireSettings(enabled=False, token=""))
        session_dir = tmp_path / "traces" / "copilot" / "s1"
        accounting = {
            "accounting_id": "acct-session",
            "usage": _usage_row(),
            "attribution_status": "pending",
        }
        assert (
            otel_export.export_session_accounting(
                config, session_dir, "s1", "copilot", "/proj", accounting
            )
            is False
        )
        assert list(otel_jobs_dir(config.root).glob("accounting-*.json")) == []

    def test_worker_round_trips_session_accounting_job(
        self,
        enabled_config: Config,
        wired_instance,
        exporter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        session_id = "s1"
        platform = "copilot"
        session_dir = tmp_path / "traces" / platform / session_id
        accounting_id = "copilot-usage-7f3a"
        job_id = f"accounting:{session_id}:{accounting_id}"
        digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        job_path = otel_jobs_dir(enabled_config.root) / f"accounting-{digest}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "kind": "session_accounting",
                    "session_dir": str(session_dir),
                    "session_id": session_id,
                    "platform": platform,
                    "cwd": "/proj",
                    "accounting_id": accounting_id,
                    "destination": "session-accounting-span",
                    "usage": _usage_row(session_id=session_id),
                    "attribution_status": "pending",
                    "agent_id": None,
                    "attributes": {},
                    "span_id": job_id,
                    "state": "queued",
                    "attempt": 0,
                }
            ),
            encoding="utf-8",
        )

        otel_worker.main([str(job_path)])

        assert not job_path.exists()
        spans = exporter.exported_spans_as_dict()
        accounting_span = next(span for span in spans if span["name"] == "accounting")
        session_span = next(span for span in spans if span["name"] == "session")

        assert accounting_span["parent"]["span_id"] == session_span["context"]["span_id"]
        assert accounting_span["attributes"]["thirdeye.accounting.id"] == accounting_id
        assert accounting_span["attributes"]["thirdeye.platform"] == platform
        assert accounting_span["attributes"]["thirdeye.cwd"] == "/proj"
        assert accounting_span["context"]["span_id"] == otel_export._accounting_span_id(
            platform, session_id, accounting_id
        )


class TestTurnAccountingExport:
    def test_queues_deterministic_job_without_duplicates(
        self, enabled_config: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        spawned: list[Path] = []
        monkeypatch.setattr(otel_export, "_spawn", spawned.append)
        session_dir = tmp_path / "traces" / "copilot" / "s1"
        accounting = {
            "accounting_id": "acct-turn",
            "usage": _usage_row(session_id="s1"),
            "attribution_status": "pending",
            "agent_id": None,
            "attributes": {"accounting.destination": "turn-accounting-span"},
        }

        first = otel_export.export_turn_accounting(
            enabled_config, session_dir, "s1", "copilot", "/proj", "turn_1", accounting
        )
        second = otel_export.export_turn_accounting(
            enabled_config, session_dir, "s1", "copilot", "/proj", "turn_1", accounting
        )

        assert first is True
        assert second is True
        jobs = list(otel_jobs_dir(enabled_config.root).glob("accounting-*.json"))
        assert len(jobs) == 1
        payload = json.loads(jobs[0].read_text(encoding="utf-8"))
        assert payload["kind"] == "turn_accounting"
        assert payload["destination"] == "turn-accounting-span"
        assert payload["state"] == "queued"
        assert payload["job_id"] == "accounting:s1:turn_1:acct-turn"
        assert payload["span_id"] == payload["job_id"]
        assert payload["turn_id"] == "turn_1"
        assert len(spawned) == 2

    def test_worker_round_trips_turn_accounting_job(
        self,
        enabled_config: Config,
        wired_instance,
        exporter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        session_id = "s1"
        platform = "copilot"
        turn_id = "turn_1"
        session_dir = tmp_path / "traces" / platform / session_id
        accounting_id = "acct-delayed"
        job_id = f"accounting:{session_id}:{turn_id}:{accounting_id}"
        turn_span_id = 4242
        job_path = _write_queued_accounting_job(
            enabled_config.root,
            job_id=job_id,
            kind="turn_accounting",
            session_dir=str(session_dir),
            session_id=session_id,
            platform=platform,
            cwd="/proj",
            turn_id=turn_id,
            turn_span_id=str(turn_span_id),
            accounting_id=accounting_id,
            destination="turn-accounting-span",
            usage=_usage_row(session_id=session_id),
            attribution_status="pending",
            agent_id=None,
            attributes={"accounting.destination": "turn-accounting-span"},
            span_id=job_id,
        )

        otel_worker.main([str(job_path)])

        assert not job_path.exists()
        spans = exporter.exported_spans_as_dict()
        accounting_span = next(span for span in spans if span["name"] == "accounting")
        invented_turns = [span for span in spans if span["name"] == "invoke_agent"]
        invented_chats = [span for span in spans if span["name"].startswith("chat")]

        assert invented_turns == []
        assert invented_chats == []
        assert accounting_span["parent"]["span_id"] == turn_span_id
        assert accounting_span["attributes"]["thirdeye.accounting.id"] == accounting_id
        assert accounting_span["attributes"]["thirdeye.turn.id"] == turn_id
        assert accounting_span["attributes"]["thirdeye.platform"] == platform
        assert accounting_span["context"]["span_id"] == otel_export._accounting_span_id(
            platform, session_id, accounting_id, turn_id
        )


class TestWorkerClaimRecovery:
    def test_stale_claim_is_recovered_and_export_retried(
        self,
        enabled_config: Config,
        wired_instance,
        exporter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        session_id = "s1"
        platform = "copilot"
        session_dir = tmp_path / "traces" / platform / session_id
        job_id = "accounting:s1:acct-retry"
        digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        job_path = otel_jobs_dir(enabled_config.root) / f"accounting-{digest}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "kind": "session_accounting",
                    "session_dir": str(session_dir),
                    "session_id": session_id,
                    "platform": platform,
                    "cwd": "/proj",
                    "accounting_id": "acct-retry",
                    "destination": "session-accounting-span",
                    "usage": _usage_row(session_id=session_id),
                    "attribution_status": "pending",
                    "state": "claimed",
                    "attempt": 0,
                }
            ),
            encoding="utf-8",
        )
        claim_path = otel_worker._job_claim_path(job_path)
        claim_path.write_text("99999", encoding="utf-8")
        stale_at = time.time() - otel_worker._JOB_CLAIM_STALE_S - 5
        import os

        os.utime(claim_path, (stale_at, stale_at))

        otel_worker.main([str(job_path)])

        assert not job_path.exists()
        assert not claim_path.exists()
        assert any(span["name"] == "accounting" for span in exporter.exported_spans_as_dict())

    def test_export_failure_retains_job_and_increments_attempt(
        self, enabled_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        session_id = "s1"
        platform = "copilot"
        session_dir = tmp_path / "traces" / platform / session_id
        job_id = "accounting:s1:acct-fail"
        digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        job_path = otel_jobs_dir(enabled_config.root) / f"accounting-{digest}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "kind": "session_accounting",
                    "session_dir": str(session_dir),
                    "session_id": session_id,
                    "platform": platform,
                    "cwd": "/proj",
                    "accounting_id": "acct-fail",
                    "destination": "session-accounting-span",
                    "usage": _usage_row(session_id=session_id),
                    "attribution_status": "pending",
                    "state": "queued",
                    "attempt": 0,
                }
            ),
            encoding="utf-8",
        )

        def _boom(**kwargs):
            raise RuntimeError("flush failed")

        monkeypatch.setattr(otel_export, "_export_session_accounting_inner", _boom)
        otel_worker.main([str(job_path)])

        assert job_path.exists()
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        assert payload["state"] == "queued"
        assert payload["attempt"] == 1
        assert not otel_worker._job_claim_path(job_path).exists()
        entries = _error_log_entries(enabled_config.root)
        assert any("kind=session_accounting" in entry["message"] for entry in entries)

    def test_fresh_claim_blocks_concurrent_worker(
        self, enabled_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        job_path = tmp_path / "accounting-job.json"
        payload = {"state": "queued", "attempt": 0}
        job_path.write_text(json.dumps(payload), encoding="utf-8")
        claim_path = otel_worker._job_claim_path(job_path)
        claim_path.write_text("42", encoding="utf-8")
        monkeypatch.setattr(
            time,
            "time",
            lambda: claim_path.stat().st_mtime + 1,
        )

        claimed = otel_worker._claim_job(job_path, payload)

        assert claimed is None

    def test_success_writes_emitted_before_unlinking(
        self,
        enabled_config: Config,
        wired_instance,
        exporter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        states: list[str] = []
        original = otel_worker._write_job_state

        def _capture(job_path: Path, payload: dict[str, Any]) -> None:
            states.append(str(payload.get("state")))
            original(job_path, payload)

        monkeypatch.setattr(otel_worker, "_write_job_state", _capture)
        job_path = _write_queued_accounting_job(
            enabled_config.root,
            job_id="accounting:s1:acct-emitted",
            kind="session_accounting",
            session_dir=str(tmp_path / "traces" / "copilot" / "s1"),
            session_id="s1",
            platform="copilot",
            cwd="/proj",
            accounting_id="acct-emitted",
            destination="session-accounting-span",
            usage=_usage_row(session_id="s1"),
            attribution_status="pending",
            span_id="accounting:s1:acct-emitted",
        )

        otel_worker.main([str(job_path)])

        assert "emitted" in states
        assert states[-1] == "emitted"
        assert not job_path.exists()
        assert any(span["name"] == "accounting" for span in exporter.exported_spans_as_dict())

    def test_missing_logfire_instance_retains_accounting_job(
        self, enabled_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: None)
        job_path = _write_queued_accounting_job(
            enabled_config.root,
            job_id="accounting:s1:acct-noop",
            kind="session_accounting",
            session_dir=str(tmp_path / "traces" / "copilot" / "s1"),
            session_id="s1",
            platform="copilot",
            cwd="/proj",
            accounting_id="acct-noop",
            destination="session-accounting-span",
            usage=_usage_row(session_id="s1"),
            attribution_status="pending",
            span_id="accounting:s1:acct-noop",
        )

        otel_worker.main([str(job_path)])

        assert job_path.exists()
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        assert payload["state"] == "queued"
        assert payload["attempt"] == 1
        assert not otel_worker._job_claim_path(job_path).exists()

    def test_worker_dispatches_turn_accounting_kind(
        self, enabled_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            otel_export,
            "_export_turn_accounting_inner",
            lambda **kwargs: calls.append(kwargs),
            raising=False,
        )
        job_path = _write_queued_accounting_job(
            enabled_config.root,
            job_id="accounting:s1:turn_1:acct-dispatch",
            kind="turn_accounting",
            session_dir=str(tmp_path / "traces" / "copilot" / "s1"),
            session_id="s1",
            platform="copilot",
            cwd="/proj",
            turn_id="turn_1",
            accounting_id="acct-dispatch",
            destination="turn-accounting-span",
            usage=_usage_row(session_id="s1"),
            attribution_status="pending",
            span_id="accounting:s1:turn_1:acct-dispatch",
        )

        otel_worker.main([str(job_path)])

        assert len(calls) == 1
        assert calls[0]["session_id"] == "s1"
        assert calls[0]["turn_id"] == "turn_1"
        assert calls[0]["accounting"]["accounting_id"] == "acct-dispatch"
        assert not job_path.exists()

    def test_exhausted_retries_mark_job_failed(
        self, enabled_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(Config, "load", lambda: enabled_config)
        job_path = _write_queued_accounting_job(
            enabled_config.root,
            job_id="accounting:s1:acct-exhausted",
            kind="session_accounting",
            session_dir=str(tmp_path / "traces" / "copilot" / "s1"),
            session_id="s1",
            platform="copilot",
            cwd="/proj",
            accounting_id="acct-exhausted",
            destination="session-accounting-span",
            usage=_usage_row(session_id="s1"),
            attribution_status="pending",
            span_id="accounting:s1:acct-exhausted",
            attempt=otel_worker._JOB_MAX_ATTEMPTS - 1,
        )

        def _boom(**kwargs):
            raise RuntimeError("flush failed")

        monkeypatch.setattr(otel_export, "_export_session_accounting_inner", _boom)
        otel_worker.main([str(job_path)])

        assert job_path.exists()
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        assert payload["state"] == "failed"
        assert payload["attempt"] == otel_worker._JOB_MAX_ATTEMPTS

        otel_worker.main([str(job_path)])
        still = json.loads(job_path.read_text(encoding="utf-8"))
        assert still["state"] == "failed"
        assert still["attempt"] == otel_worker._JOB_MAX_ATTEMPTS
