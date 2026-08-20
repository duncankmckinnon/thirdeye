from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye import otel_export, otel_worker
from thirdeye.config import Config, LogfireSettings

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
    Config.load().write_logfire_settings(
        LogfireSettings(enabled=True, token="fake-token", project="p")
    )


def _write_job(home: Path, **fields) -> Path:
    job_path = home / "job.json"
    payload = {
        "session_dir": str(home / "traces" / "claude" / "s1"),
        "session_id": "s1",
        "platform": "claude",
        "cwd": "/proj",
        "t": "user_message",
        "seq": 0,
        "ts": "2026-01-01T00:00:00.000Z",
        "data": {"text": "hi"},
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
        # `enabled` means _export_event_inner reaches a real logfire.configure()
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


class TestMainExportsThroughToLogfire:
    def test_reads_job_and_exports_a_span(
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
        assert len(spans) == 1
        assert spans[0]["name"] == "user_message"
        assert spans[0]["attributes"]["gen_ai.conversation.id"] == "s1"

    def test_disabled_config_never_reaches_logfire(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # No `enabled` fixture here: config on disk has logfire off, so
        # _get_instance's own guard (not export_event's — the worker never
        # calls export_event) must short-circuit before any real configure().
        import logfire

        calls = []
        monkeypatch.setattr(logfire, "configure", lambda **kwargs: calls.append(1) or object())

        job_path = _write_job(home)
        otel_worker.main([str(job_path)])

        assert calls == []

    def test_llm_calls_job_dispatches_to_the_batch_exporter(
        self, home: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
    ):
        """A `"kind": "llm_calls"` job must route to `_export_llm_calls_inner`,
        not the ordinary single-event path.
        """
        import logfire

        exporter = TestExporter()
        instance = logfire.configure(
            send_to_logfire=False,
            console=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
        )
        monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)

        job_path = home / "job.json"
        payload = {
            "kind": "llm_calls",
            "session_dir": str(home / "traces" / "claude" / "s1"),
            "session_id": "s1",
            "platform": "claude",
            "cwd": "/proj",
            "calls": [
                {
                    "seq": 5,
                    "ts": "2026-01-01T00:00:00.000Z",
                    "call_id": "call_1",
                    "data": {
                        "gen_ai.usage.input_tokens": 10,
                        "gen_ai.usage.output_tokens": 5,
                        "gen_ai.output.messages": [
                            {"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}
                        ],
                    },
                }
            ],
        }
        job_path.write_text(json.dumps(payload))
        otel_worker.main([str(job_path)])

        spans = exporter.exported_spans_as_dict()
        assert len(spans) == 1
        assert spans[0]["attributes"]["gen_ai.usage.input_tokens"] == 10
        assert "gen_ai.output.messages" in spans[0]["attributes"]

    @pytest.mark.parametrize("exported,expected_exists", [(True, True), (False, False)])
    def test_codex_turn_job_commits_or_releases_state(
        self,
        home: Path,
        enabled: None,
        monkeypatch: pytest.MonkeyPatch,
        exported: bool,
        expected_exists: bool,
    ):
        state_path = home / "turn-state"
        state_path.write_text("pending")
        monkeypatch.setattr(otel_export, "_export_codex_turn_inner", lambda **kwargs: exported)
        job_path = home / "job.json"
        job_path.write_text(
            json.dumps(
                {
                    "kind": "codex_turn",
                    "session_dir": str(home / "traces" / "codex" / "s1"),
                    "session_id": "s1",
                    "cwd": "/proj",
                    "seq": 1,
                    "turn": {},
                    "state_path": str(state_path),
                }
            )
        )
        otel_worker.main([str(job_path)])
        assert state_path.exists() is expected_exists
        if expected_exists:
            assert state_path.read_text() == "sent"
