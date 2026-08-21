from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        assert turn_span["name"] == "agent-turn"
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
