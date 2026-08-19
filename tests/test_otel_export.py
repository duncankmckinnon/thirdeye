from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye import otel_export
from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import otel_jobs_dir
from thirdeye.store import Store
from thirdeye.writer import utc_iso_ms

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
def exporter():
    return TestExporter()


@pytest.fixture
def wired_instance(exporter, monkeypatch: pytest.MonkeyPatch):
    """A real Logfire instance wired to an in-memory exporter, network-free."""
    import logfire

    instance = logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
    )
    monkeypatch.setattr(otel_export, "_get_instance", lambda config: instance)
    return instance


@pytest.fixture
def enabled_config(tmp_path: Path) -> Config:
    return Config(
        root=tmp_path,
        logfire=LogfireSettings(enabled=True, token="fake-token", project="p"),
    )


def _export(config: Config, session_dir_: Path, **event) -> None:
    """Call the actual export logic directly, in-process — bypassing the
    async dispatch in `export_event` (which hands off to a detached
    subprocess and so can't be observed by an in-process TestExporter).
    """
    session_dir_.mkdir(parents=True, exist_ok=True)
    event.setdefault("data", None)
    otel_export._export_event_inner(config=config, session_dir_=session_dir_, **event)


class TestFlattenAttrs:
    def test_primitives_pass_through(self):
        assert otel_export._flatten_attrs({"a": 1, "b": "x", "c": True, "d": 1.5}) == {
            "a": 1,
            "b": "x",
            "c": True,
            "d": 1.5,
        }

    def test_homogeneous_list_passes_through(self):
        assert otel_export._flatten_attrs({"tags": ["a", "b"]}) == {"tags": ["a", "b"]}

    def test_nested_dict_is_json_encoded(self):
        out = otel_export._flatten_attrs({"tool_input": {"path": "x.py"}})
        assert out["tool_input"] == '{"path": "x.py"}'

    def test_none_values_dropped(self):
        assert otel_export._flatten_attrs({"a": None, "b": 1}) == {"b": 1}

    def test_non_dict_input_yields_empty(self):
        assert otel_export._flatten_attrs("not a dict") == {}


class TestToolId:
    def test_claude_tool_use_id(self):
        assert otel_export._tool_id({"tool_use_id": "tu_1"}) == "tu_1"

    def test_codex_call_id(self):
        assert otel_export._tool_id({"call_id": "call_1"}) == "call_1"

    def test_missing(self):
        assert otel_export._tool_id({"tool_name": "Bash"}) is None


class TestBackgroundNoiseSuppression:
    """Logfire's token-check runs in a background thread and warns straight to
    stderr on a bad/unreachable token; that thread can outlive our own call
    into logfire, so scoping suppression to a `with` block around configure()
    does not reliably catch it (confirmed by hand: `catch_warnings()` +
    `redirect_stderr` around configure() let the warning through anyway, since
    it fires after that block already exited). _get_instance calls
    _silence_background_noise() to set a permanent, process-global filter
    instead. Exercised directly, not through a real configure() call, so this
    test doesn't make a real network request against a fake token.
    """

    def test_silences_opentelemetry_logger(self):
        import logging

        logging.getLogger("opentelemetry").setLevel(logging.NOTSET)
        otel_export._silence_background_noise()
        assert logging.getLogger("opentelemetry").level > logging.CRITICAL

    def test_adds_a_permanent_warnings_filter_for_logfire_module(self):
        import warnings

        before = list(warnings.filters)
        otel_export._silence_background_noise()
        after = list(warnings.filters)
        assert after != before  # a filter was added and, unlike catch_warnings(), kept

    def test_get_instance_calls_silence_before_configuring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import logfire

        calls = []
        monkeypatch.setattr(otel_export, "_silence_background_noise", lambda: calls.append(1))
        monkeypatch.setattr(logfire, "configure", lambda **kwargs: object())
        config = Config(
            root=tmp_path, logfire=LogfireSettings(enabled=True, token="bad-token", project=None)
        )
        otel_export._get_instance(config)
        assert calls == [1]


class TestExportEventDispatch:
    """`export_event` (called synchronously from inside a hook process) must
    never itself touch the network: it should only ever write a job file and
    spawn a detached, unwaited-for worker process. `subprocess.Popen` is
    mocked so these tests don't actually spawn a Python interpreter per case.
    """

    def test_disabled_spawns_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        spawned = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda *a, **k: spawned.append(a))
        config = Config(root=tmp_path)
        Store(config).append_event(session_id="s1", platform="claude", cwd="/p", t="user_message")
        assert spawned == []

    def test_tool_call_spawns_nothing(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        spawned = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda *a, **k: spawned.append(a))
        Store(enabled_config).append_event(
            session_id="s1", platform="claude", cwd="/p", t="tool_call", data={"tool_name": "Bash"}
        )
        assert spawned == []

    def test_enabled_point_event_spawns_a_detached_worker(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        calls = []

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                calls.append((argv, kwargs))

        monkeypatch.setattr(otel_export.subprocess, "Popen", _FakePopen)
        Store(enabled_config).append_event(
            session_id="s1", platform="claude", cwd="/proj", t="user_message", data={"text": "hi"}
        )
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[0] == otel_export.sys.executable
        assert argv[1:3] == ["-m", "thirdeye.otel_worker"]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is otel_export.subprocess.DEVNULL

    def test_job_file_carries_the_full_event(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        calls = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda argv, **k: calls.append(argv))
        Store(enabled_config).append_event(
            session_id="s1", platform="claude", cwd="/proj", t="user_message", data={"text": "hi"}
        )
        job_path = Path(calls[0][3])
        assert job_path.parent == otel_jobs_dir(tmp_path)
        payload = json.loads(job_path.read_text())
        assert payload["session_id"] == "s1"
        assert payload["platform"] == "claude"
        assert payload["t"] == "user_message"
        assert payload["data"] == {"text": "hi"}

    def test_spawn_failure_is_logged_not_raised(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        def _boom(*a, **k):
            raise OSError("no more processes")

        monkeypatch.setattr(otel_export.subprocess, "Popen", _boom)
        # Must not raise, even though spawning fails outright.
        Store(enabled_config).append_event(
            session_id="s1", platform="claude", cwd="/p", t="user_message"
        )


class TestExportEventInner:
    """The actual span-building logic, exercised directly and synchronously —
    this is what `thirdeye.otel_worker` calls once it's read a job file back
    off disk in its own detached process.
    """

    def test_marker_span_for_point_event(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        _export(
            enabled_config,
            sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            t="user_message",
            seq=0,
            ts="2026-01-01T00:00:00.000Z",
            data={"text": "hi"},
        )
        spans = exporter.exported_spans_as_dict()
        assert len(spans) == 1
        span = spans[0]
        assert span["name"] == "user_message"
        assert span["attributes"]["text"] == "hi"
        assert span["attributes"]["gen_ai.conversation.id"] == "s1"
        assert span["start_time"] == span["end_time"]

    def test_tool_call_alone_exports_nothing(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        _export(
            enabled_config,
            sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            t="tool_call",
            seq=0,
            ts="2026-01-01T00:00:00.000Z",
            data={"tool_name": "Bash", "tool_use_id": "tu_1"},
        )
        assert exporter.exported_spans_as_dict() == []

    def test_tool_result_pairs_with_matching_tool_call(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        # Local events must actually be on disk: the pairing logic reads them
        # back via SessionReader, same as the real worker would.
        store = Store(Config(root=tmp_path))
        store.append_event(
            session_id="s1",
            platform="claude",
            cwd="/proj",
            t="tool_call",
            data={"tool_name": "Bash", "tool_use_id": "tu_1", "command": "ls"},
        )
        sd = tmp_path / "traces" / "claude" / "s1"
        _export(
            enabled_config,
            sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            t="tool_result",
            seq=1,
            ts=utc_iso_ms(),  # after the tool_call's own wall-clock ts
            data={"tool_use_id": "tu_1", "tool_response": "file.txt"},
        )
        spans = exporter.exported_spans_as_dict()
        assert len(spans) == 1
        span = spans[0]
        assert span["name"] == "tool: Bash"
        assert span["attributes"]["command"] == "ls"
        assert span["attributes"]["tool_response"] == "file.txt"
        # <=, not <: thirdeye's ts has millisecond resolution, so two fast
        # back-to-back writes in a test can legitimately land on the same tick.
        assert span["start_time"] <= span["end_time"]

    def test_events_in_one_session_share_a_trace(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        _export(
            enabled_config,
            sd,
            session_id="s1",
            platform="claude",
            cwd="/p",
            t="session_start",
            seq=0,
            ts="2026-01-01T00:00:00.000Z",
        )
        _export(
            enabled_config,
            sd,
            session_id="s1",
            platform="claude",
            cwd="/p",
            t="user_message",
            seq=1,
            ts="2026-01-01T00:00:01.000Z",
        )
        spans = exporter.exported_spans_as_dict()
        assert len(spans) == 2
        assert spans[0]["context"]["trace_id"] == spans[1]["context"]["trace_id"]
        # The second span is parented under the first (the trace's root).
        assert spans[1]["parent"]["span_id"] == spans[0]["context"]["span_id"]

    def test_different_sessions_get_different_traces(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        _export(
            enabled_config,
            tmp_path / "traces" / "claude" / "s1",
            session_id="s1",
            platform="claude",
            cwd="/p",
            t="user_message",
            seq=0,
            ts="2026-01-01T00:00:00.000Z",
        )
        _export(
            enabled_config,
            tmp_path / "traces" / "claude" / "s2",
            session_id="s2",
            platform="claude",
            cwd="/p",
            t="user_message",
            seq=0,
            ts="2026-01-01T00:00:00.000Z",
        )
        spans = exporter.exported_spans_as_dict()
        assert spans[0]["context"]["trace_id"] != spans[1]["context"]["trace_id"]

    def test_root_persisted_across_separate_calls(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        """Simulates two separate worker processes exporting into the same
        session: each call here is independent, only otel.json on disk ties
        them together, same as two real `thirdeye.otel_worker` invocations.
        """
        sd = tmp_path / "traces" / "claude" / "s1"
        _export(
            enabled_config,
            sd,
            session_id="s1",
            platform="claude",
            cwd="/p",
            t="session_start",
            seq=0,
            ts="2026-01-01T00:00:00.000Z",
        )
        _export(
            enabled_config,
            sd,
            session_id="s1",
            platform="claude",
            cwd="/p",
            t="user_message",
            seq=1,
            ts="2026-01-01T00:00:01.000Z",
        )
        spans = exporter.exported_spans_as_dict()
        assert spans[0]["context"]["trace_id"] == spans[1]["context"]["trace_id"]
