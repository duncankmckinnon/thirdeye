from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from thirdeye import otel_export
from thirdeye.config import Config, LogfireSettings
from thirdeye.span_ids import (
    chat_span_id,
    root_span_id_for_session,
    tool_span_id,
    trace_id_for_session,
    turn_span_id,
)

pytest.importorskip("logfire")

from logfire.testing import TestExporter  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402


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
    """A real Logfire instance wired to an in-memory exporter, network-free.

    Wired to the same generator `_get_instance` would hand a real one, so the
    ids the export path pre-allocates are actually the ids spans come out
    with — otherwise every id assertion here would pass against the SDK's own
    random generator and prove nothing.
    """
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
        logfire=LogfireSettings(enabled=True, token="fake-token", project="p"),
    )


def _turn(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        turn_id="turn_1",
        start_ts="2026-01-01T00:00:00.000Z",
        end_ts="2026-01-01T00:00:05.000Z",
        input_message="",
        output_message="",
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


def _tool_call(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        tool_call_id="tu_1",
        name="Bash",
        start_ts="2026-01-01T00:00:01.100Z",
        end_ts="2026-01-01T00:00:01.900Z",
        attributes={"command": "ls"},
    )
    defaults.update(overrides)
    return defaults


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

    def test_primitives_and_homogeneous_lists_get_no_json_schema(self):
        # Without a JSON-encoded key, there's nothing to tell the backend to
        # parse back into structured data.
        out = otel_export._flatten_attrs({"a": 1, "tags": ["a", "b"]})
        assert "logfire.json_schema" not in out

    def test_nested_dict_marked_as_object_in_json_schema(self):
        out = otel_export._flatten_attrs({"tool_input": {"path": "x.py"}})
        schema = json.loads(out["logfire.json_schema"])
        assert schema == {"type": "object", "properties": {"tool_input": {"type": "object"}}}

    def test_list_of_dicts_marked_as_array_in_json_schema(self):
        out = otel_export._flatten_attrs({"gen_ai.input.messages": [{"role": "user", "parts": []}]})
        schema = json.loads(out["logfire.json_schema"])
        assert schema == {
            "type": "object",
            "properties": {"gen_ai.input.messages": {"type": "array"}},
        }

    def test_merge_raw_combines_before_flattening_so_schema_covers_both(self):
        merged = otel_export._merge_raw({"a": {"x": 1}}, {"b": ["y", "z"], "c": [{"n": 1}]})
        out = otel_export._flatten_attrs(merged)
        schema = json.loads(out["logfire.json_schema"])
        assert schema["properties"] == {"a": {"type": "object"}, "c": {"type": "array"}}


class TestScrubCallback:
    """`_scrub_callback` is passed to Logfire's ScrubbingOptions, wired in by
    `_get_instance` — see its own docstring for why "session" specifically
    needs an exemption for a coding agent's captured content (routinely full
    of legitimate uses of the word, unlike the other default patterns).
    """

    def _match(self, value: str):
        import re

        from logfire import ScrubMatch
        from logfire._internal.scrubbing import DEFAULT_PATTERNS

        pattern = re.compile("|".join(DEFAULT_PATTERNS), re.IGNORECASE | re.DOTALL)
        m = pattern.search(value)
        assert m is not None, f"expected {value!r} to trip a default scrubbing pattern"
        return ScrubMatch(path=("attributes", "text"), value=value, pattern_match=m)

    def test_session_match_is_let_through(self):
        value = "this session captured 5 tool calls"
        assert otel_export._scrub_callback(self._match(value)) == value

    def test_session_match_is_case_insensitive(self):
        value = "Session started"
        assert otel_export._scrub_callback(self._match(value)) == value

    def test_password_match_is_still_redacted(self):
        assert otel_export._scrub_callback(self._match("password: hunter2")) is None

    def test_api_key_match_is_still_redacted(self):
        assert otel_export._scrub_callback(self._match("my api_key is abc123")) is None

    def test_end_to_end_via_a_real_scrubber(self):
        """Not just the callback function in isolation — the actual Scrubber
        Logfire's SDK builds from ScrubbingOptions(callback=...).
        """
        from logfire._internal.scrubbing import Scrubber

        scrubber = Scrubber(patterns=None, callback=otel_export._scrub_callback)

        kept, notes = scrubber.scrub_value(("attributes", "text"), "the session ended cleanly")
        assert kept == "the session ended cleanly"
        assert notes == []

        redacted, notes = scrubber.scrub_value(("attributes", "text"), "password: hunter2")
        assert redacted != "password: hunter2"
        assert notes != []


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
        configured_with = {}

        def _configure(**kwargs):
            configured_with.update(kwargs)
            return object()

        monkeypatch.setattr(otel_export, "_silence_background_noise", lambda: calls.append(1))
        monkeypatch.setattr(logfire, "configure", _configure)
        config = Config(
            root=tmp_path, logfire=LogfireSettings(enabled=True, token="bad-token", project=None)
        )
        otel_export._get_instance(config, "claude")
        assert calls == [1]
        assert configured_with["advanced"].id_generator is otel_export._state["id_generator"]
        assert configured_with["scrubbing"].callback is otel_export._scrub_callback


class TestPreallocatedIdGenerator:
    """Span ids for this tree are derived rather than minted, so that a span
    emitted while a turn is still running can name a parent that hasn't been
    exported yet. `_start_span_with_id` is how a derived id actually reaches
    the SDK: it sets a one-shot slot on the generator handed to
    `logfire.configure`, and the next id drawn is the one we chose.
    """

    def _tracer(self, instance):
        return instance.config.get_tracer_provider().get_tracer("thirdeye")

    def test_preset_span_id_is_used_verbatim(self, wired_instance):
        chosen = 0x0123456789ABCDEF
        span = otel_export._start_span_with_id(
            self._tracer(wired_instance), "chosen", chosen, start_time=1, attributes={}
        )
        span.end(end_time=2)
        assert span.get_span_context().span_id == chosen

    def test_preset_trace_id_is_used_for_a_parentless_span(self, wired_instance):
        span = otel_export._start_span_with_id(
            self._tracer(wired_instance),
            "chosen",
            0xAA,
            trace_id=0xBB,
            start_time=1,
            attributes={},
        )
        span.end(end_time=2)
        context = span.get_span_context()
        assert (context.trace_id, context.span_id) == (0xBB, 0xAA)

    def test_slot_clears_after_one_use(self, wired_instance):
        tracer = self._tracer(wired_instance)
        chosen = 0x0123456789ABCDEF
        first = otel_export._start_span_with_id(
            tracer, "first", chosen, start_time=1, attributes={}
        )
        first.end(end_time=2)
        second = tracer.start_span("second", start_time=3)
        second.end(end_time=4)
        second_id = second.get_span_context().span_id
        assert second_id != chosen  # the slot is consumed, not sticky
        assert second_id != 0

    def test_unset_slot_yields_valid_random_ids(self):
        generator = otel_export._id_generator()
        span_ids = {generator.generate_span_id() for _ in range(5)}
        trace_ids = {generator.generate_trace_id() for _ in range(5)}
        assert len(span_ids) == 5
        assert all(0 < value < 2**64 for value in span_ids)
        assert len(trace_ids) == 5
        assert all(0 < value < 2**128 for value in trace_ids)

    def test_configure_draws_no_ids(self, exporter):
        """A canary on a third-party assumption the whole scheme rests on.

        Slots are set immediately before a `start_span` call, so anything else
        drawing an id in between would steal one and silently misparent a
        span. `logfire.configure` is the one thing that runs between our own
        spans without us asking; today it draws nothing (its configuration
        span defaults off), but a future release emitting one at configure
        time would break parenting in a way no other test here would notice.
        """
        import logfire

        probe = otel_export._build_id_generator()
        drawn: list[str] = []
        probe.generate_span_id = lambda: drawn.append("span") or 1
        probe.generate_trace_id = lambda: drawn.append("trace") or 1

        logfire.configure(
            send_to_logfire=False,
            console=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
            advanced=logfire.AdvancedOptions(id_generator=probe),
        )
        assert drawn == []


class TestRootOwnership:
    def test_root_creation_is_exclusive(self, tmp_path: Path):
        root_path = tmp_path / "otel.json"
        root, first_lock = otel_export._root_or_ownership(root_path)
        assert root is None
        assert first_lock is not None
        assert otel_export._create_root_atomic(root_path, 0xAA, 0xBB) == ((0xAA, 0xBB), True)
        first_lock.unlink()
        root, second_lock = otel_export._root_or_ownership(root_path)
        assert root == (0xAA, 0xBB)
        assert second_lock is None


class TestClaimTurnExport:
    def test_first_claim_succeeds_second_fails(self, tmp_path: Path):
        assert otel_export._claim_turn_export(tmp_path, "turn_1") is True
        assert otel_export._claim_turn_export(tmp_path, "turn_1") is False

    def test_distinct_turn_ids_each_claim_independently(self, tmp_path: Path):
        assert otel_export._claim_turn_export(tmp_path, "turn_1") is True
        assert otel_export._claim_turn_export(tmp_path, "turn_2") is True

    def test_turn_id_with_colon_is_a_safe_filename(self, tmp_path: Path):
        assert otel_export._claim_turn_export(tmp_path, "cum:12345") is True


class TestExportTurnDispatch:
    """`export_turn` (called synchronously from inside a hook process) must
    never itself touch the network: it should only ever write a job file and
    spawn a detached, unwaited-for worker process. `subprocess.Popen` is
    mocked so these tests don't actually spawn a Python interpreter per case.
    """

    def test_disabled_spawns_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        spawned = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda *a, **k: spawned.append(a))
        config = Config(root=tmp_path)
        otel_export.export_turn(config, tmp_path, "s1", "claude", "/p", _turn())
        assert spawned == []

    def test_no_token_spawns_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        spawned = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda *a, **k: spawned.append(a))
        config = Config(root=tmp_path, logfire=LogfireSettings(enabled=True, token=None))
        otel_export.export_turn(config, tmp_path, "s1", "claude", "/p", _turn())
        assert spawned == []

    def test_enabled_spawns_a_detached_worker(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        calls = []

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                calls.append((argv, kwargs))

        monkeypatch.setattr(otel_export.subprocess, "Popen", _FakePopen)
        otel_export.export_turn(enabled_config, tmp_path, "s1", "claude", "/proj", _turn())
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[0] == otel_export.sys.executable
        assert argv[1:3] == ["-m", "thirdeye.otel_worker"]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is otel_export.subprocess.DEVNULL

    def test_job_file_carries_the_full_turn(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        calls = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda argv, **k: calls.append(argv))
        turn = _turn(turn_id="turn_42")
        otel_export.export_turn(enabled_config, tmp_path, "s1", "claude", "/proj", turn)
        job_path = Path(calls[0][3])
        assert job_path.parent == otel_export.otel_jobs_dir(tmp_path)
        payload = json.loads(job_path.read_text())
        assert payload["kind"] == "turn"
        assert payload["session_id"] == "s1"
        assert payload["platform"] == "claude"
        assert payload["cwd"] == "/proj"
        assert payload["turn"]["turn_id"] == "turn_42"

    def test_spawn_failure_is_logged_not_raised(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        def _boom(*a, **k):
            raise OSError("no more processes")

        monkeypatch.setattr(otel_export.subprocess, "Popen", _boom)
        # Must not raise, even though spawning fails outright.
        otel_export.export_turn(enabled_config, tmp_path, "s1", "claude", "/p", _turn())


class TestExportSpansDispatch:
    """Live spans use the same detached job transport as completed turns."""

    @staticmethod
    def _span(**overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "name": "chat claude-sonnet-5",
            "span_id": 2**64 - 1,
            "parent_span_id": 2**63 + 17,
            "start_ts": "2026-01-01T00:00:01.000Z",
            "end_ts": "2026-01-01T00:00:02.000Z",
            "attributes": {"gen_ai.operation.name": "chat"},
        }
        defaults.update(overrides)
        return defaults

    @pytest.mark.parametrize(
        "settings",
        [
            LogfireSettings(enabled=False, token="fake-token"),
            LogfireSettings(enabled=True, token=None),
        ],
    )
    def test_disabled_or_tokenless_config_spawns_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings: LogfireSettings,
    ):
        spawned = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda *a, **k: spawned.append(a))
        config = Config(root=tmp_path, logfire=settings)

        otel_export.export_spans(
            config, tmp_path / "session", "s1", "claude", "/proj", 1, [self._span()]
        )

        assert spawned == []

    def test_writes_decimal_id_job_and_spawns_once(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        calls = []
        monkeypatch.setattr(
            otel_export.subprocess,
            "Popen",
            lambda argv, **kwargs: calls.append((argv, kwargs)),
        )
        trace_id = 2**128 - 1
        source_span = self._span(attributes={"nested": {"safe": True}})

        otel_export.export_spans(
            enabled_config,
            tmp_path / "traces" / "claude" / "s1",
            "s1",
            "claude",
            "/proj",
            trace_id,
            [source_span],
        )

        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[:3] == [otel_export.sys.executable, "-m", "thirdeye.otel_worker"]
        assert kwargs["start_new_session"] is True
        job_path = Path(argv[3])
        payload = json.loads(job_path.read_text())
        assert payload == {
            "kind": "spans",
            "session_dir": str(tmp_path / "traces" / "claude" / "s1"),
            "session_id": "s1",
            "platform": "claude",
            "cwd": "/proj",
            "trace_id": str(trace_id),
            "spans": [
                {
                    **source_span,
                    "span_id": str(source_span["span_id"]),
                    "parent_span_id": str(source_span["parent_span_id"]),
                }
            ],
        }
        # Serialization must not mutate the caller's already-built span.
        assert isinstance(source_span["span_id"], int)
        assert isinstance(source_span["parent_span_id"], int)

    def test_malformed_span_is_swallowed_before_spawn(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        spawned = []
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda *a, **k: spawned.append(a))

        otel_export.export_spans(
            enabled_config,
            tmp_path / "session",
            "s1",
            "claude",
            "/proj",
            1,
            [{"name": "missing ids"}],
        )

        assert spawned == []

    def test_unwritable_jobs_directory_is_swallowed(
        self, tmp_path: Path, enabled_config: Config, monkeypatch: pytest.MonkeyPatch
    ):
        def _permission_denied(*args, **kwargs):
            raise PermissionError("jobs directory is read-only")

        spawned = []
        monkeypatch.setattr(otel_export, "_write_job", _permission_denied)
        monkeypatch.setattr(otel_export.subprocess, "Popen", lambda *a, **k: spawned.append(a))

        otel_export.export_spans(
            enabled_config, tmp_path / "session", "s1", "claude", "/proj", 1, [self._span()]
        )

        assert spawned == []


class TestExportSpansBatch:
    @staticmethod
    def _batch_span(**overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "name": "chat claude-sonnet-5",
            "span_id": str(0xFEDCBA9876543210),
            "parent_span_id": str(0x123456789ABCDEF0),
            "start_ts": "2026-01-01T00:00:01.000Z",
            "end_ts": "2026-01-01T00:00:02.000Z",
            "attributes": {"gen_ai.operation.name": "chat"},
        }
        defaults.update(overrides)
        return defaults

    def _export(self, tmp_path: Path, enabled_config: Config, spans, *, trace_id=0xABCDEF):
        otel_export._export_spans_batch(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / "claude" / "s1",
            session_id="s1",
            platform="claude",
            cwd="/proj",
            trace_id=str(trace_id),
            spans=spans,
        )

    def test_honours_trace_span_and_parent_ids(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        trace_id = 0xABCDEF0123456789ABCDEF0123456789
        span_id = 0xFEDCBA9876543210
        parent_span_id = 0x123456789ABCDEF0

        self._export(
            tmp_path,
            enabled_config,
            [
                self._batch_span(
                    span_id=str(span_id),
                    parent_span_id=str(parent_span_id),
                )
            ],
            trace_id=trace_id,
        )

        [span] = exporter.exported_spans_as_dict()
        assert span["context"]["trace_id"] == trace_id
        assert span["context"]["span_id"] == span_id
        assert span["parent"]["trace_id"] == trace_id
        assert span["parent"]["span_id"] == parent_span_id

    def test_child_is_emitted_when_parent_is_absent_from_batch(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        absent_parent_id = 0x1111222233334444

        self._export(
            tmp_path,
            enabled_config,
            [self._batch_span(name="tool: Bash", parent_span_id=str(absent_parent_id))],
        )

        [child] = exporter.exported_spans_as_dict()
        assert child["name"] == "tool: Bash"
        assert child["parent"]["span_id"] == absent_parent_id
        assert all(span["context"]["span_id"] != absent_parent_id for span in [child])

    def test_force_flushes_exactly_once_after_multiple_spans(
        self,
        tmp_path: Path,
        enabled_config: Config,
        wired_instance,
        exporter,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls = []
        real_force_flush = wired_instance.force_flush

        def _force_flush(*args, **kwargs):
            calls.append((args, kwargs))
            return real_force_flush(*args, **kwargs)

        monkeypatch.setattr(wired_instance, "force_flush", _force_flush)
        self._export(
            tmp_path,
            enabled_config,
            [
                self._batch_span(span_id="101"),
                self._batch_span(name="tool: Read", span_id="102", parent_span_id="101"),
            ],
        )

        assert len(exporter.exported_spans_as_dict()) == 2
        assert calls == [((), {"timeout_millis": otel_export._FLUSH_TIMEOUT_MS})]

    def test_chat_attribute_keys_match_completed_turn_export(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        call = _llm_call()
        session_dir = tmp_path / "traces" / "claude" / "s1"
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=session_dir,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=_turn(llm_calls=[call]),
        )
        completed_chat = next(
            span for span in exporter.exported_spans_as_dict() if span["name"].startswith("chat")
        )

        live_attributes = otel_export._chat_attributes(call)
        self._export(
            tmp_path,
            enabled_config,
            [self._batch_span(span_id="999", attributes=live_attributes)],
        )
        live_chat = next(
            span for span in exporter.exported_spans_as_dict() if span["context"]["span_id"] == 999
        )

        assert set(live_chat["attributes"]) == set(completed_chat["attributes"])

    def test_tool_attributes_include_common_thirdeye_vocabulary(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        self._export(
            tmp_path,
            enabled_config,
            [
                self._batch_span(
                    name="tool: Bash",
                    attributes={"command": "pwd"},
                )
            ],
        )

        [tool_span] = exporter.exported_spans_as_dict()
        assert tool_span["attributes"]["command"] == "pwd"
        assert tool_span["attributes"]["gen_ai.conversation.id"] == "s1"
        assert tool_span["attributes"]["thirdeye.platform"] == "claude"
        assert tool_span["attributes"]["thirdeye.cwd"] == "/proj"

    def test_live_tool_attribute_named_attributes_is_preserved(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        self._export(
            tmp_path,
            enabled_config,
            [
                self._batch_span(
                    name="tool: Bash",
                    attributes={"command": "x", "attributes": {"mode": "safe"}},
                )
            ],
        )

        [tool_span] = exporter.exported_spans_as_dict()
        assert tool_span["attributes"]["command"] == "x"
        assert json.loads(tool_span["attributes"]["attributes"]) == {"mode": "safe"}


class TestExportTurnInner:
    """The actual span-building logic, exercised directly and synchronously —
    this is what `thirdeye.otel_worker` calls once it's read a job file back
    off disk in its own detached process.
    """

    def test_minimal_turn_produces_one_span_with_no_messages(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=_turn(),
        )
        spans = exporter.exported_spans_as_dict()
        # The root (session) span plus the turn span.
        assert len(spans) == 2
        turn_span = spans[-1]
        assert turn_span["name"] == "agent-turn"
        assert "gen_ai.input.messages" not in turn_span["attributes"]
        assert "gen_ai.output.messages" not in turn_span["attributes"]
        assert turn_span["attributes"]["thirdeye.turn.status"] == "completed"
        assert turn_span["attributes"]["thirdeye.turn.id"] == "turn_1"

    def test_live_chat_parent_matches_completed_turn_span_id(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        session_id = "live-parent-session"
        expected_turn_id = turn_span_id(session_id, 1)
        live_chat_id = chat_span_id(session_id, "call_live")
        session_dir = tmp_path / "traces" / "claude" / session_id

        otel_export._export_spans_batch(
            config=enabled_config,
            session_dir_=session_dir,
            session_id=session_id,
            platform="claude",
            cwd="/proj",
            trace_id=trace_id_for_session(session_id),
            spans=[
                {
                    "name": "chat claude-sonnet-5",
                    "span_id": str(live_chat_id),
                    "parent_span_id": str(expected_turn_id),
                    "start_ts": "2026-01-01T00:00:01.000Z",
                    "end_ts": "2026-01-01T00:00:02.000Z",
                    "attributes": _llm_call(call_id="call_live"),
                }
            ],
        )
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=session_dir,
            session_id=session_id,
            platform="claude",
            cwd="/proj",
            turn=_turn(turn_span_id=str(expected_turn_id)),
        )

        spans = exporter.exported_spans_as_dict()
        live_chat = next(span for span in spans if span["context"]["span_id"] == live_chat_id)
        completed_turn = next(span for span in spans if span["name"] == "agent-turn")
        assert live_chat["parent"]["span_id"] == completed_turn["context"]["span_id"]
        assert completed_turn["context"]["span_id"] == expected_turn_id

    def test_turn_with_messages_carries_them(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=_turn(input_message="fix the bug", output_message="fixed it"),
        )
        turn_span = exporter.exported_spans_as_dict()[-1]
        attrs = turn_span["attributes"]
        input_messages = json.loads(attrs["gen_ai.input.messages"])
        output_messages = json.loads(attrs["gen_ai.output.messages"])
        assert input_messages[0]["parts"][0]["content"] == "fix the bug"
        assert output_messages[0]["parts"][0]["content"] == "fixed it"

    def test_root_span_carries_no_input_output(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=_turn(input_message="hi", output_message="hello"),
        )
        root_span = exporter.exported_spans_as_dict()[0]
        assert "gen_ai.input.messages" not in root_span["attributes"]
        assert "gen_ai.output.messages" not in root_span["attributes"]
        assert root_span["attributes"]["gen_ai.conversation.id"] == "s1"
        assert root_span["attributes"]["thirdeye.platform"] == "claude"
        assert root_span["attributes"]["thirdeye.cwd"] == "/proj"

    def test_new_session_persists_and_emits_derived_root_ids(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        session_id = "session-with-derived-root"
        sd = tmp_path / "traces" / "claude" / session_id

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id=session_id,
            platform="claude",
            cwd="/proj",
            turn=_turn(),
        )

        expected_trace_id = trace_id_for_session(session_id)
        expected_span_id = root_span_id_for_session(session_id)
        persisted = json.loads(otel_export.otel_state_path(sd).read_text())
        assert persisted == {
            "trace_id": f"{expected_trace_id:032x}",
            "span_id": f"{expected_span_id:016x}",
        }

        root_span, turn_span = exporter.exported_spans_as_dict()
        assert root_span["name"] == "session"
        assert root_span["context"]["trace_id"] == expected_trace_id
        assert root_span["context"]["span_id"] == expected_span_id
        assert turn_span["context"]["trace_id"] == expected_trace_id
        assert turn_span["parent"]["span_id"] == expected_span_id

    def test_existing_root_ids_take_precedence_and_are_not_reemitted(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        session_id = "session-with-legacy-root"
        sd = tmp_path / "traces" / "claude" / session_id
        root_path = otel_export.otel_state_path(sd)
        legacy_trace_id = 0x123456789ABCDEF0123456789ABCDEF0
        legacy_span_id = 0x123456789ABCDEF0
        assert otel_export._create_root_atomic(root_path, legacy_trace_id, legacy_span_id) == (
            (legacy_trace_id, legacy_span_id),
            True,
        )
        original_payload = root_path.read_text()

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id=session_id,
            platform="claude",
            cwd="/proj",
            turn=_turn(),
        )

        spans = exporter.exported_spans_as_dict()
        assert [span["name"] for span in spans] == ["agent-turn"]
        turn_span = spans[0]
        assert turn_span["context"]["trace_id"] == legacy_trace_id
        assert turn_span["parent"]["span_id"] == legacy_span_id
        assert root_path.read_text() == original_payload

    def test_same_derived_root_race_does_not_reemit_session_span(
        self,
        tmp_path: Path,
        enabled_config: Config,
        wired_instance,
        exporter,
        monkeypatch: pytest.MonkeyPatch,
    ):
        session_id = "session-with-concurrent-derived-root"
        sd = tmp_path / "traces" / "claude" / session_id
        root_path = otel_export.otel_state_path(sd)
        expected = (
            trace_id_for_session(session_id),
            root_span_id_for_session(session_id),
        )
        real_atomic_create = otel_export._atomic_create

        def _competing_create(path: Path, payload: str) -> bool:
            if path == root_path:
                # Simulate another worker winning between our lock recovery
                # and root persistence with the same deterministic ids.
                assert real_atomic_create(path, payload) is True
                return False
            return real_atomic_create(path, payload)

        monkeypatch.setattr(otel_export, "_atomic_create", _competing_create)
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id=session_id,
            platform="claude",
            cwd="/proj",
            turn=_turn(),
        )

        spans = exporter.exported_spans_as_dict()
        assert [span["name"] for span in spans] == ["agent-turn"]
        assert spans[0]["context"]["trace_id"] == expected[0]
        assert spans[0]["parent"]["span_id"] == expected[1]

    def test_llm_call_and_tool_call_nest_correctly(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        turn = _turn(
            llm_calls=[_llm_call(tool_calls=[_tool_call()])],
        )
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )
        spans = exporter.exported_spans_as_dict()
        # root, turn, chat, tool
        assert len(spans) == 4
        root_span, turn_span, chat_span, tool_span = spans
        assert turn_span["name"] == "agent-turn"
        assert chat_span["name"] == "chat claude-sonnet-5"
        assert tool_span["name"] == "tool: Bash"
        assert chat_span["context"]["span_id"] == chat_span_id("s1", "call_1")
        assert tool_span["context"]["span_id"] == tool_span_id("s1", "tu_1")
        assert chat_span["parent"]["span_id"] == turn_span["context"]["span_id"]
        assert tool_span["parent"]["span_id"] == chat_span["context"]["span_id"]
        assert chat_span["attributes"]["gen_ai.usage.input_tokens"] == 100
        assert chat_span["attributes"]["gen_ai.usage.output_tokens"] == 50
        assert tool_span["attributes"]["command"] == "ls"
        assert tool_span["attributes"]["gen_ai.conversation.id"] == "s1"

    def test_permission_request_is_a_point_in_time_span_under_the_turn(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        turn = _turn(
            permission_requests=[
                {
                    "ts": "2026-01-01T00:00:02.000Z",
                    "tool_name": "Bash",
                    "attributes": {"command": "rm -rf /"},
                }
            ]
        )
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )
        spans = exporter.exported_spans_as_dict()
        pr_span = spans[-1]
        turn_span = spans[-2]
        assert pr_span["name"] == "permission_request: Bash"
        assert pr_span["start_time"] == pr_span["end_time"]
        assert pr_span["parent"]["span_id"] == turn_span["context"]["span_id"]
        assert pr_span["attributes"]["command"] == "rm -rf /"

    def test_subagent_nests_under_the_parent_turn_with_its_own_children(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        subagent = _turn(
            turn_id="turn_sub",
            llm_calls=[_llm_call(tool_calls=[_tool_call()])],
        )
        turn = _turn(subagents=[subagent])
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )
        spans = exporter.exported_spans_as_dict()
        # root, top turn, subagent turn, chat, tool
        assert len(spans) == 5
        top_turn_span = spans[1]
        subagent_span = spans[2]
        chat_span = spans[3]
        tool_span = spans[4]
        assert subagent_span["name"] == "agent-turn (subagent)"
        assert subagent_span["parent"]["span_id"] == top_turn_span["context"]["span_id"]
        assert chat_span["parent"]["span_id"] == subagent_span["context"]["span_id"]
        assert tool_span["parent"]["span_id"] == chat_span["context"]["span_id"]

    def test_calling_twice_with_same_turn_id_exports_only_once(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        turn = _turn()
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )
        first_count = len(exporter.exported_spans_as_dict())
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )
        assert len(exporter.exported_spans_as_dict()) == first_count

    def test_events_in_one_session_share_a_trace(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/p",
            turn=_turn(turn_id="turn_1"),
        )
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/p",
            turn=_turn(turn_id="turn_2"),
        )
        spans = exporter.exported_spans_as_dict()
        trace_ids = {s["context"]["trace_id"] for s in spans}
        assert len(trace_ids) == 1

    def test_different_sessions_get_different_traces(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / "claude" / "s1",
            session_id="s1",
            platform="claude",
            cwd="/p",
            turn=_turn(),
        )
        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=tmp_path / "traces" / "claude" / "s2",
            session_id="s2",
            platform="claude",
            cwd="/p",
            turn=_turn(),
        )
        spans = exporter.exported_spans_as_dict()
        assert spans[0]["context"]["trace_id"] != spans[-1]["context"]["trace_id"]


class TestExportTurnInnerFailureRecovery:
    """A turn's claim must only become permanent ("sent") once its whole
    subtree has actually been built and flushed — not the moment export is
    attempted. Otherwise a transient failure (root ownership never resolves,
    or the flush itself fails) would permanently suppress every future retry
    of that turn, silently losing it for good.
    """

    def test_root_ownership_failure_releases_claim_so_a_retry_can_succeed(
        self, tmp_path: Path, enabled_config: Config, wired_instance, exporter
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        turn = _turn()
        real_root_or_ownership = otel_export._root_or_ownership
        attempt = {"n": 0}

        def _fails_once(root_path):
            attempt["n"] += 1
            if attempt["n"] == 1:
                return None, None  # simulates _root_or_ownership timing out
            return real_root_or_ownership(root_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(otel_export, "_root_or_ownership", _fails_once)
            with pytest.raises(RuntimeError):
                otel_export._export_turn_inner(
                    config=enabled_config,
                    session_dir_=sd,
                    session_id="s1",
                    platform="claude",
                    cwd="/proj",
                    turn=turn,
                )
        assert exporter.exported_spans_as_dict() == []
        claim_path = otel_export._turn_claim_path(sd, turn["turn_id"])
        assert not claim_path.exists()

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )
        assert claim_path.read_text() == "sent"
        spans = exporter.exported_spans_as_dict()
        assert len(spans) == 2  # session root + turn span
        assert spans[-1]["name"] == "agent-turn"

    def test_flush_failure_releases_claim_so_a_retry_can_succeed(
        self,
        tmp_path: Path,
        enabled_config: Config,
        wired_instance,
        exporter,
        monkeypatch: pytest.MonkeyPatch,
    ):
        sd = tmp_path / "traces" / "claude" / "s1"
        turn = _turn()
        real_force_flush = wired_instance.force_flush
        attempt = {"n": 0}

        def _flush_fails_once(*args, **kwargs):
            attempt["n"] += 1
            if attempt["n"] == 1:
                return False
            return real_force_flush(*args, **kwargs)

        monkeypatch.setattr(wired_instance, "force_flush", _flush_fails_once)

        with pytest.raises(RuntimeError):
            otel_export._export_turn_inner(
                config=enabled_config,
                session_dir_=sd,
                session_id="s1",
                platform="claude",
                cwd="/proj",
                turn=turn,
            )
        claim_path = otel_export._turn_claim_path(sd, turn["turn_id"])
        assert not claim_path.exists()
        spans_after_failure = len(exporter.exported_spans_as_dict())

        otel_export._export_turn_inner(
            config=enabled_config,
            session_dir_=sd,
            session_id="s1",
            platform="claude",
            cwd="/proj",
            turn=turn,
        )
        assert claim_path.read_text() == "sent"
        assert len(exporter.exported_spans_as_dict()) > spans_after_failure
