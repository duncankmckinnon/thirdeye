"""Behavioral tests for Copilot hook payload normalization."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from thirdeye.platforms.copilot.constants import (
    CLI_HOOK_EVENT_ALIASES,
    SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from thirdeye.platforms.copilot.hook_payload import parse_hook
from thirdeye.platforms.copilot.types import SourceRecord

FIXTURES = Path(__file__).parent / "fixtures"
CLI_HOOKS = FIXTURES / "hooks.jsonl"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
CHILD_AGENT_ID = "bf8cb9f3-2097-4db0-a3c8-78a2653b2106"
OBSERVED_AT = "2026-09-10T17:08:25.626Z"

# PascalCase aliases accepted for input compatibility (Claude-style names).
PASCAL_CASE_ALIASES: dict[str, str] = {
    "SessionStart": "sessionStart",
    "UserPromptSubmit": "userPromptSubmitted",
    "PreToolUse": "preToolUse",
    "PostToolUse": "postToolUse",
    "Stop": "agentStop",
    "SubagentStart": "subagentStart",
    "SubagentStop": "subagentStop",
    "SessionEnd": "sessionEnd",
}


def _load_cli_hooks() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CLI_HOOKS.read_text(encoding="utf-8").splitlines()]


def _parse(
    event: str,
    payload: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    observation_id: str = "obs-test-1",
    observed_at: str = OBSERVED_AT,
) -> SourceRecord:
    return parse_hook(
        event,
        payload,
        context or {},
        observed_at=observed_at,
        observation_id=observation_id,
    )


def test_hook_payload_module_has_no_io_imports():
    import thirdeye.platforms.copilot.hook_payload as hook_payload

    source = Path(hook_payload.__file__).read_text(encoding="utf-8")
    forbidden = ("subprocess", "sqlite3", "open(", "Path(", "os.remove", "shutil")
    for token in forbidden:
        assert token not in source, f"hook_payload.py must not use {token!r}"


@pytest.mark.parametrize("camel_event", CLI_HOOK_EVENT_ALIASES)
def test_parse_hook_accepts_native_camel_case_events(camel_event: str):
    payload = {
        "sessionId": NATIVE_SESSION_ID,
        "timestamp": 1789060102204,
        "cwd": "/fixture/workspace",
    }
    record = _parse(camel_event, payload, observation_id=f"obs-{camel_event}")

    assert record["source_kind"] == "hook"
    assert record["native_session_id"] == NATIVE_SESSION_ID
    assert record["observed_at"] == OBSERVED_AT
    assert record["payload"]["schema_version"] == SCHEMA_VERSION
    assert record["payload"]["hook_payload"] == payload
    assert record["payload"]["event"] == camel_event
    assert f"obs-{camel_event}" in record["source_id"]


@pytest.mark.parametrize(("pascal_event", "canonical_event"), PASCAL_CASE_ALIASES.items())
def test_parse_hook_accepts_pascal_case_aliases(pascal_event: str, canonical_event: str):
    payload = {
        "sessionId": NATIVE_SESSION_ID,
        "timestamp": 1789060102204,
        "cwd": "/fixture/workspace",
    }
    record = _parse(pascal_event, payload)

    assert record["payload"]["event"] == canonical_event


def test_parse_hook_retains_unmodified_payload_from_cli_fixture():
    hook = next(entry for entry in _load_cli_hooks() if entry["registered_event"] == "preToolUse")
    payload = deepcopy(hook["payload"])
    record = _parse(hook["registered_event"], payload, observation_id="obs-pre-tool")

    assert record["payload"]["hook_payload"] == hook["payload"]
    assert record["payload"]["hook_payload"] is not payload


def test_parse_hook_does_not_mutate_input_payload_or_context():
    payload = {
        "sessionId": NATIVE_SESSION_ID,
        "timestamp": 1789060105626,
        "nested": {"items": [1, 2]},
    }
    context = {"env": {"WB_PLAN": "p"}, "unexpected": "strip-me"}
    original_payload = deepcopy(payload)
    original_context = deepcopy(context)

    _parse("agentStop", payload, context=context)

    assert payload == original_payload
    assert context == original_context


def test_parse_hook_allowlists_context_and_drops_unknown_keys():
    context = {
        "env": {"WB_PLAN": "session-trace"},
        "trace_id": "trace-abc",
        "secret_token": "must-not-appear",
    }
    record = _parse(
        "sessionStart", {"sessionId": NATIVE_SESSION_ID, "timestamp": 1}, context=context
    )

    stored = record["payload"]["context"]
    assert stored["env"] == {"WB_PLAN": "session-trace"}
    assert stored["trace_id"] == "trace-abc"
    assert "secret_token" not in stored


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"timestamp": 1},
        {"sessionId": ""},
        {"sessionId": "../escape"},
        {"sessionId": "bad/id"},
    ],
)
def test_parse_hook_rejects_missing_or_invalid_session_routing(payload: dict[str, Any]):
    with pytest.raises(ValueError):
        _parse("sessionStart", payload)


def test_parse_hook_normalizes_millisecond_timestamp_to_iso():
    payload = {"sessionId": NATIVE_SESSION_ID, "timestamp": 1789060105626}
    record = _parse("agentStop", payload)

    assert record["ts"] is not None
    assert record["ts"].endswith("Z") or "+" in record["ts"]


def test_parse_hook_leaves_ts_none_when_timestamp_missing():
    payload = {"sessionId": NATIVE_SESSION_ID, "cwd": "/fixture/workspace"}
    record = _parse("sessionEnd", payload)

    assert record["ts"] is None


def test_parse_hook_preserves_child_agent_stop_with_child_session_id():
    hook = next(
        entry
        for entry in _load_cli_hooks()
        if entry["registered_event"] == "agentStop"
        and entry["payload"]["sessionId"] == CHILD_AGENT_ID
    )
    record = _parse(hook["registered_event"], hook["payload"], observation_id="obs-child-stop")

    assert record["native_session_id"] == CHILD_AGENT_ID
    assert record["payload"]["hook_payload"]["sessionId"] == CHILD_AGENT_ID


def test_parse_hook_preserves_subagent_stop_on_parent_session_id():
    hook = next(entry for entry in _load_cli_hooks() if entry["registered_event"] == "subagentStop")
    record = _parse(hook["registered_event"], hook["payload"], observation_id="obs-subagent-stop")

    assert record["native_session_id"] == NATIVE_SESSION_ID
    assert record["payload"]["hook_payload"]["agentId"] == CHILD_AGENT_ID


def test_distinct_observation_ids_produce_distinct_source_ids():
    payload = {"sessionId": NATIVE_SESSION_ID, "timestamp": 1789060105626, "stopReason": "end_turn"}
    first = _parse("agentStop", payload, observation_id="hook-observation-1")
    second = _parse("agentStop", payload, observation_id="hook-observation-2")

    assert first["source_id"] != second["source_id"]
    assert first["payload"]["hook_payload"] == second["payload"]["hook_payload"]


def test_source_record_envelope_uses_schema_version_one():
    record = _parse(
        "sessionStart",
        {"sessionId": NATIVE_SESSION_ID, "timestamp": 1789060102204},
    )
    assert record["payload"]["schema_version"] == SOURCE_SCHEMA_VERSION == 1


def test_locator_identifies_observation():
    record = _parse(
        "userPromptSubmitted",
        {"sessionId": NATIVE_SESSION_ID, "timestamp": 1789060102173, "prompt": "hi"},
        observation_id="obs-locator",
    )
    locator = record["locator"]
    assert locator["observation_id"] == "obs-locator"
    assert locator["event"] == "userPromptSubmitted"


def test_source_id_format_includes_session_and_observation():
    record = _parse(
        "sessionStart",
        {"sessionId": NATIVE_SESSION_ID, "timestamp": 1789060102204},
        observation_id="obs-format-check",
    )
    assert record["source_id"] == f"hook/{NATIVE_SESSION_ID}/obs-format-check"


def test_parse_hook_preserves_zulu_iso_timestamp_strings():
    iso_z = "2026-09-10T17:08:25.626Z"
    assert (
        _parse("sessionStart", {"sessionId": NATIVE_SESSION_ID, "timestamp": iso_z})["ts"] == iso_z
    )


def test_parse_hook_preserves_positive_offset_iso_timestamp_strings():
    iso_offset = "2026-09-10T17:08:25.626+00:00"
    assert (
        _parse("sessionStart", {"sessionId": NATIVE_SESSION_ID, "timestamp": iso_offset})["ts"]
        == iso_offset
    )


def test_parse_hook_preserves_negative_offset_iso_timestamp_strings():
    iso_offset = "2026-09-10T10:08:25-07:00"
    assert (
        _parse("sessionStart", {"sessionId": NATIVE_SESSION_ID, "timestamp": iso_offset})["ts"]
        == iso_offset
    )


def test_parse_hook_converts_numeric_string_timestamp():
    record = _parse(
        "sessionStart",
        {"sessionId": NATIVE_SESSION_ID, "timestamp": "1789060105626"},
    )
    assert record["ts"] is not None
    assert record["ts"].endswith("Z")


@pytest.mark.parametrize(
    "timestamp",
    [
        True,
        False,
        None,
        "",
        "   ",
        "not-a-number",
        {},
        "nonsenseZ",
        "2026-13-40T99:99:99Z",
        "2026-09-10T17:08:25.626Z extra",
        "+not-an-iso-timestamp",
    ],
)
def test_parse_hook_invalid_timestamps_leave_ts_none(timestamp: object):
    record = _parse("sessionStart", {"sessionId": NATIVE_SESSION_ID, "timestamp": timestamp})
    assert record["ts"] is None


def test_parse_hook_accepts_second_epoch_timestamps():
    record = _parse("sessionStart", {"sessionId": NATIVE_SESSION_ID, "timestamp": 1_789_060_105})
    assert record["ts"] is not None
    assert record["ts"].endswith("Z")


def test_parse_hook_unknown_event_name_passes_through():
    record = _parse(
        "customFutureHook",
        {"sessionId": NATIVE_SESSION_ID, "timestamp": 1789060102204},
    )
    assert record["payload"]["event"] == "customFutureHook"
    assert record["locator"]["event"] == "customFutureHook"


def test_parse_hook_stores_all_allowlisted_context_keys():
    context = {
        "env": {"WB_PLAN": "p"},
        "trace_id": "trace-1",
        "span_id": "span-1",
        "parent_span_id": "parent-span-1",
        "trace_context": {"sampled": True},
        "traceparent": "00-abc-def-01",
    }
    stored = _parse(
        "sessionStart",
        {"sessionId": NATIVE_SESSION_ID, "timestamp": 1},
        context=context,
    )["payload"]["context"]
    assert stored == context
    assert stored is not context
    assert stored["env"] is not context["env"]
