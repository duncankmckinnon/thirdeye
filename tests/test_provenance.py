from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from thirdeye.platforms.provenance import foreign_payload_reason

_PASCAL_CASE_EVENTS = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "SubagentStop",
    "Stop",
]


@pytest.mark.parametrize("expected", ["claude", "codex"])
@pytest.mark.parametrize("marker", ["cursor_version", "composer_mode"])
@pytest.mark.parametrize("value", [None, False, ""])
def test_cursor_marker_presence_is_foreign_even_when_value_is_falsey(
    expected: str,
    marker: str,
    value: object,
):
    reason = foreign_payload_reason({marker: value}, expected)

    assert isinstance(reason, str)
    assert reason
    assert marker in reason


@pytest.mark.parametrize("expected", ["claude", "codex"])
def test_lower_camel_event_is_cursor_evidence_for_non_cursor_platforms(expected: str):
    reason = foreign_payload_reason(
        {"hook_event_name": "beforeSubmitPrompt"},
        expected,
    )

    assert isinstance(reason, str)
    assert reason
    assert "beforeSubmitPrompt" in reason


@pytest.mark.parametrize(
    "payload",
    [
        {"hook_event_name": "beforeSubmitPrompt"},
        {"hook_event_name": "subagentStop"},
        {"cursor_version": "1.0.0"},
        {"composer_mode": "agent"},
    ],
)
def test_genuine_cursor_evidence_is_accepted_for_cursor(payload: dict[str, Any]):
    assert foreign_payload_reason(payload, "cursor") is None


@pytest.mark.parametrize(
    "event_name",
    [
        "sessionStart",
        "beforeSubmitPrompt",
        "afterAgentResponse",
        "beforeShellExecution",
        "postToolUse",
        "subagentStop",
        "stop",
    ],
)
def test_genuine_cursor_camel_case_events_are_accepted_for_cursor(event_name: str):
    assert foreign_payload_reason({"hook_event_name": event_name}, expected="cursor") is None


@pytest.mark.parametrize("marker", ["cursor_version", "composer_mode"])
@pytest.mark.parametrize("value", [None, False, ""])
def test_cursor_markers_are_accepted_for_cursor(marker: str, value: object):
    assert foreign_payload_reason({marker: value}, expected="cursor") is None


@pytest.mark.parametrize("event_name", _PASCAL_CASE_EVENTS)
def test_pascal_case_event_is_foreign_for_cursor(event_name: str):
    reason = foreign_payload_reason({"hook_event_name": event_name}, "cursor")

    assert isinstance(reason, str)
    assert reason
    assert event_name in reason


@pytest.mark.parametrize("expected", ["claude", "codex"])
@pytest.mark.parametrize("event_name", _PASCAL_CASE_EVENTS)
def test_pascal_case_event_is_accepted_for_claude_and_codex(
    expected: str,
    event_name: str,
):
    assert foreign_payload_reason({"hook_event_name": event_name}, expected) is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"hook_event_name": ""},
        {"hook_event_name": None},
        {"hook_event_name": False},
        {"hook_event_name": 1},
        {"hook_event_name": 123},
        {"hook_event_name": []},
        {"hook_event_name": {}},
        {"hook_event_name": " event"},
        {"hook_event_name": "1stEvent"},
        {"hook_event_name": "_privateEvent"},
        {"event_name": "beforeSubmitPrompt"},
        {"hookEventName": "beforeSubmitPrompt"},
        {"eventName": "SessionStart"},
        {"cursorVersion": "1.0.0"},
        {"unexpected": "SessionStart"},
        {"unexpected": "beforeSubmitPrompt"},
    ],
)
@pytest.mark.parametrize("expected", ["claude", "codex", "cursor"])
def test_missing_malformed_and_unknown_evidence_fails_open(
    payload: dict[str, Any],
    expected: str,
):
    assert foreign_payload_reason(payload, expected) is None


@pytest.mark.parametrize("expected", ["", "Cursor", "claude-code", "unknown"])
def test_unrecognized_expected_platform_alias_fails_open(expected: str):
    payload = {
        "hook_event_name": "beforeSubmitPrompt",
        "cursor_version": "1.0.0",
    }

    assert foreign_payload_reason(payload, expected) is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"cursor_version": None}, "claude"),
        ({"composer_mode": False}, "codex"),
        ({"composer_mode": "agent"}, "codex"),
        ({"hook_event_name": "beforeSubmitPrompt"}, "claude"),
        ({"hook_event_name": "SessionStart"}, "cursor"),
    ],
)
def test_foreign_reason_is_stable(payload: dict[str, Any], expected: str):
    first = foreign_payload_reason(payload, expected)
    second = foreign_payload_reason(payload, expected)

    assert first == second
    assert isinstance(first, str)
    assert first


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"cursor_version": None, "nested": {"items": [1, 2]}}, "claude"),
        ({"composer_mode": False, "nested": {"items": [1, 2]}}, "codex"),
        (
            {
                "hook_event_name": "beforeSubmitPrompt",
                "nested": {"items": [1, 2]},
            },
            "claude",
        ),
        (
            {"hook_event_name": "SessionStart", "nested": {"items": [1, 2]}},
            "cursor",
        ),
        (
            {"hook_event_name": "SessionStart", "nested": {"items": [1, 2]}},
            "codex",
        ),
        ({"eventName": "unknown", "nested": {"items": [1, 2]}}, "cursor"),
        (
            {
                "hook_event_name": "beforeSubmitPrompt",
                "cursor_version": None,
                "nested": {"items": [1, {"value": False}]},
            },
            "claude",
        ),
        (
            {
                "hook_event_name": "beforeSubmitPrompt",
                "cursor_version": None,
                "nested": {"items": [1, {"value": False}]},
            },
            "codex",
        ),
        (
            {
                "hook_event_name": "beforeSubmitPrompt",
                "cursor_version": None,
                "nested": {"items": [1, {"value": False}]},
            },
            "cursor",
        ),
    ],
)
def test_classification_does_not_mutate_payload(payload: dict[str, Any], expected: str):
    original = deepcopy(payload)

    foreign_payload_reason(payload, expected)

    assert payload == original
