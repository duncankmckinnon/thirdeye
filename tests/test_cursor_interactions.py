from __future__ import annotations

import copy
from typing import Any

import pytest

from thirdeye.platforms.cursor.interactions import (
    CanonicalInteraction,
    InteractionKind,
    canonical_interactions,
    interaction_messages,
)

GENERATION = "gen-abc"
TS = "2026-09-02T12:00:00.000Z"


def _event(
    seq: int,
    event_type: str,
    *,
    ts: str = TS,
    generation_id: str = GENERATION,
    **data,
) -> dict:
    return {
        "seq": seq,
        "t": event_type,
        "ts": ts,
        "data": {"generation_id": generation_id, **data},
    }


def _canonicalize(
    events: list[dict],
    *,
    generation_id: str = GENERATION,
    through_seq: int | None = None,
) -> list[CanonicalInteraction]:
    if through_seq is None:
        through_seq = max((event["seq"] for event in events), default=0)
    return canonical_interactions(events, generation_id=generation_id, through_seq=through_seq)


def test_maps_user_message_to_user_message_kind():
    events = [_event(0, "user_message", prompt="hello")]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].kind == "user_message"
    assert result[0].source_type == "user_message"
    assert result[0].payload == events[0]["data"]


def test_maps_assistant_message_to_assistant_message_kind():
    events = [_event(0, "assistant_message", text="done")]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].kind == "assistant_message"
    assert result[0].source_type == "assistant_message"
    assert result[0].payload == events[0]["data"]


def test_maps_assistant_thought_to_reasoning_kind():
    events = [_event(0, "assistant_thought", text="thinking")]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].kind == "reasoning"
    assert result[0].source_type == "assistant_thought"
    assert result[0].payload == events[0]["data"]


def test_maps_tool_call_to_tool_call_kind():
    events = [_event(0, "tool_call", tool_name="shell", tool_use_id="call-1")]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].kind == "tool_call"
    assert result[0].source_type == "tool_call"
    assert result[0].payload == events[0]["data"]


def test_maps_tool_result_to_tool_result_kind():
    events = [_event(0, "tool_result", tool_name="shell", tool_use_id="call-1", output="ok")]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].kind == "tool_result"
    assert result[0].source_type == "tool_result"
    assert result[0].payload == events[0]["data"]


@pytest.mark.parametrize(
    "event_type",
    ["subagent_start", "subagent_message", "turn_stop"],
)
def test_maps_lifecycle_event_types_to_lifecycle_kind(event_type: str):
    events = [_event(0, event_type, subagent_id="child-1")]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].kind == "lifecycle"
    assert result[0].source_type == event_type
    assert result[0].payload == events[0]["data"]


@pytest.mark.parametrize(
    "event_type",
    ["session_start", "session_end", "notification", "error"],
)
def test_excludes_unmapped_event_types(event_type: str):
    events = [_event(0, event_type, detail="ignored")]
    result = _canonicalize(events)

    assert result == []


def test_filters_to_requested_generation_id():
    events = [
        _event(0, "user_message", generation_id=GENERATION, prompt="keep"),
        _event(1, "user_message", generation_id="other-gen", prompt="drop"),
    ]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].payload["prompt"] == "keep"
    assert result[0].generation_id == GENERATION


def test_reads_generation_id_from_generation_id_camel_case():
    events = [
        {
            "seq": 0,
            "t": "user_message",
            "ts": TS,
            "data": {"generationId": GENERATION, "prompt": "camel"},
        }
    ]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].payload["prompt"] == "camel"


def test_filters_to_through_seq_inclusive():
    events = [
        _event(0, "user_message", prompt="first"),
        _event(1, "assistant_message", text="second"),
        _event(2, "tool_call", tool_name="Read", tool_use_id="call-1"),
    ]
    result = _canonicalize(events, through_seq=1)

    assert [item.source_seq for item in result] == [0, 1]


def test_reads_correlation_id_from_tool_call_id_first():
    events = [
        _event(
            0,
            "tool_call",
            tool_call_id="primary",
            toolCallId="secondary",
            tool_use_id="tertiary",
            subagent_id="quaternary",
        )
    ]
    result = _canonicalize(events)

    assert result[0].correlation_id == "primary"
    assert result[0].interaction_id == f"{GENERATION}:tool_call:primary:0"


def test_reads_correlation_id_from_tool_call_id_camel_case():
    events = [
        {
            "seq": 0,
            "t": "tool_call",
            "ts": TS,
            "data": {
                "generation_id": GENERATION,
                "toolCallId": "camel-call",
                "toolUseId": "ignored",
            },
        }
    ]
    result = _canonicalize(events)

    assert result[0].correlation_id == "camel-call"


def test_reads_correlation_id_from_tool_use_id_when_call_id_missing():
    events = [_event(0, "tool_call", tool_use_id="use-1", subagent_id="agent-1")]
    result = _canonicalize(events)

    assert result[0].correlation_id == "use-1"


def test_reads_correlation_id_from_call_id_before_subagent_id():
    events = [_event(0, "tool_call", call_id="call-1", subagent_id="agent-1")]
    result = _canonicalize(events)

    assert result[0].correlation_id == "call-1"


def test_reads_correlation_id_from_call_id_camel_case():
    events = [
        {
            "seq": 0,
            "t": "tool_call",
            "ts": TS,
            "data": {
                "generation_id": GENERATION,
                "callId": "camel-call",
                "subagentId": "ignored",
            },
        }
    ]
    result = _canonicalize(events)

    assert result[0].correlation_id == "camel-call"


def test_reads_correlation_id_from_tool_use_id_camel_case():
    events = [
        {
            "seq": 0,
            "t": "tool_call",
            "ts": TS,
            "data": {
                "generation_id": GENERATION,
                "toolUseId": "camel-use",
                "subagentId": "ignored",
            },
        }
    ]
    result = _canonicalize(events)

    assert result[0].correlation_id == "camel-use"


def test_reads_correlation_id_from_subagent_id_when_no_tool_ids():
    events = [_event(0, "subagent_start", subagent_id="child-1")]
    result = _canonicalize(events)

    assert result[0].correlation_id == "child-1"


def test_reads_correlation_id_from_subagent_id_camel_case():
    events = [
        {
            "seq": 0,
            "t": "subagent_start",
            "ts": TS,
            "data": {"generationId": GENERATION, "subagentId": "camel-child"},
        }
    ]
    result = _canonicalize(events)

    assert result[0].correlation_id == "camel-child"


def test_uses_dash_in_interaction_id_when_correlation_id_missing():
    events = [_event(0, "user_message", prompt="hello")]
    result = _canonicalize(events)

    assert result[0].correlation_id == ""
    assert result[0].interaction_id == f"{GENERATION}:user_message:-:0"


def test_returns_empty_list_for_no_matching_events():
    assert _canonicalize([]) == []
    assert _canonicalize([_event(0, "session_start", detail="ignored")]) == []


def test_excludes_events_missing_generation_id():
    events = [
        {
            "seq": 0,
            "t": "user_message",
            "ts": TS,
            "data": {"prompt": "no generation"},
        }
    ]
    result = _canonicalize(events)

    assert result == []


def test_deduplicates_reasoning_with_same_timestamp_despite_model_metadata():
    events = [
        _event(
            0,
            "assistant_thought",
            text="plan",
            model="claude-4",
            speed="fast",
        ),
        _event(
            1,
            "assistant_thought",
            text="plan",
            model="gpt-5",
            model_id="gpt-5.6",
            model_params={"temperature": 0.2},
            speed="slow",
            fast=True,
        ),
    ]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].source_seq == 0
    assert result[0].duplicate_seqs == (1,)


def test_retains_complete_first_reasoning_payload_after_deduplication():
    first_payload = {
        "generation_id": GENERATION,
        "text": "plan",
        "model": "claude-4",
        "speed": "fast",
        "extra": {"nested": [1, 2]},
    }
    events = [
        {"seq": 0, "t": "assistant_thought", "ts": TS, "data": copy.deepcopy(first_payload)},
        {
            "seq": 1,
            "t": "assistant_thought",
            "ts": TS,
            "data": {
                "generation_id": GENERATION,
                "text": "plan",
                "model": "gpt-5",
            },
        },
    ]
    result = _canonicalize(events)

    assert result[0].payload == first_payload


def test_keeps_same_reasoning_text_at_different_timestamps_separate():
    events = [
        _event(0, "assistant_thought", ts="2026-09-02T12:00:00.000Z", text="plan"),
        _event(1, "assistant_thought", ts="2026-09-02T12:00:01.000Z", text="plan"),
    ]
    result = _canonicalize(events)

    assert len(result) == 2
    assert result[0].duplicate_seqs == ()
    assert result[1].duplicate_seqs == ()


def test_keeps_same_reasoning_text_within_same_second_separate():
    events = [
        _event(0, "assistant_thought", ts="2026-09-02T12:00:00.000Z", text="plan"),
        _event(1, "assistant_thought", ts="2026-09-02T12:00:00.900Z", text="plan"),
    ]
    result = _canonicalize(events)

    assert len(result) == 2
    assert result[0].duplicate_seqs == ()
    assert result[1].duplicate_seqs == ()


def test_records_multiple_reasoning_duplicate_sequences():
    events = [
        _event(0, "assistant_thought", text="plan", model="a"),
        _event(1, "assistant_thought", text="plan", model="b"),
        _event(2, "assistant_thought", text="plan", model="c"),
    ]
    result = _canonicalize(events)

    assert len(result) == 1
    assert result[0].source_seq == 0
    assert result[0].duplicate_seqs == (1, 2)


def test_keeps_same_reasoning_with_distinct_correlation_ids_separate():
    events = [
        _event(0, "assistant_thought", text="plan", tool_use_id="call-a"),
        _event(1, "assistant_thought", text="plan", tool_use_id="call-b"),
    ]
    result = _canonicalize(events)

    assert len(result) == 2
    assert result[0].correlation_id == "call-a"
    assert result[1].correlation_id == "call-b"


def test_does_not_deduplicate_non_reasoning_interactions():
    events = [
        _event(0, "user_message", prompt="same"),
        _event(1, "user_message", prompt="same"),
        _event(2, "tool_call", tool_name="Read", tool_use_id="call-1"),
        _event(3, "tool_call", tool_name="Read", tool_use_id="call-1"),
    ]
    result = _canonicalize(events)

    assert len(result) == 4
    assert all(item.duplicate_seqs == () for item in result)


def test_returns_interactions_in_ascending_source_sequence():
    events = [
        _event(2, "tool_result", tool_use_id="call-1", output="ok"),
        _event(0, "user_message", prompt="start"),
        _event(1, "tool_call", tool_use_id="call-1", tool_name="Read"),
    ]
    result = _canonicalize(events)

    assert [item.source_seq for item in result] == [0, 1, 2]


def test_does_not_mutate_input_events():
    original = [
        _event(0, "user_message", prompt="hello"),
        _event(1, "assistant_thought", text="think"),
    ]
    snapshot = copy.deepcopy(original)
    _canonicalize(original)

    assert original == snapshot


def test_populates_canonical_interaction_metadata_from_event():
    events = [
        _event(
            7,
            "tool_call",
            ts="2026-09-02T12:07:00.000Z",
            tool_use_id="call-7",
            tool_name="shell",
        )
    ]
    result = _canonicalize(events)

    item = result[0]
    assert item.interaction_id == f"{GENERATION}:tool_call:call-7:7"
    assert item.source_seq == 7
    assert item.ts == "2026-09-02T12:07:00.000Z"
    assert item.generation_id == GENERATION
    assert item.duplicate_seqs == ()


def _messages(
    events: list[dict],
    *,
    generation_id: str = GENERATION,
    through_seq: int | None = None,
    before_seq: int | None = None,
) -> list[dict[str, Any]]:
    interactions = _canonicalize(events, generation_id=generation_id, through_seq=through_seq)
    return interaction_messages(interactions, before_seq=before_seq)


def _make_interaction(
    seq: int,
    kind: InteractionKind,
    payload: dict[str, Any],
    *,
    correlation_id: str = "",
    generation_id: str = GENERATION,
) -> CanonicalInteraction:
    return CanonicalInteraction(
        interaction_id=f"{generation_id}:{kind}:{correlation_id or '-'}:{seq}",
        kind=kind,
        source_type=kind,
        source_seq=seq,
        duplicate_seqs=(),
        ts=TS,
        generation_id=generation_id,
        correlation_id=correlation_id,
        payload={"generation_id": generation_id, **payload},
    )


def test_projects_user_message_from_prompt():
    events = [_event(0, "user_message", prompt="hello")]
    assert _messages(events) == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
    ]


@pytest.mark.parametrize("key", ["input", "text"])
def test_projects_user_message_from_alternate_text_keys(key: str):
    events = [_event(0, "user_message", **{key: "hello"})]
    assert _messages(events) == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
    ]


def test_user_message_text_lookup_prefers_prompt_over_input_and_text():
    events = [_event(0, "user_message", prompt="first", input="second", text="third")]
    assert _messages(events) == [
        {"role": "user", "parts": [{"type": "text", "content": "first"}]},
    ]


def test_projects_assistant_message_from_text():
    events = [_event(0, "assistant_message", text="done")]
    assert _messages(events) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "done"}]},
    ]


@pytest.mark.parametrize("key", ["response", "output"])
def test_projects_assistant_message_from_alternate_text_keys(key: str):
    events = [_event(0, "assistant_message", **{key: "done"})]
    assert _messages(events) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "done"}]},
    ]


def test_assistant_message_text_lookup_prefers_text_over_response_and_output():
    events = [_event(0, "assistant_message", text="first", response="second", output="third")]
    assert _messages(events) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "first"}]},
    ]


def test_projects_reasoning_as_assistant_text_with_thirdeye_kind():
    events = [_event(0, "assistant_thought", text="thinking")]
    assert _messages(events) == [
        {
            "role": "assistant",
            "thirdeye.kind": "reasoning",
            "parts": [{"type": "text", "content": "thinking"}],
        },
    ]


def test_projects_tool_call_with_id_name_and_arguments():
    events = [
        _event(
            0,
            "tool_call",
            tool_use_id="call-1",
            tool_name="shell",
            command="pytest -q",
        )
    ]
    assert _messages(events) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "shell",
                    "arguments": "pytest -q",
                }
            ],
        },
    ]


@pytest.mark.parametrize("key", ["toolName", "name", "tool"])
def test_tool_call_name_lookup_uses_alternate_keys(key: str):
    events = [
        _event(0, "tool_call", tool_use_id="call-1", **{key: "Read"}, path="/a.py"),
    ]
    assert _messages(events)[0]["parts"][0]["name"] == "Read"


def test_tool_call_name_lookup_prefers_tool_name_over_alternate_keys():
    events = [
        _event(
            0,
            "tool_call",
            tool_use_id="call-1",
            tool_name="shell",
            toolName="ignored",
            name="ignored",
            tool="ignored",
        )
    ]
    assert _messages(events)[0]["parts"][0]["name"] == "shell"


@pytest.mark.parametrize(
    "key",
    ["tool_input", "toolInput", "arguments", "input", "command", "file_path", "filePath", "path"],
)
def test_tool_call_arguments_from_alternate_input_keys(key: str):
    value = (
        {"nested": [1, 2]}
        if key in {"tool_input", "toolInput", "arguments", "input"}
        else "/tmp/file.py"
    )
    events = [
        _event(0, "tool_call", tool_use_id="call-1", tool_name="Read", **{key: value}),
    ]
    assert _messages(events)[0]["parts"][0]["arguments"] == value


def test_tool_call_arguments_returns_dict_when_multiple_input_keys_present():
    payload = {
        "command": "pytest -q",
        "file_path": "/repo/tests/test_cursor_interactions.py",
        "input": {"line": 1},
    }
    events = [_event(0, "tool_call", tool_use_id="call-1", tool_name="shell", **payload)]
    assert _messages(events)[0]["parts"][0]["arguments"] == payload


def test_tool_call_arguments_preserves_structures_without_stringifying():
    nested = {"items": [{"id": 1}, {"id": 2}], "enabled": True}
    events = [
        _event(0, "tool_call", tool_use_id="call-1", tool_name="Task", arguments=nested),
    ]
    assert _messages(events)[0]["parts"][0]["arguments"] == nested


def test_projects_tool_result_with_tool_call_response_part():
    events = [
        _event(
            0,
            "tool_result",
            tool_use_id="call-1",
            tool_name="shell",
            output="passed",
        )
    ]
    assert _messages(events) == [
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": "passed",
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    "key",
    ["tool_output", "toolOutput", "result", "stdout", "response", "edits", "diff"],
)
def test_tool_result_response_from_alternate_output_keys(key: str):
    value = {"lines": ["+added"]} if key == "edits" else "ok"
    events = [_event(0, "tool_result", tool_use_id="call-1", **{key: value})]
    assert _messages(events)[0]["parts"][0]["response"] == value


def test_tool_result_response_returns_dict_when_multiple_output_keys_present():
    payload = {
        "stdout": "line one\n",
        "stderr": "ignored",
        "result": {"exit_code": 0},
        "output": "line one\n",
    }
    events = [_event(0, "tool_result", tool_use_id="call-1", **payload)]
    assert _messages(events)[0]["parts"][0]["response"] == {
        "stdout": "line one\n",
        "result": {"exit_code": 0},
        "output": "line one\n",
    }


def test_tool_result_response_preserves_structures_without_stringifying():
    structured = {"files": [{"path": "a.py", "diff": "---\n+++"}]}
    events = [_event(0, "tool_result", tool_use_id="call-1", edits=structured)]
    assert _messages(events)[0]["parts"][0]["response"] == structured


@pytest.mark.parametrize("event_type", ["subagent_start", "subagent_message", "turn_stop"])
def test_lifecycle_interactions_are_excluded_from_messages(event_type: str):
    events = [
        _event(0, "user_message", prompt="start"),
        _event(1, event_type, subagent_id="child-1"),
        _event(2, "assistant_message", text="done"),
    ]
    assert _messages(events) == [
        {"role": "user", "parts": [{"type": "text", "content": "start"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "done"}]},
    ]


def test_before_seq_excludes_matching_and_later_sequences():
    events = [
        _event(0, "user_message", prompt="first"),
        _event(1, "assistant_message", text="second"),
        _event(2, "tool_call", tool_use_id="call-1", tool_name="Read", path="/a.py"),
        _event(3, "tool_result", tool_use_id="call-1", output="contents"),
    ]
    assert _messages(events, before_seq=2) == [
        {"role": "user", "parts": [{"type": "text", "content": "first"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "second"}]},
    ]


def test_before_seq_none_includes_all_projectable_interactions():
    events = [
        _event(0, "user_message", prompt="only"),
        _event(1, "turn_stop", subagent_id="child-1"),
    ]
    assert _messages(events, before_seq=None) == [
        {"role": "user", "parts": [{"type": "text", "content": "only"}]},
    ]


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("user_message", {}),
        ("user_message", {"detail": "no text keys"}),
        ("assistant_message", {}),
        ("reasoning", {}),
    ],
)
def test_skips_text_interactions_missing_required_text(kind: str, payload: dict):
    event_type = {
        "user_message": "user_message",
        "assistant_message": "assistant_message",
        "reasoning": "assistant_thought",
    }[kind]
    events = [_event(0, event_type, **payload)]
    assert _messages(events) == []


@pytest.mark.parametrize("kind", ["tool_call", "tool_result"])
def test_skips_tool_interactions_missing_correlation_id(kind: str):
    event_type = kind
    payload = {"tool_name": "shell", "command": "ls"} if kind == "tool_call" else {"output": "ok"}
    interaction = _make_interaction(0, kind, payload, correlation_id="")
    assert interaction_messages([interaction]) == []


def test_skips_tool_call_missing_tool_name():
    interaction = _make_interaction(
        0,
        "tool_call",
        {"command": "ls"},
        correlation_id="call-1",
    )
    assert interaction_messages([interaction]) == []


def test_projects_tool_result_without_output_keys_as_null_response():
    interaction = _make_interaction(
        0,
        "tool_result",
        {"tool_name": "shell", "stderr": "ignored"},
        correlation_id="call-1",
    )
    assert interaction_messages([interaction]) == [
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": None,
                }
            ],
        },
    ]


def test_projects_tool_result_with_explicitly_null_output():
    events = [_event(0, "tool_result", tool_use_id="call-1", output=None)]
    assert _messages(events) == [
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": None,
                }
            ],
        },
    ]


def test_projects_tool_result_with_empty_string_output():
    events = [_event(0, "tool_result", tool_use_id="call-1", output="")]
    assert _messages(events) == [
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": "",
                }
            ],
        },
    ]


def test_projects_tool_call_without_input_keys_as_null_arguments():
    events = [_event(0, "tool_call", tool_use_id="call-1", tool_name="TodoWrite")]
    assert _messages(events) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "TodoWrite",
                    "arguments": None,
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    "event_type,payload,expected",
    [
        (
            "user_message",
            {"prompt": ""},
            {"role": "user", "parts": [{"type": "text", "content": ""}]},
        ),
        (
            "assistant_message",
            {"text": ""},
            {"role": "assistant", "parts": [{"type": "text", "content": ""}]},
        ),
        (
            "assistant_thought",
            {"text": ""},
            {
                "role": "assistant",
                "thirdeye.kind": "reasoning",
                "parts": [{"type": "text", "content": ""}],
            },
        ),
    ],
)
def test_projects_empty_string_text_instead_of_skipping(
    event_type: str, payload: dict, expected: dict
):
    events = [_event(0, event_type, **payload)]
    assert _messages(events) == [expected]


@pytest.mark.parametrize(
    "event_type,payload,expected_role",
    [
        ("user_message", {"prompt": None, "input": "hello"}, "user"),
        ("assistant_message", {"text": None, "response": "hello"}, "assistant"),
    ],
)
def test_text_lookup_falls_through_keys_present_with_null_values(
    event_type: str, payload: dict, expected_role: str
):
    events = [_event(0, event_type, **payload)]
    assert _messages(events) == [
        {"role": expected_role, "parts": [{"type": "text", "content": "hello"}]},
    ]


def test_skips_text_interaction_when_every_text_key_is_null():
    events = [_event(0, "user_message", prompt=None, input=None, text=None)]
    assert _messages(events) == []


def test_tool_call_name_lookup_falls_through_keys_present_with_null_values():
    events = [
        _event(0, "tool_call", tool_use_id="call-1", tool_name=None, toolName="Read", path="/a.py")
    ]
    assert _messages(events)[0]["parts"][0]["name"] == "Read"


@pytest.mark.parametrize("key", ["response", "output"])
def test_projects_reasoning_from_alternate_text_keys(key: str):
    events = [_event(0, "assistant_thought", **{key: "thinking"})]
    assert _messages(events) == [
        {
            "role": "assistant",
            "thirdeye.kind": "reasoning",
            "parts": [{"type": "text", "content": "thinking"}],
        },
    ]


def test_preserves_interaction_order_in_projected_messages():
    events = [
        _event(0, "user_message", prompt="start"),
        _event(1, "assistant_thought", text="plan"),
        _event(2, "tool_call", tool_use_id="call-1", tool_name="Read", path="/a.py"),
        _event(3, "tool_result", tool_use_id="call-1", output="contents"),
        _event(4, "assistant_message", text="done"),
    ]
    assert _messages(events) == [
        {"role": "user", "parts": [{"type": "text", "content": "start"}]},
        {
            "role": "assistant",
            "thirdeye.kind": "reasoning",
            "parts": [{"type": "text", "content": "plan"}],
        },
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "Read",
                    "arguments": "/a.py",
                }
            ],
        },
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": "contents",
                }
            ],
        },
        {"role": "assistant", "parts": [{"type": "text", "content": "done"}]},
    ]


def test_projects_direct_canonical_interactions_without_events():
    interactions = [
        _make_interaction(0, "user_message", {"prompt": "hi"}),
        _make_interaction(
            1,
            "tool_call",
            {"tool_name": "shell", "command": "ls"},
            correlation_id="call-9",
        ),
    ]
    assert interaction_messages(interactions) == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]},
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-9",
                    "name": "shell",
                    "arguments": "ls",
                }
            ],
        },
    ]
