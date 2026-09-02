from __future__ import annotations

import copy

import pytest

from thirdeye.platforms.cursor.interactions import (
    CanonicalInteraction,
    canonical_interactions,
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
