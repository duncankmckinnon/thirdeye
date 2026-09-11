"""Shared helpers to seed Copilot V2 projections for turn and usage view tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection_state import empty_projection_state
from thirdeye.platforms.copilot.projection_store import commit_projection
from thirdeye.platforms.copilot.types import Projection, SourceBatch, SourceRecord
from thirdeye.usage.types import UsageRow

NATIVE_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
INTERACTION_ONE = "6d2b89fd-a653-430c-b532-b0936d72eb42"
INTERACTION_TWO = "7e3c90ae-b764-541d-c643-c1047e83fc53"
TURN_ONE_ID = (
    "copilot:turn:fixture:session:"
    "6d2b89fd-a653-430c-b532-b0936d72eb42"
)
TURN_TWO_ID = (
    "copilot:turn:fixture:session:"
    "7e3c90ae-b764-541d-c643-c1047e83fc53"
)
USAGE_MODEL_ONE = "gpt-5.6-luna"
USAGE_MODEL_TWO = "gpt-4.1-copilot-sentinel"
USAGE_TOKENS_ONE = 1111
USAGE_TOKENS_TWO = 2222


def _transcript_record(
    paths_source_key: str,
    event_id: str,
    *,
    content: str,
    ts: str,
    interaction_id: str,
    message_type: str = "user.message",
    agent_id: str | None = None,
) -> SourceRecord:
    payload: dict[str, Any] = {
        "type": message_type,
        "id": event_id,
        "timestamp": ts,
        "data": {
            "content": content,
            "interactionId": interaction_id,
            "turnId": "0",
        },
    }
    if agent_id is not None:
        payload["agentId"] = agent_id
    return {
        "source_id": f"{paths_source_key}/{NATIVE_ID}/{event_id}",
        "source_kind": "transcript",
        "native_session_id": NATIVE_ID,
        "ts": ts,
        "observed_at": ts,
        "payload": payload,
        "locator": {"file": "events.jsonl", "file_generation": "fixture-gen", "byte_offset": 0},
    }


def _main_turn(
    *,
    turn_id: str,
    interaction_id: str,
    source_ids: list[str],
    start_ts: str,
    end_ts: str,
    subagents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "input_message": "prompt",
        "output_message": "done",
        "status": "completed",
        "llm_calls": [],
        "permission_requests": [],
        "subagents": subagents or [],
        "attributes": {"interaction_id": interaction_id},
        "source_ids": source_ids,
    }


def _usage_row(
    *,
    stored_id: str,
    call_id: str,
    input_tokens: int,
    response_model: str,
    ts: str,
) -> UsageRow:
    return UsageRow(
        session_id=stored_id,
        seq=0,
        call_id=call_id,
        ts=ts,
        platform=PLATFORM_NAME,
        provider_name="openai",
        response_model=response_model,
        input_tokens=input_tokens,
        output_tokens=10,
    )


def _attribution(*, logical_call_id: str, usage_source_id: str, stored_turn_id: str) -> dict[str, Any]:
    return {
        "usage_source_id": usage_source_id,
        "logical_call_id": logical_call_id,
        "stored_turn_id": stored_turn_id,
        "agent_id": None,
        "call_id": None,
        "status": "matched",
        "join_kind": "direct",
        "evidence": ["direct:usage_source_id"],
    }


def seed_two_main_interaction_projection(config: Config, tmp_path: Path) -> str:
    """Archive evidence and commit a projection with two completed main user turns."""
    paths = resolve_sources(tmp_path / "copilot-home")
    source_key = paths["source_key"]
    ts1 = "2026-09-10T17:08:22.203Z"
    ts2 = "2026-09-10T17:08:24.503Z"
    ts3 = "2026-09-10T17:08:25.503Z"
    ts4 = "2026-09-10T17:08:40.000Z"
    ts5 = "2026-09-10T17:08:42.000Z"
    user1 = "user-one"
    asst1 = "assistant-one"
    child1 = "child-one"
    user2 = "user-two"
    asst2 = "assistant-two"
    records = [
        _transcript_record(
            source_key,
            user1,
            content="Read alpha.txt and beta.txt with separate view calls in parallel.",
            ts=ts1,
            interaction_id=INTERACTION_ONE,
        ),
        _transcript_record(
            source_key,
            asst1,
            content="I'll read both files now.",
            ts=ts2,
            interaction_id=INTERACTION_ONE,
            message_type="assistant.message",
        ),
        _transcript_record(
            source_key,
            child1,
            content="Explore child reading alpha.txt for the parent task.",
            ts=ts3,
            interaction_id=INTERACTION_ONE,
            message_type="assistant.message",
            agent_id="child-agent-id",
        ),
        _transcript_record(
            source_key,
            user2,
            content="Report the final sum only.",
            ts=ts4,
            interaction_id=INTERACTION_TWO,
        ),
        _transcript_record(
            source_key,
            asst2,
            content="The final sum is 100.",
            ts=ts5,
            interaction_id=INTERACTION_TWO,
            message_type="assistant.message",
        ),
    ]
    batch: SourceBatch = {
        "source_key": source_key,
        "native_session_id": NATIVE_ID,
        "cwd": "/fixture/workspace",
        "records": records,
        "next_cursor": {"fixture": 1},
        "diagnostics": [],
    }
    commit_batch(config, paths, batch)
    stored_id = stored_session_id(paths, NATIVE_ID)

    turn_one_sources = [
        f"{source_key}/{NATIVE_ID}/{user1}",
        f"{source_key}/{NATIVE_ID}/{asst1}",
        f"{source_key}/{NATIVE_ID}/{child1}",
    ]
    turn_two_sources = [
        f"{source_key}/{NATIVE_ID}/{user2}",
        f"{source_key}/{NATIVE_ID}/{asst2}",
    ]
    turn_one = _main_turn(
        turn_id=TURN_ONE_ID,
        interaction_id=INTERACTION_ONE,
        source_ids=turn_one_sources,
        start_ts=ts1,
        end_ts=ts3,
        subagents=[
            {
                "turn_id": "copilot:turn:fixture:session:child",
                "start_ts": ts3,
                "end_ts": ts3,
                "input_message": "explore",
                "output_message": "done",
                "status": "completed",
                "llm_calls": [],
                "permission_requests": [],
                "subagents": [],
                "attributes": {
                    "interaction_id": INTERACTION_ONE,
                    "agent_id": "child-agent-id",
                },
                "source_ids": [f"{source_key}/{NATIVE_ID}/{child1}"],
            }
        ],
    )
    turn_two = _main_turn(
        turn_id=TURN_TWO_ID,
        interaction_id=INTERACTION_TWO,
        source_ids=turn_two_sources,
        start_ts=ts4,
        end_ts=ts5,
    )
    usage_one = _usage_row(
        stored_id=stored_id,
        call_id="usage-source-one",
        input_tokens=USAGE_TOKENS_ONE,
        response_model=USAGE_MODEL_ONE,
        ts=ts2,
    )
    usage_two = _usage_row(
        stored_id=stored_id,
        call_id="usage-source-two",
        input_tokens=USAGE_TOKENS_TWO,
        response_model=USAGE_MODEL_TWO,
        ts=ts5,
    )
    projection: Projection = {
        "normalized_events": [],
        "turns": [turn_one, turn_two],
        "usage_rows": [usage_one, usage_two],
        "attributions": [
            _attribution(
                logical_call_id="logical-call-one",
                usage_source_id="usage-source-one",
                stored_turn_id=TURN_ONE_ID,
            ),
            _attribution(
                logical_call_id="logical-call-two",
                usage_source_id="usage-source-two",
                stored_turn_id=TURN_TWO_ID,
            ),
        ],
        "pending": [],
        "diagnostics": [],
    }
    commit_projection(config, stored_id, projection, empty_projection_state())
    return stored_id
