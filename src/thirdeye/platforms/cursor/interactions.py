"""Pure event normalization for Cursor interactions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal

InteractionKind = Literal[
    "user_message",
    "assistant_message",
    "reasoning",
    "tool_call",
    "tool_result",
    "lifecycle",
]

# Reasoning is re-emitted by Cursor with differing model/speed metadata for the
# same thought, so these keys are excluded when comparing payloads.
_IGNORED_KEYS = frozenset({"model", "model_id", "model_params", "speed", "fast"})


@dataclass(frozen=True)
class CanonicalInteraction:
    interaction_id: str
    kind: InteractionKind
    source_type: str
    source_seq: int
    duplicate_seqs: tuple[int, ...]
    ts: str
    generation_id: str
    correlation_id: str
    payload: dict[str, Any]


def canonical_interactions(
    events: Iterable[dict[str, Any]], *, generation_id: str, through_seq: int
) -> list[CanonicalInteraction]:
    """Return ordered interactions for one generation through a sequence."""
    event_kind_map: dict[str, InteractionKind] = {
        "user_message": "user_message",
        "assistant_message": "assistant_message",
        "assistant_thought": "reasoning",
        "tool_call": "tool_call",
        "tool_result": "tool_result",
        "subagent_start": "lifecycle",
        "subagent_message": "lifecycle",
        "turn_stop": "lifecycle",
    }

    interactions_by_seq: dict[int, tuple[CanonicalInteraction, dict[str, Any]]] = {}

    for event in events:
        event_type = event.get("t")
        if event_type not in event_kind_map:
            continue

        seq = event.get("seq")
        ts = event.get("ts")
        data = event.get("data", {})

        event_generation_id = data.get("generation_id") or data.get("generationId")
        if event_generation_id != generation_id:
            continue

        if seq > through_seq:
            continue

        correlation_id = (
            data.get("tool_call_id")
            or data.get("toolCallId")
            or data.get("tool_use_id")
            or data.get("toolUseId")
            or data.get("call_id")
            or data.get("callId")
            or data.get("subagent_id")
            or data.get("subagentId")
            or ""
        )

        kind = event_kind_map[event_type]
        # Copy so the returned payload never aliases the caller's event data.
        payload = dict(data)

        interaction = CanonicalInteraction(
            interaction_id=f"{generation_id}:{kind}:{correlation_id or '-'}:{seq}",
            kind=kind,
            source_type=event_type,
            source_seq=seq,
            duplicate_seqs=(),
            ts=ts,
            generation_id=generation_id,
            correlation_id=correlation_id,
            payload=payload,
        )

        interactions_by_seq[seq] = (interaction, payload)

    # Deduplicate reasoning only: the first sequence is retained and later
    # sequences with an identical dedup key are recorded as duplicates.
    dedup_first_seq: dict[tuple[Any, ...], int] = {}
    duplicates_by_seq: dict[int, list[int]] = {}
    retained: list[CanonicalInteraction] = []

    for seq in sorted(interactions_by_seq):
        interaction, payload = interactions_by_seq[seq]

        if interaction.kind != "reasoning":
            retained.append(interaction)
            continue

        dedup_key = (
            interaction.kind,
            interaction.ts,
            interaction.generation_id,
            interaction.correlation_id,
            _normalized_payload(payload),
        )
        first_seq = dedup_first_seq.get(dedup_key)
        if first_seq is None:
            dedup_first_seq[dedup_key] = seq
            duplicates_by_seq[seq] = []
            retained.append(interaction)
        else:
            duplicates_by_seq[first_seq].append(seq)

    return [
        replace(interaction, duplicate_seqs=tuple(duplicates_by_seq[interaction.source_seq]))
        if duplicates_by_seq.get(interaction.source_seq)
        else interaction
        for interaction in retained
    ]


def _normalized_payload(payload: dict[str, Any]) -> str:
    """Return canonical JSON for a payload with model/speed metadata removed."""
    comparable = {key: value for key, value in payload.items() if key not in _IGNORED_KEYS}
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def interaction_messages(
    interactions: Iterable[CanonicalInteraction], *, before_seq: int | None = None
) -> list[dict[str, Any]]:
    """Return ordered GenAI messages before an exclusive source boundary."""
    messages = []

    for interaction in interactions:
        if before_seq is not None and interaction.source_seq >= before_seq:
            break

        if interaction.kind == "lifecycle":
            continue

        if interaction.kind == "user_message":
            message = _project_user_message(interaction)
            if message:
                messages.append(message)
        elif interaction.kind == "assistant_message":
            message = _project_assistant_message(interaction)
            if message:
                messages.append(message)
        elif interaction.kind == "reasoning":
            message = _project_reasoning(interaction)
            if message:
                messages.append(message)
        elif interaction.kind == "tool_call":
            message = _project_tool_call(interaction)
            if message:
                messages.append(message)
        elif interaction.kind == "tool_result":
            message = _project_tool_result(interaction)
            if message:
                messages.append(message)

    return messages


def _get_first_key(payload: dict[str, Any], keys: list[str]) -> Any:
    """Get the first existing key from a list of alternatives."""
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _get_multiple_keys(payload: dict[str, Any], keys: list[str]) -> Any:
    """Get values for keys, returning single value or dict of multiple values."""
    present_keys = {key: payload[key] for key in keys if key in payload}
    if len(present_keys) == 0:
        return None
    elif len(present_keys) == 1:
        return list(present_keys.values())[0]
    else:
        return present_keys


def _project_user_message(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    text = _get_first_key(interaction.payload, ["prompt", "input", "text"])
    if not text:
        return None
    return {
        "role": "user",
        "parts": [{"type": "text", "content": text}],
    }


def _project_assistant_message(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    text = _get_first_key(interaction.payload, ["text", "response", "output"])
    if not text:
        return None
    return {
        "role": "assistant",
        "parts": [{"type": "text", "content": text}],
    }


def _project_reasoning(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    text = _get_first_key(interaction.payload, ["text", "response", "output"])
    if not text:
        return None
    return {
        "role": "assistant",
        "thirdeye.kind": "reasoning",
        "parts": [{"type": "text", "content": text}],
    }


def _project_tool_call(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    if not interaction.correlation_id:
        return None

    tool_name = _get_first_key(interaction.payload, ["tool_name", "toolName", "name", "tool"])
    if not tool_name:
        return None

    input_keys = ["tool_input", "toolInput", "arguments", "input", "command", "file_path", "filePath", "path"]
    arguments = _get_multiple_keys(interaction.payload, input_keys)

    return {
        "role": "assistant",
        "parts": [
            {
                "type": "tool_call",
                "id": interaction.correlation_id,
                "name": tool_name,
                "arguments": arguments,
            }
        ],
    }


def _project_tool_result(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    if not interaction.correlation_id:
        return None

    output_keys = ["tool_output", "toolOutput", "result", "output", "stdout", "response", "edits", "diff"]
    response = _get_multiple_keys(interaction.payload, output_keys)

    if response is None:
        return None

    return {
        "role": "tool",
        "parts": [
            {
                "type": "tool_call_response",
                "id": interaction.correlation_id,
                "response": response,
            }
        ],
    }
