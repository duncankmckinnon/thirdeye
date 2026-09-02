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
