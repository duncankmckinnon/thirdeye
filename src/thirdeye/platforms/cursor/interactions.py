"""Pure event normalization for Cursor interactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

InteractionKind = Literal[
    "user_message",
    "assistant_message",
    "reasoning",
    "tool_call",
    "tool_result",
    "lifecycle",
]


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
    # Map event type to interaction kind
    event_kind_map = {
        "user_message": "user_message",
        "assistant_message": "assistant_message",
        "assistant_thought": "reasoning",
        "tool_call": "tool_call",
        "tool_result": "tool_result",
        "subagent_start": "lifecycle",
        "subagent_message": "lifecycle",
        "turn_stop": "lifecycle",
    }

    # Process events, creating normalized interactions
    interactions_by_seq: dict[int, tuple[CanonicalInteraction, dict]] = {}

    for event in events:
        # Don't mutate input
        event_type = event.get("t")
        if event_type not in event_kind_map:
            continue

        seq = event.get("seq")
        ts = event.get("ts")
        data = event.get("data", {})

        # Extract generation_id from data
        event_generation_id = data.get("generation_id") or data.get("generationId")
        if event_generation_id != generation_id:
            continue

        # Filter by through_seq
        if seq > through_seq:
            continue

        # Extract correlation_id with fallback chain
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
        source_type = event_type

        # Create interaction_id
        interaction_id = f"{generation_id}:{kind}:{correlation_id or '-'}:{seq}"

        # Make a shallow copy of data to preserve it
        payload = dict(data)

        interaction = CanonicalInteraction(
            interaction_id=interaction_id,
            kind=kind,
            source_type=source_type,
            source_seq=seq,
            duplicate_seqs=(),
            ts=ts,
            generation_id=generation_id,
            correlation_id=correlation_id,
            payload=payload,
        )

        interactions_by_seq[seq] = (interaction, payload)

    # Deduplicate reasoning only
    if interactions_by_seq:
        # Build a map of deduplication keys to (interaction, first_seq, duplicate_seqs)
        dedup_map: dict[tuple, tuple[CanonicalInteraction, int, list[int]]] = {}

        for seq in sorted(interactions_by_seq.keys()):
            interaction, original_payload = interactions_by_seq[seq]

            if interaction.kind == "reasoning":
                # Create dedup key for reasoning
                # Remove model, model_id, model_params, speed, fast
                dedup_payload = {
                    k: v
                    for k, v in original_payload.items()
                    if k not in ("model", "model_id", "model_params", "speed", "fast")
                }

                # Use canonical JSON for comparison
                normalized = json.dumps(
                    dedup_payload, sort_keys=True, separators=(",", ":")
                )
                dedup_key = (
                    interaction.kind,
                    interaction.ts,
                    interaction.generation_id,
                    interaction.correlation_id,
                    normalized,
                )

                if dedup_key not in dedup_map:
                    dedup_map[dedup_key] = (interaction, seq, [])
                else:
                    # This is a duplicate
                    existing_interaction, first_seq, duplicates = dedup_map[dedup_key]
                    duplicates.append(seq)
            else:
                # Non-reasoning: always unique
                dedup_key = (interaction.kind, seq)  # Unique key per non-reasoning
                dedup_map[dedup_key] = (interaction, seq, [])

        # Build result with duplicates updated
        result_dict = {}
        for dedup_key, (interaction, first_seq, duplicates) in dedup_map.items():
            if interaction.kind == "reasoning" and duplicates:
                # Update the first interaction with duplicates
                updated = CanonicalInteraction(
                    interaction_id=interaction.interaction_id,
                    kind=interaction.kind,
                    source_type=interaction.source_type,
                    source_seq=first_seq,
                    duplicate_seqs=tuple(sorted(duplicates)),
                    ts=interaction.ts,
                    generation_id=interaction.generation_id,
                    correlation_id=interaction.correlation_id,
                    payload=interactions_by_seq[first_seq][1],
                )
                result_dict[first_seq] = updated
            elif interaction.kind != "reasoning" or not duplicates:
                result_dict[first_seq] = interaction

        # Return sorted by sequence
        return [result_dict[seq] for seq in sorted(result_dict.keys())]

    return []
