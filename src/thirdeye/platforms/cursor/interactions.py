"""Pure event normalization for Cursor interactions."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
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

_USER_TEXT_KEYS = ("prompt", "input", "text")
_ASSISTANT_TEXT_KEYS = ("text", "response", "output")
_TOOL_NAME_KEYS = ("tool_name", "toolName", "name", "tool")
_TOOL_INPUT_KEYS = (
    "tool_input",
    "toolInput",
    "arguments",
    "input",
    "command",
    "file_path",
    "filePath",
    "path",
)
_TOOL_OUTPUT_KEYS = (
    "tool_output",
    "toolOutput",
    "result",
    "output",
    "stdout",
    "response",
    "edits",
    "diff",
)

# Sentinel separating "no matching key in the payload" from a key that is
# present with a null value; the latter is captured content and is preserved.
_MISSING = object()


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
    messages: list[dict[str, Any]] = []

    for interaction in interactions:
        if before_seq is not None and interaction.source_seq >= before_seq:
            break

        # Lifecycle interactions have no projector: they get spans, not messages.
        projector = _PROJECTORS.get(interaction.kind)
        if projector is None:
            continue

        message = projector(interaction)
        if message is not None:
            messages.append(message)

    return messages


def _first_text(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first key's value that carries content, else `_MISSING`.

    A key present with a `None` value carries no content, so the lookup falls
    through to the next alternative. Every other value is returned verbatim,
    including an empty string, which is captured text rather than a missing key.
    """
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return _MISSING


def _present_values(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the sole present key's value, a dict of several, else `_MISSING`.

    Presence is membership, not truthiness, so an explicit `{"output": None}`
    stays distinguishable from a payload carrying no matching key at all.
    """
    present = {key: payload[key] for key in keys if key in payload}
    if not present:
        return _MISSING
    if len(present) == 1:
        return next(iter(present.values()))
    return present


def _project_user_message(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    text = _first_text(interaction.payload, _USER_TEXT_KEYS)
    if text is _MISSING:
        return None
    return {
        "role": "user",
        "parts": [{"type": "text", "content": text}],
    }


def _project_assistant_message(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    text = _first_text(interaction.payload, _ASSISTANT_TEXT_KEYS)
    if text is _MISSING:
        return None
    return {
        "role": "assistant",
        "parts": [{"type": "text", "content": text}],
    }


def _project_reasoning(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    text = _first_text(interaction.payload, _ASSISTANT_TEXT_KEYS)
    if text is _MISSING:
        return None
    return {
        "role": "assistant",
        "thirdeye.kind": "reasoning",
        "parts": [{"type": "text", "content": text}],
    }


def _project_tool_call(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    if not interaction.correlation_id:
        return None

    tool_name = _first_text(interaction.payload, _TOOL_NAME_KEYS)
    if tool_name is _MISSING:
        return None

    arguments = _present_values(interaction.payload, _TOOL_INPUT_KEYS)

    return {
        "role": "assistant",
        "parts": [
            {
                "type": "tool_call",
                "id": interaction.correlation_id,
                "name": tool_name,
                "arguments": None if arguments is _MISSING else arguments,
            }
        ],
    }


def _project_tool_result(interaction: CanonicalInteraction) -> dict[str, Any] | None:
    if not interaction.correlation_id:
        return None

    response = _present_values(interaction.payload, _TOOL_OUTPUT_KEYS)

    return {
        "role": "tool",
        "parts": [
            {
                "type": "tool_call_response",
                "id": interaction.correlation_id,
                "response": None if response is _MISSING else response,
            }
        ],
    }


# Lifecycle interactions are absent here on purpose: they get spans, not messages.
_PROJECTORS: dict[str, Callable[[CanonicalInteraction], dict[str, Any] | None]] = {
    "user_message": _project_user_message,
    "assistant_message": _project_assistant_message,
    "reasoning": _project_reasoning,
    "tool_call": _project_tool_call,
    "tool_result": _project_tool_result,
}
