"""Generic, JSON-serializable turn trace model.

This is the wire format that crosses the process boundary between a platform
adapter (which assembles a completed turn from raw hook events) and the
detached OTel export worker (which turns it into spans). Plain TypedDicts are
used deliberately instead of dataclasses: the record is written to a JSON job
file and read back by a separate subprocess, so it only ever needs to be a
plain dict that ``json.dumps``/``json.loads`` round-trips without any
to_dict/from_dict machinery.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class UsageDict(TypedDict, total=False):
    """Token usage for one LLM call. All fields optional — absent, not zero, for unreported counts."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    reasoning_output_tokens: int


class ToolCallSpanDict(TypedDict):
    """One tool invocation, nested under the LLM call that requested it."""

    tool_call_id: str
    name: str
    start_ts: str
    end_ts: str
    attributes: dict[str, Any]


class LlmCallSpanDict(TypedDict):
    """One model call within a turn, plus the tool calls it requested."""

    call_id: str
    provider: str
    model: str
    start_ts: str
    end_ts: str
    input_messages: list[dict[str, Any]]
    output_messages: list[dict[str, Any]]
    usage: UsageDict
    tool_calls: list[ToolCallSpanDict]


class OrphanToolCallDict(TypedDict):
    """A tool call whose requesting LLM call's chat span was already
    exported earlier (live, or in an earlier Stop) and so must not be
    rebuilt here -- re-sending it would double-count that call's tokens --
    but whose own tool span was never resolved at the time. Parented purely
    by id (the same pattern live spans already use to nest under a parent
    exported in an entirely separate, earlier process), since the actual
    parent chat span is not, and must not be, part of this export.
    """

    parent_call_id: str
    tool_call: ToolCallSpanDict


class PermissionRequestSpanDict(TypedDict):
    """A permission prompt surfaced to the user during a turn."""

    ts: str
    tool_name: str
    attributes: dict[str, Any]


TurnStatus = Literal["completed", "interrupted", "errored"]


class TurnSpanDict(TypedDict):
    """One agent turn: a user prompt through to its final response.

    ``subagents`` is recursive — a subagent invocation is exported as a
    nested ``TurnSpanDict``, not a distinct type, since structurally a
    subagent invocation is just another turn one level deeper. This is what
    lets a single recursive exporter function handle arbitrarily-nested
    subagents with no special-casing.

    ``attributes`` (here and on the two leaf span types) is a free-form
    passthrough bag — e.g. a subagent's display name/type — that the exporter
    flattens onto the span with its existing ``_flatten_attrs``/``_merge_raw``
    helpers. Producers should put raw Python values in it, not pre-serialize.

    All ``*_ts`` fields are ISO-8601 strings in the format
    ``writer.utc_iso_ms`` already produces (e.g.
    ``"2026-01-01T00:00:00.000Z"``), since the exporter's existing
    ``_ts_to_ns`` helper parses exactly that shape.

    ``input_message``/``output_message`` are plain text, not pre-wrapped
    message dicts — the exporter wraps them into ``gen_ai.input.messages``/
    ``gen_ai.output.messages`` shape itself via its existing ``_message()``
    helper. Empty string is valid for an interrupted turn with no response.
    """

    turn_id: str
    # Present on top-level turns once their deterministic OTel id is known.
    # Optional while Codex and nested-subagent producers migrate independently.
    turn_span_id: NotRequired[str]
    start_ts: str
    end_ts: str
    input_message: str
    output_message: str
    status: TurnStatus
    llm_calls: list[LlmCallSpanDict]
    # Optional: populated only by the Claude Stop-time reconstruction path
    # that can attach a tool call to an already-committed parent. Absent or
    # empty for every other producer.
    orphan_tool_calls: NotRequired[list[OrphanToolCallDict]]
    permission_requests: list[PermissionRequestSpanDict]
    subagents: list[TurnSpanDict]
    attributes: dict[str, Any]
