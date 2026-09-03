"""Reconstruct Cursor generations as Thirdeye turns.

Cursor hooks are point-in-time JSON callbacks. This adapter keeps their raw
payloads in the local event log, pairs before/after tool callbacks, and emits
the generic turn model consumed by the OTel GenAI Logfire exporter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from thirdeye.platforms.cursor.interactions import (
    canonical_interactions,
    interaction_messages,
    session_interactions,
)
from thirdeye.platforms.cursor.subagents import (
    CursorSubagentWindow,
    events_for_subagent,
    modern_subagent_stop_seqs,
    read_cursor_transcript,
    subagent_task_parent_ids,
    window_for_stop,
)
from thirdeye.reader import SessionReader
from thirdeye.span_ids import interaction_span_id, turn_span_id
from thirdeye.tracing.model import (
    InteractionSpanDict,
    LlmCallSpanDict,
    ToolCallSpanDict,
    TurnSpanDict,
    TurnStatus,
    UsageDict,
)

_PLATFORM = "cursor"

_CALL_ID_KEYS = ("tool_call_id", "toolCallId", "tool_use_id", "toolUseId", "call_id", "callId")
_CALL_ARGUMENT_KEYS = (
    "tool_input",
    "toolInput",
    "arguments",
    "input",
    "command",
    "file_path",
    "filePath",
    "path",
)
_RESULT_ARGUMENT_KEYS = (
    "tool_output",
    "toolOutput",
    "result",
    "output",
    "stdout",
    "response",
    "edits",
    "diff",
)


def _data(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("data")
    return value if isinstance(value, dict) else {}


def _text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _integer(data: dict[str, Any], *keys: str) -> int | None:
    value = None
    for key in keys:
        value = data.get(key)
        if value is not None:
            break
    if value in (None, "", "--"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _structured(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _start_ts(event: dict[str, Any]) -> str:
    end_ts = str(event.get("ts") or "")
    # `subagentStop` reports `duration_ms`; the tool callbacks use `duration`
    # or `durationMs`. All three mean elapsed milliseconds.
    duration = _integer(_data(event), "duration", "duration_ms", "durationMs")
    if not end_ts or duration is None or duration < 0:
        return end_ts
    try:
        normalized = end_ts[:-1] + "+00:00" if end_ts.endswith("Z") else end_ts
        started = datetime.fromisoformat(normalized) - timedelta(milliseconds=duration)
        return started.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except ValueError:
        return end_ts


def _provider(model: str) -> str:
    normalized = model.lower()
    if "claude" in normalized:
        return "anthropic"
    if "gemini" in normalized:
        return "gcp.gemini"
    if "deepseek" in normalized:
        return "deepseek"
    if "grok" in normalized:
        return "x_ai"
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return ""


def usage_from_payload(data: dict[str, Any]) -> UsageDict:
    input_total = _integer(data, "input_tokens", "inputTokens")
    cache_read = _integer(data, "cache_read_tokens", "cacheReadTokens")
    cache_write = _integer(data, "cache_write_tokens", "cacheWriteTokens")
    output = _integer(data, "output_tokens", "outputTokens")
    usage: UsageDict = {}
    if input_total is not None:
        # Cursor's stop/afterAgentResponse hooks report input_tokens as the turn
        # total, already including cache read and write buckets (unlike Anthropic's
        # API, which reports input_tokens excluding cache).
        usage["input_tokens"] = input_total
    if output is not None:
        usage["output_tokens"] = output
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    return usage


def _tool_span(
    *,
    session_id: str,
    generation_id: str,
    name: str,
    start: dict[str, Any],
    end: dict[str, Any],
    unmatched: str = "",
) -> ToolCallSpanDict:
    start_data, end_data = _data(start), _data(end)
    start_seq = int(start.get("seq") or 0)
    end_seq = int(end.get("seq") or 0)
    family = str(start_data.get("cursor_tool_family") or end_data.get("cursor_tool_family") or name)

    call_id = _text(start_data, *_CALL_ID_KEYS) or _text(end_data, *_CALL_ID_KEYS)
    if unmatched == "result":
        # A result with no call of its own cannot borrow the call id it echoes:
        # the absent call may still arrive live and claim that id.
        call_id = f"{generation_id}:{name}:result:{end_seq}"
    elif unmatched == "call":
        call_id = call_id or f"{generation_id}:{family}:{start_seq}"
    else:
        call_id = call_id or f"{generation_id}:{name}:{start_seq}"

    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": name,
        "gen_ai.tool.call.id": call_id,
    }
    # The raw payloads are the lossless record; the semantic attributes below
    # are a best-effort reading of them for querying.
    if unmatched != "result":
        attributes["thirdeye.tool.call.payload"] = start_data
        attributes["thirdeye.event.call_seq"] = start_seq
    if unmatched != "call":
        attributes["thirdeye.tool.result.payload"] = end_data
        attributes["thirdeye.event.result_seq"] = end_seq

    arguments = _semantic_value(start_data, _CALL_ARGUMENT_KEYS)
    if arguments is not None:
        attributes["gen_ai.tool.call.arguments"] = arguments
    result = _semantic_value(end_data, _RESULT_ARGUMENT_KEYS)
    if result is not None:
        attributes["gen_ai.tool.call.result"] = result

    exit_code = end_data.get("exit_code", end_data.get("exitCode"))
    if exit_code is not None:
        attributes["cursor.tool.exit_code"] = exit_code
    if unmatched:
        attributes["thirdeye.tool.unmatched"] = unmatched

    if start is end:
        span_start, span_end = _start_ts(end), str(end.get("ts") or "")
    else:
        span_start, span_end = _span_window(str(start.get("ts") or ""), str(end.get("ts") or ""))
    return {
        "tool_call_id": call_id,
        "name": name,
        "start_ts": span_start,
        "end_ts": span_end,
        "attributes": attributes,
    }


def _span_window(call_ts: str, result_ts: str) -> tuple[str, str]:
    """Order a call/result pair into a span window that cannot run backwards.

    Cursor sometimes delivers the after-callback before the before-callback, so
    the call is not always the earlier event. The span covers the interval the
    two callbacks actually observed; `thirdeye.event.call_seq` and
    `thirdeye.event.result_seq` keep the roles legible either way.
    """
    span_start, span_end = call_ts or result_ts, result_ts or call_ts
    first, second = _parse_ts(span_start), _parse_ts(span_end)
    if first is not None and second is not None and second < first:
        return span_end, span_start
    return span_start, span_end


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None


def _pair_key(data: dict[str, Any], family: str, name: str) -> str:
    """Identify a tool invocation well enough to pair its before/after events.

    Cursor supplies no tool call id on some callbacks, so fall back to the
    payload body (the command, tool input, or read path), which the matching
    after-callback may echo. Degrades to `family:name` when nothing is echoed.
    """
    explicit = _text(data, *_CALL_ID_KEYS)
    if explicit:
        return f"id:{explicit}"
    signature = _text(
        data,
        "command",
        "tool_input",
        "toolInput",
        "arguments",
        "file_path",
        "filePath",
        "path",
    )
    return f"{family}:{name}:{signature}" if signature else f"{family}:{name}"


def _semantic_value(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Read the payload's arguments (or its result) under any spelling Cursor uses.

    One present candidate is that value; several are kept together under their
    own keys, since guessing which spelling is canonical would drop the rest.
    Values stay structured -- embedded JSON is decoded, never pre-stringified,
    because `_flatten_attrs()` owns serialization.
    """
    present = {key: _structured(data[key]) for key in keys if data.get(key) not in (None, "")}
    if not present:
        return None
    if len(present) == 1:
        return next(iter(present.values()))
    return present


def _take_pending(
    pending: list[tuple[str, str, dict[str, Any]]],
    key: str,
    family: str,
    *,
    family_fifo: bool,
) -> dict[str, Any] | None:
    """Claim the pending counterpart matching `key`, else the oldest of the same family.

    The family pass is a last resort for callbacks that identify nothing, so it
    runs only when `family_fifo` says this side carries neither an explicit id
    nor a signature. Cursor's after-callbacks routinely echo nothing while the
    before-callback carried a command or path, so a pending signature does not
    disqualify the match -- an explicit id does: that invocation's other
    callback would have echoed the same id, so a callback without one belongs
    elsewhere.

    The family pass is strictly positional: it takes the oldest same-family
    counterpart, and gives up when that one carries an explicit id rather than
    reaching past it. Skipping ahead would silently reorder concurrent tools,
    which is the one thing positional pairing has no evidence for.

    Both passes take the earliest match: tools that complete in dispatch order
    are the common case, and a LIFO match would reverse exactly those.
    """
    for index, (pending_key, _, _event) in enumerate(pending):
        if pending_key == key:
            return pending.pop(index)[2]
    if not family_fifo:
        return None
    for index, (pending_key, pending_family, _event) in enumerate(pending):
        if pending_family != family:
            continue
        if pending_key.startswith("id:"):
            return None
        return pending.pop(index)[2]
    return None


def tool_calls_for_generation(
    events: list[dict[str, Any]], session_id: str, generation_id: str
) -> list[ToolCallSpanDict]:
    open_calls: list[tuple[str, str, dict[str, Any]]] = []
    open_results: list[tuple[str, str, dict[str, Any]]] = []
    completed: list[ToolCallSpanDict] = []

    def span(name: str, start: dict[str, Any], end: dict[str, Any], unmatched: str = "") -> None:
        completed.append(
            _tool_span(
                session_id=session_id,
                generation_id=generation_id,
                name=name,
                start=start,
                end=end,
                unmatched=unmatched,
            )
        )

    for event in events:
        event_type = str(event.get("t") or "")
        data = _data(event)
        name = _text(data, "tool_name", "toolName", "name", "tool") or "unknown"
        family = str(data.get("cursor_tool_family") or name)
        if event_type not in {"tool_call", "tool_result"}:
            continue
        if data.get("cursor_instant"):
            span(name, event, event)
            continue

        key = _pair_key(data, family, name)
        # A signatureless key identifies nothing beyond the family, which is the
        # only case where positional (FIFO) pairing is allowed to guess.
        family_fifo = key == f"{family}:{name}"
        if event_type == "tool_call":
            # Cursor can deliver the after-callback first, so a call also looks
            # back at results still waiting for one.
            result_event = _take_pending(open_results, key, family, family_fifo=family_fifo)
            if result_event is None:
                open_calls.append((key, family, event))
            else:
                span(name, event, result_event)
            continue

        call_event = _take_pending(open_calls, key, family, family_fifo=family_fifo)
        if call_event is None:
            open_results.append((key, family, event))
        else:
            span(name, call_event, event)

    for _key, _family, call_event in open_calls:
        name = _text(_data(call_event), "tool_name", "toolName", "name", "tool") or "unknown"
        span(name, call_event, call_event, unmatched="call")
    for _key, _family, result_event in open_results:
        name = _text(_data(result_event), "tool_name", "toolName", "name", "tool") or "unknown"
        span(name, result_event, result_event, unmatched="result")
    return completed


_ERROR_STATUSES = {"error", "failed", "failure"}
_INTERRUPTED_STATUSES = {"aborted", "cancelled", "canceled", "interrupted"}


def _subagent_status(data: dict[str, Any]) -> TurnStatus:
    value = _text(data, "status", "reason").lower()
    if value in _ERROR_STATUSES:
        return "errored"
    if value in _INTERRUPTED_STATUSES:
        return "interrupted"
    return "completed"


def _last_owned_response(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("t") != "assistant_message":
            continue
        text = _text(_data(event), "text", "response", "output")
        if text:
            return text
    return ""


# Attribute name -> the payload keys that may carry it, most preferred first.
# Cursor declares the callback as `agent.v1.SubagentStopRequestQuery`, whose
# fields are `subagent_id`, `subagent_type`, `status`, `duration_ms`, `summary`,
# `parent_conversation_id`, `message_count`, `tool_call_count`, `error_message`,
# `modified_files`, `git_branch`, `conversation_id`, `generation_id`, `model`,
# `loop_count`, `task`, `description`, and `model_id`. Both spellings are read
# because the proto reaches a command hook as JSON, which may camel-case it.
_SUBAGENT_TEXT_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cursor.subagent.id", ("subagent_id", "subagentId", "agent_id", "agentId")),
    ("cursor.subagent.type", ("subagent_type", "subagentType", "agent_type", "agentType")),
    ("cursor.subagent.description", ("description",)),
    (
        "cursor.subagent.model",
        ("subagent_model", "subagentModel", "model", "model_id", "modelId"),
    ),
    ("cursor.subagent.git_branch", ("git_branch", "gitBranch")),
    ("cursor.subagent.error_message", ("error_message", "errorMessage")),
)
_SUBAGENT_COUNT_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cursor.subagent.message_count", ("message_count", "messageCount")),
    ("cursor.subagent.tool_call_count", ("tool_call_count", "toolCallCount")),
    ("cursor.subagent.loop_count", ("loop_count", "loopCount")),
)


def _status(data: dict[str, Any]) -> TurnStatus:
    return "errored" if _text(data, "status", "reason").lower() in _ERROR_STATUSES else "completed"


def _event_generation_id(event: dict[str, Any]) -> str:
    return str(_data(event).get("generation_id") or _data(event).get("generationId") or "")


def bogus_generation_id(generation_id: str, session_id: str) -> bool:
    """Cursor sometimes sets ``generation_id`` to the conversation/session id.

    ``subagentStop`` is the main offender; treating that value as a real
    generation key would orphan subagents from their dispatching turn.
    """
    return bool(generation_id) and generation_id == session_id


def resolve_turn_seq(
    events: list[dict[str, Any]], *, generation_id: str, session_id: str, through_seq: int
) -> int | None:
    """Return the ``user_message`` seq anchoring ``generation_id``'s turn.

    Live tool export and turn reconstruction both need the same turn anchor.
    Never fall back to ``events[0]`` when it is not a user prompt — that
    produced orphan ``agent-turn`` spans keyed to arbitrary tool seqs.
    """
    prompt: dict[str, Any] | None = None
    for event in events:
        seq = int(event.get("seq") or 0)
        if seq > through_seq:
            break
        if event.get("t") != "user_message":
            continue
        if _event_generation_id(event) == generation_id:
            prompt = event
    if prompt is None:
        return None
    return int(prompt.get("seq") or 0)


def resolve_turn_generation(
    events: list[dict[str, Any]],
    *,
    generation_id: str,
    session_id: str,
    stop_seq: int,
) -> str:
    """Return the generation live tools already parented their chat span to.

    Cursor ``Stop`` often carries a successor ``generation_id`` (the next
    loop), not the ``user_message`` / tool generation. Building the turn from
    Stop's id emits a chat-less degenerate span and leaves those tools
    dangling.
    """
    if generation_id and not bogus_generation_id(generation_id, session_id):
        if (
            resolve_turn_seq(
                events,
                generation_id=generation_id,
                session_id=session_id,
                through_seq=stop_seq,
            )
            is not None
        ):
            return generation_id
    for event in reversed(events):
        seq = int(event.get("seq") or 0)
        if seq > stop_seq:
            continue
        if event.get("t") != "user_message":
            continue
        prompt_gen = _event_generation_id(event)
        if prompt_gen and not bogus_generation_id(prompt_gen, session_id):
            return prompt_gen
    return generation_id


def _subagents_in_turn(
    events: list[dict[str, Any]],
    *,
    turn_seq: int,
    stop_seq: int,
    session_id: str,
    generation_id: str,
) -> list[TurnSpanDict]:
    # A modern lifecycle window (paired subagent_start + subagent_message) is
    # exported independently and parented to its dispatching Task span, so it
    # must never also be embedded here. Only unmatched historical stops remain.
    modern_stop_seqs = modern_subagent_stop_seqs(events)
    subagents: list[TurnSpanDict] = []
    for event in events:
        if event.get("t") != "subagent_message":
            continue
        seq = int(event.get("seq") or 0)
        if seq in modern_stop_seqs:
            continue
        if not (turn_seq < seq <= stop_seq):
            continue
        gen = _event_generation_id(event)
        if gen and gen != generation_id and not bogus_generation_id(gen, session_id):
            continue
        subagents.append(_subagent_turn(session_id, event))
    return subagents


def _subagent_turn(session_id: str, event: dict[str, Any]) -> TurnSpanDict:
    """Project one Cursor `subagentStop` callback into a leaf subagent turn.

    Cursor reports a subagent only once it has finished: a single callback
    carrying the dispatched task, an elapsed `duration_ms`, and summary counts.
    Nothing in it describes the model calls or tools the child actually ran,
    and Cursor fires no callback for the dispatching tool either, so the leaf
    stays empty rather than inventing interior spans that were never observed
    or hanging itself off a `Task` tool span that does not exist.
    """
    data = _data(event)
    seq = int(event.get("seq") or 0)
    attributes: dict[str, Any] = {}
    for name, keys in _SUBAGENT_TEXT_ATTRS:
        text = _text(data, *keys)
        if text:
            attributes[name] = text
    for name, keys in _SUBAGENT_COUNT_ATTRS:
        count = _integer(data, *keys)
        if count is not None:
            attributes[name] = count
    return {
        "turn_id": f"subagent:{session_id}:{seq}",
        # Seq is unique within the session, so this never collides with the
        # dispatching turn's id (derived from its own first event's seq).
        "turn_span_id": str(turn_span_id(_PLATFORM, session_id, seq)),
        "start_ts": _start_ts(event),
        "end_ts": str(event.get("ts") or ""),
        "input_message": _text(data, "task"),
        "output_message": "",
        "status": _subagent_status(data),
        "llm_calls": [],
        "permission_requests": [],
        "subagents": [],
        "attributes": attributes,
    }


def subagent_turn_from_event(session_id: str, event: dict[str, Any]) -> TurnSpanDict:
    """Public wrapper for ``_subagent_turn`` used by hooks."""
    return _subagent_turn(session_id, event)


@dataclass(frozen=True)
class CursorSubagentExport:
    """A modern child turn plus how to parent it.

    ``tool_call_id`` is non-empty when the dispatching ``Task`` call id is
    known and the child should hang off that deterministic tool span.
    ``parent_turn_seq`` is populated only when the start carries no Task id
    but its own ``generation_id`` resolves to a user turn; otherwise ``None``.
    Export wiring owns the diagnostic/no-export decision when neither is set.
    """

    turn: TurnSpanDict
    tool_call_id: str
    parent_turn_seq: int | None


def _modern_subagent_turn(
    session_id: str,
    window: CursorSubagentWindow,
    owned_events: list[dict[str, Any]],
) -> TurnSpanDict:
    """Build one aggregate child turn from an exact-generation lifecycle window.

    Cursor exposes no boundaries or usage to split a child into multiple
    truthful model calls, so the child is one aggregate LLM call whose
    ``call_id`` is the derived subagent generation, carrying every tool call
    that exactly matched that generation between the lifecycle start and stop.
    """
    start_event = window.start_event or {}
    stop_event = window.stop_event
    start_data = _data(start_event)
    stop_data = _data(stop_event)

    attributes: dict[str, Any] = {}
    for name, keys in _SUBAGENT_TEXT_ATTRS:
        # Start metadata wins over the stop payload for every text attribute.
        text = _text(start_data, *keys) or _text(stop_data, *keys)
        if text:
            attributes[name] = text
    for name, keys in _SUBAGENT_COUNT_ATTRS:
        # Completed counts win: prefer the stop value, fall back to the start.
        count = _integer(stop_data, *keys)
        if count is None:
            count = _integer(start_data, *keys)
        if count is not None:
            attributes[name] = count
    parallel = start_data.get("is_parallel_worker", start_data.get("isParallelWorker"))
    if isinstance(parallel, bool):
        attributes["cursor.subagent.is_parallel_worker"] = parallel
    modified = stop_data.get("modified_files", stop_data.get("modifiedFiles"))
    if isinstance(modified, list) and all(isinstance(item, str) for item in modified):
        attributes["cursor.subagent.modified_files"] = modified

    transcript = read_cursor_transcript(
        _text(stop_data, "agent_transcript_path", "agentTranscriptPath") or None
    )
    input_text = _text(start_data, "task") or transcript.input_text
    output_text = (
        _last_owned_response(owned_events) or transcript.output_text or _text(stop_data, "summary")
    )
    model = _text(
        start_data,
        "subagent_model",
        "subagentModel",
        "model",
        "model_id",
        "modelId",
    ) or _text(stop_data, "model", "model_id", "modelId")

    tools = _owned_tool_calls(owned_events, session_id, window.generation_id)
    usage = usage_from_payload(stop_data)
    llm_calls: list[LlmCallSpanDict] = []
    if input_text or output_text or model or usage or tools:
        llm_calls.append(
            {
                "call_id": window.generation_id,
                "provider": _provider(model),
                "model": model,
                "start_ts": str(start_event.get("ts") or ""),
                "end_ts": str(stop_event.get("ts") or start_event.get("ts") or ""),
                "input_messages": (
                    [{"role": "user", "parts": [{"type": "text", "content": input_text}]}]
                    if input_text
                    else []
                ),
                "output_messages": (
                    [
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": output_text}],
                        }
                    ]
                    if output_text
                    else []
                ),
                "usage": usage,
                "tool_calls": tools,
            }
        )

    start_seq = int(start_event.get("seq") or 0)
    turn: TurnSpanDict = {
        "turn_id": str(start_seq),
        "turn_span_id": str(turn_span_id(_PLATFORM, session_id, start_seq)),
        "start_ts": str(start_event.get("ts") or ""),
        "end_ts": str(stop_event.get("ts") or start_event.get("ts") or ""),
        "input_message": input_text,
        "output_message": output_text,
        "status": _subagent_status(stop_data),
        "llm_calls": llm_calls,
        "permission_requests": [],
        "subagents": [],
        "attributes": attributes,
    }
    return turn


def _owned_tool_calls(
    owned_events: list[dict[str, Any]], session_id: str, derived_generation: str
) -> list[ToolCallSpanDict]:
    generations: list[str] = []
    for event in owned_events:
        gen = _event_generation_id(event)
        if gen and gen not in generations:
            generations.append(gen)
    if not generations:
        generations = [derived_generation] if derived_generation else []
    tools: list[ToolCallSpanDict] = []
    seen: set[str] = set()
    for generation in generations:
        for tool in tool_calls_for_generation(owned_events, session_id, generation):
            if tool["tool_call_id"] in seen:
                continue
            seen.add(tool["tool_call_id"])
            tools.append(tool)
    nested_task_ids = subagent_task_parent_ids(owned_events)
    return [tool for tool in tools if tool["tool_call_id"] not in nested_task_ids]


def resolve_subagent_export(
    session_dir_: Path, session_id: str, stop_event: dict[str, Any]
) -> CursorSubagentExport | None:
    """Resolve a just-finished Cursor child into an independently exported turn.

    A modern child (paired ``subagent_start`` + ``subagent_message``) is built
    from its exact-generation events and parented to the deterministic Task
    span from the start's ``tool_call_id``. When the start omits that id, the
    parent turn is resolved from the start event's own ``generation_id``.
    Duplicate or unmatched modern stops -- and every legacy stop-only window --
    return ``None`` here; the legacy summary-only ``_subagent_turn`` fallback
    handles unmatched historical stops elsewhere.
    """
    try:
        stop_seq = int(stop_event.get("seq") or 0)
    except (TypeError, ValueError):
        return None
    if stop_seq <= 0:
        return None

    events = list(SessionReader(session_dir_).iter_events(seq_range=(0, stop_seq + 1)))
    window = window_for_stop(events, stop_event)
    if window is None or window.start_event is None:
        return None

    owned_events = events_for_subagent(events, window)
    child_sid = _text(_data(stop_event), "child_session_id")
    if child_sid:
        child_dir = session_dir_.parent / child_sid
        if child_dir.is_dir():
            owned_events = list(SessionReader(child_dir).iter_events())
    turn = _modern_subagent_turn(session_id, window, owned_events)

    if window.tool_call_id:
        return CursorSubagentExport(turn, window.tool_call_id, None)

    start_data = _data(window.start_event)
    parent_generation = _text(start_data, "generation_id", "generationId")
    parent_turn_seq: int | None = None
    if parent_generation and not bogus_generation_id(parent_generation, session_id):
        start_seq = int(window.start_event.get("seq") or 0)
        parent_turn_seq = resolve_turn_seq(
            events,
            generation_id=parent_generation,
            session_id=session_id,
            through_seq=start_seq,
        )
    return CursorSubagentExport(turn, "", parent_turn_seq)


def build_turn(
    *, session_dir_: Path, session_id: str, generation_id: str, stop_seq: int
) -> TurnSpanDict | None:
    all_events = list(SessionReader(session_dir_).iter_events(seq_range=(0, stop_seq + 1)))
    generation_id = resolve_turn_generation(
        all_events,
        generation_id=generation_id,
        session_id=session_id,
        stop_seq=stop_seq,
    )
    if not generation_id or bogus_generation_id(generation_id, session_id):
        return None
    turn_seq = resolve_turn_seq(
        all_events, generation_id=generation_id, session_id=session_id, through_seq=stop_seq
    )
    events = [event for event in all_events if _event_generation_id(event) == generation_id]
    if not events:
        return None
    if turn_seq is None:
        # Tool-only generations have no user_message; anchor on the first observed event.
        turn_seq = int(events[0].get("seq") or 0)
    prompt_event = next((event for event in events if event.get("t") == "user_message"), None)
    response_events = [event for event in events if event.get("t") == "assistant_message"]
    stop_event = next(
        (
            event
            for event in reversed(all_events)
            if event.get("t") == "turn_stop" and int(event.get("seq") or 0) == stop_seq
        ),
        None,
    ) or next((event for event in reversed(events) if event.get("t") == "turn_stop"), events[-1])
    response_event = response_events[-1] if response_events else None
    prompt_data = _data(prompt_event or {})
    response_data = _data(response_event or {})
    stop_data = _data(stop_event)
    prompt = _text(prompt_data, "prompt", "input", "text")
    response = _text(response_data, "text", "response", "output")
    model = _text(stop_data, "model", "model_name") or _text(response_data, "model", "model_name")
    usage = usage_from_payload(stop_data)
    start_event = prompt_event or events[0]
    start_ts = str(start_event.get("ts") or "")
    end_ts = str(stop_event.get("ts") or start_ts)
    tools = tool_calls_for_generation(events, session_id, generation_id)
    # Tools already dispatched live keep the same deterministic parent IDs and
    # must not be emitted again with the completed turn.
    from thirdeye.platforms.cursor.live_spans import (
        committed_interaction_ids,
        committed_tool_call_ids,
    )

    committed_interactions = committed_interaction_ids(session_dir_, generation_id)
    committed_tools = committed_tool_call_ids(session_dir_, generation_id)
    tools = [tool for tool in tools if tool["tool_call_id"] not in committed_tools]

    # Build interaction records and compute input_messages from all session interactions.
    all_interactions = session_interactions(all_events, through_seq=stop_seq)
    response_seq = int(response_event.get("seq") or 0) if response_event else stop_seq
    input_msgs = interaction_messages(all_interactions, before_seq=response_seq)
    output_msgs = (
        [{"role": "assistant", "parts": [{"type": "text", "content": response}]}]
        if response
        else []
    )

    # Recovery records cover the active generation only; input_messages use full session history.
    turn_span_id_str = str(turn_span_id(_PLATFORM, session_id, turn_seq))
    active_interactions = canonical_interactions(
        all_events, generation_id=generation_id, through_seq=stop_seq
    )
    interaction_records: list[InteractionSpanDict] = []
    for interaction in active_interactions:
        if interaction.kind in {"tool_call", "tool_result"}:
            continue
        if (
            interaction.kind in {"reasoning", "assistant_message"}
            and interaction.interaction_id in committed_interactions
        ):
            continue
        interaction_records.append(
            {
                "interaction_id": interaction.interaction_id,
                "kind": interaction.kind,
                "span_id": str(interaction_span_id(_PLATFORM, session_id, interaction.interaction_id)),
                "parent_span_id": turn_span_id_str,
                "start_ts": interaction.ts,
                "end_ts": interaction.ts,
                "attributes": {
                    "thirdeye.interaction.kind": interaction.kind,
                    "thirdeye.interaction.payload": interaction.payload,
                    "thirdeye.interaction.correlation_id": interaction.correlation_id,
                    "thirdeye.interaction.source_type": interaction.source_type,
                    "thirdeye.interaction.source_seq": interaction.source_seq,
                    "thirdeye.interaction.timestamp": interaction.ts,
                    "thirdeye.interaction.generation_id": interaction.generation_id,
                    "thirdeye.interaction.duplicate_seqs": list(interaction.duplicate_seqs),
                },
            }
        )

    llm_calls = []
    if prompt or response or model or usage or tools:
        llm_calls.append(
            {
                "call_id": generation_id,
                "provider": _provider(model),
                "model": model,
                "start_ts": start_ts,
                "end_ts": str((response_event or stop_event).get("ts") or end_ts),
                "input_messages": input_msgs,
                "output_messages": output_msgs,
                "usage": usage,
                "tool_calls": tools,
            }
        )
    subagents = _subagents_in_turn(
        all_events,
        turn_seq=turn_seq,
        stop_seq=stop_seq,
        session_id=session_id,
        generation_id=generation_id,
    )
    if not llm_calls and not subagents and not interaction_records:
        return None
    turn: TurnSpanDict = {
        "turn_id": str(turn_seq),
        "turn_span_id": turn_span_id_str,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "input_message": prompt,
        "output_message": response,
        "status": _status(stop_data),
        "llm_calls": llm_calls,
        "permission_requests": [],
        "subagents": subagents,
        "attributes": {"cursor.generation.id": generation_id},
    }
    if interaction_records:
        turn["interactions"] = interaction_records
    return turn
