from __future__ import annotations

import dataclasses
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thirdeye.paths import session_dir
from thirdeye.tracing.model import LlmCallSpanDict, UsageDict
from thirdeye.usage.errlog import log_capture_error, safe_capture
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


@safe_capture(phase="parse_transcript", platform="claude")
def parse_new_usage_rows_claude(
    *,
    thirdeye_home: Path,
    session_id: str,
    transcript_path: str | None,
) -> tuple[list[UsageRow], int | None] | None:
    """Tail-parse the Claude transcript for new usage rows since the last
    stored offset. Pure with respect to `UsageStore`: reads state but does
    not advance it or append anything — pass the result to
    `persist_usage_rows_claude` (with the seq of whatever event this usage
    belongs to) to actually commit it.

    Rows are stamped with a placeholder `seq=0`; `persist_usage_rows_claude`
    replaces it with the real triggering seq. `capture_usage_claude` combines
    this with `persist_usage_rows_claude` for callers with nothing else to do
    in between; the two are split out separately mainly for testability.

    Returns `(new_rows, new_offset)`. `new_offset` is `None` when there is
    nothing to parse (no transcript_path, or the file doesn't exist yet) —
    expected, not an error; the caller should skip persisting in that case.
    The whole return value is `None` only on a genuine unexpected error
    (caught and logged to usage-errors.jsonl by @safe_capture) — callers must
    distinguish the two: `0`/no-op for the former, `None`-propagating for the
    latter, same as this function's undivided predecessor did.

    One row is parsed per assistant frame - Claude writes one frame per content
    block, all carrying the identical ``message.usage``. Collapsing those
    duplicates is ``usage/read.py``'s job, never this writer's.
    """
    if not transcript_path:
        return [], None
    tp = Path(transcript_path)
    if not tp.is_file():
        log_capture_error(
            thirdeye_home=thirdeye_home,
            phase="open_source",
            message=f"transcript file does not exist: {transcript_path}",
            platform="claude",
            session_id=session_id,
            source_path=str(transcript_path),
        )
        return [], None

    sd = session_dir(thirdeye_home, "claude", session_id)
    offset = int(UsageStore(sd).read_state().get("transcript_offset", 0))

    new_rows: list[UsageRow] = []
    with tp.open("rb") as f:
        f.seek(offset)
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = _extract_row(frame, session_id, 0)
            if row is not None:
                new_rows.append(row)
        new_offset = f.tell()
    return new_rows, new_offset


@safe_capture(phase="parse_transcript", platform="claude")
def persist_usage_rows_claude(
    *,
    thirdeye_home: Path,
    session_id: str,
    rows: list[UsageRow],
    new_offset: int,
    triggering_seq: int,
) -> int:
    """Stamp `rows` (from `parse_new_usage_rows_claude`) with the real
    triggering seq and commit them, and the new transcript offset, to
    UsageStore. Returns the number of rows persisted.
    """
    sd = session_dir(thirdeye_home, "claude", session_id)
    store = UsageStore(sd)
    stamped = [dataclasses.replace(row, seq=triggering_seq) for row in rows]
    if stamped:
        store.append(stamped)
    state = store.read_state()
    store.write_state(
        transcript_offset=new_offset,
        last_seq=triggering_seq if stamped else state.get("last_seq", -1),
    )
    return len(stamped)


def capture_usage_claude(
    *,
    thirdeye_home: Path,
    session_id: str,
    transcript_path: str | None,
    triggering_seq: int,
) -> int | None:
    """Parse and persist new usage rows in one call — thin composition of
    `parse_new_usage_rows_claude` and `persist_usage_rows_claude`, which are
    also usable independently. This is purely local token accounting for
    `thirdeye usage` reporting; `build_turn` (claude/tracing.py) reads the
    transcript for `extract_calls_from_transcript`'s sake completely
    separately, via the open-turn marker's own captured offset, not this
    function's `UsageStore` bookmark.

    Returns `None` on a genuine parse error (already logged), `0` if there
    was nothing to parse, else the number of rows persisted.
    """
    parsed = parse_new_usage_rows_claude(
        thirdeye_home=thirdeye_home, session_id=session_id, transcript_path=transcript_path
    )
    if parsed is None:
        return None
    rows, new_offset = parsed
    if new_offset is None:
        return 0
    return persist_usage_rows_claude(
        thirdeye_home=thirdeye_home,
        session_id=session_id,
        rows=rows,
        new_offset=new_offset,
        triggering_seq=triggering_seq,
    )


@dataclasses.dataclass(frozen=True)
class ParsedCalls:
    """One `extract_calls_from_transcript` result.

    `offset` and `last_frame_ts` are a matched pair describing where the parse
    stopped: `offset` is the byte position of the first frame *not* consumed,
    and `last_frame_ts` the timestamp of the last frame before it. Feeding both
    back into the next call resumes exactly where this one left off, with the
    dispatch point for the first group intact.
    """

    calls: list[LlmCallSpanDict]
    offset: int
    last_frame_ts: str | None


def extract_calls_from_transcript(
    transcript_path: str | None,
    offset: int,
    *,
    initial_prev_ts: str | None = None,
    incremental: bool = False,
) -> ParsedCalls:
    """Tail-parse the Claude transcript for new LLM calls since `offset`,
    building `LlmCallSpanDict` records for `thirdeye.platforms.claude.tracing
    .build_turn` to assemble into a `TurnSpanDict` for `otel_export
    .export_turn`.

    Claude Code logs each content block of one API response as its own JSONL
    line — several consecutive frames can share one `message.id`, each
    carrying one block (text/thinking/tool_use) of that single call's output.
    Those are grouped into one call here. Each call is also paired with the
    non-assistant content (user text, tool results) that preceded it, as its
    input.

    `tool_calls` is always empty on every returned call: a tool_use/
    tool_result content block carries only arguments/response content, not
    real start/end timestamps, which live in thirdeye's own `tool_call`/
    `tool_result` events instead — pairing those in against the local event
    store is `build_turn`'s job, not this function's.

    A call's span runs from the frame immediately *preceding* the group — the
    dispatch point, i.e. the user's prompt frame for a turn's first call and
    the `tool_result` frame for every later one — to the *last* frame folded
    into the group, where the response finished arriving. That window includes
    time-to-first-token, usually the dominant term.

    `offset` is the caller's responsibility, not read from state here — see
    `build_turn`, which sources it from the open-turn marker it already reads
    for `turn_seq`/`prompt`, independently of `parse_new_usage_rows_claude`'s
    own `UsageStore` bookmark.

    `initial_prev_ts` supplies the dispatch point for the first group of this
    parse, whose preceding frame sits behind `offset` and so is never read.
    Without it that group falls back to its own first frame's timestamp — a
    zero-width span, but never a nonsensical one.

    `incremental` makes the call safe to repeat while the turn it is reading is
    still running. A `message.id` group at the very end of the file may still be
    receiving frames, so committing it would emit a half-built call — and
    advancing past it would let its remaining frames open a *second* call
    carrying the same id. Under `incremental=True` that trailing group is
    abandoned instead of flushed, and the returned offset points back at its
    first frame so the next parse rebuilds it whole. A group is known to be
    complete once a frame with a different `message.id`, or any `user` frame,
    follows it — exactly the two conditions the loop already flushes on — so
    everything returned is committed and the offset only ever advances across
    committed calls. `incremental=False` (the default) is the one-shot parse at
    Stop: the trailing group is flushed and the offset is EOF.

    An incremental cursor lands on the trailing group's first *assistant* frame
    (or EOF, when no group is open), so the non-assistant frames feeding that
    group its input sit behind it: the call the next parse rebuilds carries the
    right timestamps and output, but an empty `input_messages`. Stopping instead
    at the last committed call would keep them, at the cost of re-reading those
    frames on every parse.

    Returns a `ParsedCalls`. Never raises: any error here should not be able to
    block the ordinary usage/event capture paths that already run in the same
    hook call, so unlike the other functions in this module this one is not
    `@safe_capture`-wrapped by itself — call it inside a broader try/except,
    same as the rest of a hook's body.
    """
    # Timestamp of the last frame consumed, whatever its type. A group opening
    # on the next frame takes this as its dispatch point.
    prev_frame_ts: str | None = _iso_timestamp(initial_prev_ts) or None

    if not transcript_path:
        return ParsedCalls([], offset, prev_frame_ts)
    tp = Path(transcript_path)
    if not tp.is_file():
        return ParsedCalls([], offset, prev_frame_ts)

    new_calls: list[LlmCallSpanDict] = []
    pending_input_parts: list[dict] = []
    pending_input_role = "user"
    current: dict[str, Any] | None = None
    # Byte position of the open group's first frame, so `incremental` can rewind
    # to it rather than commit a group that may still be growing.
    current_group_offset = offset
    current_group_prev_ts = prev_frame_ts

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        usage: UsageDict = {
            "input_tokens": current["input_tokens"],
            "output_tokens": current["output_tokens"],
        }
        if current["cache_read"] is not None:
            usage["cache_read_input_tokens"] = current["cache_read"]
        if current["cache_creation"] is not None:
            usage["cache_creation_input_tokens"] = current["cache_creation"]
        input_messages = (
            [{"role": current["input_role"], "parts": current["input_parts"]}]
            if current["input_parts"]
            else []
        )
        output_messages = (
            [{"role": "assistant", "parts": current["output_parts"]}]
            if current["output_parts"]
            else []
        )
        # Both ends are `_iso_timestamp` output: a real ISO-8601 timestamp, or
        # "" when no frame carried a usable one. Collapse rather than emit a
        # backwards or half-empty span — and compare the parsed instants, since
        # frames may carry different UTC offsets, under which the string order
        # is not the chronological one.
        start_ts = current["start_ts"]
        end_ts = current["last_ts"]
        if not end_ts:
            end_ts = start_ts
        elif not start_ts or _is_after(start_ts, end_ts):
            start_ts = end_ts
        new_calls.append(
            {
                "call_id": current["call_id"],
                "provider": "anthropic",
                "model": current["model"],
                "start_ts": start_ts,
                "end_ts": end_ts,
                "input_messages": input_messages,
                "output_messages": output_messages,
                "usage": usage,
                "tool_calls": [],
            }
        )
        current = None

    with tp.open("rb") as f:
        f.seek(offset)
        while True:
            # Read line by line rather than iterate the handle, so this position
            # is the exact start of the frame about to be consumed — a group
            # opening below records it as its own.
            line_offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(frame, dict):
                continue

            frame_ts = _iso_timestamp(frame.get("timestamp"))
            # Every consumed frame becomes the dispatch point for whatever
            # group opens next — hence the finally, since the body below bails
            # out early on most frame types. A frame with a missing or
            # malformed timestamp leaves the cursor where it was.
            try:
                message = frame.get("message")

                if frame.get("type") == "user" and isinstance(message, dict):
                    # A user turn (real text, or a tool result feeding back)
                    # always ends whatever call was accumulating — it can't
                    # contribute more output after this.
                    flush_current()
                    content = message.get("content")
                    if isinstance(content, str):
                        if content:
                            pending_input_parts.append({"type": "text", "content": content})
                        pending_input_role = "user"
                    elif isinstance(content, list):
                        has_tool_result = False
                        for block in content:
                            part = _map_content_block(block)
                            if part is not None:
                                pending_input_parts.append(part)
                                has_tool_result = (
                                    has_tool_result or part["type"] == "tool_call_response"
                                )
                        pending_input_role = "tool" if has_tool_result else "user"
                    continue

                if frame.get("type") != "assistant" or not isinstance(message, dict):
                    continue
                model = message.get("model")
                if not model or model == "<synthetic>":
                    continue
                call_id = message.get("id") or frame.get("requestId") or frame.get("uuid")
                if not call_id:
                    continue
                usage = message.get("usage")
                if not isinstance(usage, dict) or (
                    usage.get("input_tokens") is None and usage.get("output_tokens") is None
                ):
                    continue
                call_id = str(call_id)

                if current is not None and current["call_id"] != call_id:
                    # A new message.id with no intervening user frame: Claude
                    # Code re-invoked the model with no new visible input (e.g.
                    # a continuation) — the new call legitimately has nothing
                    # to report as input.
                    flush_current()

                if current is None:
                    current_group_offset = line_offset
                    current_group_prev_ts = prev_frame_ts
                    raw_input = int(usage.get("input_tokens") or 0)
                    cache_read = usage.get("cache_read_input_tokens")
                    cache_crea = usage.get("cache_creation_input_tokens")
                    current = {
                        "call_id": call_id,
                        # No preceding frame to dispatch from (start of the
                        # transcript, or an unseeded incremental parse) leaves
                        # the span zero-width rather than inventing a start.
                        "start_ts": frame_ts if prev_frame_ts is None else prev_frame_ts,
                        "last_ts": frame_ts,
                        "model": str(model),
                        "input_tokens": raw_input + int(cache_read or 0) + int(cache_crea or 0),
                        "output_tokens": int(usage.get("output_tokens") or 0),
                        "cache_read": int(cache_read) if cache_read is not None else None,
                        "cache_creation": (int(cache_crea) if cache_crea is not None else None),
                        "input_parts": pending_input_parts,
                        "input_role": pending_input_role,
                        "output_parts": [],
                    }
                    pending_input_parts = []
                elif frame_ts:
                    # Frames sharing a message.id arrive over the life of the
                    # response, so the group's end follows the last of them.
                    current["last_ts"] = frame_ts

                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        part = _map_content_block(block)
                        if part is not None:
                            current["output_parts"].append(part)  # type: ignore[union-attr]
            finally:
                if frame_ts:
                    prev_frame_ts = frame_ts

        if incremental and current is not None:
            # The trailing group may still be growing: leave it uncommitted and
            # rewind to its first frame. Its dispatch point goes back out as
            # `last_frame_ts` — the last frame before the returned offset — so
            # the next parse rebuilds the whole group with the same `start_ts`.
            new_offset = current_group_offset
            prev_frame_ts = current_group_prev_ts
        else:
            flush_current()
            new_offset = f.tell()

    return ParsedCalls(new_calls, new_offset, prev_frame_ts)


def _iso_timestamp(value: object) -> str:
    """Return `value` as an offset-carrying ISO-8601 timestamp, else "".

    A frame's `timestamp` is whatever the transcript happens to hold — absent,
    null, a number, a truncated string, a bare date — and anything that isn't
    a real timestamp must not become a span boundary or a dispatch point, so
    it maps to "" and leaves the caller's cursor where it was.

    A value that *is* a timestamp but carries no UTC offset is returned with
    an explicit `+00:00` appended. The exporter's `_ts_to_ns` reads a naive
    timestamp in the worker's local timezone while the comparison below reads
    it as UTC, so an un-pinned bound could pass `_is_after` here and still
    come out backwards after export; pinning the offset makes both readings
    the same instant.
    """
    if not isinstance(value, str):
        return ""
    parsed = _parse_iso(value)
    if parsed is None:
        return ""
    return value if parsed.tzinfo is not None else value + "+00:00"


# A date-time, not merely something `fromisoformat` accepts: it also takes a
# bare "2026-08-22" (as midnight) and an hour-only "2026-08-22T10", neither of
# which is a timestamp a frame would legitimately carry. The separators are
# optional because basic format ("20260822T100000") is a date-time too, and
# `fromisoformat` — which every candidate must also satisfy — is the judge of
# which combinations of them are well-formed; all this asks for is a date
# followed by a time of day resolved at least to the minute.
_DATETIME_RE = re.compile(r"\d{4}-?\d{2}-?\d{2}[T ]\d{2}:?\d{2}")


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 date-time to a `datetime`, or `None` if `value` isn't
    one. The result is aware only if `value` carried an offset."""
    if not _DATETIME_RE.match(value):
        return None
    try:
        # `fromisoformat` only learned to accept a trailing "Z" in 3.11.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_after(start: str, end: str) -> bool:
    """Whether `start` is a later instant than `end`. Both are `_iso_timestamp`
    output, so both parse and both carry an offset; if one somehow doesn't, say
    no rather than collapse a span on the strength of an unreadable bound, and
    read any naive value the way the exporter would — as local time."""
    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    if start_dt is None or end_dt is None:
        return False
    return start_dt.astimezone(UTC) > end_dt.astimezone(UTC)


def _map_content_block(block: object) -> dict | None:
    """Map one Anthropic content block to an OTel GenAI semconv message part.

    Mirrors Logfire's own `claude_agent_sdk` integration's block mapping —
    `thinking` becomes a `reasoning` part, matching their naming.
    """
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    if kind == "text":
        text = block.get("text")
        return {"type": "text", "content": text} if text else None
    if kind == "thinking":
        thinking = block.get("thinking")
        return {"type": "reasoning", "content": thinking} if thinking else None
    if kind == "tool_use":
        return {
            "type": "tool_call",
            "id": block.get("id") or "",
            "name": block.get("name") or "",
            "arguments": block.get("input"),
        }
    if kind == "tool_result":
        return {
            "type": "tool_call_response",
            "id": block.get("tool_use_id") or "",
            "response": block.get("content"),
        }
    return None


def _extract_row(frame: dict, session_id: str, triggering_seq: int) -> UsageRow | None:
    """Return a UsageRow for an assistant frame carrying real API usage, else None.

    Emits a row per source frame; deduplication happens on read. Cache-inclusive
    input tokens follow the OTel GenAI convention: ``gen_ai.usage.input_tokens``
    includes cache reads and cache creation, which Anthropic reports separately.
    """
    if not isinstance(frame, dict):
        return None
    if frame.get("type") != "assistant":
        return None

    message = frame.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict) or not usage:
        return None

    # "<synthetic>" is Claude's zero-token placeholder for injected messages,
    # not a real API call.
    model = message.get("model")
    if not model or model == "<synthetic>":
        return None

    # Claude carries the timestamp at the top level, never inside `message`.
    ts = frame.get("timestamp") or ""

    # message.id is the most reliable key (no nulls over a real transcript);
    # fall back to requestId, then the frame uuid.
    call_id = message.get("id") or frame.get("requestId") or frame.get("uuid")
    if not call_id:
        return None

    # Anthropic reports input_tokens EXCLUDING cache; add both cache classes back
    # in for a comparable, cache-inclusive total.
    raw_input = int(usage.get("input_tokens") or 0)
    cache_read = usage.get("cache_read_input_tokens")
    cache_crea = usage.get("cache_creation_input_tokens")
    output = usage.get("output_tokens")

    # Absent both token fields means this frame carries no usage worth recording.
    if usage.get("input_tokens") is None and output is None:
        return None

    input_tokens = raw_input + int(cache_read or 0) + int(cache_crea or 0)

    return UsageRow(
        session_id=session_id,
        seq=triggering_seq,
        call_id=str(call_id),
        ts=str(ts),
        platform="claude",
        provider_name="anthropic",
        response_model=str(model),
        input_tokens=input_tokens,
        output_tokens=int(output or 0),
        operation_name="chat",
        # Absent vs zero: pass cache fields through only when reported.
        cache_read_input_tokens=int(cache_read) if cache_read is not None else None,
        cache_creation_input_tokens=(int(cache_crea) if cache_crea is not None else None),
        # Anthropic does not break out thinking/reasoning tokens.
        reasoning_output_tokens=None,
    )
