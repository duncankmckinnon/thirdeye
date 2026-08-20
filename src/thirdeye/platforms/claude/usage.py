from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from thirdeye.paths import session_dir
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
    also usable independently (`stop()` in claude/hooks.py reads the
    transcript offset before calling this, for `extract_new_calls_claude`'s
    sake, but does that with its own separate `UsageStore` read rather than
    needing the split itself).

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


def extract_new_calls_claude(
    *,
    transcript_path: str | None,
    offset: int,
) -> tuple[list[dict], int]:
    """Tail-parse the Claude transcript for new LLM calls since `offset`,
    building the rich per-call span data (content, not just token counts)
    that `otel_export.export_llm_calls` sends to Logfire.

    Claude Code logs each content block of one API response as its own JSONL
    line — several consecutive frames can share one `message.id`, each
    carrying one block (text/thinking/tool_use) of that single call's output.
    Those are grouped into one call here. Each call is also paired with the
    non-assistant content (user text, tool results) that preceded it, as its
    input.

    `offset` is the caller's responsibility, not read from state here: this
    walks the *same* transcript range `parse_new_usage_rows_claude` already
    reads for token accounting (pass it the same offset that came from
    `UsageStore.read_state()`), so the two must agree on what "new" means for
    a given hook invocation rather than each independently reading state.

    Returns `(new_calls, new_offset)`. Each call is `{"seq": 0, "ts": str,
    "call_id": str, "data": dict}` — `data` holds the fully-formed gen_ai.*
    span attributes, including `gen_ai.input.messages` /
    `gen_ai.output.messages` when there was anything to report. `seq` is a
    placeholder; the caller stamps it with the real triggering seq before
    passing calls to `export_llm_calls`. Never raises: any error here should
    not be able to block the ordinary usage/event capture paths that already
    run in the same hook call, so unlike the other functions in this module
    this one is not `@safe_capture`-wrapped by itself — call it inside a
    broader try/except, same as the rest of a hook's body.
    """
    if not transcript_path:
        return [], offset
    tp = Path(transcript_path)
    if not tp.is_file():
        return [], offset

    new_calls: list[dict] = []
    pending_input_parts: list[dict] = []
    pending_input_role = "user"
    current: dict[str, object] | None = None

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        data: dict[str, object] = {
            "gen_ai.provider.name": "anthropic",
            "gen_ai.operation.name": "chat",
            "gen_ai.response.model": current["model"],
            "gen_ai.usage.input_tokens": current["input_tokens"],
            "gen_ai.usage.output_tokens": current["output_tokens"],
        }
        if current["cache_read"] is not None:
            data["gen_ai.usage.cache_read.input_tokens"] = current["cache_read"]
        if current["cache_creation"] is not None:
            data["gen_ai.usage.cache_creation.input_tokens"] = current["cache_creation"]
        if current["input_parts"]:
            data["gen_ai.input.messages"] = [
                {"role": current["input_role"], "parts": current["input_parts"]}
            ]
        if current["output_parts"]:
            data["gen_ai.output.messages"] = [
                {"role": "assistant", "parts": current["output_parts"]}
            ]
        new_calls.append(
            {"seq": 0, "ts": current["ts"], "call_id": current["call_id"], "data": data}
        )
        current = None

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
            if not isinstance(frame, dict):
                continue

            message = frame.get("message")

            if frame.get("type") == "user" and isinstance(message, dict):
                # A user turn (real text, or a tool result feeding back) always
                # ends whatever call was accumulating — it can't contribute
                # more output after this.
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
                # A new message.id with no intervening user frame: Claude Code
                # re-invoked the model with no new visible input (e.g. a
                # continuation) — the new call legitimately has nothing to
                # report as input.
                flush_current()

            if current is None:
                raw_input = int(usage.get("input_tokens") or 0)
                cache_read = usage.get("cache_read_input_tokens")
                cache_crea = usage.get("cache_creation_input_tokens")
                current = {
                    "call_id": call_id,
                    "ts": str(frame.get("timestamp") or ""),
                    "model": str(model),
                    "input_tokens": raw_input + int(cache_read or 0) + int(cache_crea or 0),
                    "output_tokens": int(usage.get("output_tokens") or 0),
                    "cache_read": int(cache_read) if cache_read is not None else None,
                    "cache_creation": int(cache_crea) if cache_crea is not None else None,
                    "input_parts": pending_input_parts,
                    "input_role": pending_input_role,
                    "output_parts": [],
                }
                pending_input_parts = []

            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    part = _map_content_block(block)
                    if part is not None:
                        current["output_parts"].append(part)  # type: ignore[union-attr]

        flush_current()
        new_offset = f.tell()

    return new_calls, new_offset


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
