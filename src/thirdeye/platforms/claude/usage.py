from __future__ import annotations

import json
from pathlib import Path

from thirdeye.config import Config
from thirdeye.otel_export import export_usage_rows
from thirdeye.paths import session_dir
from thirdeye.usage.errlog import log_capture_error, safe_capture
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


@safe_capture(phase="parse_transcript", platform="claude")
def capture_usage_claude(
    *,
    thirdeye_home: Path,
    session_id: str,
    transcript_path: str | None,
    triggering_seq: int,
    config: Config | None = None,
    cwd: str | None = None,
) -> int:
    """Tail-parse the Claude transcript, append new UsageRows, advance offset.

    Returns the number of rows appended. Wrapped in @safe_capture so any error
    is logged to usage-errors.jsonl and the function returns None instead of
    raising.

    One row is appended per assistant frame - Claude writes one frame per content
    block, all carrying the identical ``message.usage``. Collapsing those
    duplicates is ``usage/read.py``'s job, never this writer's.

    `config` and `cwd`, when both given, additionally mirror each new row to
    Logfire as a `usage` span (see `otel_export.export_usage_rows`); `stop`
    passes both today.
    """
    if not transcript_path:
        return 0
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
        return 0

    sd = session_dir(thirdeye_home, "claude", session_id)
    store = UsageStore(sd)
    state = store.read_state()
    offset = int(state.get("transcript_offset", 0))

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
            row = _extract_row(frame, session_id, triggering_seq)
            if row is not None:
                new_rows.append(row)
        new_offset = f.tell()

    if new_rows:
        store.append(new_rows)
        export_usage_rows(config, sd, session_id, "claude", cwd, new_rows)
    store.write_state(
        transcript_offset=new_offset,
        last_seq=triggering_seq if new_rows else state.get("last_seq", -1),
    )
    return len(new_rows)


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
