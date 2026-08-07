from __future__ import annotations

from pathlib import Path

from thirdeye.paths import session_dir
from thirdeye.platforms.codex.rollout import (
    CODEX_SESSIONS_ROOT,
    end_offset,
    iter_frames,
    resolve_rollout,
)
from thirdeye.usage.errlog import log_capture_error, safe_capture
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow

__all__ = ["CODEX_SESSIONS_ROOT", "capture_usage_codex"]


@safe_capture(phase="parse_rollout", platform="codex")
def capture_usage_codex(
    *,
    thirdeye_home: Path,
    session_id: str,
    triggering_seq: int,
    sessions_root: Path | None = None,
    rollout_path: str | None = None,
    model: str | None = None,
) -> int:
    """Tail-parse the Codex rollout file for session_id, append new rows.

    `sessions_root` is overrideable for testing (default: ~/.codex/sessions).
    `rollout_path`, when given, skips resolution; `model`, when given, skips the
    turn_context model carry-forward and is used verbatim. Both exist so a future
    hooks integration can pass them without reworking this function; `notify`
    passes neither today.

    Returns the number of rows appended (one per ``token_count`` frame). Codex
    re-reports the same call, so distinct rows may carry duplicate
    ``call_id``s — ``usage/read.py`` collapses those on read; this writer never
    deduplicates.
    """
    root = sessions_root if sessions_root is not None else CODEX_SESSIONS_ROOT
    sd = session_dir(thirdeye_home, "codex", session_id)
    store = UsageStore(sd)
    state = store.read_state()

    resolved_path = rollout_path or state.get("rollout_path")
    if not resolved_path or not Path(resolved_path).is_file():
        rollout = resolve_rollout(session_id, root)
        if rollout is None:
            log_capture_error(
                thirdeye_home=thirdeye_home,
                phase="open_source",
                message=f"no rollout file found for session {session_id}",
                platform="codex",
                session_id=session_id,
            )
            return 0
        resolved_path = str(rollout)

    rp = Path(resolved_path)
    offset = int(state.get("rollout_offset", 0))

    # When `model` is given we use it verbatim and never carry forward from
    # turn_context frames; otherwise start from the persisted last_model.
    last_model: str | None = model if model is not None else state.get("last_model")

    new_rows: list[UsageRow] = []
    for _line_offset, frame in iter_frames(rp, offset):
        if model is None:
            inferred = _extract_model(frame)
            if inferred:
                last_model = inferred
        row = _extract_usage_row(frame, session_id, triggering_seq, last_model)
        if row is not None:
            new_rows.append(row)
    new_offset = end_offset(rp, offset)

    if new_rows:
        store.append(new_rows)
    store.write_state(
        rollout_path=resolved_path,
        rollout_offset=new_offset,
        last_model=last_model,
        last_seq=triggering_seq if new_rows else state.get("last_seq", -1),
    )
    return len(new_rows)


def _extract_model(frame: dict) -> str | None:
    if not isinstance(frame, dict):
        return None
    payload = frame.get("payload") if isinstance(frame.get("payload"), dict) else {}
    for d in (frame, payload):
        for key in ("model", "model_name"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
    return None


def _extract_usage_row(
    frame: dict,
    session_id: str,
    triggering_seq: int,
    last_model: str | None,
) -> UsageRow | None:
    """Return a UsageRow for a ``token_count`` frame, else None.

    Reads the per-call delta ``last_token_usage`` — never the cumulative
    ``total_token_usage``, which is used only to mint the ``call_id`` watermark.
    Codex's ``input_tokens`` already includes cached tokens (unlike Claude), so
    no cache arithmetic is applied here.
    """
    if not isinstance(frame, dict):
        return None
    if frame.get("type") != "event_msg":
        return None
    payload = frame.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "token_count":
        return None

    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    last = info.get("last_token_usage")
    total = info.get("total_token_usage")
    if not isinstance(last, dict) or last.get("total_tokens") is None:
        return None

    # The cumulative total is the call_id watermark, NOT a value to sum. The
    # series is strictly non-decreasing and advances only when tokens were
    # consumed, so distinct calls get distinct watermarks while a repeat report
    # reuses its predecessor's — and read.py's last-wins dedup collapses repeats
    # with no Codex-specific logic in the read layer. A byte offset would be
    # unique per frame and thus preserve the repeats, so it must not be used.
    cumulative = total.get("total_tokens") if isinstance(total, dict) else None
    if cumulative is None:
        return None
    call_id = f"cum:{cumulative}"

    # No cache arithmetic: Codex's input_tokens is already cache-inclusive, and
    # input + output == total_tokens. Absent cache/reasoning keys stay None
    # (absent vs zero); pre-2026-07-30 rollouts lack cache_write_input_tokens.
    input_tokens = int(last["input_tokens"])
    output_tokens = int(last["output_tokens"])
    cache_read = last.get("cached_input_tokens")
    cache_crea = last.get("cache_write_input_tokens")
    reasoning = last.get("reasoning_output_tokens")

    model = _extract_model(frame) or last_model or "unknown"
    ts = frame.get("timestamp") or ""
    return UsageRow(
        session_id=session_id,
        seq=triggering_seq,
        call_id=call_id,
        ts=str(ts),
        platform="codex",
        provider_name="openai",
        response_model=str(model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        operation_name="chat",
        cache_read_input_tokens=int(cache_read) if cache_read is not None else None,
        cache_creation_input_tokens=int(cache_crea) if cache_crea is not None else None,
        reasoning_output_tokens=int(reasoning) if reasoning is not None else None,
    )
