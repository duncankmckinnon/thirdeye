from __future__ import annotations

import dataclasses
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
def parse_new_usage_rows_codex(
    *,
    thirdeye_home: Path,
    session_id: str,
    sessions_root: Path | None = None,
    rollout_path: str | None = None,
    model: str | None = None,
) -> tuple[list[UsageRow], str | None, int | None, str | None] | None:
    """Tail-parse the Codex rollout for session_id since the last stored
    offset. Pure with respect to `UsageStore`: reads state but does not
    advance it or append anything — pass the result to
    `persist_usage_rows_codex` (with the seq of whatever event this usage
    belongs to) to actually commit it.

    `sessions_root` is overrideable for testing (default: ~/.codex/sessions).
    `rollout_path`, when given, skips resolution; `model`, when given, skips
    the turn_context model carry-forward and is used verbatim.

    Rows are stamped with a placeholder `seq=0`; `persist_usage_rows_codex`
    replaces it with the real triggering seq. `capture_usage_codex` combines
    this with `persist_usage_rows_codex` for callers with nothing else to do
    in between; the two are split out separately mainly for testability.

    Returns `(new_rows, resolved_path, new_offset, last_model)`.
    `resolved_path` is `None` when the rollout couldn't be found — expected,
    not an error; the caller should skip persisting in that case. The whole
    return value is `None` only on a genuine unexpected error (caught and
    logged to usage-errors.jsonl by @safe_capture) — callers must distinguish
    the two, same as this function's undivided predecessor did.

    Codex re-reports the same call, so distinct rows may carry duplicate
    ``call_id``s — ``usage/read.py`` collapses those on read; this writer
    never deduplicates.
    """
    root = sessions_root if sessions_root is not None else CODEX_SESSIONS_ROOT
    sd = session_dir(thirdeye_home, "codex", session_id)
    state = UsageStore(sd).read_state()

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
            return [], None, None, None
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
        row = _extract_usage_row(frame, session_id, 0, last_model)
        if row is not None:
            new_rows.append(row)
    new_offset = end_offset(rp, offset)
    return new_rows, resolved_path, new_offset, last_model


@safe_capture(phase="parse_rollout", platform="codex")
def persist_usage_rows_codex(
    *,
    thirdeye_home: Path,
    session_id: str,
    rows: list[UsageRow],
    resolved_path: str,
    new_offset: int,
    last_model: str | None,
    triggering_seq: int,
) -> int:
    """Stamp `rows` (from `parse_new_usage_rows_codex`) with the real
    triggering seq and commit them, and the new rollout offset/path/model, to
    UsageStore. Returns the number of rows persisted.
    """
    sd = session_dir(thirdeye_home, "codex", session_id)
    store = UsageStore(sd)
    stamped = [dataclasses.replace(row, seq=triggering_seq) for row in rows]
    if stamped:
        store.append(stamped)
    state = store.read_state()
    store.write_state(
        rollout_path=resolved_path,
        rollout_offset=new_offset,
        last_model=last_model,
        last_seq=triggering_seq if stamped else state.get("last_seq", -1),
    )
    return len(stamped)


def capture_usage_codex(
    *,
    thirdeye_home: Path,
    session_id: str,
    triggering_seq: int,
    sessions_root: Path | None = None,
    rollout_path: str | None = None,
    model: str | None = None,
) -> int | None:
    """Parse and persist new usage rows in one call — thin composition of
    `parse_new_usage_rows_codex` and `persist_usage_rows_codex`, which are
    also usable independently.

    Returns `None` on a genuine parse error (already logged), `0` if there
    was nothing to parse, else the number of rows persisted.
    """
    parsed = parse_new_usage_rows_codex(
        thirdeye_home=thirdeye_home,
        session_id=session_id,
        sessions_root=sessions_root,
        rollout_path=rollout_path,
        model=model,
    )
    if parsed is None:
        return None
    rows, resolved_path, new_offset, last_model = parsed
    if resolved_path is None or new_offset is None:
        return 0
    return persist_usage_rows_codex(
        thirdeye_home=thirdeye_home,
        session_id=session_id,
        rows=rows,
        resolved_path=resolved_path,
        new_offset=new_offset,
        last_model=last_model,
        triggering_seq=triggering_seq,
    )


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
