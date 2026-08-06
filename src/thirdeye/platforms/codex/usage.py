from __future__ import annotations

import json
from pathlib import Path

from thirdeye.paths import session_dir
from thirdeye.usage.errlog import log_capture_error, safe_capture
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow

CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"


@safe_capture(phase="parse_rollout", platform="codex")
def capture_usage_codex(
    *,
    thirdeye_home: Path,
    session_id: str,
    triggering_seq: int,
    sessions_root: Path | None = None,
) -> int:
    """Tail-parse the Codex rollout file for session_id, append new rows.

    `sessions_root` is overrideable for testing (default: ~/.codex/sessions).
    Returns the number of rows appended.
    """
    root = sessions_root if sessions_root is not None else CODEX_SESSIONS_ROOT
    sd = session_dir(thirdeye_home, "codex", session_id)
    store = UsageStore(sd)
    state = store.read_state()

    rollout_path = state.get("rollout_path")
    if not rollout_path or not Path(rollout_path).is_file():
        rollout = _resolve_rollout(root, session_id)
        if rollout is None:
            log_capture_error(
                thirdeye_home=thirdeye_home,
                phase="open_source",
                message=f"no rollout file found for session {session_id}",
                platform="codex",
                session_id=session_id,
            )
            return 0
        rollout_path = str(rollout)

    offset = int(state.get("rollout_offset", 0))
    rp = Path(rollout_path)
    last_model: str | None = state.get("last_model")
    new_rows: list[UsageRow] = []
    # Track each frame's absolute byte offset for a stable, unique call_id. Codex
    # frames carry no natural id, and the offset is monotonic across incremental
    # captures, so no two frames ever collide under the (session_id, call_id) key.
    pos = offset
    with rp.open("rb") as f:
        f.seek(offset)
        for raw in f:
            frame_offset = pos
            pos += len(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            inferred = _extract_model(frame)
            if inferred:
                last_model = inferred
            row = _extract_usage_row(frame, session_id, triggering_seq, last_model, frame_offset)
            if row is not None:
                new_rows.append(row)
        new_offset = f.tell()

    if new_rows:
        store.append(new_rows)
    store.write_state(
        rollout_path=rollout_path,
        rollout_offset=new_offset,
        last_model=last_model,
        last_seq=triggering_seq if new_rows else state.get("last_seq", -1),
    )
    return len(new_rows)


def _resolve_rollout(sessions_root: Path, session_id: str) -> Path | None:
    if not sessions_root.exists():
        return None
    matches = list(sessions_root.rglob(f"rollout-*-{session_id}.jsonl"))
    return matches[0] if matches else None


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
    frame_offset: int,
) -> UsageRow | None:
    if not isinstance(frame, dict):
        return None
    payload = frame.get("payload")
    if not isinstance(payload, dict):
        return None
    input_tokens = payload.get("input_tokens")
    output_tokens = payload.get("output_tokens")
    total_tokens = payload.get("total_tokens")
    # A Codex usage frame reports all three counts; the presence of total_tokens
    # identifies it. Totals are derived, never stored, so it is not persisted.
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None
    model = _extract_model(frame) or last_model or "unknown"
    ts = frame.get("timestamp") or ""
    return UsageRow(
        session_id=session_id,
        seq=triggering_seq,
        call_id=f"{session_id}:{frame_offset}",
        ts=str(ts),
        platform="codex",
        provider_name="openai",
        response_model=str(model),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        operation_name="chat",
        # Codex rollout frames do not break out cache or reasoning tokens.
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        reasoning_output_tokens=None,
    )
