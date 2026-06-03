from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from thirdeye.paths import session_dir
from thirdeye.usage.errlog import safe_capture
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


def _get_str(payload: dict, *keys: str, default: str = "") -> str:
    """Return the first non-empty string-coerced value among the keys."""
    for k in keys:
        v = payload.get(k)
        if v is not None and v != "":
            return str(v)
    return default


def _get_int(payload: dict, *keys: str) -> int | None:
    """Return the first parseable int among the keys; None if none parse.

    Accepts numeric strings (including "0") but rejects "" and "--".
    """
    for k in keys:
        v = payload.get(k)
        if v is None or v == "" or v == "--":
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return None


@safe_capture(phase="extract_usage", platform="cursor")
def capture_usage_cursor(
    *,
    thirdeye_home: Path,
    session_id: str,
    payload: dict[str, Any],
    triggering_seq: int,
) -> int:
    """Append a UsageRow extracted from a Cursor stop/sessionEnd payload.

    Returns the number of rows appended (0 or 1). Wrapped in @safe_capture so
    any exception is logged to usage-errors.jsonl and the function returns None
    instead of raising.
    """
    model = _get_str(payload, "model", "model_name")
    input_tokens = _get_int(payload, "input_tokens", "inputTokens")
    output_tokens = _get_int(payload, "output_tokens", "outputTokens")

    if not model or input_tokens is None or output_tokens is None:
        return 0

    sd = session_dir(thirdeye_home, "cursor", session_id)
    store = UsageStore(sd)
    state = store.read_state()
    if state.get("last_seq", -1) >= 0:
        return 0

    row = UsageRow(
        session_id=session_id,
        seq=triggering_seq,
        ts=_now_iso(),
        platform="cursor",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    store.append([row])
    store.write_state(last_seq=triggering_seq)
    return 1


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
