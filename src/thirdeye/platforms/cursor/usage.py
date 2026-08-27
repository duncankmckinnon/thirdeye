from __future__ import annotations

from pathlib import Path
from typing import Any

from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.tracing import _provider, _text, usage_from_payload
from thirdeye.usage.errlog import safe_capture
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow
from thirdeye.writer import utc_iso_ms


@safe_capture(phase="extract_usage", platform="cursor")
def capture_usage_cursor(
    *, thirdeye_home: Path, session_id: str, payload: dict[str, Any], triggering_seq: int
) -> int:
    generation_id = _text(payload, "generation_id", "generationId")
    model = _text(payload, "model", "model_name")
    usage = usage_from_payload(payload)
    if not generation_id or not model or "input_tokens" not in usage or "output_tokens" not in usage:
        return 0
    store = UsageStore(session_dir(thirdeye_home, "cursor", session_id))
    if store.read_state().get("last_cursor_generation_id") == generation_id:
        return 0
    store.append(
        [
            UsageRow(
                session_id=session_id,
                seq=triggering_seq,
                call_id=generation_id,
                ts=utc_iso_ms(),
                platform="cursor",
                provider_name=_provider(model) or "cursor",
                response_model=model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            )
        ]
    )
    store.write_state(last_cursor_generation_id=generation_id, last_seq=triggering_seq)
    return 1
