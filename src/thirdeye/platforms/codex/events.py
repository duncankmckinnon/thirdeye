"""Derive high-fidelity events from a Codex rollout file.

Codex's ``notify`` hook fires once per turn and carries no transcript path, so
this module takes the *same* rollout JSONL that usage tail-parsing already reads
and maps its frames onto thirdeye's existing event vocabulary. Mapping onto the
existing types (``tool_call``/``tool_result``/...) is deliberate: the web UI and
eval runner already pair and consume those, so Codex sessions gain UI and eval
support with no changes elsewhere.

Ordering and idempotency
------------------------
The caller performs one pass over each new rollout range: it appends events
first, then usage rows, then advances the stored offset in a single
``write_state``. A crash *between* appending and advancing replays the range on
the next run. Usage rows collapse by ``call_id`` in ``usage/read.py``, but
``events.alog`` is append-only with no dedup, so replayed events may duplicate.
This window is **accepted**, not engineered away — thirdeye already tolerates
re-emitted hook events — and each event's ``data`` carries ``rollout_offset``
(the frame's byte position) so post-hoc repair remains possible. Do not add
locking or an event-dedup index here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from thirdeye.platforms.codex.rollout import end_offset, iter_frames
from thirdeye.store import Store
from thirdeye.usage.errlog import log_capture_error, safe_capture

if TYPE_CHECKING:
    from thirdeye.config import Config

# Frame types whose top-level ``type`` alone determines the event; their payload
# carries no ``type`` discriminator.
_TOP_LEVEL_MAP = {
    "session_meta": "session_start",
    "turn_context": "notification",
}

# ``event_msg`` frames, keyed on ``payload.type``. ``token_count`` is
# deliberately absent — it is usage-only and must never become an event.
_EVENT_MSG_MAP = {
    "user_message": "user_message",
    "agent_message": "assistant_message",
    "task_started": "notification",
    "task_complete": "notification",
    "turn_aborted": "error",
    "exec_command_end": "tool_result",
    "patch_apply_end": "tool_result",
    "mcp_tool_call_end": "tool_result",
    "web_search_end": "tool_result",
}

# ``response_item`` frames, keyed on ``payload.type``. Both the ``function_call``
# and ``custom_tool_call`` families map to ``tool_call``/``tool_result`` — tool
# representation shifted across Codex versions, so both must be handled.
_RESPONSE_ITEM_MAP = {
    "function_call": "tool_call",
    "custom_tool_call": "tool_call",
    "local_shell_call": "tool_call",
    "web_search_call": "tool_call",
    "image_generation_call": "tool_call",
    "function_call_output": "tool_result",
    "custom_tool_call_output": "tool_result",
}


def _event_type_for(frame: dict) -> str | None:
    """Map a rollout frame to a thirdeye event type, or None to skip it.

    Any frame type not in the tables is skipped without error so a new Codex
    version cannot break capture.
    """
    ftype = frame.get("type")
    if ftype in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[ftype]
    payload = frame.get("payload")
    subtype = payload.get("type") if isinstance(payload, dict) else None
    if ftype == "event_msg":
        return _EVENT_MSG_MAP.get(subtype)
    if ftype == "response_item":
        return _RESPONSE_ITEM_MAP.get(subtype)
    return None


def _event_data(frame: dict, offset: int) -> dict:
    """The event's data: the frame's payload (minus ``type``) plus the offset.

    Preserving the payload keeps ``call_id`` intact so ``tool_call`` and
    ``tool_result`` pair, and ``rollout_offset`` records the source byte position
    for post-hoc dedup after a crash-window replay.
    """
    payload = frame.get("payload")
    data = {k: v for k, v in payload.items() if k != "type"} if isinstance(payload, dict) else {}
    data["rollout_offset"] = offset
    return data


@safe_capture(phase="parse_rollout_events", platform="codex")
def capture_events_codex(
    *,
    config: Config,
    session_id: str,
    cwd: str,
    rollout_path: str,
    offset: int,
) -> tuple[int, int]:
    """Append events for new rollout frames. Returns (events_appended, new_offset).

    Wrapped in ``@safe_capture`` so any unexpected error is logged and swallowed;
    on the fail-soft paths below the offset is returned unchanged so nothing is
    silently skipped.
    """
    rp = Path(rollout_path)
    if not rp.is_file():
        log_capture_error(
            thirdeye_home=config.root,
            phase="open_source",
            message=f"rollout file does not exist: {rollout_path}",
            platform="codex",
            session_id=session_id,
            source_path=str(rollout_path),
        )
        return 0, offset

    store = Store(config)
    appended = 0
    for frame_offset, frame in iter_frames(rp, offset):
        event_type = _event_type_for(frame)
        if event_type is None:
            continue
        store.append_event(
            session_id=session_id,
            platform="codex",
            cwd=cwd,
            t=event_type,
            data=_event_data(frame, frame_offset),
        )
        appended += 1

    new_offset = end_offset(rp, offset)
    return appended, new_offset
