"""Fail-open runtime for the ``thirdeye-copilot-hook EVENT`` dispatcher."""

from __future__ import annotations

import io
import json
import sys
from typing import Any

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.paths import session_dir
from thirdeye.platforms.provenance import foreign_payload_reason
from thirdeye.reader import SessionReader
from thirdeye.tags import TagStore
from thirdeye.usage.errlog import log_capture_error

from .capture import record_hook
from .constants import CLI_HOOK_EVENT_ALIASES, PLATFORM_NAME
from .followup import schedule_followup
from .identity import resolve_sources, stored_session_id
from .types import SourcePaths

_PASCAL_CASE_ALIASES = {
    "SessionStart": "sessionStart",
    "UserPromptSubmit": "userPromptSubmitted",
    "PreToolUse": "preToolUse",
    "PostToolUse": "postToolUse",
    "Stop": "agentStop",
    "SubagentStart": "subagentStart",
    "SubagentStop": "subagentStop",
    "SessionEnd": "sessionEnd",
}
_TRACE_CONTEXT_KEYS = ("trace_id", "span_id", "parent_span_id", "trace_context", "traceparent")


def _read_stdin() -> dict[str, Any]:
    try:
        buffer = getattr(sys.stdin, "buffer", None)
        raw = (
            io.TextIOWrapper(buffer, encoding="utf-8").read()
            if buffer is not None
            else sys.stdin.read()
        )
        value = json.loads(raw) if raw else {}
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def _canonical_event(event: str) -> str | None:
    if event in CLI_HOOK_EVENT_ALIASES:
        return event
    return _PASCAL_CASE_ALIASES.get(event)


def _context(config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {"env": capture_env(config.capture_env_patterns)}
    # Trace data is supplied explicitly by Copilot/the invoking integration;
    # never infer it by harvesting arbitrary process environment variables.
    for key in _TRACE_CONTEXT_KEYS:
        if key in payload:
            context[key] = payload[key]
    return context


def _tag_observation(
    config: Config,
    paths: SourcePaths,
    native_id: str,
    payload: dict[str, Any],
    env: dict[str, str],
) -> None:
    """Attach opt-in environment tags to the just-written raw hook event."""

    tags = [tag for name, value in env.items() if (tag := env_to_tag(name, value)) is not None]
    if not tags:
        return
    directory = session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, native_id))
    try:
        matching = [
            event
            for event in SessionReader(directory).iter_events(types=("copilot_hook",))
            if isinstance(event.get("data"), dict)
            and isinstance(event["data"].get("source_record"), dict)
            and event["data"]["source_record"].get("payload", {}).get("hook_payload") == payload
        ]
        if not matching:
            return
        store = TagStore(directory)
        for tag in tags:
            store.add(int(matching[-1]["seq"]), tag, source="auto")
    except Exception:
        return


def _log(config: Config, phase: str, exc: BaseException, native_id: str = "") -> None:
    try:
        log_capture_error(
            thirdeye_home=config.root,
            phase=phase,
            error=exc,
            platform=PLATFORM_NAME,
            session_id=native_id,
            silent_fallback=True,
        )
    except Exception:
        return


def _schedule(config: Config, paths: SourcePaths, native_id: str) -> None:
    try:
        schedule_followup(config, paths, native_id)
    except Exception as exc:
        _log(config, "copilot_followup_schedule", exc, native_id)


def main() -> None:
    """Receive one passive Copilot hook invocation and always return silently."""

    try:
        # The installed dispatcher provides the event explicitly.  Do not
        # classify a payload based on casing: that would confuse Copilot and
        # Cursor hook conventions.
        event = sys.argv[1] if len(sys.argv) > 1 else ""
        canonical_event = _canonical_event(event)
        if canonical_event is None:
            return
        payload = _read_stdin()
        if foreign_payload_reason(payload, expected=PLATFORM_NAME) is not None:
            return
        config = Config.load()
        paths = resolve_sources()
        context = _context(config, payload)
        native_id = payload.get("sessionId")
        try:
            record_hook(config, paths, canonical_event, payload, context)
        except Exception as exc:
            _log(
                config,
                "copilot_hook_capture",
                exc,
                native_id if isinstance(native_id, str) else "",
            )
            # ``record_hook`` spools before capture.  A later worker may pick
            # up a receipt whose immediate archive attempt encountered a
            # transient source or lock failure.
            if isinstance(native_id, str):
                _schedule(config, paths, native_id)
            return

        if not isinstance(native_id, str):
            return
        _tag_observation(config, paths, native_id, payload, context["env"])
        _schedule(config, paths, native_id)
    except Exception:
        # Hooks must never decide a Copilot permission or emit protocol output.
        return


if __name__ == "__main__":
    main()
