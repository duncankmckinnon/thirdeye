"""Fail-open runtime for the ``thirdeye-copilot-hook EVENT`` dispatcher."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from thirdeye._compat.locking import LockTimeout
from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.index import IndexReader
from thirdeye.paths import index_path, session_dir
from thirdeye.platforms.provenance import foreign_payload_reason
from thirdeye.reader import SessionReader
from thirdeye.tags import TagStore
from thirdeye.usage.errlog import log_capture_error

from .capture import record_hook
from .constants import CLI_HOOK_EVENT_ALIASES, PLATFORM_NAME
from .followup import nonblocking_archive_lock, schedule_followup
from .identity import resolve_sources, stored_session_id
from .types import SourcePaths

# Accepted PascalCase aliases.  Canonicalization is owned by parse_hook;
# this set is filter-only so the explicit dispatcher argument is passed through.
_PASCAL_CASE_ALIASES = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SubagentStart",
        "SubagentStop",
        "SessionEnd",
    }
)
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


def _accepted_event(event: str) -> bool:
    return event in CLI_HOOK_EVENT_ALIASES or event in _PASCAL_CASE_ALIASES


def _context(config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {"env": capture_env(config.capture_env_patterns)}
    # Trace data is supplied explicitly by Copilot/the invoking integration;
    # never infer it by harvesting arbitrary process environment variables.
    for key in _TRACE_CONTEXT_KEYS:
        if key in payload:
            context[key] = payload[key]
    return context


@contextlib.contextmanager
def _bound_observation_id(observation_id: str) -> Iterator[None]:
    """Force ``record_hook`` to persist this observation id so tags can target it."""

    from thirdeye.platforms.copilot import capture as capture_mod

    original = capture_mod.uuid4
    capture_mod.uuid4 = lambda: SimpleNamespace(hex=observation_id)
    try:
        yield
    finally:
        capture_mod.uuid4 = original


def _tag_observation(
    config: Config,
    paths: SourcePaths,
    native_id: str,
    env: dict[str, str],
    *,
    observation_id: str,
) -> None:
    """Attach opt-in environment tags to the observation just written."""

    tags = [tag for name, value in env.items() if (tag := env_to_tag(name, value)) is not None]
    if not tags:
        return
    directory = session_dir(config.root, PLATFORM_NAME, stored_session_id(paths, native_id))
    try:
        reader = SessionReader(directory)
        count = IndexReader(index_path(directory)).count()
        for seq in range(count - 1, -1, -1):
            event = reader.get_event(seq)
            if event.get("t") != "copilot_hook":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            record = data.get("source_record")
            if not isinstance(record, dict):
                continue
            locator = record.get("locator")
            if not isinstance(locator, dict):
                continue
            if locator.get("observation_id") != observation_id:
                continue
            store = TagStore(directory)
            for tag in tags:
                store.add(int(event["seq"]), tag, source="auto")
            return
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
        if not _accepted_event(event):
            return
        payload = _read_stdin()
        if foreign_payload_reason(payload, expected=PLATFORM_NAME) is not None:
            return
        config = Config.load()
        paths = resolve_sources()
        context = _context(config, payload)
        native_id = payload.get("sessionId")
        observation_id = uuid4().hex
        try:
            with _bound_observation_id(observation_id), nonblocking_archive_lock():
                record_hook(config, paths, event, payload, context)
        except LockTimeout:
            # Spool is durable before capture.  A later worker retries import.
            if isinstance(native_id, str):
                _schedule(config, paths, native_id)
            return
        except Exception as exc:
            _log(
                config,
                "copilot_hook_capture",
                exc,
                native_id if isinstance(native_id, str) else "",
            )
            if isinstance(native_id, str):
                _schedule(config, paths, native_id)
            return

        if not isinstance(native_id, str):
            return
        _tag_observation(
            config,
            paths,
            native_id,
            context["env"],
            observation_id=observation_id,
        )
        _schedule(config, paths, native_id)
    except Exception:
        # Hooks must never decide a Copilot permission or emit protocol output.
        return


if __name__ == "__main__":
    main()
