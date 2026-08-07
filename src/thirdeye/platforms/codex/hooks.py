from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.paths import session_dir
from thirdeye.store import Store
from thirdeye.tags import TagStore
from thirdeye.usage.errlog import log_capture_error
from thirdeye.usage.store import UsageStore

_PLATFORM = "codex"

# Strip routing keys from stored event data because they're routing fields
# OR camel/kebab variants we don't need duplicated in storage.
_STRIP_KEYS = frozenset(
    {
        "thread-id",
        "thread_id",
        "threadId",
        "cwd",
        "working-directory",
        "working_directory",
    }
)


def _read_argv() -> dict:
    if len(sys.argv) < 2:
        return {}
    raw = sys.argv[1]
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _flex_get(d: dict, *keys, default=None):
    for key in keys:
        v = d.get(key)
        if v is not None and v != "":
            return v
    return default


def _strip_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _STRIP_KEYS}


def _emit(t: str, payload: dict) -> int | None:
    sid = _flex_get(payload, "thread-id", "thread_id", "threadId")
    if not sid:
        return None
    cwd = _flex_get(payload, "cwd", "working-directory", "working_directory") or os.getcwd()
    return Store(Config.load()).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t=t,
        data=_strip_payload(payload),
    )


def session_start() -> None:
    payload = _read_argv()
    sid = _flex_get(payload, "thread-id", "thread_id", "threadId")
    seq = _emit("session_start", payload)
    if seq is None:
        return
    config = Config.load()
    captured = capture_env(config.capture_env_patterns)
    if not captured:
        return
    sd = session_dir(config.root, _PLATFORM, sid)
    tagstore = TagStore(sd)
    for name, value in captured.items():
        tag = env_to_tag(name, value)
        if tag is None:
            continue
        tagstore.add(seq, tag, source="auto")


def notify() -> None:
    from thirdeye.platforms.codex.events import capture_events_codex
    from thirdeye.platforms.codex.rollout import resolve_rollout
    from thirdeye.platforms.codex.usage import capture_usage_codex

    try:
        payload = _read_argv()
        if payload.get("type") != "agent-turn-complete":
            return
        sid = _flex_get(payload, "thread-id", "thread_id", "threadId")
        if not sid:
            return
        cwd = _flex_get(payload, "cwd", "working-directory", "working_directory") or os.getcwd()
        config = Config.load()

        # The agent_turn event uniquely carries `input-messages` and
        # `last-assistant-message`; no rollout frame duplicates them in that
        # form, so it is always recorded even when the rollout is unresolvable.
        seq = Store(config).append_event(
            session_id=sid,
            platform=_PLATFORM,
            cwd=cwd,
            t="agent_turn",
            data=_strip_payload(payload),
        )

        sd = session_dir(config.root, _PLATFORM, sid)
        usage_store = UsageStore(sd)
        state = usage_store.read_state()

        # Resolve the rollout: cached path from state, else by session id. The
        # notify payload carries no transcript path, so the rollout JSONL is the
        # only source of per-turn events and token usage.
        rollout_path = state.get("rollout_path")
        if not rollout_path or not Path(rollout_path).is_file():
            resolved = resolve_rollout(sid)
            if resolved is None:
                log_capture_error(
                    thirdeye_home=config.root,
                    phase="open_source",
                    message=f"no rollout file found for session {sid}",
                    platform=_PLATFORM,
                    session_id=sid,
                )
                return
            rollout_path = str(resolved)

        offset = int(state.get("rollout_offset", 0))

        # One pass over the new rollout range: events first, then usage, then
        # advance the shared bookmark in a single write. Ordering is the one the
        # events module documents; do not reorder.
        _n_events, new_offset = capture_events_codex(
            config=config,
            session_id=sid,
            cwd=cwd,
            rollout_path=rollout_path,
            offset=offset,
        )
        capture_usage_codex(
            thirdeye_home=config.root,
            session_id=sid,
            triggering_seq=seq,
            rollout_path=rollout_path,
        )
        usage_store.write_state(
            rollout_path=rollout_path,
            rollout_offset=new_offset,
        )
    except Exception:
        pass
