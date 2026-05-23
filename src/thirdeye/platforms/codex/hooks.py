from __future__ import annotations

import json
import os
import sys

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.paths import session_dir
from thirdeye.store import Store
from thirdeye.tags import TagStore

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
        seq = Store(config).append_event(
            session_id=sid,
            platform=_PLATFORM,
            cwd=cwd,
            t="agent_turn",
            data=_strip_payload(payload),
        )
        capture_usage_codex(
            thirdeye_home=config.root,
            session_id=sid,
            triggering_seq=seq,
        )
    except Exception:
        pass
