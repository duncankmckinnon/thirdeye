"""Handlers for Codex CLI's native ``hooks.json`` mechanism.

Distinct from ``codex/hooks.py``, which serves Codex's older, argv-invoked
``notify`` callback (``thread-id``-keyed JSON on argv) and the rollout-based
turn/usage reconstruction built on top of it. ``hooks.json`` is a newer,
separate Codex mechanism that mirrors Claude Code's own hook contract exactly
— the same event names (``SessionStart``, ``PostToolUse``, ...), the same
``session_id``-keyed JSON payload, delivered the same way: on stdin, one
invocation per event. So these handlers mirror ``claude/hooks.py``'s shape,
not ``codex/hooks.py``'s.

Deliberately missing: ``pre_tool_use``, ``post_tool_use``, and ``stop``.
Codex's rollout already gives ``notify`` a complete, richer picture of tool
calls and turn usage than a live per-event hook could (real token counts,
reconstructed message content) — wiring these here too would capture the same
tool calls and turns twice. ``install.py`` accordingly never points Codex's
``PreToolUse``/``PostToolUse``/``Stop`` hooks at thirdeye; see there for the
default handling for those three. Also missing: ``notification`` /
``permission_denied`` — Claude Code hook events with no Codex hooks.json
counterpart (Codex's own recognized event set, visible as ``[hooks.state]``
trust entries in ``config.toml`` once approved, has no ``Notification`` or
``PermissionDenied``).
"""

from __future__ import annotations

import json
import os
import sys

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.platforms.provenance import foreign_payload_reason
from thirdeye.store import Store
from thirdeye.tags import TagStore, extract_hashtags
from thirdeye.usage.errlog import log_capture_error

_PLATFORM = "codex"

# Same rationale as claude/hooks.py's _STRIP_KEYS: routing fields (redundant
# once passed to Store.append_event) and long paths that are pure noise.
_STRIP_KEYS = frozenset({"session_id", "cwd", "transcript_path", "agent_transcript_path"})


def _read_stdin() -> dict:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    reason = foreign_payload_reason(payload, _PLATFORM)
    if reason is not None:
        config = Config.load()
        log_capture_error(
            thirdeye_home=config.root,
            phase="foreign_payload",
            message=reason,
            platform=_PLATFORM,
            session_id=str(payload.get("session_id") or ""),
        )
        return {}
    return payload


def _strip_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _STRIP_KEYS}


def _emit(t: str, payload: dict) -> int | None:
    sid = payload.get("session_id")
    if not sid:
        return None
    cwd = payload.get("cwd") or os.getcwd()
    return Store(Config.load()).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t=t,
        data=_strip_payload(payload),
    )


def _reap_mid_turn_marker(payload: dict) -> None:
    from thirdeye.platforms.codex.interrupt_marker import reap_marker_for_event

    sid = payload.get("session_id")
    if not sid:
        return
    config = Config.load()
    cwd = payload.get("cwd") or os.getcwd()
    reap_marker_for_event(
        config,
        session_dir(config.root, _PLATFORM, sid),
        sid,
        cwd,
        prompt_id=payload.get("prompt_id"),
    )


def session_start() -> None:
    payload = _read_stdin()
    _reap_mid_turn_marker(payload)
    sid = payload.get("session_id")
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


def user_prompt_submit() -> None:
    from thirdeye.platforms.codex.interrupt_marker import replace_open_turn

    payload = _read_stdin()
    sid = payload.get("session_id")
    if not sid:
        return
    cwd = payload.get("cwd") or os.getcwd()
    config = Config.load()
    sd = session_dir(config.root, _PLATFORM, sid)
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t="user_message",
        data=_strip_payload(payload),
    )
    replace_open_turn(
        config,
        sd,
        sid,
        cwd,
        prompt=str(payload.get("prompt") or ""),
        prompt_id=payload.get("prompt_id"),
        turn_seq=seq,
    )
    try:
        prompt = payload.get("prompt") or ""
        tags = extract_hashtags(prompt)
        if not tags:
            return
        tagstore = TagStore(sd)
        for tag in tags:
            tagstore.add(seq, tag, source="auto")
        mp = meta_path(sd)
        m = read_meta(mp)
        if m is not None:
            m.tag_count = tagstore.tagged_seq_count()
            write_meta(mp, m)
    except Exception:
        pass


def subagent_start() -> None:
    payload = _read_stdin()
    _reap_mid_turn_marker(payload)
    _emit("subagent_start", payload)


def subagent_stop() -> None:
    # Matches claude/hooks.py's "subagent_message" naming for the same event
    # concept, so cross-platform tooling that treats event types generically
    # doesn't need a second vocabulary for the same thing.
    payload = _read_stdin()
    _reap_mid_turn_marker(payload)
    _emit("subagent_message", payload)


def permission_request() -> None:
    payload = _read_stdin()
    _reap_mid_turn_marker(payload)
    _emit("permission_request", payload)


def pre_compact() -> None:
    payload = _read_stdin()
    _reap_mid_turn_marker(payload)
    _emit("compact_start", payload)


def post_compact() -> None:
    payload = _read_stdin()
    _reap_mid_turn_marker(payload)
    _emit("compact_end", payload)


def session_end() -> None:
    from thirdeye.platforms.codex.interrupt_marker import close_stale_turn_if_open

    payload = _read_stdin()
    sid = payload.get("session_id")
    if sid:
        config = Config.load()
        close_stale_turn_if_open(
            config,
            session_dir(config.root, _PLATFORM, sid),
            sid,
            payload.get("cwd") or os.getcwd(),
        )
    if _emit("session_end", payload) is not None:
        Store(Config.load()).close_session(payload["session_id"], platform=_PLATFORM)
