"""Fail-open dispatcher for Cursor IDE and CLI hooks."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.platforms.cursor.constants import DEDICATED_TOOL_NAMES, STRIP_KEYS
from thirdeye.store import Store
from thirdeye.tags import TagStore, extract_hashtags

_PLATFORM = "cursor"


def _read_stdin() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _get_str(payload: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _event_name(payload: dict[str, Any]) -> str:
    return _get_str(payload, "hook_event_name", "hookEventName", "event_name", "eventName")


def _session_id(payload: dict[str, Any]) -> str:
    return _get_str(payload, "conversation_id", "conversationId")


def _generation_id(payload: dict[str, Any]) -> str:
    return _get_str(payload, "generation_id", "generationId")


def _cwd(payload: dict[str, Any]) -> str:
    explicit = payload.get("cwd")
    if explicit:
        return str(explicit)
    roots = payload.get("workspace_roots")
    if isinstance(roots, list) and roots:
        return str(roots[0])
    return os.getcwd()


def _strip(payload: dict[str, Any]) -> dict[str, Any]:
    data = {key: value for key, value in payload.items() if key not in STRIP_KEYS}
    generation_id = _generation_id(payload)
    if generation_id:
        data["generation_id"] = generation_id
    return data


def _emit(
    payload: dict[str, Any], event_type: str, extra: dict[str, Any] | None = None
) -> int | None:
    session_id = _session_id(payload)
    if not session_id:
        return None
    data = _strip(payload)
    if extra:
        data.update(extra)
    return Store(Config.load()).append_event(
        session_id=session_id,
        platform=_PLATFORM,
        cwd=_cwd(payload),
        t=event_type,
        data=data,
    )


def _print_permissive(event: str) -> None:
    output = sys.__stdout__ or sys.stdout
    try:
        output.write(
            '{"permission": "allow"}' if event.startswith("before") else '{"continue": true}'
        )
        output.flush()
    except Exception:
        pass


def _session_start(payload: dict[str, Any]) -> None:
    seq = _emit(payload, "session_start")
    session_id = _session_id(payload)
    if seq is None or not session_id:
        return
    config = Config.load()
    captured = capture_env(config.capture_env_patterns)
    if not captured:
        return
    tags = TagStore(session_dir(config.root, _PLATFORM, session_id))
    for name, value in captured.items():
        tag = env_to_tag(name, value)
        if tag is not None:
            tags.add(seq, tag, source="auto")


def _session_end(payload: dict[str, Any]) -> None:
    session_id = _session_id(payload)
    if not session_id:
        return
    config = Config.load()
    seq = _emit(payload, "session_end")
    if seq is not None:
        _capture_usage(config, session_id, payload, seq)
    Store(config).close_session(session_id, platform=_PLATFORM)


def _before_submit(payload: dict[str, Any]) -> None:
    seq = _emit(payload, "user_message")
    session_id = _session_id(payload)
    if seq is None or not session_id:
        return
    try:
        prompt = _get_str(payload, "prompt", "input", "text")
        found = extract_hashtags(prompt)
        if not found:
            return
        config = Config.load()
        sd = session_dir(config.root, _PLATFORM, session_id)
        tags = TagStore(sd)
        for tag in found:
            tags.add(seq, tag, source="auto")
        meta = read_meta(meta_path(sd))
        if meta is not None:
            meta.tag_count = tags.tagged_seq_count()
            write_meta(meta_path(sd), meta)
    except Exception:
        pass


def _after_response(payload: dict[str, Any]) -> None:
    _emit(payload, "assistant_message")


def _after_thought(payload: dict[str, Any]) -> None:
    _emit(payload, "assistant_thought")


def _tool_name(payload: dict[str, Any], default: str) -> str:
    return _get_str(payload, "tool_name", "toolName", "name", "tool", default=default)


def _before_shell(payload: dict[str, Any]) -> None:
    _emit(payload, "tool_call", {"tool_name": "shell", "cursor_tool_family": "shell"})


def _after_shell(payload: dict[str, Any]) -> None:
    seq = _emit(payload, "tool_result", {"tool_name": "shell", "cursor_tool_family": "shell"})
    _emit_live(payload, seq)


def _before_mcp(payload: dict[str, Any]) -> None:
    name = _tool_name(payload, "mcp")
    _emit(payload, "tool_call", {"tool_name": name, "cursor_tool_family": "mcp"})


def _after_mcp(payload: dict[str, Any]) -> None:
    name = _tool_name(payload, "mcp")
    seq = _emit(payload, "tool_result", {"tool_name": name, "cursor_tool_family": "mcp"})
    _emit_live(payload, seq)


def _instant_tool(payload: dict[str, Any], event_type: str, name: str) -> None:
    seq = _emit(payload, event_type, {"tool_name": name, "cursor_instant": True})
    _emit_live(payload, seq)


def _emit_live(payload: dict[str, Any], seq: int | None) -> None:
    session_id = _session_id(payload)
    generation_id = _generation_id(payload)
    if seq is None or not session_id or not generation_id:
        return
    try:
        from thirdeye.platforms.cursor.live_spans import emit_live_tools

        config = Config.load()
        emit_live_tools(
            config,
            session_dir(config.root, _PLATFORM, session_id),
            session_id,
            _cwd(payload),
            generation_id,
            seq,
        )
    except Exception:
        pass


def _post_tool_use(payload: dict[str, Any]) -> None:
    name = _tool_name(payload, "unknown")
    if name.lower() in DEDICATED_TOOL_NAMES:
        return
    _instant_tool(payload, "tool_result", name)


def _capture_usage(config: Config, session_id: str, payload: dict[str, Any], seq: int) -> None:
    try:
        from thirdeye.platforms.cursor.usage import capture_usage_cursor

        capture_usage_cursor(
            thirdeye_home=config.root,
            session_id=session_id,
            payload=payload,
            triggering_seq=seq,
        )
    except Exception:
        pass


def _stop(payload: dict[str, Any]) -> None:
    session_id = _session_id(payload)
    if not session_id:
        return
    config = Config.load()
    cwd = _cwd(payload)
    seq = _emit(payload, "turn_stop")
    if seq is None:
        return
    _capture_usage(config, session_id, payload, seq)
    # `turn_stop` carries the model and token counts, so it is persisted even
    # when Cursor omits the generation_id. Only the turn span needs the id.
    generation_id = _generation_id(payload)
    if not generation_id:
        return
    try:
        from thirdeye.otel_export import export_turn
        from thirdeye.platforms.cursor.tracing import build_turn

        sd = session_dir(config.root, _PLATFORM, session_id)
        turn = build_turn(
            session_dir_=sd,
            session_id=session_id,
            generation_id=generation_id,
            stop_seq=seq,
        )
        if turn is not None:
            export_turn(config, sd, session_id, _PLATFORM, cwd, turn)
    except Exception:
        pass


def _subagent_stop(payload: dict[str, Any]) -> None:
    _emit(payload, "subagent_message")


_HANDLERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "sessionStart": _session_start,
    "sessionEnd": _session_end,
    "beforeSubmitPrompt": _before_submit,
    "afterAgentResponse": _after_response,
    "afterAgentThought": _after_thought,
    "beforeShellExecution": _before_shell,
    "afterShellExecution": _after_shell,
    "beforeMCPExecution": _before_mcp,
    "afterMCPExecution": _after_mcp,
    "beforeReadFile": lambda p: _instant_tool(p, "tool_call", "read_file"),
    "afterFileEdit": lambda p: _instant_tool(p, "tool_result", "edit_file"),
    "beforeTabFileRead": lambda p: _instant_tool(p, "tool_call", "read_file_tab"),
    "afterTabFileEdit": lambda p: _instant_tool(p, "tool_result", "edit_file_tab"),
    "postToolUse": _post_tool_use,
    "subagentStop": _subagent_stop,
    "stop": _stop,
}


def main() -> int:
    event = ""
    try:
        payload = _read_stdin()
        event = _event_name(payload)
        handler = _HANDLERS.get(event)
        if handler is not None:
            handler(payload)
    except Exception:
        pass
    finally:
        _print_permissive(event)
    return 0
