from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable

from thirdeye.config import Config
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.platforms.cursor.constants import DEDICATED_TOOL_NAMES, STRIP_KEYS
from thirdeye.platforms.cursor.usage import capture_usage_cursor
from thirdeye.store import Store
from thirdeye.tags import TagStore, extract_hashtags

_PLATFORM = "cursor"


def _read_stdin() -> dict:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _get_str(p: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = p.get(k)
        if v is not None and v != "":
            return str(v)
    return default


def _event_name(p: dict) -> str:
    return _get_str(p, "hook_event_name", "hookEventName")


def _session_id(p: dict) -> str:
    return _get_str(p, "conversation_id", "conversationId")


def _cwd(p: dict) -> str:
    cwd = p.get("cwd")
    if cwd:
        return str(cwd)
    roots = p.get("workspace_roots")
    if isinstance(roots, list) and roots:
        return str(roots[0])
    return os.getcwd()


def _strip(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in STRIP_KEYS}


def _emit(payload: dict, t: str, extra: dict | None = None) -> int | None:
    sid = _session_id(payload)
    if not sid:
        return None
    data = _strip(payload)
    if extra:
        data = {**data, **extra}
    return Store(Config.load()).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=_cwd(payload),
        t=t,
        data=data,
    )


def _print_permissive(event: str) -> None:
    out = sys.__stdout__ or sys.stdout
    try:
        if event.startswith("before"):
            out.write('{"permission": "allow"}')
        else:
            out.write('{"continue": true}')
        out.flush()
    except Exception:
        pass


def _h_session_start(payload: dict) -> None:
    _emit(payload, "session_start")


def _h_session_end(payload: dict) -> None:
    sid = _session_id(payload)
    if not sid:
        return
    config = Config.load()
    store = Store(config)
    seq = store.append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=_cwd(payload),
        t="session_end",
        data=_strip(payload),
    )
    store.close_session(sid, platform=_PLATFORM)
    capture_usage_cursor(
        thirdeye_home=config.root,
        session_id=sid,
        payload=payload,
        triggering_seq=seq if seq is not None else 0,
    )


def _h_before_submit(payload: dict) -> None:
    sid = _session_id(payload)
    if not sid:
        return
    config = Config.load()
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=_cwd(payload),
        t="user_message",
        data=_strip(payload),
    )
    try:
        prompt = _get_str(payload, "prompt", "input", "text")
        tags = extract_hashtags(prompt)
        if not tags:
            return
        sd = session_dir(config.root, _PLATFORM, sid)
        ts = TagStore(sd)
        for tag in tags:
            ts.add(seq, tag, source="auto")
        mp = meta_path(sd)
        m = read_meta(mp)
        if m is not None:
            m.tag_count = ts.tagged_seq_count()
            write_meta(mp, m)
    except Exception:
        pass


def _h_after_response(payload: dict) -> None:
    _emit(payload, "assistant_message")


def _h_before_shell(payload: dict) -> None:
    _emit(payload, "tool_call", {"tool_name": "shell"})


def _h_after_shell(payload: dict) -> None:
    _emit(payload, "tool_result", {"tool_name": "shell"})


def _h_before_mcp(payload: dict) -> None:
    name = _get_str(payload, "tool_name", "toolName", "name", default="mcp")
    _emit(payload, "tool_call", {"tool_name": name})


def _h_after_mcp(payload: dict) -> None:
    name = _get_str(payload, "tool_name", "toolName", "name", default="mcp")
    _emit(payload, "tool_result", {"tool_name": name})


def _h_before_read_file(payload: dict) -> None:
    _emit(payload, "tool_call", {"tool_name": "read_file"})


def _h_after_file_edit(payload: dict) -> None:
    _emit(payload, "tool_result", {"tool_name": "edit_file"})


def _h_before_tab_read(payload: dict) -> None:
    _emit(payload, "tool_call", {"tool_name": "tab_read_file"})


def _h_after_tab_edit(payload: dict) -> None:
    _emit(payload, "tool_result", {"tool_name": "tab_edit_file"})


def _h_post_tool_use(payload: dict) -> None:
    name = _get_str(payload, "tool_name", "toolName", "name", "tool")
    if name and name.lower() in DEDICATED_TOOL_NAMES:
        return
    if not name:
        name = "unknown"
    _emit(payload, "tool_result", {"tool_name": name})


def _h_stop(payload: dict) -> None:
    sid = _session_id(payload)
    if not sid:
        return
    config = Config.load()
    # TODO: Store does not expose a max_seq(session_id) helper. Pass 0 until
    # such an API is added rather than introducing a new Store method here.
    triggering_seq = 0
    capture_usage_cursor(
        thirdeye_home=config.root,
        session_id=sid,
        payload=payload,
        triggering_seq=triggering_seq,
    )


_HANDLERS: dict[str, Callable[[dict], None]] = {
    "sessionStart": _h_session_start,
    "sessionEnd": _h_session_end,
    "beforeSubmitPrompt": _h_before_submit,
    "afterAgentResponse": _h_after_response,
    "beforeShellExecution": _h_before_shell,
    "afterShellExecution": _h_after_shell,
    "beforeMCPExecution": _h_before_mcp,
    "afterMCPExecution": _h_after_mcp,
    "beforeReadFile": _h_before_read_file,
    "afterFileEdit": _h_after_file_edit,
    "beforeTabFileRead": _h_before_tab_read,
    "afterTabFileEdit": _h_after_tab_edit,
    "postToolUse": _h_post_tool_use,
    "stop": _h_stop,
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
