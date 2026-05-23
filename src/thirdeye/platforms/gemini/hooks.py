from __future__ import annotations

import json
import os
import sys

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.store import Store
from thirdeye.tags import TagStore, extract_hashtags

_PLATFORM = "gemini"

# Routing keys we strip from event data because they're already used as
# routing fields when calling Store.append_event, OR because they're
# variants we don't want duplicated in storage.
_STRIP_KEYS = frozenset(
    {
        "session_id",
        "sessionId",
        "cwd",
        "workingDir",
        "working_dir",
    }
)


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


def _flex_get(d: dict, *keys, default=None):
    """Try multiple key names, return first non-None/non-empty value."""
    for key in keys:
        v = d.get(key)
        if v is not None and v != "":
            return v
    return default


def _strip_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _STRIP_KEYS}


def _print_response() -> None:
    """Gemini hooks must print {} to stdout when they finish."""
    print(json.dumps({}))


def _emit(t: str, payload: dict) -> int | None:
    sid = _flex_get(payload, "session_id", "sessionId")
    if not sid:
        return None
    cwd = _flex_get(payload, "cwd", "workingDir", "working_dir") or os.getcwd()
    return Store(Config.load()).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t=t,
        data=_strip_payload(payload),
    )


def session_start() -> None:
    try:
        payload = _read_stdin()
        sid = _flex_get(payload, "session_id", "sessionId")
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
    except Exception:
        pass
    finally:
        _print_response()


def session_end() -> None:
    try:
        payload = _read_stdin()
        if _emit("session_end", payload) is not None:
            sid = _flex_get(payload, "session_id", "sessionId")
            Store(Config.load()).close_session(sid, platform=_PLATFORM)
    except Exception:
        pass
    finally:
        _print_response()


def before_agent() -> None:
    try:
        payload = _read_stdin()
        sid = _flex_get(payload, "session_id", "sessionId")
        if sid:
            cwd = _flex_get(payload, "cwd", "workingDir", "working_dir") or os.getcwd()
            config = Config.load()
            seq = Store(config).append_event(
                session_id=sid,
                platform=_PLATFORM,
                cwd=cwd,
                t="user_message",
                data=_strip_payload(payload),
            )
            try:
                prompt = _flex_get(payload, "prompt", "input", "userInput", "message", "user_input")
                if isinstance(prompt, str) and prompt:
                    tags = extract_hashtags(prompt)
                    if tags:
                        sd = session_dir(config.root, _PLATFORM, sid)
                        tag_store = TagStore(sd)
                        for tag in sorted(tags):
                            tag_store.add(seq, tag, source="auto")
                        mp = meta_path(sd)
                        m = read_meta(mp)
                        if m is not None:
                            m.tag_count = tag_store.tagged_seq_count()
                            write_meta(mp, m)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        _print_response()


def after_agent() -> None:
    try:
        _emit("assistant_message", _read_stdin())
    except Exception:
        pass
    finally:
        _print_response()


def before_model() -> None:
    try:
        _emit("model_request", _read_stdin())
    except Exception:
        pass
    finally:
        _print_response()


def after_model() -> None:
    from thirdeye.platforms.gemini.usage import capture_usage_gemini

    try:
        payload = _read_stdin()
        sid = _flex_get(payload, "session_id", "sessionId")
        if not sid:
            return
        cwd = _flex_get(payload, "cwd", "workingDir", "working_dir") or os.getcwd()
        config = Config.load()
        seq = Store(config).append_event(
            session_id=sid,
            platform=_PLATFORM,
            cwd=cwd,
            t="model_response",
            data=_strip_payload(payload),
        )
        capture_usage_gemini(
            thirdeye_home=config.root,
            session_id=sid,
            payload=payload,
            triggering_seq=seq,
        )
    except Exception:
        pass
    finally:
        _print_response()


def before_tool() -> None:
    try:
        _emit("tool_call", _read_stdin())
    except Exception:
        pass
    finally:
        _print_response()


def after_tool() -> None:
    try:
        _emit("tool_result", _read_stdin())
    except Exception:
        pass
    finally:
        _print_response()
