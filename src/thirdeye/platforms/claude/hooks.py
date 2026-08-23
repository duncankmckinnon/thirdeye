from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict, cast

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.reader import SessionReader
from thirdeye.span_ids import turn_span_id
from thirdeye.store import Store
from thirdeye.tags import TagStore, extract_hashtags

_PLATFORM = "claude"

# Keys removed from the payload before storing the event:
# - session_id, cwd: used as routing fields when calling Store.append_event,
#   so they're redundant in event data
# - transcript_path, agent_transcript_path: long absolute paths Claude
#   includes in nearly every payload; pure noise for default rendering
_STRIP_KEYS = frozenset({"session_id", "cwd", "transcript_path", "agent_transcript_path"})


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


def _strip_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _STRIP_KEYS}


def _emit(t: str, payload: dict) -> int | None:
    sid = payload.get("session_id")
    if not sid:
        return None
    cwd = payload.get("cwd") or os.getcwd()
    config = Config.load()
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t=t,
        data=_strip_payload(payload),
    )
    if seq is not None:
        _close_stale_turn_if_open(config, sid, cwd, seq, payload.get("prompt_id"))
    return seq


def _open_turn_path(session_dir_: Path) -> Path:
    return session_dir_ / "claude-open-turn.json"


def _open_turn_lock_path(session_dir_: Path) -> Path:
    return session_dir_ / "claude-open-turn.lock"


class OpenTurnMarker(TypedDict):
    turn_seq: int
    turn_span_id: str
    start_ts: str
    prompt: str
    prompt_id: str | None
    transcript_path: str | None
    transcript_offset: int
    last_frame_ts: str | None


_OPEN_TURN_FIELDS = frozenset(OpenTurnMarker.__required_keys__)
_OPEN_TURN_LOCK_STATE = threading.local()


@contextlib.contextmanager
def _locked_open_turn(session_dir_: Path, operation: int) -> Iterator[None]:
    key = str(session_dir_.absolute())
    held = getattr(_OPEN_TURN_LOCK_STATE, "held", {})
    current = held.get(key)
    if current is not None:
        current_operation, depth = current
        if current_operation != fcntl.LOCK_EX and operation == fcntl.LOCK_EX:
            raise RuntimeError("cannot upgrade a shared open-turn lock")
        held[key] = (current_operation, depth + 1)
        try:
            yield
        finally:
            held[key] = (current_operation, depth)
        return

    session_dir_.mkdir(parents=True, exist_ok=True)
    with _open_turn_lock_path(session_dir_).open("a+") as lock:
        fcntl.flock(lock.fileno(), operation)
        held[key] = (operation, 1)
        _OPEN_TURN_LOCK_STATE.held = held
        try:
            yield
        finally:
            held.pop(key, None)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_open_turn_unlocked(session_dir_: Path) -> OpenTurnMarker | None:
    try:
        marker = json.loads(_open_turn_path(session_dir_).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or not _OPEN_TURN_FIELDS.issubset(marker):
        return None
    turn_seq = marker.get("turn_seq")
    span_id = marker.get("turn_span_id")
    transcript_offset = marker.get("transcript_offset")
    if (
        not isinstance(turn_seq, int)
        or isinstance(turn_seq, bool)
        or turn_seq < 0
        or not isinstance(span_id, str)
        or not span_id.isascii()
        or not span_id.isdecimal()
        or not 0 < int(span_id) < 2**64
        or not isinstance(marker.get("start_ts"), str)
        or not isinstance(marker.get("prompt"), str)
        or not (marker.get("prompt_id") is None or isinstance(marker.get("prompt_id"), str))
        or not (
            marker.get("transcript_path") is None or isinstance(marker.get("transcript_path"), str)
        )
        or not isinstance(transcript_offset, int)
        or isinstance(transcript_offset, bool)
        or transcript_offset < 0
        or not (marker.get("last_frame_ts") is None or isinstance(marker.get("last_frame_ts"), str))
    ):
        return None
    return cast(OpenTurnMarker, marker)


def _read_open_turn(session_dir_: Path) -> OpenTurnMarker | None:
    """Read the current marker without observing a cursor write in progress."""
    try:
        with _locked_open_turn(session_dir_, fcntl.LOCK_SH):
            return _read_open_turn_unlocked(session_dir_)
    except OSError:
        return None


def _write_open_turn(session_dir_: Path, marker: OpenTurnMarker) -> None:
    with _locked_open_turn(session_dir_, fcntl.LOCK_EX):
        _open_turn_path(session_dir_).write_text(json.dumps(marker))


def _advance_turn_cursor(
    session_dir_: Path,
    *,
    expected_turn_seq: int,
    offset: int,
    last_frame_ts: str | None,
) -> bool:
    """Advance a marker only if it still belongs to the expected turn."""
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or (last_frame_ts is not None and not isinstance(last_frame_ts, str))
    ):
        return False
    try:
        with _locked_open_turn(session_dir_, fcntl.LOCK_EX):
            marker = _read_open_turn_unlocked(session_dir_)
            if marker is None:
                return False
            try:
                marker_turn_seq = int(marker["turn_seq"])
            except (KeyError, TypeError, ValueError):
                return False
            if marker_turn_seq != expected_turn_seq:
                return False
            marker["transcript_offset"] = offset
            marker["last_frame_ts"] = last_frame_ts
            _open_turn_path(session_dir_).write_text(json.dumps(marker))
            return True
    except OSError:
        return False


def _delete_open_turn(session_dir_: Path, *, expected_turn_seq: int) -> bool:
    """Delete only the marker belonging to ``expected_turn_seq``."""
    try:
        with _locked_open_turn(session_dir_, fcntl.LOCK_EX):
            try:
                raw = json.loads(_open_turn_path(session_dir_).read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                return False
            if not isinstance(raw, dict):
                return False
            marker_turn_seq = raw.get("turn_seq")
            if (
                not isinstance(marker_turn_seq, int)
                or isinstance(marker_turn_seq, bool)
                or marker_turn_seq != expected_turn_seq
            ):
                return False
            _open_turn_path(session_dir_).unlink()
            return True
    except OSError:
        return False


def _close_stale_turn_if_open(
    config: Config,
    session_id: str,
    cwd: str,
    proving_seq: int,
    prompt_id: str | None = None,
    *,
    force: bool = False,
) -> None:
    """`proving_seq` is the just-recorded event proving the previous turn
    never reached a real `Stop`; its own seq/ts become the interrupted
    turn's `stop_seq`/`end_ts`.
    """
    try:
        from thirdeye.otel_export import export_turn
        from thirdeye.platforms.claude.tracing import build_turn

        sd = session_dir(config.root, _PLATFORM, session_id)
        marker = _read_open_turn(sd)
        if marker is None and not force:
            return
        marker_prompt_id = marker.get("prompt_id") if marker is not None else None
        if (
            marker is not None
            and not force
            and (not prompt_id or not marker_prompt_id or prompt_id == marker_prompt_id)
        ):
            return
        proving_ts = str(SessionReader(sd).get_event(proving_seq).get("ts") or "")
        turn = build_turn(
            config=config,
            session_dir_=sd,
            session_id=session_id,
            cwd=cwd,
            stop_seq=proving_seq,
            stop_ts=proving_ts,
            transcript_path=None,
            final_response="",
            status="interrupted",
            marker_snapshot=marker,
        )
        if turn is not None:
            export_turn(config, sd, session_id, _PLATFORM, cwd, turn)
            _delete_open_turn(sd, expected_turn_seq=int(turn["turn_id"]))
    except Exception:
        pass


def session_start() -> None:
    payload = _read_stdin()
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
    payload = _read_stdin()
    sid = payload.get("session_id")
    if not sid:
        return
    cwd = payload.get("cwd") or os.getcwd()
    config = Config.load()
    sd = session_dir(config.root, _PLATFORM, sid)

    prompt = payload.get("prompt") or ""
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t="user_message",
        data=_strip_payload(payload),
    )
    if seq is None:
        return

    # If the previous turn was interrupted, this new prompt is the proof:
    # close it out and export it as "interrupted", using this event's own
    # seq/ts, before this turn's own marker overwrites the file.
    _close_stale_turn_if_open(config, sid, cwd, seq, force=True)

    try:
        tags = extract_hashtags(prompt)
        if tags:
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

    try:
        start_ts = SessionReader(sd).get_event(seq).get("ts", "")
        # Measure the transcript's current size directly rather than reusing the
        # usage bookmark: that bookmark only advances inside `stop()`, so an
        # interrupted turn leaves it pointing at the *previous* turn's start and
        # this turn would re-parse — and re-emit — the previous turn's LLM calls.
        # A fresh measurement consults no prior state, so it cannot inherit that
        # staleness. `st_size` is the byte offset the transcript parser seeks to.
        transcript_path = payload.get("transcript_path")
        tp = Path(transcript_path) if transcript_path else None
        offset = tp.stat().st_size if tp is not None and tp.is_file() else 0
        _write_open_turn(
            sd,
            {
                "turn_seq": seq,
                "turn_span_id": str(turn_span_id(sid, seq)),
                "start_ts": start_ts,
                "prompt": prompt,
                "prompt_id": payload.get("prompt_id"),
                "transcript_path": transcript_path,
                "transcript_offset": offset,
                "last_frame_ts": start_ts,
            },
        )
    except Exception:
        pass


def pre_tool_use() -> None:
    _emit("tool_call", _read_stdin())


def post_tool_use() -> None:
    payload = _read_stdin()
    _emit("tool_result", payload)

    try:
        from thirdeye.platforms.claude.live_spans import emit_live_spans

        sid = payload.get("session_id")
        tool_use_id = payload.get("tool_use_id")
        if not sid or not tool_use_id:
            return
        cwd = payload.get("cwd") or os.getcwd()
        config = Config.load()
        sd = session_dir(config.root, _PLATFORM, sid)
        emit_live_spans(config, sd, sid, cwd, str(tool_use_id))
    except Exception:
        pass


def stop() -> None:
    from thirdeye.otel_export import export_turn
    from thirdeye.platforms.claude.tracing import build_turn
    from thirdeye.platforms.claude.usage import capture_usage_claude

    payload = _read_stdin()
    sid = payload.get("session_id")
    if not sid:
        return
    cwd = payload.get("cwd") or os.getcwd()
    config = Config.load()
    sd = session_dir(config.root, _PLATFORM, sid)
    marker = _read_open_turn(sd)
    expected_turn_seq = marker["turn_seq"] if marker is not None else None
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t="assistant_message",
        data=_strip_payload(payload),
    )

    transcript_path = payload.get("transcript_path")

    capture_usage_claude(
        thirdeye_home=config.root,
        session_id=sid,
        transcript_path=transcript_path,
        triggering_seq=seq,
    )

    try:
        stop_ts = SessionReader(sd).get_event(seq).get("ts", "")
        final_response = str(payload.get("last_assistant_message") or payload.get("response") or "")
        turn = build_turn(
            config=config,
            session_dir_=sd,
            session_id=sid,
            cwd=cwd,
            stop_seq=seq,
            stop_ts=stop_ts,
            transcript_path=transcript_path,
            final_response=final_response,
            marker_snapshot=marker,
        )
        if turn is not None:
            expected_turn_seq = int(turn["turn_id"])
            export_turn(config, sd, sid, _PLATFORM, cwd, turn)
    except Exception:
        pass
    finally:
        if expected_turn_seq is not None:
            _delete_open_turn(sd, expected_turn_seq=expected_turn_seq)


def subagent_stop() -> None:
    # SubagentStop keeps its historical "subagent_message" type on purpose:
    # renaming it would orphan the 966 sessions already recorded under it.
    # SubagentStart (below) uses a distinct "subagent_start" type; the
    # start/stop asymmetry is intentional, not an oversight.
    payload = _read_stdin()
    # `_STRIP_KEYS` drops `agent_transcript_path` as noise from every stored
    # event, but `build_turn` needs it here to locate the subagent's own
    # transcript — re-add it under a distinct key so it survives stripping.
    transcript_path = payload.get("agent_transcript_path")
    if transcript_path:
        payload = {**payload, "agent_transcript": transcript_path}
    _emit("subagent_message", payload)


def stop_failure() -> None:
    # Deliberately bypasses `_emit`'s generic catch-all: that always exports
    # as "interrupted", but a StopFailure is a distinct terminal state and
    # must export as "errored" instead.
    from thirdeye.otel_export import export_turn
    from thirdeye.platforms.claude.tracing import build_turn

    payload = _read_stdin()
    sid = payload.get("session_id")
    if not sid:
        return
    cwd = payload.get("cwd") or os.getcwd()
    config = Config.load()
    sd = session_dir(config.root, _PLATFORM, sid)
    marker = _read_open_turn(sd)
    expected_turn_seq = marker["turn_seq"] if marker is not None else None
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t="error",
        data=_strip_payload(payload),
    )
    if seq is None:
        return
    try:
        stop_ts = SessionReader(sd).get_event(seq).get("ts", "")
        turn = build_turn(
            config=config,
            session_dir_=sd,
            session_id=sid,
            cwd=cwd,
            stop_seq=seq,
            stop_ts=stop_ts,
            transcript_path=payload.get("transcript_path"),
            final_response="",
            status="errored",
            marker_snapshot=marker,
        )
        if turn is not None:
            expected_turn_seq = int(turn["turn_id"])
            export_turn(config, sd, sid, _PLATFORM, cwd, turn)
    except Exception:
        pass
    finally:
        if expected_turn_seq is not None:
            _delete_open_turn(sd, expected_turn_seq=expected_turn_seq)


def notification() -> None:
    _emit("notification", _read_stdin())


def permission_request() -> None:
    _emit("permission_request", _read_stdin())


def session_end() -> None:
    payload = _read_stdin()
    seq = _emit("session_end", payload)
    if seq is not None:
        sid = payload["session_id"]
        _close_stale_turn_if_open(
            Config.load(), sid, payload.get("cwd") or os.getcwd(), seq, force=True
        )
        Store(Config.load()).close_session(sid, platform=_PLATFORM)


def post_tool_use_failure() -> None:
    # Emits "tool_result", not "error", on purpose: web/routes/sessions.py pairs
    # each "tool_call" with a "tool_result", so a distinct type would leave failed
    # tool calls rendering as dangling. The failure is evident from the payload.
    _emit("tool_result", _read_stdin())


def subagent_start() -> None:
    _emit("subagent_start", _read_stdin())


def user_prompt_expansion() -> None:
    _emit("user_prompt_expansion", _read_stdin())


def pre_compact() -> None:
    _emit("compact_start", _read_stdin())


def post_compact() -> None:
    _emit("compact_end", _read_stdin())


def permission_denied() -> None:
    _emit("permission_denied", _read_stdin())
