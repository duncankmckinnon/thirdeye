from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict, cast

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.platforms.provenance import foreign_payload_reason
from thirdeye.reader import SessionReader
from thirdeye.span_ids import turn_span_id
from thirdeye.store import Store
from thirdeye.tags import TagStore, extract_hashtags
from thirdeye.usage.errlog import log_capture_error

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
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    # Valid JSON that is not an object (array, scalar, null) would make every
    # caller's payload.get() raise; fail open with an empty payload instead.
    return value if isinstance(value, dict) else {}


def _strip_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _STRIP_KEYS}


def _foreign_session_id(payload: dict) -> str:
    """Best-effort session identifier for a payload of unknown origin.

    A foreign payload is by definition not shaped like Claude's, so it need not
    carry session_id at all: Cursor names the same field conversation_id (or
    conversationId). Reading only session_id would log an empty id for exactly
    the payloads this diagnostic exists to correlate.
    """
    for key in ("session_id", "conversation_id", "conversationId"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _reject_foreign_payload(payload: dict) -> bool:
    """Reject only payloads carrying positive evidence of another platform."""
    try:
        reason = foreign_payload_reason(payload, expected=_PLATFORM)
    except Exception:
        return False
    if reason is None:
        return False

    try:
        config = Config.load()
        log_capture_error(
            thirdeye_home=config.root,
            phase="foreign_payload",
            level="warn",
            platform=_PLATFORM,
            session_id=_foreign_session_id(payload),
            message=reason,
        )
    except Exception:
        pass
    return True


def _log_hook_invocation(t: str, payload: dict) -> None:
    """Breadcrumb that a hook actually fired, before any Claude processing
    that could discard the payload. A background subagent's dispatching
    PostToolUse and its own SubagentStart can fire concurrently -- unlike a
    foreground one, where they're strictly sequential -- and one of the two is
    observed to silently go missing under that concurrency. Without this
    there's no way to tell "the hook was never invoked" from "it ran but this
    code discarded what it got", since both look identical: nothing.

    Foreign-payload rejection is the one step that runs earlier: a payload
    from another platform is not a Claude hook invocation, so claiming one
    here would corrupt the very signal this breadcrumb exists to provide. Such
    payloads get the warn-level foreign_payload entry instead.
    """
    try:
        config = Config.load()
        log_capture_error(
            thirdeye_home=config.root,
            phase="hook_invoked",
            level="info",
            platform=_PLATFORM,
            session_id=str(payload.get("session_id") or ""),
            message=(
                f"t={t} tool_use_id={payload.get('tool_use_id')!r} "
                f"agent_id={payload.get('agent_id')!r} prompt_id={payload.get('prompt_id')!r}"
            ),
        )
    except Exception:
        pass


def _emit(t: str, payload: dict) -> int | None:
    # Deliberately ahead of the breadcrumb: rejected payloads must leave no
    # hook_invoked trace, only the foreign_payload warning.
    if _reject_foreign_payload(payload):
        return None
    _log_hook_invocation(t, payload)
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


class _OptionalOpenTurnFields(TypedDict, total=False):
    """Split out rather than annotated `NotRequired` on the main TypedDict:
    this module uses `from __future__ import annotations`, so annotations are
    strings at runtime and `NotRequired` goes undetected, silently making the
    field required and invalidating every marker written before it existed.
    `total=False` inheritance is unaffected by stringized annotations.
    """

    # `message.id`s this turn has already exported a chat span for. Claude
    # writes a parallel tool batch as separate assistant frames with each
    # tool_result interleaved between them, so one `message.id` lands either
    # side of a `user` frame; the parser closes the group on that frame and the
    # trailing fragment reopens it under the same id. Both fragments derive the
    # same deterministic `chat_span_id`, so without this the call is exported
    # twice and its tokens counted twice.
    committed_call_ids: list[str]

    # `tool_use_id`s this turn has already exported a tool span for. Unlike a
    # chat span's `message.id`, a `tool_use_id` has no reopening/reparse path
    # to guard against -- this instead guards against `post_tool_use` itself
    # firing more than once for the same tool call, which derives the same
    # deterministic `tool_span_id` and would otherwise export it twice.
    committed_tool_use_ids: list[str]

    # `transcript_offset`'s value at turn start, before any live advancement.
    # `transcript_offset` itself only ever moves forward as live spans commit,
    # so once it has advanced past a parallel tool-dispatch message's transcript
    # lines, nothing live-side can ever see those bytes again -- even a
    # tool_use_id whose own span was never resolved. Stop-time reconstruction
    # uses this fixed start point instead, so it can still find and attach a
    # tool call the live path left behind. Absent on a marker written before
    # this field existed; `turn_start_offset()` falls back to the current
    # (possibly already-advanced) `transcript_offset` in that case, same as
    # the field always behaved before it existed.
    turn_start_offset: int


class OpenTurnMarker(_OptionalOpenTurnFields):
    turn_seq: int
    turn_span_id: str
    start_ts: str
    prompt: str
    prompt_id: str | None
    transcript_path: str | None
    transcript_offset: int
    last_frame_ts: str | None


# Only the reopen of a group that was just closed can duplicate a call, and
# that fragment is always a few frames behind, so a short window suffices to
# suppress it while keeping the marker small.
_COMMITTED_CALL_ID_LIMIT = 64

_OPEN_TURN_FIELDS = frozenset(OpenTurnMarker.__required_keys__)
_OPEN_TURN_LOCK_STATE = threading.local()

# A background subagent's dispatching PostToolUse and its own SubagentStart
# can fire concurrently, both wanting this lock -- see `_log_hook_invocation`.
# Blocking indefinitely risks the harness's own hook timeout killing this
# process with zero signal; every caller already tolerates a raised error
# (each wraps its `_locked_open_turn` use in `except OSError`/`except
# Exception`) and falls back to Stop-time reconstruction, so giving up after
# a bounded wait -- comfortably under any realistic hook timeout -- is
# strictly safer than blocking. Measured critical sections under this lock
# are sub-millisecond local disk I/O even under 5-way contention, so this
# budget is generous, not tight.
_LOCK_RETRY_BUDGET_S = 0.3
_LOCK_RETRY_INITIAL_DELAY_S = 0.005
_LOCK_RETRY_MAX_DELAY_S = 0.025


def _acquire_with_bounded_retry(fd: int, operation: int) -> None:
    deadline = time.monotonic() + _LOCK_RETRY_BUDGET_S
    delay = _LOCK_RETRY_INITIAL_DELAY_S
    while True:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out after {_LOCK_RETRY_BUDGET_S}s waiting for claude-open-turn.lock"
                ) from None
            time.sleep(max(0.0, min(delay, remaining)))
            delay = min(delay * 2, _LOCK_RETRY_MAX_DELAY_S)


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
        _acquire_with_bounded_retry(lock.fileno(), operation)
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
        or not _valid_committed_call_ids(marker.get("committed_call_ids"))
        or not _valid_committed_call_ids(marker.get("committed_tool_use_ids"))
        or not _valid_turn_start_offset(marker.get("turn_start_offset"))
    ):
        return None
    return cast(OpenTurnMarker, marker)


def _valid_committed_call_ids(value: object) -> bool:
    """Absent is valid: the field postdates the marker format."""
    return value is None or (
        isinstance(value, list) and all(isinstance(item, str) for item in value)
    )


def _valid_turn_start_offset(value: object) -> bool:
    """Absent is valid: the field postdates the marker format."""
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def committed_call_ids(marker: OpenTurnMarker) -> list[str]:
    """Call ids this turn has already exported a chat span for."""
    return list(marker.get("committed_call_ids") or [])


def committed_tool_use_ids(marker: OpenTurnMarker) -> list[str]:
    """Tool use ids this turn has already exported a tool span for."""
    return list(marker.get("committed_tool_use_ids") or [])


def turn_start_offset(marker: OpenTurnMarker) -> int:
    """The transcript offset at turn start, before any live advancement.

    Falls back to the current `transcript_offset` for a marker written
    before this field existed -- the same starting point Stop-time
    reconstruction always used before this field existed, so an in-flight
    turn spanning an upgrade degrades to old behavior rather than breaking.
    """
    value = marker.get("turn_start_offset")
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else marker["transcript_offset"]
    )


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
    newly_committed_call_ids: list[str] | None = None,
    newly_committed_tool_use_ids: list[str] | None = None,
) -> bool:
    """Advance a marker only if it still belongs to the expected turn."""
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or (last_frame_ts is not None and not isinstance(last_frame_ts, str))
        or not _valid_committed_call_ids(newly_committed_call_ids)
        or not _valid_committed_call_ids(newly_committed_tool_use_ids)
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
            if newly_committed_call_ids:
                # Merge against the marker just re-read under the lock, not a
                # copy the caller read earlier: another hook process may have
                # committed ids in between.
                merged = committed_call_ids(marker)
                merged.extend(i for i in newly_committed_call_ids if i not in merged)
                marker["committed_call_ids"] = merged[-_COMMITTED_CALL_ID_LIMIT:]
            if newly_committed_tool_use_ids:
                merged_tools = committed_tool_use_ids(marker)
                merged_tools.extend(
                    i for i in newly_committed_tool_use_ids if i not in merged_tools
                )
                marker["committed_tool_use_ids"] = merged_tools[-_COMMITTED_CALL_ID_LIMIT:]
            _open_turn_path(session_dir_).write_text(json.dumps(marker))
            return True
    except OSError:
        return False


def _delete_open_turn(session_dir_: Path, *, expected_turn_seq: int) -> bool:
    """Delete only the marker belonging to ``expected_turn_seq``."""
    try:
        with _locked_open_turn(session_dir_, fcntl.LOCK_EX):
            marker = _read_open_turn_unlocked(session_dir_)
            if marker is None:
                return False
            if marker["turn_seq"] != expected_turn_seq:
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
    if _reject_foreign_payload(payload):
        return
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
                "turn_span_id": str(turn_span_id(_PLATFORM, sid, seq)),
                "start_ts": start_ts,
                "prompt": prompt,
                "prompt_id": payload.get("prompt_id"),
                "transcript_path": transcript_path,
                "transcript_offset": offset,
                "turn_start_offset": offset,
                "last_frame_ts": start_ts,
            },
        )
    except Exception:
        pass


def pre_tool_use() -> None:
    _emit("tool_call", _read_stdin())


def post_tool_use() -> None:
    payload = _read_stdin()
    # Guarded here as well as inside _emit: _emit returns None both for a
    # rejected payload and for an ordinary one lacking session_id, so its
    # return value cannot gate the emit_live_spans call below. Double
    # rejection is harmless -- _reject_foreign_payload is pure apart from the
    # log write, and the first call has already returned by then.
    if _reject_foreign_payload(payload):
        return
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
    if _reject_foreign_payload(payload):
        return
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
    if _reject_foreign_payload(payload):
        return
    _log_hook_invocation("subagent_message", payload)
    sid = payload.get("session_id")
    if not sid:
        return
    cwd = payload.get("cwd") or os.getcwd()
    # `_STRIP_KEYS` drops `agent_transcript_path` as noise from every stored
    # event, but `build_turn` needs it here to locate the subagent's own
    # transcript — re-add it under a distinct key so it survives stripping.
    transcript_path = payload.get("agent_transcript_path")
    if transcript_path:
        payload = {**payload, "agent_transcript": transcript_path}
    # Deliberately bypasses `_emit`: its stale-open-turn check treats any
    # event whose `prompt_id` doesn't match the currently open turn's as
    # proof that turn was abandoned. A subagent can run well past its own
    # dispatching turn's Stop, so by the time this fires a *different*,
    # still-legitimately-open later turn may exist -- carrying the earlier
    # turn's prompt_id here would wrongly force that unrelated turn closed
    # and exported as "interrupted".
    config = Config.load()
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t="subagent_message",
        data=_strip_payload(payload),
    )
    if seq is None:
        return

    try:
        from thirdeye.otel_export import export_subagent_turn
        from thirdeye.platforms.claude.tracing import resolve_subagent_export

        sd = session_dir(config.root, _PLATFORM, sid)
        stop_ev = SessionReader(sd).get_event(seq)
        resolved = resolve_subagent_export(sd, sid, stop_ev)
        if resolved is None:
            return
        turn, tool_use_id = resolved
        export_subagent_turn(config, sd, sid, _PLATFORM, cwd, turn, tool_use_id)
    except Exception:
        pass


def stop_failure() -> None:
    # Deliberately bypasses `_emit`'s generic catch-all: that always exports
    # as "interrupted", but a StopFailure is a distinct terminal state and
    # must export as "errored" instead.
    from thirdeye.otel_export import export_turn
    from thirdeye.platforms.claude.tracing import build_turn

    payload = _read_stdin()
    if _reject_foreign_payload(payload):
        return
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
