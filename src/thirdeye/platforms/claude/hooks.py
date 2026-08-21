from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from thirdeye.config import Config
from thirdeye.env_capture import capture_env, env_to_tag
from thirdeye.meta import read_meta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.reader import SessionReader
from thirdeye.store import Store
from thirdeye.tags import TagStore, extract_hashtags
from thirdeye.usage.store import UsageStore
from thirdeye.writer import utc_iso_ms

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
    return Store(Config.load()).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t=t,
        data=_strip_payload(payload),
    )


def _open_turn_path(session_dir_: Path) -> Path:
    return session_dir_ / "claude-open-turn.json"


def _close_stale_turn_if_open(config: Config, session_id: str, cwd: str) -> None:
    """Claude Code's `Stop` hook is documented to never fire on a user
    interrupt, so an interrupted turn's open-turn marker (written by
    `user_prompt_submit`, deleted by `stop`) is the only signal that a turn
    never reached its normal close. If one is still on disk, this closes it
    out as `status="interrupted"` and exports whatever was captured up to now.

    Deliberately called only from `user_prompt_submit` and `session_end`, not
    from the shared `_emit()` that `pre_tool_use`/`post_tool_use`/
    `permission_request`/`subagent_start`/etc. all funnel through: those fire
    routinely *during* a still-open, un-interrupted turn, so treating "marker
    present" as proof of staleness there would export every ongoing turn
    early as a bogus "interrupted" one — and since `otel_export.export_turn`
    claims each turn id only once ever, that would permanently suppress the
    real, complete export `stop()` would otherwise produce for it. Only a
    hook that cannot legitimately fire while the marker's own turn is still
    in flight — a new prompt, the session ending — is safe to wire up here.
    """
    try:
        from thirdeye.otel_export import export_turn
        from thirdeye.platforms.claude.tracing import build_turn

        sd = session_dir(config.root, _PLATFORM, session_id)
        marker_path = _open_turn_path(sd)
        if not marker_path.exists():
            return
        turn = build_turn(
            config=config,
            session_dir_=sd,
            session_id=session_id,
            cwd=cwd,
            stop_seq=2**62,
            stop_ts=utc_iso_ms(),
            transcript_path=None,
            final_response="",
        )
        if turn is not None:
            turn["status"] = "interrupted"
            export_turn(config, sd, session_id, _PLATFORM, cwd, turn)
        marker_path.unlink(missing_ok=True)
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

    # If the previous turn was interrupted, this new prompt is the proof:
    # close it out and export it as "interrupted" before this turn's own
    # marker overwrites the file.
    _close_stale_turn_if_open(config, sid, cwd)

    prompt = payload.get("prompt") or ""
    seq = Store(config).append_event(
        session_id=sid,
        platform=_PLATFORM,
        cwd=cwd,
        t="user_message",
        data=_strip_payload(payload),
    )
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
        offset = int(UsageStore(sd).read_state().get("transcript_offset", 0))
        _open_turn_path(sd).write_text(
            json.dumps(
                {
                    "turn_seq": seq,
                    "start_ts": start_ts,
                    "prompt": prompt,
                    "transcript_path": payload.get("transcript_path"),
                    "transcript_offset": offset,
                }
            )
        )
    except Exception:
        pass


def pre_tool_use() -> None:
    _emit("tool_call", _read_stdin())


def post_tool_use() -> None:
    _emit("tool_result", _read_stdin())


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
        final_response = str(
            payload.get("last_assistant_message") or payload.get("response") or ""
        )
        turn = build_turn(
            config=config,
            session_dir_=sd,
            session_id=sid,
            cwd=cwd,
            stop_seq=seq,
            stop_ts=stop_ts,
            transcript_path=transcript_path,
            final_response=final_response,
        )
        if turn is not None:
            export_turn(config, sd, sid, _PLATFORM, cwd, turn)
    except Exception:
        pass
    finally:
        _open_turn_path(sd).unlink(missing_ok=True)


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
    _emit("error", _read_stdin())


def notification() -> None:
    _emit("notification", _read_stdin())


def permission_request() -> None:
    _emit("permission_request", _read_stdin())


def session_end() -> None:
    payload = _read_stdin()
    sid = payload.get("session_id")
    if sid:
        cwd = payload.get("cwd") or os.getcwd()
        _close_stale_turn_if_open(Config.load(), sid, cwd)
    if _emit("session_end", payload) is not None:
        Store(Config.load()).close_session(payload["session_id"], platform=_PLATFORM)


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
