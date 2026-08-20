"""Mirror thirdeye events into Logfire as OpenTelemetry spans, live.

Every thirdeye event (tool call, message, notification, ...) funnels through
one call site, ``Store.append_event`` — including calls made from inside
Claude Code / Codex hook subprocesses. That makes it the one place to add
Logfire export and have it cover every platform automatically, with no
separate sync step.

Each underlying LLM call within a turn (Claude's transcript / Codex's rollout
can both hold several — an agentic turn routinely makes multiple model calls
before it's done) gets its *own* span, populated the way Logfire's own
GenAI-provider instrumentations populate theirs: ``gen_ai.usage.*`` for that
call's own tokens (not a turn-wide sum), plus ``gen_ai.input.messages`` /
``gen_ai.output.messages`` holding the actual conversation content — text,
tool calls and their results, and (Claude only, for now) reasoning/thinking
content — as OTel GenAI semantic-convention message parts. See
``platforms/claude/usage.py``'s ``extract_new_calls_claude`` for how those are
built from the raw transcript, and ``export_llm_calls`` below for how they
reach Logfire. This is a second, explicit entry point into export — call
content is captured by a separate pipeline from ordinary session events
(straight from the transcript/rollout, never through ``Store.append_event``).

Each thirdeye session becomes one Logfire trace. The first event exported for
a session becomes that trace's root span; its real (SDK-generated) trace_id
and span_id are persisted to ``otel.json`` in the session directory so later
events — emitted by separate, later hook subprocesses — can reference the same
trace and parent under it. Every exported event's own span id is *also*
persisted, individually, to ``otel-spans/<seq>.json`` — this is what lets a
turn's LLM-call spans parent directly under the specific ``assistant_message``
/ ``agent_turn`` span for that turn, rather than dangling off the trace root
as unconnected siblings. A ``tool_call`` is never exported on its own; it is
folded into the matching ``tool_result`` as one span with an accurate
start/end duration, found by scanning back through the session's own stored
events for the nearest event sharing the same tool id (Claude's
``tool_use_id`` / Codex's ``call_id``).

The actual Logfire call — including a flush, which is a real network round
trip — never happens in the hook process itself. ``export_event`` (called
from the hook process) only ever writes a small job file and spawns a
detached, unwaited-for child process (``thirdeye.otel_worker``) to do the
work, so a slow or unreachable Logfire endpoint adds no latency to the tool
call that triggered it. ``start_new_session=True`` detaches the child from
the hook's process group so it survives even if Claude Code kills that group
once the hook returns.

Safety: this module runs inside hook subprocesses whose stdout Claude Code
treats as a hook decision. Every public function here must never raise and
never write to stdout — a Logfire hiccup must be invisible to the tool call it
rode in on. Failures are logged via ``usage.errlog`` (a file), same as every
other capture-side failure path in thirdeye.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.ids import new_ulid
from thirdeye.paths import otel_jobs_dir, otel_span_path, otel_state_path
from thirdeye.usage.errlog import log_capture_error

# Cache across calls *within one process*. Each hook invocation is its own
# short-lived process, so this only saves repeat configure() calls when a
# single process exports more than one event (e.g. the eval runner).
_state: dict[str, Any] = {"attempted": False, "instance": None}

_TOOL_ID_KEYS = ("tool_use_id", "call_id")
_ATTR_PRIMITIVES = (str, bool, int, float)
# Logfire's own convention for telling its backend which JSON-encoded string
# attributes to parse back into structured data (objects/arrays) rather than
# render as opaque text. Spans built via `logfire.span()` get this set
# automatically; ours go through the raw OTel API, so `_flatten_attrs` sets
# it by hand. Without it, `gen_ai.input.messages` / `.output.messages` never
# render as a chat view in the UI, just a flat "prompt" text field.
_LOGFIRE_JSON_SCHEMA_KEY = "logfire.json_schema"

# Keyed by the *end* event type -> the matching *start* event type it pairs
# with to form one real-duration span. A start-side type with no entry here
# (tool_call) is never exported on its own.
_PAIR_START_FOR_END = {"tool_result": "tool_call"}

# How long export_llm_calls waits for the triggering event's own span record
# to show up before giving up and parenting under the session root instead.
# Both workers are spawned within moments of each other from the same hook
# call, and each pays real (if brief) network setup cost before either one
# gets as far as persisting anything, so a handful of short polls is usually
# enough to observe the other side land first.
_CALL_PARENT_RETRIES = 6
_CALL_PARENT_RETRY_DELAY_S = 0.1

_SCAN_CAP = 500  # bound on how far back to search for a matching start event
_FLUSH_TIMEOUT_MS = 2000


def is_available() -> bool:
    try:
        import logfire  # noqa: F401
    except ImportError:
        return False
    return True


def status(config: Config) -> dict[str, Any]:
    """Cheap, side-effect-free summary for `thirdeye logfire status` / the UI."""
    return {
        "package_installed": is_available(),
        "enabled": config.logfire.enabled,
        "has_token": bool(config.logfire.token),
        "project": config.logfire.project,
        "token_suffix": config.logfire.token[-4:] if config.logfire.token else None,
    }


def _silence_background_noise() -> None:
    """Logfire/OTel log network hiccups (bad token, unreachable host) on their
    own, straight to stderr, from a background thread — a `warnings.warn()`
    from Logfire's token-check thread, and an `opentelemetry` logger call from
    the OTLP exporter. Both fire asynchronously, sometimes after our own call
    into logfire has already returned, so scoping suppression to a `with`
    block around that call does not reliably catch them (confirmed by hand:
    `warnings.catch_warnings()` + `redirect_stderr` around configure() let the
    warning through anyway). A permanent, process-global filter is the only
    thing that reliably catches a write timed after our call site.

    This runs inside hook subprocesses whose stdout Claude Code may read as a
    hook decision; even stderr noise here is a leak into the user's actual
    coding session, so this is deliberately permanent for the process, not
    scoped to one export call.
    """
    warnings.filterwarnings("ignore", module=r"logfire\..*")
    logging.getLogger("opentelemetry").setLevel(logging.CRITICAL + 1)


def _scrub_callback(match: Any) -> Any:
    """Let through only Logfire's own `"session"` default scrubbing pattern;
    every other pattern (password, secret, api key, credential, ...) is left
    to redact as normal.

    thirdeye's captured content is a coding agent's own tool calls and
    messages — routinely full of legitimate uses of the word "session" (file
    paths, transcript discussion, thirdeye's own attribute names) with no
    sensitivity to it at all, which is why "most of them" (per the user who
    asked for this) were getting blanked out. `match.pattern_match.group(0)`
    is the literal substring that triggered the match; comparing it against
    "session" (case-insensitively, since the pattern is compiled with
    re.IGNORECASE) is how the docs' own examples distinguish which pattern
    fired — no other default pattern's match text can equal "session".

    Caveat inherited from Logfire's own scrubbing design, not introduced
    here: it finds only the *first* (leftmost) match in a value and redacts
    the value wholesale, so a value containing both "session" and a genuine
    secret later in the same string — with "session" appearing first — would
    still be exempted here. That's the same granularity Logfire's own
    documented callback pattern operates at.
    """
    if match.pattern_match.group(0).lower() == "session":
        return match.value
    return None


def _get_instance(config: Config, platform: str):
    """Return a configured Logfire instance, or None if export is inactive.

    Never raises. Cached for the life of this process. The service name is
    fixed to the *first* platform seen by this process, so Claude Code and
    Codex sessions show up as distinct services (``claude`` / ``codex``) in
    Logfire instead of one indistinguishable ``thirdeye`` — the OTel scope
    (the tracer name below) already says "thirdeye", so it needn't be
    repeated in the service name too. Each hook invocation is its own
    short-lived, single-platform process, so this only matters for a process
    that exports more than one platform's events (e.g. the eval runner),
    where the first platform wins.
    """
    if _state["attempted"]:
        return _state["instance"]
    _state["attempted"] = True
    if not config.logfire.enabled or not config.logfire.token:
        return None
    try:
        import logfire
    except ImportError:
        return None
    _silence_background_noise()
    try:
        instance = logfire.configure(
            token=config.logfire.token,
            send_to_logfire=True,
            console=False,
            service_name=platform,
            scrubbing=logfire.ScrubbingOptions(callback=_scrub_callback),
        )
    except Exception as exc:
        log_capture_error(thirdeye_home=config.root, phase="logfire_configure", error=exc)
        return None
    _state["instance"] = instance
    return instance


def _ts_to_ns(ts: str) -> int:
    s = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    return int(datetime.fromisoformat(s).timestamp() * 1_000_000_000)


def _flatten_attrs(data: Any) -> dict[str, Any]:
    """OTel attribute values must be a primitive or a homogeneous sequence of
    one; anything else (nested dicts, mixed lists) is JSON-encoded to a
    string, and its key recorded under a `logfire.json_schema` companion
    attribute so Logfire's backend parses it back into structured data
    (object/array) instead of rendering it as opaque text.
    """
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    schema_properties: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, _ATTR_PRIMITIVES):
            out[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, _ATTR_PRIMITIVES) for v in value
        ):
            out[key] = list(value)
        else:
            try:
                out[key] = json.dumps(value, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                out[key] = str(value)
            else:
                schema_properties[key] = {"type": "object" if isinstance(value, dict) else "array"}
    if schema_properties:
        out[_LOGFIRE_JSON_SCHEMA_KEY] = json.dumps(
            {"type": "object", "properties": schema_properties}
        )
    return out


def _merge_raw(*parts: Any) -> dict[str, Any]:
    """Merge zero or more raw (pre-flatten) attribute dicts, later keys
    overriding earlier ones. Used to combine several sources into one dict
    before a single `_flatten_attrs` pass, so the resulting
    `logfire.json_schema` covers every JSON-encoded key instead of just
    whichever source was flattened last.
    """
    merged: dict[str, Any] = {}
    for part in parts:
        if isinstance(part, dict):
            merged.update(part)
    return merged


def _tool_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in _TOOL_ID_KEYS:
        v = data.get(key)
        if v:
            return str(v)
    return None


def _read_root(path: Path) -> tuple[int, int] | None:
    try:
        raw = json.loads(path.read_text())
        return int(raw["trace_id"], 16), int(raw["span_id"], 16)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _atomic_create(path: Path, payload: str) -> bool:
    """Write `payload` to `path` iff it doesn't already exist. Returns whether
    this call was the one that created it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, payload.encode())
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        return False


def _root_or_ownership(root_path: Path) -> tuple[tuple[int, int] | None, Path | None]:
    """Return an existing root or exclusive ownership of creating it.

    Span ids are SDK-generated, so the root cannot be reserved directly. A
    short-lived sibling lock serializes the read/start/persist sequence across
    detached workers and prevents two first spans from creating split traces.
    """
    lock_path = root_path.with_name(f"{root_path.name}.lock")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        root = _read_root(root_path)
        if root is not None:
            return root, None
        if _atomic_create(lock_path, str(os.getpid())):
            return None, lock_path
        time.sleep(0.02)

    # A worker may have died while owning the lock. Reclaim only an old lock;
    # otherwise decline this export rather than emitting a split trace.
    try:
        if time.time() - lock_path.stat().st_mtime > 2.0:
            lock_path.unlink(missing_ok=True)
            if _atomic_create(lock_path, str(os.getpid())):
                return None, lock_path
    except OSError:
        pass
    return _read_root(root_path), None


def _span_payload(trace_id: int, span_id: int) -> str:
    return json.dumps({"trace_id": f"{trace_id:032x}", "span_id": f"{span_id:016x}"})


def _create_root_atomic(path: Path, trace_id: int, span_id: int) -> tuple[int, int]:
    """Persist (trace_id, span_id) as the session's root, first writer wins.

    A losing writer's own generated ids are simply discarded in favor of the
    winner's, so every process agrees on one root going forward.
    """
    if _atomic_create(path, _span_payload(trace_id, span_id)):
        return trace_id, span_id
    return _read_root(path) or (trace_id, span_id)


def _persist_span(path: Path, trace_id: int, span_id: int) -> None:
    """Best-effort record of one exported event's own span id, keyed by its
    seq — so a later, separate process (LLM-call export) can parent under
    this exact span rather than only ever the session root. Every seq is
    exported at most once, so a collision here would mean something else is
    wrong; either way this must never raise, so a loss is silently ignored
    rather than reconciled the way `_create_root_atomic` reconciles a root
    collision.
    """
    _atomic_create(path, _span_payload(trace_id, span_id))


def _parent_context(trace_id: int, span_id: int):
    from opentelemetry import trace as otel_trace

    span_context = otel_trace.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=otel_trace.TraceFlags(otel_trace.TraceFlags.SAMPLED),
    )
    return otel_trace.set_span_in_context(otel_trace.NonRecordingSpan(span_context))


def _find_matching_start(
    reader: Any, before_seq: int, start_type: str, tool_id: str | None
) -> dict[str, Any] | None:
    """Scan backward from before_seq-1 for the matching start event.

    An id match (Claude's tool_use_id / Codex's call_id) wins immediately.
    With no id on either side, falls back to the nearest preceding event of
    start_type — a best-effort pairing for older payload shapes that carry no
    stable id. Bounded by _SCAN_CAP so a very long session can't turn every
    tool_result into an O(session length) scan.
    """
    fallback = None
    lo = max(0, before_seq - _SCAN_CAP)
    for seq in range(before_seq - 1, lo - 1, -1):
        try:
            event = reader.get_event(seq)
        except (IndexError, OSError):
            continue
        if event.get("t") != start_type:
            continue
        if tool_id is not None and _tool_id(event.get("data")) == tool_id:
            return event
        if fallback is None:
            fallback = event
    return fallback


def export_event(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    t: str,
    seq: int,
    ts: str,
    data: Any,
) -> None:
    """Hand this event off for background export. Never raises, never blocks
    on network I/O — the actual Logfire call happens in a detached child
    process this spawns and does not wait for. See module docstring.
    """
    if not config.logfire.enabled or not config.logfire.token:
        return
    # Codex tools are reconstructed as children of the completed turn's
    # inference span from the rollout JSONL. Exporting their local event-store
    # representation too would create a second, generic copy elsewhere in the
    # trace. Claude still uses event pairing here.
    if (
        platform == "codex"
        and t in {"tool_call", "tool_result"}
        and isinstance(data, dict)
        and data.get("thirdeye.codex_turn_batched") is True
    ):
        return
    if t == "tool_call":
        return  # nothing to export yet; folded into the matching tool_result
    try:
        _spawn_worker(
            thirdeye_home=config.root,
            session_dir_=session_dir_,
            session_id=session_id,
            platform=platform,
            cwd=cwd,
            t=t,
            seq=seq,
            ts=ts,
            data=data,
        )
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="logfire_export_spawn",
            error=exc,
            platform=platform,
            session_id=session_id,
        )


def _write_job(thirdeye_home: Path, payload: dict[str, Any]) -> Path:
    jobs_dir = otel_jobs_dir(thirdeye_home)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_path = jobs_dir / f"{new_ulid()}.json"
    job_path.write_text(json.dumps(payload, default=str))
    return job_path


def _spawn(job_path: Path) -> None:
    """Hand a job file to a detached ``thirdeye.otel_worker``.

    A job *file*, not a pipe: writing to a subprocess's stdin can itself block
    the caller if the payload is large and the child hasn't started reading
    yet (a big tool output could fill the pipe buffer). A local file write has
    no such risk. ``start_new_session=True`` gives the child its own process
    group so it isn't killed alongside the hook that spawned it.
    """
    subprocess.Popen(
        [sys.executable, "-m", "thirdeye.otel_worker", str(job_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _spawn_worker(
    *,
    thirdeye_home: Path,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    t: str,
    seq: int,
    ts: str,
    data: Any,
) -> None:
    job_path = _write_job(
        thirdeye_home,
        {
            "session_dir": str(session_dir_),
            "session_id": session_id,
            "platform": platform,
            "cwd": cwd,
            "t": t,
            "seq": seq,
            "ts": ts,
            "data": data,
        },
    )
    _spawn(job_path)


def export_llm_calls(
    config: Config | None,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str | None,
    calls: list[dict[str, Any]],
) -> None:
    """Hand a whole batch of new LLM calls (from one turn's transcript/rollout
    segment) off for background export as children of the interaction span
    they belong to (see module docstring). Never raises, never blocks on
    network I/O — same guarantee as `export_event`, which this deliberately
    does *not* delegate to: every call in one batch shares the same `seq`
    (the triggering event's), so they belong in one job and one flush, not
    one detached subprocess each — a turn with several LLM calls would
    otherwise fan out into that many concurrent processes all hitting
    Logfire's ingest endpoint for a single hook invocation.

    Each item in `calls` is `{"seq": int, "ts": str, "call_id": str, "data":
    dict}`, where `data` is the fully-formed span attributes (gen_ai.usage.*,
    gen_ai.input.messages, gen_ai.output.messages, ...) — see
    `platforms/claude/usage.py`'s `extract_new_calls_claude`.

    No-op if `config` or `cwd` is missing (older call sites that predate this
    integration don't pass them), or no call carries a timestamp. Calls whose
    `call_id` has already been exported — Codex repeat-reports and
    same-offset capture races can hand this the same call more than once —
    are silently dropped rather than exported again as a duplicate span
    nested under the same interaction.
    """
    if config is None or cwd is None:
        return
    if not config.logfire.enabled or not config.logfire.token:
        return
    claimed = [
        call
        for call in calls
        if call.get("ts") and _claim_call_export(session_dir_, call["call_id"])
    ]
    if not claimed:
        return
    try:
        job_path = _write_job(
            config.root,
            {
                "kind": "llm_calls",
                "session_dir": str(session_dir_),
                "session_id": session_id,
                "platform": platform,
                "cwd": cwd,
                "calls": claimed,
            },
        )
        _spawn(job_path)
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="logfire_llm_calls_export_spawn",
            error=exc,
            platform=platform,
            session_id=session_id,
        )


def export_codex_turn(
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    seq: int,
    turn: dict[str, Any],
) -> bool:
    """Export one rollout-reconstructed Codex turn and its tool children."""
    if not config.logfire.enabled or not config.logfire.token:
        return False
    turn_id = str(turn.get("turn_id") or "")
    if not turn_id or not turn.get("start_ts") or not turn.get("end_ts"):
        return False
    state_path = _call_claim_path(session_dir_, f"codex-turn:{turn_id}")
    try:
        state = state_path.read_text()
    except OSError:
        state = ""
    if state == "pending":
        try:
            stale = time.time() - state_path.stat().st_mtime > 30.0
        except OSError:
            stale = True
        if stale:
            state_path.unlink(missing_ok=True)
            state = ""
        else:
            return True
    if state in {"sent", "1"}:
        return True
    if not _atomic_create(state_path, "pending"):
        return True
    try:
        job_path = _write_job(
            config.root,
            {
                "kind": "codex_turn",
                "session_dir": str(session_dir_),
                "session_id": session_id,
                "cwd": cwd,
                "seq": seq,
                "turn": turn,
                "state_path": str(state_path),
            },
        )
        _spawn(job_path)
        return True
    except Exception as exc:
        state_path.unlink(missing_ok=True)
        log_capture_error(
            thirdeye_home=config.root,
            phase="logfire_codex_turn_export_spawn",
            error=exc,
            platform="codex",
            session_id=session_id,
        )
        return False


def _message(role: str, content: str) -> list[dict[str, Any]]:
    return [{"role": role, "parts": [{"type": "text", "content": content}]}]


def _preceding_user_message_ns(session_dir_: Path, seq: int) -> int | None:
    """Return the timestamp of the latest user event before ``seq``.

    Codex's rollout timestamps can precede the hook timestamp which recorded
    the prompt by a small amount. Logfire sorts by span start time, so without
    a floor the model call can appear above the user input that initiated it.
    """
    from thirdeye.reader import SessionReader

    try:
        events = list(
            SessionReader(session_dir_).iter_events(types={"user_message"}, seq_range=(0, seq))
        )
    except (OSError, IndexError, ValueError):
        return None
    if not events:
        return None
    ts = events[-1].get("ts")
    return _ts_to_ns(ts) if ts else None


def _export_codex_turn_inner(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    cwd: str,
    seq: int,
    turn: dict[str, Any],
) -> bool:
    """Emit a semconv inference span with semconv execute-tool children."""
    from opentelemetry.trace import SpanKind

    instance = _get_instance(config, "codex")
    if instance is None:
        return False
    tracer = instance.config.get_tracer_provider().get_tracer("thirdeye")
    root_path = otel_state_path(session_dir_)
    parent, root_lock = _root_or_ownership(root_path)
    if parent is None and root_lock is None:
        return False
    model = str(turn.get("model") or "")
    usage_keys = {
        "input_tokens": "gen_ai.usage.input_tokens",
        "output_tokens": "gen_ai.usage.output_tokens",
        "cached_input_tokens": "gen_ai.usage.cache_read.input_tokens",
        "cache_write_input_tokens": "gen_ai.usage.cache_creation.input_tokens",
        "reasoning_output_tokens": "gen_ai.usage.reasoning.output_tokens",
    }
    calls = turn.get("calls") or [
        {
            "start_ts": turn["start_ts"],
            "end_ts": turn["end_ts"],
            "input_messages": (
                _message("user", str(turn["user_prompt"])) if turn.get("user_prompt") else []
            ),
            "output_messages": (
                _message("assistant", str(turn["assistant_output"]))
                if turn.get("assistant_output")
                else []
            ),
            "usage": turn.get("usage") or {},
            "tools": turn.get("tools") or [],
        }
    ]
    context = _parent_context(*parent) if parent else None
    user_message_ns = _preceding_user_message_ns(session_dir_, seq)
    try:
        for call_index, call in enumerate(calls):
            attrs: dict[str, Any] = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.conversation.id": session_id,
                "thirdeye.platform": "codex",
                "thirdeye.cwd": cwd,
                "thirdeye.seq": seq,
                "codex.turn.id": str(turn.get("turn_id") or ""),
                "codex.call.index": call_index,
            }
            if model:
                attrs["gen_ai.request.model"] = model
            if call.get("input_messages"):
                attrs["gen_ai.input.messages"] = call["input_messages"]
            if call.get("output_messages"):
                attrs["gen_ai.output.messages"] = call["output_messages"]
            for source, target in usage_keys.items():
                value = (call.get("usage") or {}).get(source)
                if isinstance(value, int):
                    attrs[target] = value

            start_ns = _ts_to_ns(call.get("start_ts") or turn["start_ts"])
            if call_index == 0 and user_message_ns is not None:
                # Preserve rollout timing unless hook/rollout clock skew would
                # render the initiating prompt after this call.
                start_ns = max(start_ns, user_message_ns + 1)
            span = tracer.start_span(
                f"chat {model}" if model else "chat",
                context=context,
                kind=SpanKind.CLIENT,
                start_time=start_ns,
                attributes=_flatten_attrs(attrs),
            )
            span.end(end_time=_ts_to_ns(call.get("end_ts") or turn["end_ts"]))
            span_ctx = span.get_span_context()
            if parent is None and call_index == 0:
                _create_root_atomic(root_path, span_ctx.trace_id, span_ctx.span_id)
                # Later calls in a rootless session share the first call's
                # trace, while remaining siblings rather than nesting calls.
                context = _parent_context(span_ctx.trace_id, span_ctx.span_id)

            tool_parent = _parent_context(span_ctx.trace_id, span_ctx.span_id)
            for tool in call.get("tools") or []:
                name = str(tool.get("name") or "unknown_tool")
                tool_attrs: dict[str, Any] = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": name,
                    "gen_ai.conversation.id": session_id,
                    "thirdeye.platform": "codex",
                }
                if tool.get("call_id"):
                    tool_attrs["gen_ai.tool.call.id"] = str(tool["call_id"])
                if tool.get("arguments") not in (None, ""):
                    tool_attrs["gen_ai.tool.call.arguments"] = tool["arguments"]
                if tool.get("result") not in (None, ""):
                    tool_attrs["gen_ai.tool.call.result"] = tool["result"]
                child = tracer.start_span(
                    f"execute_tool {name}",
                    context=tool_parent,
                    kind=SpanKind.INTERNAL,
                    start_time=_ts_to_ns(
                        tool.get("start_ts") or call.get("start_ts") or turn["start_ts"]
                    ),
                    attributes=_flatten_attrs(tool_attrs),
                )
                child.end(
                    end_time=_ts_to_ns(
                        tool.get("end_ts")
                        or tool.get("start_ts")
                        or call.get("end_ts")
                        or turn["end_ts"]
                    )
                )
    finally:
        if root_lock is not None:
            root_lock.unlink(missing_ok=True)

    return instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS) is not False


def _call_claim_path(session_dir_: Path, call_id: str) -> Path:
    # Hashed, not the raw call_id, as a filename: nothing guarantees any
    # platform's call_id is filesystem- or path-traversal-safe (Codex's, for
    # instance, contains a raw colon: `cum:<n>`).
    digest = hashlib.sha256(call_id.encode()).hexdigest()
    return session_dir_ / "otel-calls-sent" / f"{digest}.json"


def _claim_call_export(session_dir_: Path, call_id: str) -> bool:
    """First-wins claim on exporting this call_id's span, ever, for this
    session. Same-offset capture races (two hook invocations independently
    rediscovering the same not-yet-persisted rows) can otherwise hand the
    same call to export twice; every duplicate describes the same underlying
    call identically, so first-wins is correct and needs no reconciliation —
    it just answers whether THIS call is the one that gets to export.
    """
    return _atomic_create(_call_claim_path(session_dir_, call_id), "1")


def _export_event_inner(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    t: str,
    seq: int,
    ts: str,
    data: Any,
) -> None:
    if t == "tool_call":
        return  # folded into the matching tool_result below

    instance = _get_instance(config, platform)
    if instance is None:
        return

    tracer = instance.config.get_tracer_provider().get_tracer("thirdeye")
    end_ns = _ts_to_ns(ts)

    start_type = _PAIR_START_FOR_END.get(t)
    tool_id: str | None = None
    if start_type is not None:
        from thirdeye.reader import SessionReader

        tool_id = _tool_id(data)
        matched = _find_matching_start(SessionReader(session_dir_), seq, start_type, tool_id)
        start_ns = _ts_to_ns(matched["ts"]) if matched else end_ns
        attrs = _flatten_attrs(_merge_raw(matched.get("data") if matched else None, data))
        tool_name = attrs.get("tool_name")
        name = f"tool: {tool_name}" if tool_name else t
    else:
        start_ns = end_ns
        raw_attrs = data if isinstance(data, dict) else {}
        if t == "user_message":
            # Retain the raw payload while also supplying the GenAI semantic
            # attribute Logfire uses to recognize and render user messages.
            prompt = raw_attrs.get("prompt") or raw_attrs.get("message") or raw_attrs.get("content")
            if isinstance(prompt, str) and prompt:
                raw_attrs = dict(raw_attrs)
                raw_attrs["gen_ai.input.messages"] = _message("user", prompt)
        attrs = _flatten_attrs(raw_attrs)
        name = t

    # gen_ai.conversation.id, not thirdeye.session_id: Logfire's default
    # scrubber redacts any attribute whose key or value matches /session/,
    # which would blank out our own session identifier. This mirrors the
    # OTel GenAI semconv key thirdeye.usage.types.UsageRow already uses for
    # the same concept.
    attrs["gen_ai.conversation.id"] = session_id
    attrs["thirdeye.platform"] = platform
    attrs["thirdeye.cwd"] = cwd
    attrs["thirdeye.seq"] = seq

    root_path = otel_state_path(session_dir_)
    # A tool span nests under the specific chat call that requested it, so
    # tool execution reads as a child of the model call that made it rather
    # than a flat sibling under the whole session; every other event type
    # (user_message, assistant_message, ...) still nests directly under root.
    if tool_id is not None:
        parent = _resolve_tool_parent(session_dir_, root_path, tool_id)
    else:
        parent = _read_root(root_path)

    if parent is None:
        parent, root_lock = _root_or_ownership(root_path)
        if parent is None and root_lock is None:
            return
        if parent is None:
            try:
                span = tracer.start_span(name, start_time=start_ns, attributes=attrs)
                span.end(end_time=end_ns)
                ctx = span.get_span_context()
                _create_root_atomic(root_path, ctx.trace_id, ctx.span_id)
            finally:
                root_lock.unlink(missing_ok=True)
        else:
            span = tracer.start_span(
                name, context=_parent_context(*parent), start_time=start_ns, attributes=attrs
            )
            span.end(end_time=end_ns)
            ctx = span.get_span_context()
    else:
        span = tracer.start_span(
            name, context=_parent_context(*parent), start_time=start_ns, attributes=attrs
        )
        span.end(end_time=end_ns)
        ctx = span.get_span_context()

    _persist_span(otel_span_path(session_dir_, seq), ctx.trace_id, ctx.span_id)
    instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS)


def _resolve_parent(specific_path: Path, root_path: Path) -> tuple[int, int] | None:
    """Poll briefly for a specific span record, falling back to the session
    root (whatever it is *right now* — possibly still None, if this is
    racing to be the very first export in a brand-new session too) if it
    never shows up in time.

    The record is built by a separate worker process racing this one with no
    ordering guarantee between them, hence the poll rather than a single read.
    """
    for attempt in range(_CALL_PARENT_RETRIES):
        found = _read_root(specific_path)
        if found is not None:
            return found
        if attempt < _CALL_PARENT_RETRIES - 1:
            time.sleep(_CALL_PARENT_RETRY_DELAY_S)
    return _read_root(root_path)


def _resolve_call_parent(
    session_dir_: Path, root_path: Path, triggering_seq: int
) -> tuple[int, int] | None:
    """Find the span a turn's LLM-call spans should nest under: the specific
    triggering event's own span (typically its `assistant_message`).
    """
    return _resolve_parent(otel_span_path(session_dir_, triggering_seq), root_path)


def _tool_call_span_path(session_dir_: Path, tool_use_id: str) -> Path:
    # Hashed, not the raw tool_use_id, as a filename: nothing guarantees any
    # platform's tool call id is filesystem- or path-traversal-safe.
    digest = hashlib.sha256(tool_use_id.encode()).hexdigest()
    return session_dir_ / "otel-tool-call-spans" / f"{digest}.json"


def _resolve_tool_parent(
    session_dir_: Path, root_path: Path, tool_use_id: str
) -> tuple[int, int] | None:
    """Find the span a tool call should nest under: the specific `chat` LLM
    call whose response requested it, so tool execution spans read as
    children of the model call that made them rather than flat siblings
    under the whole session.
    """
    return _resolve_parent(_tool_call_span_path(session_dir_, tool_use_id), root_path)


def _tool_call_ids(data: dict[str, Any]) -> list[str]:
    """Extract every tool_call id from a chat call's raw (pre-flatten)
    gen_ai.output.messages, so each can be recorded as a lookup key for the
    span that produced it.
    """
    ids: list[str] = []
    for message in data.get("gen_ai.output.messages") or []:
        if not isinstance(message, dict):
            continue
        for part in message.get("parts") or []:
            if isinstance(part, dict) and part.get("type") == "tool_call" and part.get("id"):
                ids.append(str(part["id"]))
    return ids


def _export_llm_calls_inner(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    calls: list[dict[str, Any]],
) -> None:
    """Export a whole batch of LLM-call spans in one process: one shared
    Logfire instance, one flush — never one subprocess and one network round
    trip per call (see `export_llm_calls`). Every call shares the same
    triggering seq, so there's exactly one parent to resolve for the batch.
    """
    if not calls:
        return
    instance = _get_instance(config, platform)
    if instance is None:
        return

    tracer = instance.config.get_tracer_provider().get_tracer("thirdeye")
    root_path = otel_state_path(session_dir_)
    parent = _resolve_call_parent(session_dir_, root_path, calls[0]["seq"])
    user_message_ns = _preceding_user_message_ns(session_dir_, calls[0]["seq"])

    for call_index, call in enumerate(calls):
        ts_ns = _ts_to_ns(call["ts"])
        if call_index == 0 and user_message_ns is not None:
            ts_ns = max(ts_ns, user_message_ns + 1)
        # gen_ai.input.messages / .output.messages are nested lists of dicts —
        # raw OTel attributes can only hold a primitive or a homogeneous
        # sequence of one, so _flatten_attrs JSON-encodes them to a string,
        # same as it already does for ordinary event data. This is the same
        # wire format Logfire's own SDK produces from its convenience
        # `span.set_attribute(OUTPUT_MESSAGES, [...])` API, which does this
        # encoding internally before anything reaches OTLP.
        attrs = _flatten_attrs(call["data"])
        attrs["gen_ai.conversation.id"] = session_id
        attrs["thirdeye.platform"] = platform
        attrs["thirdeye.cwd"] = cwd
        attrs["thirdeye.seq"] = call["seq"]
        model = attrs.get("gen_ai.response.model")
        name = f"chat {model}" if model else "chat"

        if parent is None:
            span = tracer.start_span(name, start_time=ts_ns, attributes=attrs)
            span.end(end_time=ts_ns)
            ctx = span.get_span_context()
            parent = _create_root_atomic(root_path, ctx.trace_id, ctx.span_id)
        else:
            span = tracer.start_span(
                name, context=_parent_context(*parent), start_time=ts_ns, attributes=attrs
            )
            span.end(end_time=ts_ns)
            ctx = span.get_span_context()

        # So the tool: X spans this call's tool_calls eventually produce (via
        # _export_event_inner, exported later by a separate hook invocation)
        # can nest under this exact call instead of the flat session root.
        for tool_use_id in _tool_call_ids(call["data"]):
            _persist_span(
                _tool_call_span_path(session_dir_, tool_use_id), ctx.trace_id, ctx.span_id
            )

    instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS)
