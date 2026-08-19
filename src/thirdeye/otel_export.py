"""Mirror thirdeye events into Logfire as OpenTelemetry spans, live.

Every thirdeye event (tool call, message, notification, ...) funnels through
one call site, ``Store.append_event`` — including calls made from inside
Claude Code / Codex hook subprocesses. That makes it the one place to add
Logfire export and have it cover every platform automatically, with no
separate sync step. Token usage is the one exception: it's captured by a
separate pipeline straight into ``UsageStore``, never through
``Store.append_event``, so ``export_usage_rows`` is a second, explicit entry
point the two platforms' usage-capture functions call directly.

Each thirdeye session becomes one Logfire trace. The first event exported for
a session becomes that trace's root span; its real (SDK-generated) trace_id
and span_id are persisted to ``otel.json`` in the session directory so later
events — emitted by separate, later hook subprocesses — can reference the same
trace and parent under it. Every exported event's own span id is *also*
persisted, individually, to ``otel-spans/<seq>.json`` — this is what lets
usage rows (see below) parent directly under the specific interaction they
belong to rather than dangling off the trace root as unconnected siblings. A
``tool_call`` is never exported on its own; it is folded into the matching
``tool_result`` as one span with an accurate start/end duration, found by
scanning back through the session's own stored events for the nearest event
sharing the same tool id (Claude's ``tool_use_id`` / Codex's ``call_id``).

Usage rows are exported as children of the ``assistant_message`` /
``agent_turn`` span whose seq they were captured against (that seq is exactly
what ``UsageRow.seq`` holds — see ``platforms/*/usage.py``). That parent span
is built by a *separate* detached worker process racing this one, so
``export_usage_rows`` polls briefly for its ``otel-spans/<seq>.json`` record
before falling back to the session root — a bounded wait that costs nothing
since it happens off the hook's critical path either way.

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
from thirdeye.usage.types import UsageRow

# Cache across calls *within one process*. Each hook invocation is its own
# short-lived process, so this only saves repeat configure() calls when a
# single process exports more than one event (e.g. the eval runner).
_state: dict[str, Any] = {"attempted": False, "instance": None}

_TOOL_ID_KEYS = ("tool_use_id", "call_id")
_ATTR_PRIMITIVES = (str, bool, int, float)

# Keyed by the *end* event type -> the matching *start* event type it pairs
# with to form one real-duration span. A start-side type with no entry here
# (tool_call) is never exported on its own.
_PAIR_START_FOR_END = {"tool_result": "tool_call"}

_SCAN_CAP = 500  # bound on how far back to search for a matching start event
_FLUSH_TIMEOUT_MS = 2000

# How long export_usage_rows waits for the triggering event's own span record
# to show up before giving up and parenting under the session root instead.
# Both workers are spawned within moments of each other from the same hook
# call, and each pays real (if brief) network setup cost before either one
# gets as far as persisting anything, so a handful of short polls is usually
# enough to observe the other side land first.
_USAGE_PARENT_RETRIES = 6
_USAGE_PARENT_RETRY_DELAY_S = 0.1


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
    one; anything else (nested dicts, mixed lists) is JSON-encoded to a string.
    """
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
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
    return out


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
    seq — so a later, separate process (usage export) can parent under this
    exact span rather than only ever the session root. Every seq is exported
    at most once, so a collision here would mean something else is wrong;
    either way this must never raise, so a loss is silently ignored rather
    than reconciled the way `_create_root_atomic` reconciles a root collision.
    """
    _atomic_create(path, _span_payload(trace_id, span_id))


def _usage_claim_path(session_dir_: Path, call_id: str) -> Path:
    # Hashed, not the raw call_id, as a filename: Codex's call_id already
    # contains a colon (`cum:<n>`), and nothing guarantees any platform's
    # call_id is otherwise filesystem- or path-traversal-safe.
    digest = hashlib.sha256(call_id.encode()).hexdigest()
    return session_dir_ / "otel-usage-sent" / f"{digest}.json"


def _claim_usage_export(session_dir_: Path, call_id: str) -> bool:
    """First-wins claim on exporting this call_id's usage span, ever, for this
    session. `UsageStore` is a deliberately dumb, faithful mirror that never
    deduplicates writes — Codex re-reports the same call verbatim, and two
    capture calls racing on the same not-yet-advanced transcript offset can
    each independently discover and append the same rows (see
    `usage/read.py`'s module docstring) — so the *same* new row can legitimately
    reach `export_usage_rows` more than once. Every duplicate carries identical
    token values (same invariant `usage/read.py`'s last-wins read-side dedup
    relies on), so first-wins here is equally correct and doesn't need to
    reconcile which copy survives — it just answers whether THIS call is the
    one that gets to export.
    """
    return _atomic_create(_usage_claim_path(session_dir_, call_id), "1")


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


def export_usage_rows(
    config: Config | None,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str | None,
    rows: list[UsageRow],
) -> None:
    """Hand a whole batch of new UsageRows off for background export as
    children of the interaction span they belong to (see module docstring).
    Never raises, never blocks on network I/O — same guarantee as
    `export_event`, which this deliberately does *not* delegate to: every row
    in one capture call shares the same `seq` (the triggering event's), so
    they belong in one job and one flush, not one detached subprocess each —
    a session's transcript is tail-parsed in a burst, so a turn with a dozen
    tool calls would otherwise fan out into a dozen concurrent processes all
    hitting Logfire's ingest endpoint for a single hook invocation.

    No-op if `config` or `cwd` is missing (older call sites that predate this
    integration don't pass them), or no row carries a timestamp (some usage
    frames don't report one; `_ts_to_ns` can't place a span without one).
    Rows whose `call_id` has already been exported — Codex repeat-reports and
    same-offset capture races can hand this the same row more than once, see
    `_claim_usage_export` — are silently dropped rather than exported again as
    a duplicate span nested under the same interaction.
    """
    if config is None or cwd is None:
        return
    if not config.logfire.enabled or not config.logfire.token:
        return
    claimed = [
        row for row in rows if row.ts and _claim_usage_export(session_dir_, row.call_id)
    ]
    if not claimed:
        return
    try:
        _spawn_usage_worker(
            thirdeye_home=config.root,
            session_dir_=session_dir_,
            session_id=session_id,
            platform=platform,
            cwd=cwd,
            rows=[{"seq": row.seq, "ts": row.ts, "data": row.attributes()} for row in claimed],
        )
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="logfire_usage_export_spawn",
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


def _spawn_usage_worker(
    *,
    thirdeye_home: Path,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    rows: list[dict[str, Any]],
) -> None:
    job_path = _write_job(
        thirdeye_home,
        {
            "kind": "usage_rows",
            "session_dir": str(session_dir_),
            "session_id": session_id,
            "platform": platform,
            "cwd": cwd,
            "rows": rows,
        },
    )
    _spawn(job_path)


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
    if start_type is not None:
        from thirdeye.reader import SessionReader

        tool_id = _tool_id(data)
        matched = _find_matching_start(SessionReader(session_dir_), seq, start_type, tool_id)
        start_ns = _ts_to_ns(matched["ts"]) if matched else end_ns
        attrs = _flatten_attrs(matched.get("data") if matched else None)
        attrs.update(_flatten_attrs(data))
        tool_name = attrs.get("tool_name")
        name = f"tool: {tool_name}" if tool_name else t
    else:
        start_ns = end_ns
        attrs = _flatten_attrs(data)
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
    root = _read_root(root_path)

    if root is None:
        span = tracer.start_span(name, start_time=start_ns, attributes=attrs)
        span.end(end_time=end_ns)
        ctx = span.get_span_context()
        _create_root_atomic(root_path, ctx.trace_id, ctx.span_id)
    else:
        span = tracer.start_span(
            name, context=_parent_context(*root), start_time=start_ns, attributes=attrs
        )
        span.end(end_time=end_ns)
        ctx = span.get_span_context()

    _persist_span(otel_span_path(session_dir_, seq), ctx.trace_id, ctx.span_id)
    instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS)


def _resolve_usage_parent(
    session_dir_: Path, root_path: Path, triggering_seq: int
) -> tuple[int, int] | None:
    """Find the span usage rows for `triggering_seq` should nest under.

    Polls briefly for that event's own span record, since it's built by a
    separate worker process racing this one with no ordering guarantee
    between them. Falls back to the session root (whatever it is *right now*
    — possibly still None, if this is racing to be the very first export in
    a brand-new session too) rather than dropping the data if the specific
    span never shows up in time.
    """
    span_path = otel_span_path(session_dir_, triggering_seq)
    for attempt in range(_USAGE_PARENT_RETRIES):
        found = _read_root(span_path)
        if found is not None:
            return found
        if attempt < _USAGE_PARENT_RETRIES - 1:
            time.sleep(_USAGE_PARENT_RETRY_DELAY_S)
    return _read_root(root_path)


def _export_usage_rows_inner(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    rows: list[dict[str, Any]],
) -> None:
    """Export a whole batch of usage rows in one process: one shared Logfire
    instance, one flush — never one subprocess and one network round trip per
    row (see `export_usage_rows`). Every row shares the same triggering seq,
    so there's exactly one parent to resolve for the whole batch.
    """
    if not rows:
        return
    instance = _get_instance(config, platform)
    if instance is None:
        return

    tracer = instance.config.get_tracer_provider().get_tracer("thirdeye")
    root_path = otel_state_path(session_dir_)
    parent = _resolve_usage_parent(session_dir_, root_path, rows[0]["seq"])

    for row in rows:
        ts_ns = _ts_to_ns(row["ts"])
        attrs = dict(row["data"])
        attrs["gen_ai.conversation.id"] = session_id
        attrs["thirdeye.platform"] = platform
        attrs["thirdeye.cwd"] = cwd
        attrs["thirdeye.seq"] = row["seq"]
        model = attrs.get("gen_ai.response.model")
        name = f"usage: {model}" if model else "usage"

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

    instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS)
