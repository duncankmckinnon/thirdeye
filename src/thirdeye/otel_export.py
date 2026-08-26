"""Turn a completed ``thirdeye.tracing.model.TurnSpanDict`` into OTel spans.

thirdeye mirrors coding-agent sessions into Logfire. The previous design
exported every raw hook event live, one at a time, from whichever short-lived
hook subprocess happened to fire — which meant a tool span and the LLM call
that requested it were built by two different, unordered processes, and
nesting one under the other required persisting span ids to disk and racing/
polling for them across processes.

That's no longer necessary: a platform adapter (Claude's ``Stop`` hook,
Codex's ``notify`` hook) now assembles an entire turn — every LLM call, tool
call, permission request, and nested subagent invocation it produced — into
one ``TurnSpanDict`` *before* calling into this module, and hands the whole
thing to ``export_turn`` as a single atomic unit. That is what eliminates
almost all of the cross-process span-resolution machinery this module used to
need: there is exactly one process, exporting a fully-known tree, so parent
spans always already exist by the time their children are built.

The resulting trace shape, per thirdeye session:

- **Session** — a root span, no input/output, purely an anchor other spans
  nest under. Persisted (trace_id, span_id) to ``otel.json`` in the session
  directory, same as before, since later turns are still separate processes
  that need to find the same root.
- **Agent-turn** — one span per user prompt through to its final response (or
  point of interruption), carrying the turn's own input/output messages and
  status.
- Each turn's LLM calls nest under the turn span; each LLM call's tool calls
  nest under *that specific* LLM call's span, not flatly under the turn.
  Permission requests nest directly under the turn as point-in-time spans.
- Subagent invocations nest recursively: structurally a subagent invocation
  is just another turn one level deeper, so the same recursive exporter
  (``_export_turn_subtree``) handles them with no special-casing beyond the
  span name.

Platform branching does not belong in this module — the harness-specific
adapters own building the ``TurnSpanDict``; this module's job is only turning
one into spans.

The actual Logfire call — including a flush, a real network round trip —
never happens in the hook process itself. ``export_turn`` only ever writes a
small job file and spawns a detached, unwaited-for child process
(``thirdeye.otel_worker``) to do the work, so a slow or unreachable Logfire
endpoint adds no latency to the hook invocation that triggered it.
``start_new_session=True`` detaches the child from the hook's process group
so it survives even if the harness kills that group once the hook returns.

Safety: this module runs inside hook subprocesses whose stdout the harness
may treat as a hook decision. Every public function here must never raise and
never write to stdout — a Logfire hiccup must be invisible to the turn it
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
from functools import lru_cache
from pathlib import Path
from typing import Any

from thirdeye.config import Config
from thirdeye.ids import new_ulid
from thirdeye.paths import otel_jobs_dir, otel_state_path
from thirdeye.span_ids import (
    chat_span_id,
    root_span_id_for_session,
    tool_span_id,
    trace_id_for_session,
)
from thirdeye.tracing.model import TurnSpanDict
from thirdeye.usage.errlog import log_capture_error

# Cache across calls *within one process*. Each hook invocation is its own
# short-lived process, so this only saves repeat configure() calls when a
# single process exports more than one turn (e.g. the eval runner).
# `id_generator` is the one instance handed to `logfire.configure`; the emit
# path must reach *that* object, since setting a slot on any other one would
# silently do nothing.
_state: dict[str, Any] = {"attempted": False, "instance": None, "id_generator": None}

_ATTR_PRIMITIVES = (str, bool, int, float)
# Logfire's own convention for telling its backend which JSON-encoded string
# attributes to parse back into structured data (objects/arrays) rather than
# render as opaque text. Spans built via `logfire.span()` get this set
# automatically; ours go through the raw OTel API, so `_flatten_attrs` sets
# it by hand. Without it, `gen_ai.input.messages` / `.output.messages` never
# render as a chat view in the UI, just a flat "prompt" text field.
_LOGFIRE_JSON_SCHEMA_KEY = "logfire.json_schema"

_FLUSH_TIMEOUT_MS = 2000

# The name a platform goes by on Logfire's Agents page. Only platforms whose
# internal key differs from the name their CLI is known by need an entry; any
# other platform is used verbatim. Deliberately separate from `thirdeye.platform`
# and from the configured service name, which both stay the internal key.
_AGENT_NAMES = {"claude": "claude-code"}

# Maps a key in an `UsageDict` to the OTel GenAI semantic-convention attribute
# it becomes. `UsageDict` is `total=False`, so only keys actually present are
# ever emitted — an absent count means "not reported", not zero.
_USAGE_KEYS = {
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cache_read_input_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_creation_input_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "reasoning_output_tokens": "gen_ai.usage.reasoning.output_tokens",
}


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


def _build_id_generator() -> Any:
    """Build a generator that lets us hand the SDK ids of our own choosing.

    Span ids for this trace tree are *derived*, not minted (see
    ``thirdeye.span_ids``), because a tool span emitted while a turn is still
    running has to name the ``chat`` span that requested it as its parent —
    and that chat span isn't exported until the turn ends. But OTel's
    ``IdGenerator`` interface is context-free (``generate_span_id()`` takes no
    arguments), so there is nothing to key an id off of. What can be
    controlled is the *sequence*: a mutable slot, set immediately before a
    ``start_span`` call and cleared as it's read, so the next id the SDK draws
    is the one we just chose. Anything the SDK starts that we didn't
    pre-allocate for still gets an ordinary random id.

    Worker processes are single-threaded and every span here is started
    synchronously, so "the next id drawn" is unambiguous. `logfire.configure`
    itself draws zero ids, so no internal span can consume a pending slot —
    a third-party assumption that `TestPreallocatedIdGenerator` keeps a canary
    on, because a regression in it would misparent spans invisibly.

    The class is defined inside this function because its base class lives in
    ``opentelemetry``, present only with the optional ``logfire`` extra;
    importing it at module scope would break every hook invocation without it.
    """
    from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator

    class PreallocatedIdGenerator(IdGenerator):
        """Returns a pre-set id if one is pending, else a random one."""

        def __init__(self) -> None:
            self.next_span_id: int | None = None
            self.next_trace_id: int | None = None
            self._random = RandomIdGenerator()

        def generate_span_id(self) -> int:
            value, self.next_span_id = self.next_span_id, None
            return value if value is not None else self._random.generate_span_id()

        def generate_trace_id(self) -> int:
            value, self.next_trace_id = self.next_trace_id, None
            return value if value is not None else self._random.generate_trace_id()

    return PreallocatedIdGenerator()


def _id_generator() -> Any:
    """This process's generator instance, created on first use and cached."""
    if _state["id_generator"] is None:
        _state["id_generator"] = _build_id_generator()
    return _state["id_generator"]


def _start_span_with_id(
    tracer: Any,
    name: str,
    span_id: int,
    *,
    parent_ctx: Any = None,
    start_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    trace_id: int | None = None,
    kind: Any = None,
) -> Any:
    """Start a span carrying a span id we chose rather than one the SDK minted.

    `trace_id` is only honored for a span with no parent — a child always
    inherits its parent's trace — so it is passed for the session root and
    nowhere else. It's assigned unconditionally so a value left pending by an
    earlier call can never leak into a later root span.
    """
    generator = _id_generator()
    generator.next_span_id = span_id
    generator.next_trace_id = trace_id
    kwargs = {
        "context": parent_ctx,
        "start_time": start_time,
        "attributes": attributes,
    }
    if kind is not None:
        kwargs["kind"] = kind
    return tracer.start_span(name, **kwargs)


def _get_instance(config: Config, platform: str):
    """Return a configured Logfire instance, or None if export is inactive.

    Never raises. Cached for the life of this process. The service name is
    fixed to the *first* platform seen by this process, so Claude Code and
    Codex sessions show up as distinct services (``claude`` / ``codex``) in
    Logfire instead of one indistinguishable ``thirdeye`` — the OTel scope
    (the tracer name below) already says "thirdeye", so it needn't be
    repeated in the service name too. Each hook invocation is its own
    short-lived, single-platform process, so this only matters for a process
    that exports more than one platform's turns (e.g. the eval runner),
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
            # Keeping the SDK is what preserves scrubbing, retries and the
            # OTLP wire format for free; the one thing it doesn't give us is
            # ids of our own choosing, and this injects those through its
            # normal path rather than around it.
            advanced=logfire.AdvancedOptions(id_generator=_id_generator()),
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


def _message(role: str, content: str) -> list[dict[str, Any]]:
    return [{"role": role, "parts": [{"type": "text", "content": content}]}]


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

    A short-lived sibling lock serializes the read/persist/start sequence
    across detached workers. Derived root ids mean two concurrent first
    exports can no longer produce *split* traces — they'd derive the same
    ids — but without the lock they would each emit their own copy of the
    session root span, since neither's atomic create would tell it apart from
    a win. Sessions rooted before derivation landed keep whatever ids
    ``otel.json`` already holds, so the lock still guards those too.
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


def _create_root_atomic(path: Path, trace_id: int, span_id: int) -> tuple[tuple[int, int], bool]:
    """Persist (trace_id, span_id) as the session's root, first writer wins.

    A losing writer's own generated ids are simply discarded in favor of the
    winner's, so every process agrees on one root going forward. The boolean
    records whether this call created the file; comparing ids cannot answer
    that once every worker derives the same deterministic root.
    """
    if _atomic_create(path, _span_payload(trace_id, span_id)):
        return (trace_id, span_id), True
    return _read_root(path) or (trace_id, span_id), False


def _parent_context(trace_id: int, span_id: int):
    from opentelemetry import trace as otel_trace

    span_context = otel_trace.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=otel_trace.TraceFlags(otel_trace.TraceFlags.SAMPLED),
    )
    return otel_trace.set_span_in_context(otel_trace.NonRecordingSpan(span_context))


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


# How long a "pending" turn claim is honored before being treated as
# abandoned (the worker that made it died mid-export) and reclaimed by a
# later retry — same threshold and pending/sent shape the old per-turn Codex
# export claim used.
_TURN_CLAIM_STALE_S = 30.0


def _turn_claim_path(session_dir_: Path, turn_id: str) -> Path:
    # Hashed, not the raw turn_id, as a filename: nothing guarantees any
    # platform's turn id is filesystem- or path-traversal-safe.
    digest = hashlib.sha256(turn_id.encode()).hexdigest()
    return session_dir_ / "otel-turns-sent" / f"{digest}.json"


def _claim_turn_export(session_dir_: Path, turn_id: str) -> bool:
    """First-wins claim on exporting this turn's span tree, ever, for this
    session. A replayed/duplicate hook invocation for the same turn (e.g. the
    open-turn-marker catch-all firing after the turn was already closed out
    normally) must not export it a second time — but a claim only becomes
    permanent ("sent") once the whole subtree has actually been built and
    flushed; until then it's "pending", so a crash or a failed flush partway
    through releases the claim (see `_export_turn_inner`) rather than losing
    the turn forever.
    """
    claim_path = _turn_claim_path(session_dir_, turn_id)
    try:
        state = claim_path.read_text()
    except OSError:
        state = ""
    if state == "sent":
        return False
    if state == "pending":
        try:
            stale = time.time() - claim_path.stat().st_mtime > _TURN_CLAIM_STALE_S
        except OSError:
            stale = True
        if not stale:
            return False
        claim_path.unlink(missing_ok=True)
    return _atomic_create(claim_path, "pending")


def export_turn(
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    turn: TurnSpanDict,
) -> None:
    """Hand a completed turn off for background export. Never raises, never
    blocks on network I/O — the actual Logfire call happens in a detached
    child process this spawns and does not wait for. See module docstring.
    """
    if not config.logfire.enabled or not config.logfire.token:
        return
    try:
        job_path = _write_job(
            config.root,
            {
                "kind": "turn",
                "session_dir": str(session_dir_),
                "session_id": session_id,
                "platform": platform,
                "cwd": cwd,
                "turn": turn,
            },
        )
        _spawn(job_path)
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="logfire_turn_export_spawn",
            error=exc,
            platform=platform,
            session_id=session_id,
        )


def export_spans(
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    trace_id: int,
    spans: list[dict[str, Any]],
) -> bool:
    """Hand already-built spans off for background export.

    IDs are serialized as decimal strings so their full unsigned 64-/128-bit
    values survive the JSON boundary without relying on a consumer's numeric
    precision. As with :func:`export_turn`, this function only performs local
    file/process work and never lets a capture failure reach its caller. The
    return value says whether the job was durably written and dispatched, so a
    live cursor is never advanced past a failed local write.
    """
    if not config.logfire.enabled or not config.logfire.token:
        return False
    try:
        serialized_spans = []
        for span in spans:
            serialized = dict(span)
            serialized["span_id"] = str(int(span["span_id"]))
            serialized["parent_span_id"] = str(int(span["parent_span_id"]))
            serialized_spans.append(serialized)
        job_path = _write_job(
            config.root,
            {
                "kind": "spans",
                "session_dir": str(session_dir_),
                "session_id": session_id,
                "platform": platform,
                "cwd": cwd,
                "trace_id": str(int(trace_id)),
                "spans": serialized_spans,
            },
        )
        _spawn(job_path)
        return True
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="logfire_spans_export_spawn",
            error=exc,
            platform=platform,
            session_id=session_id,
        )
        return False


def export_subagent_turn(
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    turn: TurnSpanDict,
    tool_use_id: str,
) -> None:
    """Hand a completed subagent turn off for background export, nested
    under the tool span (already exported, live, when the dispatching tool
    call itself completed) that dispatched it -- not under whichever turn
    happens to be open in this session when the subagent's own Stop fires.

    A subagent can run well past its dispatching turn's own Stop, so by the
    time this is called that turn's span tree may already be exported (with
    no way to graft a child onto it after the fact) or a wholly unrelated
    later turn may be in progress. Parenting is independent of turn export
    for exactly the same reason a live tool/chat span's is: the deterministic
    `tool_span_id` gives Logfire everything it needs to place this in the
    tree without its parent needing to still be open, or even in this
    process. Never raises, never blocks on network I/O -- same contract as
    `export_turn`.
    """
    if not config.logfire.enabled or not config.logfire.token:
        return
    try:
        try:
            state = json.loads(otel_state_path(session_dir_).read_text())
            trace_id = int(state["trace_id"], 16)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            trace_id = trace_id_for_session(session_id)
        job_path = _write_job(
            config.root,
            {
                "kind": "subagent_turn",
                "session_dir": str(session_dir_),
                "session_id": session_id,
                "platform": platform,
                "cwd": cwd,
                "trace_id": str(trace_id),
                "parent_span_id": str(int(tool_span_id(session_id, tool_use_id))),
                "turn": turn,
            },
        )
        _spawn(job_path)
    except Exception as exc:
        log_capture_error(
            thirdeye_home=config.root,
            phase="logfire_subagent_turn_export_spawn",
            error=exc,
            platform=platform,
            session_id=session_id,
        )


def _export_turn_inner(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    turn: TurnSpanDict,
) -> None:
    instance = _get_instance(config, platform)
    if instance is None:
        return
    if not _claim_turn_export(session_dir_, turn["turn_id"]):
        return
    claim_path = _turn_claim_path(session_dir_, turn["turn_id"])

    try:
        tracer = instance.config.get_tracer_provider().get_tracer("thirdeye")
        root_path = otel_state_path(session_dir_)
        parent, root_lock = _root_or_ownership(root_path)
        try:
            if parent is None and root_lock is None:
                raise RuntimeError("could not resolve or create session root")
            if parent is None:
                # First export for this session: the root is purely an
                # anchor, so it carries none of the turn's own input/output
                # content. Its ids are derived from the session id and
                # persisted *before* the span is emitted — the reverse of the
                # old mint-then-record order — so any other process can name
                # this trace and this parent without having read the file back
                # first, which is what makes mid-turn emission possible.
                root_ns = _ts_to_ns(turn["start_ts"])
                root_attrs = _flatten_attrs(
                    {
                        "gen_ai.conversation.id": session_id,
                        "thirdeye.platform": platform,
                        "thirdeye.cwd": cwd,
                    }
                )
                derived = (
                    trace_id_for_session(session_id),
                    root_span_id_for_session(session_id),
                )
                parent, created_root = _create_root_atomic(root_path, *derived)
                # Another writer can win after stale-lock recovery with either
                # legacy ids or these same deterministic ids. In both cases
                # its root span exists already; emit only if our atomic create
                # actually persisted the root file.
                if created_root:
                    root_span = _start_span_with_id(
                        tracer,
                        "session",
                        derived[1],
                        trace_id=derived[0],
                        start_time=root_ns,
                        attributes=root_attrs,
                    )
                    root_span.end(end_time=root_ns)
        finally:
            if root_lock is not None:
                root_lock.unlink(missing_ok=True)

        parent_ctx = _parent_context(*parent)
        _export_turn_subtree(
            tracer, parent_ctx, turn, session_id=session_id, platform=platform, cwd=cwd
        )
        if instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS) is False:
            raise RuntimeError("turn export was not flushed")
    except Exception:
        claim_path.unlink(missing_ok=True)
        raise
    claim_path.write_text("sent")


def _export_subagent_turn_inner(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    trace_id: int | str,
    parent_span_id: int | str,
    turn: TurnSpanDict,
) -> None:
    """Export one subagent's turn under an already-known parent span id.

    Uses the same first-wins claim as `_export_turn_inner`, keyed separately
    (`subagent:<turn_id>`) so this and a same-session `build_turn` embedding
    that also happens to see this subagent (the fast, fully-synchronous case)
    can't both export it -- whichever claims first wins, the other is a no-op.
    Unlike a top-level turn, there is no session-root bootstrapping here: the
    parent (the dispatching tool's span) was already exported live by the
    time the subagent even started, so this only ever attaches to an existing
    remote parent context, never creates one.
    """
    instance = _get_instance(config, platform)
    if instance is None:
        return
    claim_id = f"subagent:{turn['turn_id']}"
    if not _claim_turn_export(session_dir_, claim_id):
        return
    claim_path = _turn_claim_path(session_dir_, claim_id)

    try:
        tracer = instance.config.get_tracer_provider().get_tracer("thirdeye")
        parent_ctx = _parent_context(int(trace_id), int(parent_span_id))
        _export_turn_subtree(
            tracer,
            parent_ctx,
            turn,
            session_id=session_id,
            platform=platform,
            cwd=cwd,
            span_name="agent-turn (subagent)",
        )
        if instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS) is False:
            raise RuntimeError("subagent turn export was not flushed")
    except Exception:
        claim_path.unlink(missing_ok=True)
        raise
    claim_path.write_text("sent")


@lru_cache(maxsize=128)
def _repo_name(cwd: str) -> str | None:
    """The name of the git repository `cwd` sits in, or None outside one.

    Walks upward from `cwd` looking for `.git` — a directory in an ordinary
    clone, a file in a worktree or submodule, so existence is the test rather
    than being a directory. Complements `thirdeye.cwd`: the directory an agent
    was started from is often some subdirectory of the project, and only the
    repo root groups those sessions together.

    Cached because it is consulted once per span, and it swallows OS errors
    rather than raising: export runs in a background worker that can outlive
    the directory the session ran in.
    """
    if not cwd:
        return None
    try:
        path = Path(cwd).resolve()
        for candidate in (path, *path.parents):
            if (candidate / ".git").exists():
                # The filesystem root is not a project name worth reporting.
                return candidate.name or None
    except OSError:
        return None
    return None


def _agent_name(platform: str) -> str:
    """The `gen_ai.agent.name` a platform's spans are attributed to."""
    return _AGENT_NAMES.get(platform, platform)


def _identity_attributes(
    *,
    session_id: str,
    platform: str,
    cwd: str,
    turn_id: Any = None,
    turn_span_id: Any = None,
) -> dict[str, Any]:
    """Attributes naming the session and turn a span belongs to.

    A live span is exported while its `agent-turn` parent is still open, so it
    has no parent row to inherit this from and cannot otherwise be attributed
    to a turn until the turn ends. Applied on the completed-turn path too, so
    the two paths keep one vocabulary.

    `gen_ai.agent.name` is the platform rather than anything per-session, so
    one Logfire agent covers every session a given CLI ever ran.
    """
    attributes: dict[str, Any] = {
        "gen_ai.conversation.id": session_id,
        "gen_ai.agent.name": _agent_name(platform),
        "thirdeye.platform": platform,
        "thirdeye.cwd": cwd,
        # Dropped by `_flatten_attrs` when None, i.e. outside a repository.
        "thirdeye.repo": _repo_name(cwd),
    }
    if turn_id is not None:
        attributes["thirdeye.turn.id"] = str(turn_id)
    if turn_span_id is not None:
        attributes["thirdeye.turn.span_id"] = str(turn_span_id)
    return attributes


def _cost_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Price a chat span's token usage into `operation.cost`, or `{}`.

    Logfire surfaces cost from this attribute; when it is missing the UI falls
    back to pricing `gen_ai.usage.input_tokens` at the model's full input rate.
    That count is cache-inclusive (the convention Logfire's own Anthropic
    instrumentation uses), so on a cache-heavy session — where nearly every
    input token is a cache read billed at a tenth of the rate — the fallback
    overstates cost by roughly 9x.

    `genai_prices` is the same library Logfire's instrumentation prices with,
    so the two agree by construction. It also requires the cache-inclusive
    convention: it rejects a `cache_read_tokens` larger than `input_tokens`.

    Best-effort by design, matching Logfire's own handling — an unpriceable
    model or a missing dependency costs the span its cost attribute, never the
    span itself.
    """
    model = attributes.get("gen_ai.response.model")
    input_tokens = attributes.get("gen_ai.usage.input_tokens")
    if not model or not isinstance(input_tokens, int):
        return {}
    try:
        from genai_prices import calc_price
        from genai_prices.types import Usage

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=attributes.get("gen_ai.usage.output_tokens"),
            cache_read_tokens=attributes.get("gen_ai.usage.cache_read.input_tokens"),
            cache_write_tokens=attributes.get("gen_ai.usage.cache_creation.input_tokens"),
        )
        price = calc_price(
            usage,
            model_ref=str(model),
            provider_id=attributes.get("gen_ai.provider.name"),
        )
        return {"operation.cost": float(price.total_price)}
    except Exception:
        return {}


def _chat_attributes(
    call_or_attributes: dict[str, Any],
    *,
    session_id: str,
    platform: str,
    cwd: str,
    turn_id: Any = None,
    turn_span_id: Any = None,
) -> dict[str, Any]:
    """Build flattened attributes for a chat span.

    Completed-turn export passes an LLM-call record, while live batch export
    passes the already-built semantic attributes from its job. Accepting both
    forms keeps the vocabulary and JSON handling in one place.
    """
    if all(
        key in call_or_attributes
        for key in ("input_messages", "output_messages", "provider", "usage")
    ):
        model = call_or_attributes.get("model") or ""
        provider = call_or_attributes["provider"]
        attributes: dict[str, Any] = {
            "gen_ai.input.messages": call_or_attributes["input_messages"],
            "gen_ai.output.messages": call_or_attributes["output_messages"],
            "gen_ai.provider.name": provider,
            # Superseded by `provider.name`, but still what pydantic-ai's own
            # instrumentation emits alongside it, and what Logfire's older
            # views read the provider from. Cheap enough to carry both.
            "gen_ai.system": provider,
            "gen_ai.operation.name": "chat",
        }
        if model:
            # `request.model` is the attribute the convention builds the span
            # name from and the LLM views group by; `response.model` on its own
            # is only Recommended, so a span carrying just that one has no
            # model as far as those views are concerned. When the model is
            # unknown both are left absent rather than emitted as "", which
            # would collect every modelless call under one blank name.
            attributes["gen_ai.request.model"] = model
            attributes["gen_ai.response.model"] = model
        usage = call_or_attributes["usage"]
        for source, target in _USAGE_KEYS.items():
            if source in usage:
                attributes[target] = usage[source]
    else:
        attributes = call_or_attributes
    return _flatten_attrs(
        _merge_raw(
            attributes,
            _identity_attributes(
                session_id=session_id,
                platform=platform,
                cwd=cwd,
                turn_id=turn_id,
                turn_span_id=turn_span_id,
            ),
            _cost_attributes(attributes),
        )
    )


def _tool_attributes(
    attributes: dict[str, Any],
    *,
    session_id: str,
    platform: str,
    cwd: str,
    turn_id: Any = None,
    turn_span_id: Any = None,
) -> dict[str, Any]:
    """Enrich and flatten a tool span's raw attributes."""
    return _flatten_attrs(
        _merge_raw(
            attributes,
            _identity_attributes(
                session_id=session_id,
                platform=platform,
                cwd=cwd,
                turn_id=turn_id,
                turn_span_id=turn_span_id,
            ),
        )
    )


def _export_spans_batch(
    *,
    config: Config,
    session_dir_: Path,
    session_id: str,
    platform: str,
    cwd: str,
    trace_id: int | str,
    spans: list[dict[str, Any]],
) -> None:
    """Emit a batch of independently-parented spans and flush exactly once.

    A parent need not be present in this batch. The remote parent context is
    sufficient for Logfire to reconstruct the tree when the parent arrives.
    ``session_dir_`` is part of the common job envelope and intentionally
    unused here; unlike turn export, live spans have no turn-level claim.
    """
    del session_dir_
    instance = _get_instance(config, platform)
    if instance is None:
        return

    tracer = instance.config.get_tracer_provider().get_tracer("thirdeye")
    trace_id_int = int(trace_id)
    for span_data in spans:
        name = span_data["name"]
        raw_attributes = span_data.get("attributes", {})
        turn_id = span_data.get("turn_seq")
        turn_span_id_ = span_data.get("turn_span_id")
        if name == "chat" or name.startswith("chat "):
            attributes = _chat_attributes(
                raw_attributes,
                session_id=session_id,
                platform=platform,
                cwd=cwd,
                turn_id=turn_id,
                turn_span_id=turn_span_id_,
            )
        elif name.startswith("tool:"):
            attributes = _tool_attributes(
                raw_attributes,
                session_id=session_id,
                platform=platform,
                cwd=cwd,
                turn_id=turn_id,
                turn_span_id=turn_span_id_,
            )
        else:
            attributes = _flatten_attrs(raw_attributes)
        parent_ctx = _parent_context(trace_id_int, int(span_data["parent_span_id"]))
        span = _start_span_with_id(
            tracer,
            name,
            int(span_data["span_id"]),
            parent_ctx=parent_ctx,
            start_time=_ts_to_ns(span_data["start_ts"]),
            attributes=attributes,
        )
        span.end(end_time=_ts_to_ns(span_data["end_ts"]))

    if instance.force_flush(timeout_millis=_FLUSH_TIMEOUT_MS) is False:
        raise RuntimeError("span batch was not flushed")


def _export_turn_subtree(
    tracer: Any,
    parent_ctx: Any,
    turn: TurnSpanDict,
    *,
    session_id: str,
    platform: str,
    cwd: str,
    span_name: str = "agent-turn",
) -> None:
    """Export one turn and its whole subtree (LLM calls, their tool calls,
    permission requests, and recursively any nested subagent turns).

    A subagent invocation is structurally just another turn one level deeper,
    so it is exported by recursing into this same function rather than any
    dedicated subagent-handling logic.
    """
    from opentelemetry.trace import SpanKind

    turn_attrs: dict[str, Any] = {
        "thirdeye.turn.status": turn["status"],
        "thirdeye.turn.id": turn["turn_id"],
        "gen_ai.conversation.id": session_id,
        # What Logfire's Agents page matches a span on: an `invoke_agent`
        # operation carrying the agent's name. A pydantic-ai agent gets both
        # from its own instrumentation; spans built through the raw OTel API
        # get neither, and without them a turn is just an anonymous span and
        # the session never registers as an agent run at all.
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": _agent_name(platform),
        "thirdeye.platform": platform,
        "thirdeye.cwd": cwd,
        "thirdeye.repo": _repo_name(cwd),
    }
    if turn["input_message"]:
        turn_attrs["gen_ai.input.messages"] = _message("user", turn["input_message"])
    if turn["output_message"]:
        turn_attrs["gen_ai.output.messages"] = _message("assistant", turn["output_message"])

    turn_id = turn.get("turn_span_id")
    if turn_id is None:
        turn_span = tracer.start_span(
            span_name,
            context=parent_ctx,
            start_time=_ts_to_ns(turn["start_ts"]),
            attributes=_flatten_attrs(_merge_raw(turn_attrs, turn.get("attributes"))),
        )
    else:
        turn_span = _start_span_with_id(
            tracer,
            span_name,
            int(turn_id),
            parent_ctx=parent_ctx,
            start_time=_ts_to_ns(turn["start_ts"]),
            attributes=_flatten_attrs(_merge_raw(turn_attrs, turn.get("attributes"))),
        )
    turn_span.end(end_time=_ts_to_ns(turn["end_ts"]))
    turn_ctx = turn_span.get_span_context()
    turn_parent_ctx = _parent_context(turn_ctx.trace_id, turn_ctx.span_id)

    for llm_call in turn["llm_calls"]:
        model = llm_call.get("model") or ""
        call_span = _start_span_with_id(
            tracer,
            f"chat {model}" if model else "chat",
            chat_span_id(session_id, llm_call["call_id"]),
            parent_ctx=turn_parent_ctx,
            start_time=_ts_to_ns(llm_call["start_ts"]),
            attributes=_chat_attributes(
                llm_call,
                session_id=session_id,
                platform=platform,
                cwd=cwd,
                turn_id=turn["turn_id"],
                turn_span_id=turn.get("turn_span_id"),
            ),
        )
        call_span.end(end_time=_ts_to_ns(llm_call["end_ts"]))
        call_ctx = call_span.get_span_context()
        call_parent_ctx = _parent_context(call_ctx.trace_id, call_ctx.span_id)

        for tool_call in llm_call["tool_calls"]:
            tool_span = _start_span_with_id(
                tracer,
                f"tool: {tool_call['name']}",
                tool_span_id(session_id, tool_call["tool_call_id"]),
                parent_ctx=call_parent_ctx,
                kind=SpanKind.INTERNAL,
                start_time=_ts_to_ns(tool_call["start_ts"]),
                attributes=_tool_attributes(
                    tool_call["attributes"],
                    session_id=session_id,
                    platform=platform,
                    cwd=cwd,
                    turn_id=turn["turn_id"],
                    turn_span_id=turn.get("turn_span_id"),
                ),
            )
            tool_span.end(end_time=_ts_to_ns(tool_call["end_ts"]))

    for orphan in turn.get("orphan_tool_calls") or []:
        parent_call_id = orphan["parent_call_id"]
        tool_call = orphan["tool_call"]
        orphan_parent_ctx = _parent_context(
            turn_ctx.trace_id, int(chat_span_id(session_id, parent_call_id))
        )
        orphan_span = _start_span_with_id(
            tracer,
            f"tool: {tool_call['name']}",
            tool_span_id(session_id, tool_call["tool_call_id"]),
            parent_ctx=orphan_parent_ctx,
            kind=SpanKind.INTERNAL,
            start_time=_ts_to_ns(tool_call["start_ts"]),
            attributes=_tool_attributes(
                tool_call["attributes"],
                session_id=session_id,
                platform=platform,
                cwd=cwd,
                turn_id=turn["turn_id"],
                turn_span_id=turn.get("turn_span_id"),
            ),
        )
        orphan_span.end(end_time=_ts_to_ns(tool_call["end_ts"]))

    for permission_request in turn["permission_requests"]:
        pr_ts = _ts_to_ns(permission_request["ts"])
        pr_attrs = _flatten_attrs(permission_request["attributes"])
        pr_attrs["gen_ai.conversation.id"] = session_id
        pr_attrs["thirdeye.platform"] = platform
        pr_attrs["thirdeye.cwd"] = cwd
        pr_repo = _repo_name(cwd)
        if pr_repo:
            pr_attrs["thirdeye.repo"] = pr_repo
        pr_span = tracer.start_span(
            f"permission_request: {permission_request['tool_name']}",
            context=turn_parent_ctx,
            start_time=pr_ts,
            attributes=pr_attrs,
        )
        pr_span.end(end_time=pr_ts)

    for subagent in turn["subagents"]:
        _export_turn_subtree(
            tracer,
            turn_parent_ctx,
            subagent,
            session_id=session_id,
            platform=platform,
            cwd=cwd,
            span_name="agent-turn (subagent)",
        )
