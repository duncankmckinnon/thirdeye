# Handoff: bound the Codex interrupt-marker lock

**Branch:** `fix/codex-interrupt-marker-lock` (already created, currently just this doc)
**Prior work this follows on from:** PR #35 (merged to main) — "Fix concurrent-session data loss and corruption under subagent hooks." Read that PR's description first; it explains the general pattern (concurrent hook processes, one-shot resolution, silent failure) this task mirrors.

## Context

PR #35 fixed four bugs in Claude's concurrency handling, found and confirmed via live testing with real concurrent subagent dispatch. One of those fixes — bounding `claude-open-turn.lock`'s acquisition instead of blocking forever — has a sibling gap in the Codex platform adapter that was **not** part of that PR, by explicit scope decision (see the last few messages of that session's conversation, or just this doc).

While auditing whether PR #35's fixes could affect Codex, I found:

- Bugs #1 (`write_meta` crash) and #2 (seq/index corruption) live in **shared** code (`thirdeye/store.py`, `writer.py`, `meta.py`, `index.py`) used by both platforms. Codex was equally exposed and is now equally protected — no Codex-specific work needed there.
- Bug #4 (tool-span sweep + Stop-time orphan fallback) is **structurally specific to Claude** — it depends on Claude's incremental external-transcript parser and a shared advancing byte cursor, which Codex's turn-building (reads directly from thirdeye's own local event log by seq range) doesn't have. Not applicable to Codex.
- Bug #3 (`claude-open-turn.lock`'s blocking `fcntl.flock` with no timeout) **does** have a Codex counterpart: `platforms/codex/interrupt_marker.py`'s `_locked_marker()` uses the identical blocking pattern, unbounded, no retry. This is the one thing left to fix.

## The bug

`src/thirdeye/platforms/codex/interrupt_marker.py:38-49`:

```python
@contextlib.contextmanager
def _locked_marker(session_dir_: Path) -> Iterator[int]:
    session_dir_.mkdir(parents=True, exist_ok=True)
    fd = os.open(_marker_path(session_dir_), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
```

Every caller of this (`has_open_marker`, `mark_turn_open`, `clear_marker_not_after`, `_reap` — the shared implementation behind `close_stale_turn_if_open`/`reap_abandoned_marker`/`reap_marker_for_event`/`replace_open_turn`) blocks indefinitely if another process holds the lock. Under concurrent Codex subagents (Codex does support subagents — see `_subagents_in_range` in `platforms/codex/tracing.py` — dispatched via `SubagentStart`/`SubagentStop` hooks), two hook processes for the same session can contend for this lock at the same time, exactly the scenario that motivated the Claude fix.

**Lower risk than the Claude version, but not zero:** every caller here already wraps its lock use in `try: ... except OSError: pass` or `return`/no-op equivalents (check each function — `has_open_marker` catches `OSError`, `mark_turn_open`/`clear_marker_not_after`/`_reap` all do too), so a raised `TimeoutError` (a subclass of `OSError`) would be swallowed exactly as gracefully as blocking-forever is dangerous — same "verify every caller already tolerates a raised error" analysis PR #35 did for the Claude lock, and the same reason that fix needed zero caller-side changes. Confirm this holds here too before assuming it — don't just take this doc's word for it, re-derive it the way the reference implementation's commit message did.

One structural difference worth checking as you go: `_reap`'s lock-held critical section only does local JSON marker reads/writes (no transcript parsing, no span building), and the actual `export_turn(...)` call happens *after* the `with _locked_marker(...)` block exits, not inside it — so the blast radius of a long hold is smaller here than it was for Claude's `emit_live_spans`. That's a reason this is lower-severity, not a reason to skip fixing it.

## The fix

Mirror `_acquire_with_bounded_retry` from `src/thirdeye/platforms/claude/hooks.py:170-189` (on main after PR #35 merged):

```python
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
```

Port this into `interrupt_marker.py` (update the error message to reference `codex-open-turn.json`, not `claude-open-turn.lock`), and swap `_locked_marker`'s blocking `fcntl.flock(fd, fcntl.LOCK_EX)` for a call to it. `_locked_marker` here is always exclusive (no shared-lock variant like Claude's `_locked_open_turn` has), so this is actually simpler than the Claude version — no `operation` parameter needed, no reentrant thread-local tracking to preserve (check whether `_locked_marker` has any reentrancy today; skimming the code above, it doesn't look like it does, unlike Claude's lock).

Don't copy the retry budget constants without thinking about whether they're right for this call site — the Claude budget (~300ms) was justified by measuring actual critical-section duration under 5-way contention (sub-millisecond, local disk I/O only). Do the same empirical check here (or reason from the fact that `_reap`'s critical section is *smaller* than Claude's, so the same or a smaller budget is very likely still generous) rather than assuming the number transfers unexamined.

## Testing approach (mirror what PR #35 did)

1. **TDD, in this order**: write a test that forces contention (hold the lock via a separate open file description on the same path — flock is scoped per open-file-description, not per-process, so this reliably simulates a different process without needing real subprocesses — see `TestLockedOpenTurnBoundedRetry` in `tests/test_claude_hooks.py:454+` for the exact pattern: uncontended-still-works, gives-up-with-TimeoutError-instead-of-hanging (with a real watchdog when running it manually, since the unfixed code will genuinely hang), succeeds-once-contention-clears-within-budget), confirm it hangs/fails against the current code, then implement.
2. **Where to put the tests**: `tests/test_codex_tracing.py` already has substantial `interrupt_marker` coverage starting around line 329 ("interrupt_marker: fallback for a turn that never gets a notify call") — add the bounded-retry tests there, matching that section's existing style, rather than creating a new file (no `test_codex_interrupt_marker.py` exists yet; there's no strong reason to introduce one for this alone).
3. **Verify every caller's existing exception handling actually catches `TimeoutError`** the way PR #35 did for Claude's callers — read `has_open_marker`, `mark_turn_open`, `clear_marker_not_after`, and `_reap` again with this specific question, don't assume the earlier "this doc says it's fine" claim above is sufficient.
4. **Live verification, if you want the same confidence level PR #35 reached**: that session hot-patched the changed files directly into the installed Homebrew package (`/opt/homebrew/opt/thirdeye/libexec/lib/python3.12/site-packages/thirdeye/...`, keeping `.bak` copies, restored after verification) and drove real concurrent Codex subagents through the actual CLI, cross-checking the local breadcrumb log (`~/.thirdeye/logs/usage-errors.jsonl`, phase `hook_invoked`) against exported Logfire spans. Worth doing the same here if Codex subagent concurrency is easy to reproduce; not required if you're satisfied with unit-level confidence given the smaller blast radius.
5. Full suite must stay green (`.venv/bin/python -m pytest tests/ -q --no-cov`) and `ruff check` clean on changed files, same bar as PR #35.

## Definition of done

- [x] `_locked_marker` in `interrupt_marker.py` uses bounded retry, not blocking `flock`.
- [x] Every caller's exception handling confirmed (not assumed) to tolerate the new `TimeoutError`: `has_open_marker`, `mark_turn_open`, `clear_marker_not_after`, and `_reap` each wrap their `_locked_marker` use in `except OSError`, and `TimeoutError` is a builtin `OSError` subclass.
- [x] New tests in `tests/test_codex_tracing.py` covering: uncontended acquisition unchanged, contended acquisition gives up within budget (not hangs), acquisition succeeds if contention clears within the budget. Confirmed the hang test genuinely hung against the unfixed code (killed via watchdog) before implementing the fix.
- [x] Full test suite green (1944 passed, 1 skipped, 2 xpassed), ruff clean.
- [ ] PR opened against `main`, description explains the gap this closes and links back to PR #35 for the pattern it mirrors.
