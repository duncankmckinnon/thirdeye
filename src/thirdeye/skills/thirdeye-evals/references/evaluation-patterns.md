# Evaluation patterns

Five recurring rubric shapes. Each section describes when to use the pattern,
a sample directive snippet you can paste into a YAML definition, and what
the resulting findings tend to look like.

## Token efficiency

**Use when**: investigating high-cost sessions, comparing prompt strategies,
debugging runaway tool output, or evaluating cache utilization.

```yaml
directive: |
  Evaluate this session for token efficiency. Surface:
    - turns whose tool result payload is large and likely could be summarized
      or filtered (e.g. full directory dumps, full file reads when grep would
      suffice).
    - repeated reads of the same file in a short window.
    - low prompt-cache hit ratio across turns (use `thirdeye usage <sid>`).
    - oversized system / context blocks vs. the work actually performed.
  Verdict: pass if no warnings; warn if any warning-level finding; fail if
  total tokens exceed 2x the median of comparable sessions AND at least one
  warning was emitted.
```

**Expected findings**: `category: token-efficiency`, often `seq` set to the
oversized tool call; severity `warn` for individual offenses, `info` for
session-level cache-ratio commentary.

## Tool quality

**Use when**: agents are picking the wrong tool, repeating tool calls
unnecessarily, or missing batching opportunities.

```yaml
directive: |
  Evaluate this session for tool-selection quality. Flag turns where:
    - a better tool existed (e.g. Bash `find` when Glob would have worked,
      Read of full file when Grep was sufficient).
    - the same tool was called with identical inputs in adjacent turns.
    - multiple independent tool calls in a row could have been batched in
      one assistant message.
  Verdict: warn on any single occurrence, fail if any input-identical
  repeat happened three or more times.
```

**Expected findings**: `category: tool-quality`, `seq` always pinned to the
offending event; severity `warn` per occurrence, `error` for pathological
repeats.

## Error recovery

**Use when**: studying how agents react to failing commands, broken tests,
or unexpected tool output.

```yaml
directive: |
  Evaluate this session for error-recovery behavior. For each turn where a
  tool failed (non-zero exit, exception text, "not found" responses) check
  the next 1-3 turns and grade:
    - did the agent investigate the failure (read the error, check
      adjacent state, run a diagnostic command)?
    - or did it blindly retry the same command, possibly with trivial
      variation?
  Emit warning findings on blind retries, info findings on competent
  recoveries (so the run history has positive evidence too).
  Verdict: pass if every failure was investigated; warn on any blind
  retry; fail on a blind-retry loop of 3+ attempts.
```

**Expected findings**: `category: error-recovery`, `seq` on the failure
event; positive `info` findings on the recovery turn.

## Task adherence

**Use when**: validating that the agent actually addressed the user's
stated intent and didn't drift into unrelated changes.

```yaml
directive: |
  Evaluate whether the agent stayed on task. Read the user's first prompt
  (the first user_prompt_submit event) as the canonical statement of
  intent. Then for each subsequent change-making turn (Edit, Write,
  Bash with side effects) judge whether that change advances the stated
  intent or constitutes scope creep / drift.
  Verdict: pass if all change-making turns advance intent; warn if any
  drift was small and reversible; fail if material work was done outside
  the stated scope.
```

**Expected findings**: `category: task-drift`, `seq` on the off-scope
change; session-level `info` finding restating the original intent for
quick comparison.

## Redundancy

**Use when**: looking for wasted work that prompt-caching alone won't
fix — re-reading state the agent already has, re-injecting context
the conversation already contains, ignoring earlier tool results.

```yaml
directive: |
  Evaluate this session for redundant work. Flag turns where:
    - a file or directory was read again after being read earlier in the
      session, with no intervening edit.
    - the agent re-stated context (e.g. re-summarized the task, re-listed
      files) that was already established in earlier turns.
    - a previous tool result already contained the answer the agent then
      went and re-derived.
  Verdict: warn on any single occurrence, fail if redundant reads
  account for >25% of all tool calls.
```

**Expected findings**: `category: redundancy`, `seq` on the redundant
turn, note referencing the earlier turn where the same info was already
available.
