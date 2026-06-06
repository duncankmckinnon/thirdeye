# Token overuse and error analysis

A waste pattern is only worth fixing if it *costs something* — tokens,
wrong answers, or repeated retries. This step crosses the patterns
from [pattern-audit.md](pattern-audit.md) with per-turn token usage
and tool-error events.

## Token rollups

### Per-session total

```bash
thirdeye usage <sid>
```

### Per-turn detail

Anchors token spend to event `seq`, so you can correlate spikes against
the tool-mix audit.

```bash
thirdeye usage <sid> --json | jq -c '{seq, model, in: .input_tokens, out: .output_tokens}'
```

### Find the spike turns

```bash
thirdeye usage <sid> --json \
  | jq -c 'select(.output_tokens > 4000 or .input_tokens > 80000) | {seq, in: .input_tokens, out: .output_tokens, model}'
```

Then open the spike turns in context:

```bash
thirdeye event <sid> <seq>
thirdeye event <sid> $((seq - 1))   # the tool result that fed it
```

A token spike at turn N is almost always caused by the tool result at
turn N-1: an unbounded Read, a giant `grep` dump, an MCP server
returning a paginated payload all at once. That's the fix target.

## Cohort-wide cost view

```bash
thirdeye usage --platform claude --since 14d --sort total --top 20
```

For each high-cost session, classify by task shape (from
[task-identification.md](task-identification.md)) and check whether the
cost is justified by the task or by waste:

| Cost / task | Read it as |
|-------------|------------|
| High cost, large surface task | Probably justified — verify edits land. |
| High cost, small task | Almost always waste — read the tool mix. |
| Low cost, failure | Likely an early abort or wrong tool; check `tool_use_error`. |

## Tool errors

Tool errors are first-class events. Cluster them by tool and message:

```bash
thirdeye events <sid> --json \
  | jq -c 'select(.t == "tool_result" and (.data.error // .data.is_error // false))' \
  | jq -r '.data.tool_name + "\t" + ((.data.error // .data.content // "")|tostring)[0:120]' \
  | sort | uniq -c | sort -rn
```

### Recovery patterns

After each error, check the next 2–3 tool calls. Recovery shapes:

- **Same tool, same args** → agent retried blindly. Bad sign.
- **Same tool, narrowed args** → agent learned (e.g. added `limit`).
  Good.
- **Different tool** → agent re-chose. Often good, sometimes panic
  (Bash fallback when Grep failed).
- **Long text-only stretch then giving up** → agent gave up; the task
  ends incorrectly.

```bash
thirdeye events <sid> --json \
  | jq -c 'select(.t == "tool_call" or (.t == "tool_result" and (.data.error // .data.is_error // false)))' \
  | head -60
```

## Erroneous results that *aren't* errors

The hardest waste: the agent confidently produces a wrong answer with
no error event. Signals:

- A long final assistant message with no preceding `Edit` / `Write`
  when the task asked for code changes.
- An eval verdict of `fail` on this session (see
  `thirdeye eval show <sid>`).
- A user follow-up event saying "no, that's not right" or starting
  over.

```bash
# Sessions with a fail verdict from any rubric.
thirdeye eval list --since 14d --verdict fail --platform claude --json
```

When you find these, the recommendation usually targets the *prompt /
invocation* rather than tool-mix.

## Tie patterns to cost

Build the bridge between [pattern-audit.md](pattern-audit.md) and the
report's "waste patterns found" section by quantifying each pattern's
token cost where possible:

| Pattern observed | How to quantify |
|------------------|-----------------|
| Re-reads of same file | sum input_tokens on Read tool_result turns for that file |
| Unbounded Reads | sum input_tokens on turns following `Read` with no `limit` |
| `grep_via_bash` | sum output_tokens on those Bash tool_results (often very large) |
| MCP dump | sum input_tokens on the turn after the MCP tool_result |
| Retry loop | sum input_tokens across the loop's duration |

A pattern with no measurable cost is still worth surfacing, but the
recommendation prioritization in the report should weight by tokens
saved.

## Carry forward to the report

For each waste pattern that survives this step, you should have:

- The pattern name (from pattern-audit)
- An example `<sid>:<seq>` (from pattern-audit)
- An estimated token cost or error count (from this step)
- A signal of "did this lead to a wrong answer?" (eval verdict, user
  follow-up, or "no — just slow")

The next reference turns these into recommendations.
