# Running evals

`thirdeye eval run` dispatches a judge agent (`claude`, `codex`, or `gemini`)
to grade a recorded session against a definition. The judge runs as a normal
local CLI invocation in read-only mode — no Python LLM SDK is used.

## Choosing an agent

| Agent    | Pros                                                                 | Cons                                                |
|----------|----------------------------------------------------------------------|-----------------------------------------------------|
| `claude` | Rich tool use, dependable structured-output adherence.               | Costs more per run; slower in some configurations.  |
| `gemini` | Cheaper, generally faster.                                           | Less consistent on structured output and reasoning. |
| `codex`  | Useful when you already have an OpenAI/Codex billing relationship.   | NDJSON wire format, less granular cost data.        |

The eval invocation itself is a fully traced thirdeye session (your hooks
are installed), so you can always review what the judge did with
`thirdeye events <judge_sid>`.

## Foreground (blocking) run

```bash
thirdeye eval run <sid> --using default --agent claude
```

The CLI blocks until the judge finishes, appends the result row to
`<session>/evals.jsonl`, and then prints the formatted result. Exit code is
0 even if the verdict is `fail` — a non-zero exit code is reserved for
dispatcher errors (judge crashed, definition not found, etc.).

If `--agent` is omitted, the definition's `default_agent` is used.

## Background run

```bash
thirdeye eval run <sid> --using default --agent claude --background
# → job_id: 9f3a1b2c
```

A background run forks the judge subprocess and returns immediately with a
`job_id`. The stub at `<session>/evals.jobs/<job_id>.json` tracks state.

Check progress:

```bash
thirdeye eval status            # all in-flight jobs
thirdeye eval status <sid>      # jobs for one session
```

When the judge finishes, the result lands in `evals.jsonl` and the job
stub flips to `done` (or `error`).

## `--using` vs `--rubric`

These two are mutually exclusive — pick one:

- `--using NAME` — load a stored definition by name.
- `--rubric FILE` — supply a one-off YAML directive on disk without
  installing it as a named definition.

```bash
# stored
thirdeye eval run <sid> --using token-efficiency

# one-off
thirdeye eval run <sid> --rubric ./adhoc-rubric.yaml --agent gemini
```

Prefer `--using` whenever you'll run the rubric more than once — installed
definitions are versionable, discoverable via `thirdeye eval def list`, and
serve as the unit `thirdeye eval list --using` filters on.

## Worked example: kick off background evals across many sessions

Evaluate every currently-open session in parallel:

```bash
thirdeye list --status open --json \
  | jq -r '.session_id' \
  | xargs -I {} thirdeye eval run {} --agent claude --background
```

Then poll status:

```bash
thirdeye eval status
```

When all jobs report `done`, surface the histogram:

```bash
thirdeye eval list --using default --since today --json \
  | jq -s 'group_by(.verdict) | map({verdict: .[0].verdict, n: length})'
```

## Read-only posture per agent

Each adapter sets per-CLI sandboxing flags before invoking the judge so the
evaluator cannot mutate your repo or session store. The exact flags
(allowlists, sandbox mode, plan-only mode) are documented in the full design
spec under "Adapter pattern → read-only flags". The contract is the same
across all three agents: the judge can read the session and run
`thirdeye`/`sqlite3`, but writes, edits, and shell escapes are denied.

If a judge appears to have made changes outside read-only scope, that is a
bug in the adapter — file it with the session id of the eval invocation
itself (visible via `thirdeye list --platform claude` immediately after the
run completes).
