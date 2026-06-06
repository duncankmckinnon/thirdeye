# Eval definitions

An **eval definition** is a named, reusable rubric stored on disk. It tells
the dispatched judge agent what to look for in a session. Definitions are
plain YAML — no code — and live in `<thirdeye_home>/evals/defs/<name>.yaml`.

## Anatomy

```yaml
name: token-efficiency
description: Grades a session on prompt cost, cache utilization, and result-size hygiene.
default_agent: claude
output_schema: v1
directive: |
  You are evaluating a recorded agent session for token efficiency.
  Focus on:
    - oversized system / context prompts vs. work performed
    - repeated reads of the same file or directory listing
    - tool results that dump large payloads when a summary would do
    - low prompt-cache hit ratio across turns
  Emit findings keyed to event seq numbers where wasteful behavior occurred.
  Verdict: pass if no warnings; warn if any warning-level finding; fail if
  any error-level finding.
```

Field reference:

- `name` — kebab-case identifier; also the filename stem.
- `description` — one line, surfaced by `thirdeye eval def list`.
- `default_agent` — `claude` | `codex` | `gemini`. Used when `--agent` is
  omitted on `thirdeye eval run`.
- `output_schema` — pinned to `v1`. The dispatcher uses this to know which
  output contract block to append.
- `directive` — the freeform rubric. Becomes block 1 of the assembled
  prompt sent to the judge.

## Prompt assembly

The judge agent never sees just your directive. The dispatcher concatenates
three blocks:

1. **Your directive** — verbatim from the YAML.
2. **Session context** — paths, commands, and a summary of how to read the
   session via `thirdeye events <sid>` and `sqlite3`.
3. **Output contract** — the strict JSON shape the judge must return,
   matching `output_schema`.

Because blocks 2 and 3 are added automatically, your directive should NOT:

- re-describe what a thirdeye session is or how to read events,
- re-state the output JSON shape or field names,
- include closing instructions like "now produce JSON".

Keep the directive focused on *what to evaluate* and *how to grade*.

## Workflows

### Inspect a shipped default and copy it for customization

```bash
thirdeye eval def show default > /tmp/d.yaml
# edit /tmp/d.yaml in your editor
thirdeye eval def create my-default --directive-file /tmp/d.yaml --force
```

### Branch from an existing definition

```bash
thirdeye eval def create strict-default --from default
thirdeye eval def edit strict-default
```

`--from` copies the source definition's directive, description, and
default_agent into the new file; you then edit only what you want to change.

### Create a one-off definition inline

```bash
thirdeye eval def create no-todo-spam \
  --directive 'Flag any turn where the agent called TodoWrite with no actual state change.' \
  --default-agent claude
```

### Discover what's already installed

```bash
thirdeye eval def list
thirdeye eval def show token-efficiency
```

## Worked example: test-driven workflow adherence

This rubric flags agents that claim a task is done without first running
the relevant test suite.

```yaml
name: tdd-adherence
description: Grades whether the agent ran tests before claiming completion of any code change.
default_agent: claude
output_schema: v1
directive: |
  Evaluate this session for test-driven discipline. For every turn where the
  agent claimed an implementation, fix, or refactor was complete, verify by
  inspecting earlier events that BOTH of the following held:
    1. The agent invoked a test runner (pytest, npm test, go test, cargo
       test, etc.) on the affected code path.
    2. The test run succeeded — exit 0 or equivalent.
  Emit a warning finding (seq = the completion-claim event) when a claim
  was made without a preceding successful test run. Emit an error finding
  if completion was claimed after a failing test run with no subsequent
  fix-and-retest cycle.
  Verdict:
    pass  — every completion claim had a prior successful test run.
    warn  — at least one claim lacked a test run but no failing tests were
            ignored.
    fail  — any completion claim followed a known-failing test.
```

Save to `~/.thirdeye/evals/defs/tdd-adherence.yaml` (or use `thirdeye eval
def create tdd-adherence --directive-file ...`) and invoke with:

```bash
thirdeye eval run <sid> --using tdd-adherence
```

## Narrow vs broad rubrics

Prefer narrow, single-axis rubrics (one for token efficiency, one for tool
quality, one for tdd-adherence) over a single mega-rubric. Benefits:

- Each judge call has tighter focus, fewer false positives.
- You can score the same session against several axes and see them
  side-by-side with `thirdeye eval list <sid>`.
- Findings are easier to act on when each finding's category is implied by
  the rubric name.

Use a broad rubric (`default`) when you want a quick triage signal across
many sessions and don't yet know which dimensions matter.
