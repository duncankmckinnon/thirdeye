# Task identification

A waste pattern only matters relative to a task. "47 tool calls" is
high for a one-line fix and low for a multi-file refactor. Before
counting anything, label what each session was *trying* to do.

## What counts as "the task"

The task is whatever the user or orchestrator asked for in the first
1–3 user-role events. Anything the agent decided to do on its own
(exploration, planning, retries) is part of the *execution*, not the
task.

For workbench-dispatched sessions the first user event is the rendered
task description — the cleanest possible task label.

## Pull the first user turns

```bash
thirdeye events <sid> --json \
  | jq -c 'select(.t == "user") | {seq, text: (.data.text // .data.content // "")[0:300]}' \
  | head -3
```

For Claude Code sessions, the first user event after a hook stop is the
real prompt; the `system-reminder` blocks above it are environmental.
Strip them by grepping for the first non-reminder text.

## Cluster sessions by task shape

For a cohort of 5–15 sessions, read the first user turn of each and
write a one-line label. Then group:

```
task-shape:
  workbench-dispatch    [a3f2c8, 91e4d2, 7d12c9, 4e0f6b]   4
  bug-investigation     [b8c402, 71d9aa]                   2
  refactor-multi-file   [3f9e2a]                           1
  one-shot-question     [a2c1d3, ee0011]                   2
```

Comparable sessions cluster together. Singletons should usually be
dropped from numeric analysis (they're anecdotes, not patterns).

## Programmatic task fingerprints

For larger cohorts, a coarse classifier on the first user turn cuts
manual labeling:

```bash
thirdeye list --json --platform claude --since 14d \
  | jq -r '.session_id' \
  | while read sid; do
      first=$(thirdeye events "$sid" --json 2>/dev/null \
              | jq -c 'select(.t == "user") | (.data.text // .data.content // "")[0:200]' \
              | head -1)
      printf '%s\t%s\n' "$sid" "$first"
    done \
  | python3 -c '
import sys, re
for line in sys.stdin:
    sid, _, text = line.partition("\t")
    text_l = text.lower()
    if re.search(r"task description|dispatch|workbench", text_l): tag = "wb"
    elif re.search(r"fix|bug|error|fails|broken|regression", text_l): tag = "bug"
    elif re.search(r"refactor|rename|extract|move", text_l): tag = "refactor"
    elif re.search(r"\?$|how (do|does|can)", text_l): tag = "question"
    elif re.search(r"add|implement|build|create", text_l): tag = "feature"
    else: tag = "other"
    print(f"{tag}\t{sid[:8]}\t{text.strip()[:80]}")
' | sort
```

This is intentionally crude — the goal is to **bucket** sessions for
comparison, not to perfectly classify them.

## Task-shape baselines

Once you've clustered, expected ranges per shape (Claude Code, no
skills involved):

| Task shape | Events | Tool calls | Notes |
|------------|--------|------------|-------|
| one-shot question | 4–20 | 0–4 | Often zero tool calls if it's answered from prompt context. |
| bug investigation | 20–80 | 8–30 | Read-heavy. Should end in 1 edit, not many. |
| feature add | 40–150 | 20–60 | Read + edit + verify. Above 60 tool calls usually means thrash. |
| refactor (multi-file) | 60–200 | 30–80 | Edit-heavy. High Read count is expected. |
| workbench task | varies | varies | Use the cohort's own median as the baseline, not a global one. |

These are starting points — your repo will have its own envelope.
Always derive baselines from comparable sessions in the cohort.

## When the task isn't clear

Some sessions have a vague first turn ("continue", "keep going"), are
resumed mid-conversation, or were started by a sub-process. For these:

1. Check `thirdeye events <sid> --json | jq 'select(.t == "user")' | head -5`
   to see if the second or third user turn carries the task.
2. If still unclear, drop the session — it can't be benchmarked.
3. Tag it `task-unclear` so the next reviewer doesn't waste time on it.

## Output for the report

Step 2 of the report (see SKILL.md) is a one-line task summary of the
cohort:

> *4 × workbench dispatch, 3 × bug investigation, 2 × ad-hoc refactor,
> 1 unclear (dropped).*

Carry the task labels into pattern audit — they make every count
comparable.
