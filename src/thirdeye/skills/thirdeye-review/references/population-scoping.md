# Population scoping

The cohort is the most important decision in a review. A bad cohort
makes every downstream number meaningless: too narrow and findings are
about one quirky session; too wide and you average across task types
that have nothing to compare.

## The four axes

Pick a value for each before looking at any data.

| Axis | Examples | Rule of thumb |
|------|----------|---------------|
| **Agent type** | `claude`, `codex`, `gemini` | Always pick exactly one. Tool conventions differ enough that mixing platforms hides per-agent waste. |
| **Repo / cwd** | `$PWD`, `~/code/thrdi-web` | Constrain to repos with similar conventions. Cross-repo reviews need a common task type. |
| **Time window** | `7d`, `14d`, ISO range | Wider = more sessions, but older sessions may predate skill/convention changes. Default 14d. |
| **Task shape** | workbench dispatch, free-form, eval, planning | The hardest axis to filter on — usually requires reading the first user turn. See [task-identification.md](task-identification.md). |

## Recipes

### Sessions by platform + repo + window

```bash
thirdeye list --json \
  --platform claude \
  --cwd "$PWD" \
  --since 14d \
  | jq -c '{sid: .session_id[0:8], events: .event_count, started: .started_at, cwd: .cwd}'
```

### Substantive sessions only

Anything under ~20 events is usually setup, abort, or a one-shot question
— filter it out unless the task itself is short.

```bash
thirdeye list --json --platform claude --cwd "$PWD" --since 14d \
  | jq -c 'select(.event_count > 20)'
```

### Workbench task sessions

The `wb` CLI dispatches each task into its own working directory under
`<repo>/.workbench/<plan>/task-*`, so cwd is a perfect filter.

```bash
thirdeye list --json --since 14d \
  | jq -r 'select(.cwd | test("/\\.workbench/.+/task-")) | .session_id'
```

### Long-tail (cost) sessions

For a cost-focused review, start from the global usage rollup, not from
`thirdeye list`:

```bash
thirdeye usage --platform claude --since 14d --top 20
```

Then pull session metadata for each ID returned.

## Sizing the cohort

| N sessions | Use case |
|-----------|----------|
| 1–2 | Diagnosing a single incident — write up that session, do not generalize. |
| 3–5 | Spot-check: enough to see a pattern, not enough for confident recommendations. Caveat the report. |
| 6–15 | **Default review size.** Findings here are usually actionable. |
| 15+ | Population-level analysis. Worth aggregating numerically; consider sub-clustering by task. |

## When to widen vs narrow

- **Findings look like one weird session?** Narrow (drop that session
  or filter on its task type), then re-scan.
- **Same waste pattern in every session?** Widen across cwds or
  platforms to test whether it's actually agent-level or just this
  repo.
- **Nothing interesting happens?** Either the agents are fine, or the
  cohort is too tame — try a noisier window (a known incident week) or
  filter to `event_count > 80` for cost-dominant sessions.

## Save the cohort

Once you have a cohort that's worth reviewing, tag the sessions so the
next review can compare against the same baseline:

```bash
for sid in $(thirdeye list --json --platform claude --cwd "$PWD" --since 14d \
             | jq -r 'select(.event_count > 40) | .session_id'); do
  thirdeye tag "$sid" 0 --add review-2026-06-cohort
done
```

Later: `thirdeye list --tag review-2026-06-cohort`.
