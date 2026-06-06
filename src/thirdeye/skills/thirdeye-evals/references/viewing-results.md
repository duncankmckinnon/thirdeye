# Viewing eval results

Eval results live in `<session>/evals.jsonl`, append-only, one JSON row per
run. Bytes never move once written; multiple runs against the same session
stack, and the latest row of a given definition is the default surfaced.

## `thirdeye eval show <sid>`

Prints the latest eval result for the session. Output anatomy:

```
session: 9f3a1b2c-...                          platform: claude
eval:    default (claude)                      finished: 2026-05-16T18:02:11Z
verdict: warn

summary
  The agent completed the requested refactor but re-read the same 4 files
  twice and emitted an unnecessary full-tree listing.

scores
  overall              0.72
  task_completion      0.95
  efficiency           0.55
  tool_quality         0.80

findings (3)
  seq    severity  category      note
  ----   --------  ------------  -----------------------------------------
  12     warn      redundancy    File X read twice in 4 turns
  47     warn      tool-quality  Used find . instead of git ls-files
  (none) info      summary       Session would benefit from /compact mid-run

narrative
  <multi-paragraph evaluator commentary>
```

## Pinning to a specific run

```bash
thirdeye eval show <sid> --id <eval_id>      # specific past run
thirdeye eval show <sid> --using default     # latest of one definition
```

`--id` and `--using` are mutually exclusive. Omit both to get the most
recent eval of any definition.

## JSON output

```bash
thirdeye eval show <sid> --json
```

Emits the raw row from `evals.jsonl`. Pipe through `jq` for scripting:

```bash
thirdeye eval show <sid> --json | jq '.verdict, .scores.overall'
```

## `thirdeye eval list`

Lists eval rows across sessions. With no `<sid>` it scans the whole store.

```bash
thirdeye eval list                              # all evals everywhere
thirdeye eval list <sid>                        # one session's history
thirdeye eval list --using default              # filter by definition
thirdeye eval list --agent gemini               # filter by judge agent
thirdeye eval list --verdict fail               # only failures
thirdeye eval list --since 2026-05-01 --until 2026-05-10
```

All filters compose. Add `--json` for parseable output.

## Worked example: verdict histogram

```bash
thirdeye eval list --since 2026-05-01 --json \
  | jq -s 'group_by(.verdict) | map({verdict: .[0].verdict, n: length})'
```

## Worked example: average overall score for one definition

```bash
thirdeye eval list --using default --json \
  | jq -s 'map(.scores.overall) | add / length'
```

## Worked example: find sessions where two rubrics disagree

```bash
thirdeye eval list --json \
  | jq -s '
      group_by(.session_id)
      | map(select((map(.verdict) | unique | length) > 1))
      | map({sid: .[0].session_id, verdicts: map({(.using): .verdict})})
    '
```

Useful when a narrow rubric (e.g. `token-efficiency`) flags problems the
broad `default` rubric missed, or vice versa.
