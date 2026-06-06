# Per-turn findings

Each finding the judge emits has a shape:

```json
{
  "seq": 47,
  "severity": "warn",
  "category": "redundancy",
  "note": "Re-reads file X for the third time in 5 turns."
}
```

Fields:

- `seq` — integer event sequence number this finding attaches to. `null`
  means session-level (no specific turn).
- `severity` — `info` | `warn` | `error`.
- `category` — short free-form tag (e.g. `redundancy`, `tool-quality`).
- `note` — one-line human-readable description.

## Severity glyphs in the events viewer

When `thirdeye events <sid>` renders a session, findings appear inline
beneath the event they're attached to:

```
[12] tool_use   Read     src/foo.py
    · info   redundancy   File read for the second time
[47] tool_use   Bash     find . -name '*.py'
    ⚠ warn   tool-quality Used find . instead of git ls-files
[91] tool_use   Edit     src/foo.py
    ✖ error  task-drift   Edit changed unrelated section
```

Glyphs: `·` info, `⚠` warn, `✖` error.

## Default behavior

Annotations are on by default for both `thirdeye events <sid>` and
`thirdeye event <sid> <seq>`. The renderer pulls all findings from
`evals.jsonl` and groups them by `seq`.

If a session has multiple eval results (e.g. ran `default` then
`token-efficiency` later), all of their findings render together. To narrow:

```bash
# only findings from the token-efficiency rubric
thirdeye events <sid> --eval token-efficiency

# suppress all findings (clean diff-friendly output)
thirdeye events <sid> --no-findings
```

## Drilling from session-level to turn-level

Session-level findings (where `seq` is `null`) appear under a "session"
heading at the top of `thirdeye events`. They typically reference the
narrative; cross-reference with `thirdeye eval show <sid>` to see the
evaluator's full reasoning, then jump to the specific turn:

```bash
thirdeye eval show <sid>          # read narrative, note seq numbers
thirdeye event <sid> 47           # inspect the offending turn in detail
```

## Worked example: list every event with at least one warning

```bash
sid=<your-session-id>
thirdeye eval show "$sid" --json \
  | jq -r '.findings[] | select(.severity == "warn" and .seq != null) | .seq' \
  | sort -u \
  | while read -r seq; do
      echo "=== seq $seq ==="
      thirdeye event "$sid" "$seq"
    done
```

## Worked example: count findings by category across one session

```bash
thirdeye eval show <sid> --json \
  | jq '.findings | group_by(.category) | map({cat: .[0].category, n: length})'
```

Use this to spot which dimension (redundancy, tool-quality, task-drift)
dominated a given session.
