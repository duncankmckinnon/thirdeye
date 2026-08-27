# Searching and retrieval

Find sessions and events across the unified store. Every session ID accepts any
unique prefix — typing the first 4-8 characters is usually enough.

## List sessions

```bash
thirdeye list                                # every session, newest first
thirdeye list --json                         # one JSON object per line (default)
thirdeye list --tree                         # human-readable
```

### Filters

All filters AND together. `--tag` is repeatable and tags are AND'd.

```bash
thirdeye list --platform claude              # by platform
thirdeye list --harness codex                # alias for --platform
thirdeye list --cwd /path/to/repo            # by working directory
thirdeye list --status open                  # only open sessions
thirdeye list --since 2026-05-01             # active at/after this time
thirdeye list --until 2026-05-13             # active at/before this time
thirdeye list --tag review                   # at least one event tagged 'review'
thirdeye list --tag review --tag bug         # events tagged both 'review' AND 'bug'
```

## Per-session reads

```bash
thirdeye events <sid>                        # all events, terse
thirdeye events <sid> --json                 # JSON-per-line, machine-parseable
thirdeye events <sid> --type tool_use        # filter by event type (repeatable)
thirdeye show <sid>                          # alias of `events`
thirdeye tail <sid> -n 5                     # last 5 events
thirdeye event <sid> <seq>                   # one event, fully expanded
thirdeye event <sid> <seq> --field input     # print only one field of data
```

## Search across sessions

```bash
thirdeye search "migration"                  # substring across all events
thirdeye search "tool error" \
    --platform claude \
    --tag review \
    --since 2026-05-01                       # filters AND together
```

Search filters: `--platform` / `--harness`, `--cwd`, `--tag` (repeatable), `--since`,
`--until`.

## Composing pipelines

`--json` output is one event/session per line, so it composes cleanly with `jq`:

```bash
# Enumerate session IDs from the last week and tail each one
thirdeye list --json --since 2026-05-01 \
  | jq -r '.session_id' \
  | while read -r sid; do
      echo "=== $sid ==="
      thirdeye tail "$sid" -n 3
    done

# Count tool calls per session
thirdeye list --json --platform claude \
  | jq -r '.session_id' \
  | while read -r sid; do
      count=$(thirdeye events "$sid" --json --type tool_use | wc -l)
      echo "$sid $count"
    done

# Pull just the inputs of tool_use events for a session
thirdeye events <sid> --json --type tool_use \
  | jq '.data.input'
```
