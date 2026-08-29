---
name: use-thirdeye
description: Use when an agent needs to inspect, search, or evaluate past agent sessions captured by the thirdeye CLI — including debugging tool calls, analyzing token usage, reviewing session efficiency to suggest skill improvements, retrieving session events, and running evaluations across recorded traces.
---

## Overview

`thirdeye` (PyPI: `thrdi`) captures events from agentic tools (Claude Code, Codex, Cursor)
into a unified per-session event store on disk. Each session's data lives under
`<thirdeye_home>/traces/<platform>/<sid>/` and contains a sequential log of all events the
agent emitted during that session. Sessions are addressable by any unique prefix of their session ID,
so `thirdeye events abc123` works as long as `abc123` is unambiguous.

## Setting up tracing

Use this when you need to enable or disable thirdeye hooks for a specific platform so that future
sessions are captured automatically. See [setup-and-tracing.md](references/setup-and-tracing.md)
for full instructions on installing hooks, verifying data flow, and removing hooks when done.

```bash
# Enable hooks for a platform
thirdeye add --claude
thirdeye add --codex
thirdeye add --cursor

# Remove hooks
thirdeye remove --claude
thirdeye remove --codex
thirdeye remove --cursor
```

## Searching and retrieving session data

Use this to find sessions, page through events, or pull specific events for further inspection.
See [searching-and-retrieval.md](references/searching-and-retrieval.md) for full filter and
output format details.

Key commands:

```bash
thirdeye list                          # list recent sessions
thirdeye list --platform claude --since 2024-01-01
thirdeye events <id>                   # stream all events for a session
thirdeye tail <id>                     # follow live events
thirdeye event <id> <seq>              # fetch one event by sequence number
thirdeye search "<query>"              # full-text search across sessions
thirdeye search "<query>" --platform claude --cwd /my/repo --tag reviewed
```

Available filters: `--platform`, `--cwd`, `--tag`, `--since`, `--until`. Add `--json` to any
command for machine-parseable output.

## Debugging tool calls

When a past agent ran a tool that failed, hung, or returned wrong data, use this workflow to
locate the relevant events and reconstruct the tool's input and output. See
[tool-call-debugging.md](references/tool-call-debugging.md) for the step-by-step workflow,
including how to narrow to tool events and read input/output payloads.

```bash
thirdeye events <id> --json | jq 'select(.type == "tool_use")'
thirdeye event <id> <seq>
```

## Analyzing token usage

When investigating high-cost sessions or comparing token efficiency across runs, use the token
analysis workflow in [token-use-analysis.md](references/token-use-analysis.md). It covers
reading usage fields from events, aggregating across a session, and comparing sessions with
`thirdeye stats`.

```bash
thirdeye stats <id>
thirdeye stats <id> --json
```

## Reviewing session efficiency

When asked to improve agent performance, reduce token usage, or suggest new skills based on
recorded behavior, use the workflow in
[session-efficiency-review.md](references/session-efficiency-review.md). It covers population
selection, tool-mix analysis, red-flag thresholds, and how to turn findings into skill or
plan updates.

```bash
thirdeye list --json --cwd "$PWD" --since 2026-05-01 | jq 'select(.event_count > 80)'
thirdeye events <id> --json | jq -r 'select(.t == "tool_call") | .data.tool_name' | sort | uniq -c
```

## Running evaluations

When grading agent behavior against a rubric — accuracy of edits, adherence to instructions,
tool-selection correctness — across a set of sessions, use the workflow in
[evaluation-workflows.md](references/evaluation-workflows.md). It covers selecting a session
set, extracting relevant events, and applying structured criteria.

```bash
thirdeye list --tag eval-candidate --json
thirdeye events <id> --json
```

## Tagging and curation

Tags let you annotate individual events and then use those annotations as filters in search
and evaluation workflows.

```bash
thirdeye tag <id> <seq> --add reviewed,correct
thirdeye tag <id> <seq> --add needs-investigation
thirdeye tags                          # list all tags in use
thirdeye search "<query>" --tag reviewed
```

Use `thirdeye tags` to discover what tags exist across your sessions. Any tag added with
`thirdeye tag` is immediately available as a `--tag` filter on `list`, `events`, and `search`.
