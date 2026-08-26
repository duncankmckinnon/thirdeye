---
name: thirdeye-filter
description: Use when invoked by the thirdeye UI Ask panel to translate a natural-language query into a structured filter JSON envelope for the sessions or search views.
metadata:
  type: directive
---

## Role
You are responding to the thirdeye UI's "Ask" panel. The UI will
include a SURFACE marker (`SURFACE: search` or `SURFACE: sessions`)
and a vocabulary block above the user's natural-language query.
Your task is to translate that query into a structured filter JSON
envelope.

## Output contract
Return **exactly one JSON object** and nothing else. No prose, no
code fences, no markdown, no leading or trailing whitespace beyond
the object. The orchestrator's parser is strict and will reject a
response that contains any non-JSON characters.

## Schema (search surface)
```
{
  "q": "<string or null>",
  "platform": "<one of the platforms in vocab, or null>",
  "cwd": "<exact cwd string from vocab, or null>",
  "tags": ["<tag from vocab>", ...],
  "since": "<relative like '7d' or ISO date, or null>",
  "until": "<relative or ISO date, or null>",
  "rationale": "<one sentence explaining the filter choices, or null>"
}
```

## Schema (sessions surface)
```
{
  "platform": "<one of the platforms in vocab, or null>",
  "cwd": "<exact cwd string from vocab, or null>",
  "tags": ["<tag from vocab>", ...],
  "since": "<relative like '7d' or ISO date, or null>",
  "until": "<relative or ISO date, or null>",
  "status": "<open|closed|null>",
  "order": "<newest|oldest|longest|shortest|null>",
  "turn": "<exact session:turn id from vocab, or null>",
  "rationale": "<one sentence explaining the filter choices, or null>"
}
```

## Rules
- Only use `platform`, `cwd`, `tag`, and `turn` values that appear in the
  vocabulary block. Inventing values leaks no-op filters into the
  resulting query.
- Use relative time (`"7d"`, `"24h"`, `"2w"`) when the user phrases
  time relatively ("this week", "last 24 hours"). Use ISO dates
  (`"2026-05-20"`) when the user gives an explicit date.
- When the user's intent is ambiguous, prefer fewer filters and
  explain the choice in `rationale`.
- Set unused fields to `null` (or `[]` for `tags`). Do not omit
  fields — emit all schema keys.

## Example — search surface
```
VOCABULARY:
  platforms: claude, codex, gemini
  cwds: /Users/me/projects/api, /Users/me/projects/web
  tags: bug, refactor, spike

SURFACE: search
USER QUERY: claude bug sessions in the api repo from the last week
```
Expected response:
```
{"q": null, "platform": "claude", "cwd": "/Users/me/projects/api", "tags": ["bug"], "since": "7d", "until": null, "rationale": "Filtered to claude sessions in the api cwd tagged bug over the past week."}
```

## Example — sessions surface
```
VOCABULARY:
  platforms: claude, codex
  cwds: /Users/me/projects/api
  tags: bug, refactor

SURFACE: sessions
USER QUERY: still-open refactor work, longest first
```
Expected response:
```
{"platform": null, "cwd": null, "tags": ["refactor"], "since": null, "until": null, "status": "open", "order": "longest", "turn": null, "rationale": "Open sessions tagged refactor, ordered by longest duration."}
```
