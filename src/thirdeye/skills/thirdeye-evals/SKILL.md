---
name: thirdeye-evals
description: Use when an agent needs to create, run, view, or manage evaluations against thirdeye-recorded sessions — defining named rubrics, dispatching agent-based evaluators (claude/codex/gemini), interpreting per-turn findings, or comparing eval results across sessions.
---

# Evaluating thirdeye sessions

This skill teaches you how to use `thirdeye eval` to grade recorded sessions
with an LLM-as-judge agent. It assumes the `use-thirdeye` skill is also
installed for basic session navigation (`thirdeye list`, `thirdeye events`,
`thirdeye usage`).

## Overview

An evaluation is one run of an **eval definition** (a named directive) against
a **session**, dispatched to one of: `claude`, `codex`, or `gemini`. Results
are append-only at `<session>/evals.jsonl`. The dispatched agent runs in
read-only mode and can use `thirdeye` and `sqlite3` to verify findings.

## Eval definitions

`thirdeye eval def list / show / create / edit / rm`. Three shipped defaults
(`default`, `token-efficiency`, `tool-quality`) are lazily materialized into
`<thirdeye_home>/evals/defs/` so you can edit them. Create your own with
`thirdeye eval def create <name> --directive ...`. See
[eval-definitions.md](references/eval-definitions.md).

## Running an eval

`thirdeye eval run <sid> --using <name> --agent claude|codex|gemini`. Add
`--background` to detach. Exit code is 0 regardless of verdict. The eval
invocation is itself a traced thirdeye session for audit. See
[running-evals.md](references/running-evals.md).

## Viewing results

`thirdeye eval show <sid>` prints the latest result with verdict, scores, and
findings table. `thirdeye eval list` shows history across sessions.
`thirdeye eval status` shows background jobs. See
[viewing-results.md](references/viewing-results.md).

## Per-turn findings

Findings keyed by event `seq` annotate `thirdeye events <sid>` and
`thirdeye event <sid> <seq>` output by default. Filter with `--eval NAME` or
hide with `--no-findings`. See
[per-turn-findings.md](references/per-turn-findings.md).

## Common evaluation patterns

Token efficiency, tool quality, error recovery, task adherence, redundancy.
Each has a sample directive snippet. See
[evaluation-patterns.md](references/evaluation-patterns.md).
