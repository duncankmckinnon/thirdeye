---
name: thirdeye-review
description: Use when asked to find inefficiencies in agent behavior, reduce token spend, improve agent results, or suggest invocation/skill changes by analyzing past thirdeye-traced sessions. Filters traces by agent type, identifies the tasks performed, audits tool choices and execution patterns, surfaces token overuse and erroneous results, ties each finding back to how the agent was invoked, and proposes concrete invocation tweaks or new skills.
---

# Reviewing thirdeye traces for agent inefficiencies

This skill is for an agent reviewing **other** agents' traces in the local
thirdeye store and turning the findings into actionable improvements:
invocation tweaks, new skills, sharper skill descriptions, or convention
edits. It assumes [`use-thirdeye`](../use-thirdeye/SKILL.md) is installed
for the underlying CLI surface (`thirdeye list / events / usage / search`).

## The five-step workflow

Follow these in order. Each step has a dedicated reference with concrete
recipes — open it when you start that step, not all at once.

1. **Scope the population.** Pick a cohort: one platform (`claude` /
   `codex`), one cwd / repo, one time window. Wider cohorts dilute
   signal; narrower cohorts overfit to a single quirky session. Target
   5–20 sessions of comparable task type.
   → [population-scoping.md](references/population-scoping.md)

2. **Identify the tasks.** Read each session's first 1–3 user/instruction
   events and cluster sessions by what the agent was actually being asked
   to do. Without a task label, "the agent wasted tokens" can't be tied
   to anything fixable.
   → [task-identification.md](references/task-identification.md)

3. **Audit tools and execution patterns.** Count tool calls per session,
   look for the known waste signatures (re-reads, bash-grep, broad
   exploration, retry loops, oversized payloads), and compare to a
   reference baseline for the same task type.
   → [pattern-audit.md](references/pattern-audit.md)

4. **Quantify token overuse and surface errors.** Cross the tool-mix
   findings with per-turn token usage and tool-error events. Findings
   that don't bend the cost curve or fix a wrong answer aren't worth a
   recommendation.
   → [token-and-error-analysis.md](references/token-and-error-analysis.md)

5. **Tie back to invocation and recommend.** Map each waste pattern to
   the *invocation surface* that drove it — the user prompt, available
   skills, conventions file, plan task description, MCP server set — and
   propose a targeted change. Generic advice ("write better prompts") is
   not a recommendation.
   → [invocation-tieback-and-recommendations.md](references/invocation-tieback-and-recommendations.md)

## Quick start

```bash
# 1. Scope: last two weeks of claude sessions in this repo, over 40 events.
thirdeye list --json --platform claude --cwd "$PWD" --since 14d \
  | jq -c 'select(.event_count > 40)'

# 2. Pull the first few user turns from each to label task type.
thirdeye events <sid> --json | jq -c 'select(.t == "user") | {seq, text: (.data.text // "")[0:200]}' | head -3

# 3. Tool mix.
thirdeye events <sid> --json | jq -r 'select(.t == "tool_call") | .data.tool_name' | sort | uniq -c | sort -rn

# 4. Token spend correlated to that session.
thirdeye usage <sid>

# 5. Tag findings so you can revisit and compare.
thirdeye tag <sid> 0 --add review-2026-06, redundant-reads
```

## Output: the review report

Every review must end with a report of this shape — anything less is
just an observation log. The orchestrator / reader cares only about
**what to change**.

1. **Population.** Cohort definition, N sessions, agent type, time window,
   repo/task type. One paragraph.
2. **Tasks observed.** 2–6 task clusters with counts. ("4 × workbench
   dispatch, 3 × bug investigation, 2 × ad-hoc refactor.")
3. **Waste patterns found.** Top 2–4 patterns with: signal name, the
   threshold or number, an example session ID + seq, and the token /
   error cost where measurable.
4. **Recommendations.** Each item must name the invocation surface
   (skill name, conventions file, plan task description, system prompt,
   MCP allowlist) and the exact change. Group as:
   - *New skills* — name + 1-line description + the trigger sentence.
   - *Skill description tightening* — existing skill + before/after of
     the `description:` field.
   - *Convention / plan changes* — bullets to add to
     `.workbench/conventions.md` or plan Context.
   - *Invocation flags* — `--allowedTools`, MCP set, sandbox mode,
     `--max-retries`, etc.
5. **Followups.** Tags written, sessions worth re-running with a fix.

## Anti-patterns

- **Reviewing one session.** A single session can't separate "agent did
  badly" from "task was weird." Always look at ≥ 3 comparable sessions.
- **Counting without comparing.** "47 tool calls" is noise without a
  baseline. Always cite a reference — same task type with fewer calls,
  or the same agent on a smaller subtask.
- **Suggesting "be more careful."** Every recommendation must change a
  file or flag. If you can't name the file or flag, you don't have a
  recommendation yet.
- **Ignoring the invocation.** Tool-mix problems often live in the
  prompt, not the agent. Always pull the first user turn before
  blaming the agent's choices.
- **Skipping the tag step.** Without tags, the next review can't tell
  which sessions you've already audited or what you concluded.
