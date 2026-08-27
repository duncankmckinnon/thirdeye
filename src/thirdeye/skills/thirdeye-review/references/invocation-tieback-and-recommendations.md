# Tying findings to invocation, and writing recommendations

This is the only step that produces output the reader can act on.
Everything before it is measurement; this is the prescription. A
recommendation that doesn't name a file or flag isn't a
recommendation — it's commentary.

## What "the invocation" is

The set of things that shape an agent's behavior on a given run:

| Surface | Where it lives | Edited by |
|---------|----------------|-----------|
| User prompt / task description | First user turn of the session; for `wb`-dispatched tasks, the rendered task body. | Plan author (`.workbench/<plan>/plan.md`). |
| System prompt / harness defaults | The agent harness (Claude Code, Codex). | Usually fixed per platform. |
| Available skills | `~/.claude/skills/`, `.agents/skills/`, `.claude/skills/` and per-plugin registries. | Skill author. |
| Skill descriptions | Each skill's `description:` frontmatter — drives whether the harness loads it. | Skill author. |
| Conventions file | `.workbench/conventions.md` or repo `CLAUDE.md` / `AGENTS.md`. | Repo maintainer. |
| Tool allowlist / MCP set | `--allowedTools`, `--sandbox`, MCP server registration. | Caller. |
| Retry / sandbox flags | `--max-retries`, `--sandbox read-only`, eval invocation flags. | Caller. |

For every waste pattern, the question is: **which of these would I
edit to make this pattern less likely?**

## Read the actual invocation

You can't recommend a prompt change without seeing the prompt. For
each example session you cite, pull:

```bash
# Initial task / prompt
thirdeye events <sid> --json \
  | jq -c 'select(.t == "user") | .data' | head -2

# System reminders / skill list visible to the agent
thirdeye events <sid> --json \
  | jq -c 'select(.t == "system_reminder" or (.t == "user" and (.data.text // "") | test("available-skills")))' \
  | head -5
```

If the relevant skill *was* listed for the agent and it still didn't
trigger, the recommendation is "tighten the skill description" — not
"the agent should know better".

## Pattern → recommendation map

These are the recurring mappings. Use as a starting point; calibrate to
what you actually saw.

| Pattern | Likely invocation cause | Recommendation type |
|---------|------------------------|---------------------|
| `grep_via_bash` > threshold | No conventions guidance on tool preference | Convention bullet |
| `filesystem_explore` > threshold | No codebase map in Context | New skill (repo nav) **or** plan Context section |
| Re-reads of same file | Task lacks specific file/line citations | Plan task description — add `Files: path:line-line` |
| Unbounded Reads | No guidance on Read sizing | Convention bullet (prefer Grep + targeted Read) |
| Wrong tool choice on errors | Tool failure pattern not in conventions | Convention bullet with the specific error → tool |
| 0 Skill invocations on complex task | Skill description too narrow | Tighten skill `description:` field |
| Long retry loop on test failures | `--max-retries` ceiling too high; test output not in Context | Lower retry cap; include failing assertion in plan |
| MCP tool dumps causing token spikes | MCP server returns unbounded payloads | Pagination flag, or remove MCP from `--allowedTools` for this task |
| Wrong final answer with no error | Task prompt under-specified | Plan task description — add success criteria |
| Repeated identical edits | Test/spec ambiguity | Add failing test or expected behavior to Context |

## Writing each recommendation

A recommendation must answer four questions:

1. **What surface to edit.** Exact file path or flag.
2. **What to change.** The literal addition or replacement.
3. **Which pattern it addresses.** Reference back by name.
4. **Expected impact.** "Should reduce Read count by ~40%" or "should
   eliminate the bash-grep loop entirely". Don't promise token
   savings you can't estimate.

### Example: tighten a skill description

> **Surface.** `.claude/skills/repo-tour/SKILL.md` frontmatter
> `description:` field.
>
> **Change.** Replace the current single-sentence description with one
> that names the symptoms: "Use when an agent is about to run `find`,
> `ls`, or broad Grep to discover repo layout — provides the module
> map and conventional entry points so the exploration is unnecessary."
>
> **Addresses.** `filesystem_explore` count of 9, 7, 6 in three of
> five workbench sessions where this skill was installed but not
> invoked.
>
> **Expected impact.** Should bring `filesystem_explore` to 0–1 per
> session and save the ~1.5k tokens currently spent on `find` output.

### Example: propose a new skill

> **Surface.** New `.claude/skills/test-failure-loop/SKILL.md`.
>
> **Change.** Skill body: when a test command returns a failing
> assertion, read the failing test file in full, then the asserted
> module, then propose a fix as a single Edit. Do not re-run tests
> until at least one Edit has been made.
>
> **Addresses.** Test-failure ping-pong pattern (Edit → Bash(test) →
> Edit → Bash(test) repeating 5+ times) seen in 3 of 8 bug-investigation
> sessions, costing 8–14k tokens each.
>
> **Expected impact.** Should cap loops at 2 cycles and cut per-loop
> token cost by ~60%.

### Example: convention bullet

> **Surface.** `.workbench/conventions.md` under "Tool preferences".
>
> **Change.** Add: "Use the Grep tool for code search. Do not use
> `bash rg` or `bash grep` — they paginate poorly and dump 10k+ tokens
> per call."
>
> **Addresses.** `grep_via_bash` median of 11 across the cohort, with
> output sizes 4–12k tokens.
>
> **Expected impact.** Should eliminate `bash rg` entirely; Grep
> output is bounded to 100 lines by default.

## Report assembly

The final report (per SKILL.md step 5) has five sections. This
reference fills the last three. The first two come from
[population-scoping.md](population-scoping.md) and
[task-identification.md](task-identification.md).

Keep "Waste patterns found" to 2–4 items — readers won't act on more.
Keep "Recommendations" tied 1:1 to those patterns. If you can't tie a
recommendation to a pattern in your data, drop it.

## After the report

Tag the sessions you cited so the next reviewer can find them:

```bash
for sid in a3f2c8 91e4d2 7d12c9; do
  thirdeye tag "$sid" 0 --add review-2026-06, cited
done
```

If the user adopts a recommendation, re-run the same audit a week or
two later filtered to sessions after the change date. The expected
impact is a falsifiable claim — verify it.
