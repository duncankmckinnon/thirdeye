# Session efficiency review

Analyze recorded sessions to find wasted tool calls, redundant exploration, and skill gaps.
Use this when asked to improve agent performance, token usage, or suggest new skills.

## Quick population scan

```bash
# High-activity sessions in a repo (event count ≈ cost proxy)
thirdeye list --json --cwd "$PWD" --since 2026-05-01 \
  | jq 'select(.event_count > 80) | {id: .session_id[0:8], events: .event_count, cwd: .cwd}'

# Workbench task sessions only
thirdeye list --json --since 2026-05-01 \
  | jq 'select(.cwd | test("/\\.workbench/.+/task-")) | .session_id' -r
```

## Event schema note

Claude Code events use `t` (not `type`) and `data.tool_name` (not `data.name`):

```bash
thirdeye events <sid> --json | jq -r 'select(.t == "tool_call") | .data.tool_name' \
  | sort | uniq -c | sort -rn
```

## Tool-mix analysis

Count tool usage and bash sub-patterns across a session:

```bash
thirdeye events <sid> --json | python3 - <<'PY'
import json, sys, re, collections
tools, bash = collections.Counter(), collections.Counter()
reads = collections.Counter()
for line in sys.stdin:
    e = json.loads(line)
    if e.get("t") != "tool_call": continue
    name = e["data"].get("tool_name", "?")
    tools[name] += 1
    inp = e["data"].get("tool_input", {})
    if name == "Read":
        reads[inp.get("file_path", "")] += 1
    elif name == "Bash":
        cmd = inp.get("command", "")
        if re.search(r"\bfind\b|\bls\b|\btree\b", cmd): bash["filesystem"] += 1
        elif re.search(r"\brg\b|\bgrep\b", cmd): bash["grep_via_bash"] += 1
        elif re.search(r"\bpytest\b|\bnpm test\b", cmd): bash["test"] += 1
        elif re.search(r"\bgit\b", cmd): bash["git"] += 1
        else: bash["other"] += 1
print("tools:", dict(tools.most_common(10)))
print("bash:", dict(bash))
print("re-reads:", [(p, c) for p, c in reads.most_common(8) if c > 1])
PY
```

## Red flags (what to look for)

| Signal | Threshold | Likely cause | Skill/fix |
|--------|-----------|--------------|-----------|
| `grep_via_bash` > 10 | High | Agent uses Bash instead of Grep | Add exploration guidance to plan conventions |
| Same file Read ≥ 3× | High | Lost context or no note-taking | Task description should cite line ranges; agent should cite, not re-read |
| `filesystem` (find/ls) > 5 | Medium | No codebase map | Add module map to plan Context or repo skill |
| 0 Skill invocations on complex task | Medium | Skill not triggered | Improve skill description; `@`-mention in prompt |
| Tool/event ratio > 0.75 | High | Chatty loop (read-edit-test cycles) | Tighter task scope; `--max-retries` may be too high |
| Large tool_result payloads | Check token-use-analysis.md | Unpaginated reads or MCP dumps | Use `head_limit`, `offset`, or targeted Grep |

## Compare sessions with vs without skills

```bash
thirdeye list --json --since 2026-05-01 | python3 - <<'PY'
import json, sys, subprocess, collections
sessions = [json.loads(l) for l in sys.stdin if l.strip()]
skill_used, no_skill = [], []
for s in sessions:
    if s["event_count"] < 30: continue
    out = subprocess.check_output(["thirdeye", "events", s["session_id"], "--json"], text=True)
    skill = sum(1 for l in out.splitlines() if '"tool_name": "Skill"' in l)
    tools = sum(1 for l in out.splitlines() if '"t": "tool_call"' in l)
    bucket = skill_used if skill else no_skill
    bucket.append((s["event_count"], tools))
if skill_used:
    print(f"with Skill: n={len(skill_used)} avg_events={sum(x[0] for x in skill_used)/len(skill_used):.0f}")
if no_skill:
    print(f"without Skill: n={len(no_skill)} avg_events={sum(x[0] for x in no_skill)/len(no_skill):.0f}")
PY
```

Note: sessions that invoke skills tend to be longer tasks (selection bias). Compare within the same task type (e.g. all `project-conventions/task-*`) rather than globally.

## Rubrics for skill suggestions

After scanning 5–10 representative sessions, ask:

1. **Repeated exploration** — Did multiple sessions run the same `find`/`grep`/`ls` commands? → Encapsulate in a repo navigation skill or plan Context section.
2. **Missing upfront context** — Did agents discover test commands, module layout, or config paths via exploration? → Add to `.workbench/conventions.md` or task `Files:` lines.
3. **Wrong tool choice** — Bash grep vs Grep, find vs Glob, broad Read vs targeted Grep? → Add explicit tool-preference bullets to conventions.
4. **Workbench task overhead** — Do dispatched tasks spend >30% of tool calls on orientation? → Plan task descriptions need more `Files:` and interface specs (see use-workbench skill).
5. **Skill underuse** — Is a relevant skill available but never invoked? → Tighten the skill `description` frontmatter (trigger phrases) or `@`-reference it in the plan Context.

## Tag findings for iteration

```bash
thirdeye tag <sid> 0 --add efficiency-review,high-reread
thirdeye tag <sid> 0 --add efficiency-review,bash-grep-waste
```

Filter later:

```bash
thirdeye list --tag efficiency-review --json
```

## Report template

When presenting findings, structure as:

1. **Population** — N sessions, date range, repos/task types
2. **Aggregate metrics** — tool mix, avg events/session, re-read count, bash-explore count
3. **Top waste patterns** — 2–3 concrete examples with session ID prefix and what happened
4. **Skill recommendations** — new skills or updates to existing ones, tied to observed patterns
5. **Plan/convention changes** — bullets to add to `.workbench/conventions.md` or plan Context
