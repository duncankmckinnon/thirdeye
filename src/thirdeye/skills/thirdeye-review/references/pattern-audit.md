# Tool and execution pattern audit

Count what the agent did, then look for known waste signatures. The
patterns below are the ones that have repeatedly tied back to a fixable
invocation issue in past reviews.

## Schema reminder

thirdeye events use `t` (not `type`). For Claude Code tool calls:
`t == "tool_call"` and `data.tool_name`, `data.tool_input`. Other
platforms (Codex) use the same envelope.

```bash
thirdeye events <sid> --json | jq -r 'select(.t == "tool_call") | .data.tool_name' | sort | uniq -c | sort -rn
```

## Per-session tool mix

A single Python pass that surfaces the most common waste signals:

```bash
thirdeye events <sid> --json | python3 - <<'PY'
import json, sys, re, collections
tools, bash, reads, big_reads = (
    collections.Counter(), collections.Counter(),
    collections.Counter(), [],
)
total_tool = 0
for line in sys.stdin:
    e = json.loads(line)
    if e.get("t") != "tool_call":
        continue
    total_tool += 1
    name = e["data"].get("tool_name", "?")
    tools[name] += 1
    inp = e["data"].get("tool_input", {}) or {}
    if name == "Read":
        path = inp.get("file_path", "")
        reads[path] += 1
        if not inp.get("limit") and not inp.get("offset"):
            big_reads.append(path)
    elif name == "Bash":
        cmd = inp.get("command", "") or ""
        if re.search(r"\bfind\b|\bls\b|\btree\b", cmd):
            bash["filesystem_explore"] += 1
        elif re.search(r"\brg\b|\bgrep\b", cmd):
            bash["grep_via_bash"] += 1
        elif re.search(r"\bpytest\b|\bnpm (test|run test)\b|\bvitest\b", cmd):
            bash["test_run"] += 1
        elif re.search(r"\bgit\b", cmd):
            bash["git"] += 1
        else:
            bash["other"] += 1
print(f"total tool_calls: {total_tool}")
print("top tools     :", dict(tools.most_common(8)))
print("bash buckets  :", dict(bash))
print("re-reads      :", [(p, c) for p, c in reads.most_common(6) if c > 1])
print("unbounded reads:", len(big_reads), "(first 3:", big_reads[:3], ")")
PY
```

## Waste signatures and thresholds

These are starting thresholds — calibrate to the task baseline you
established in [task-identification.md](task-identification.md).

| Signature | Threshold (default cohort) | Read it as | Surface to fix |
|-----------|---------------------------|------------|----------------|
| `grep_via_bash` count | > 8 per session | Agent reaches for `bash rg` instead of the dedicated Grep tool. | Conventions file: explicit tool-preference bullet. |
| `filesystem_explore` count | > 6 per session | No codebase map; agent is orienting from scratch every run. | Repo navigation skill, or plan Context module map. |
| Same file Read ≥ 3× | Any | Lost context, no note-taking, or unbounded reads paged off in earlier turns. | Tighter task scope; explicit `Files: path:line-line` in plan tasks. |
| Unbounded reads (no `limit`/`offset`) | > 30% of Reads | Agent paging whole files when a slice would do. | Conventions bullet: prefer Grep + targeted Read. |
| `tool_call` / `event` ratio | > 0.75 | Chatty read-edit-test loop; little planning. | Smaller task scope; raise `--max-retries` ceiling carefully. |
| Skill invocations on complex task | 0 when a relevant skill exists | Skill is installed but not triggering. | Tighten the skill's `description:` frontmatter. |
| Repeated identical edits to one file | ≥ 2 | Indecision or test-failure ping-pong. | Add the failing test as Context; tighten the spec. |
| Long stretch of `tool_use_error` events | ≥ 3 consecutive | Wrong tool choice or wrong arguments — agent isn't recovering. | Surface error pattern in conventions or skill. |

## Cohort-wide aggregation

After per-session counts, aggregate across the cohort. A pattern only
matters if it shows up across multiple sessions of the same task shape.

```bash
# For each session in cohort.txt, emit a counts row.
while read sid; do
  thirdeye events "$sid" --json \
    | python3 - "$sid" <<'PY'
import json, sys, re
sid = sys.argv[1]
counts = {"total": 0, "Read": 0, "Bash_grep": 0, "Bash_fs": 0,
          "Skill": 0, "Edit": 0, "TodoWrite": 0}
for line in sys.stdin:
    e = json.loads(line)
    if e.get("t") != "tool_call": continue
    counts["total"] += 1
    name = e["data"].get("tool_name", "")
    if name in counts: counts[name] += 1
    if name == "Bash":
        cmd = (e["data"].get("tool_input", {}) or {}).get("command", "")
        if re.search(r"\brg\b|\bgrep\b", cmd): counts["Bash_grep"] += 1
        elif re.search(r"\bfind\b|\bls\b|\btree\b", cmd): counts["Bash_fs"] += 1
print("\t".join([sid[:8]] + [f"{counts[k]}" for k in
      ("total","Read","Bash_grep","Bash_fs","Skill","Edit","TodoWrite")]))
PY
done < cohort.txt
```

Sort columns to find outliers; share medians in the report, not means
(one runaway session distorts the average).

## Execution pattern: order, not just counts

Counts hide ordering problems. Skim the timeline for shape:

```bash
thirdeye events <sid> --json \
  | jq -r 'select(.t == "tool_call") | .data.tool_name' \
  | uniq -c | head -40
```

Patterns to flag:

- **Read-Read-Read-Read-…** with no Grep/Edit in between → agent is
  paging blindly; suggest a planning step or targeted Grep.
- **Edit-Bash(test)-Edit-Bash(test)-…** for 5+ cycles → test-failure
  loop; the spec or failing assertion needs to be in Context.
- **Bash(rg)-Read-Bash(rg)-Read-…** → agent rediscovers the same area
  each turn. Suggest note-taking or a Glob-then-Grep workflow.

## When to dig into one session

If a signature scores way above cohort median in one session, open
that session's event stream and read 10–20 events around the spike:

```bash
thirdeye events <sid> --tree | less
# or, focused:
thirdeye events <sid> --json \
  | jq -c '. | select(.t == "tool_call") | {seq, tool: .data.tool_name, input: (.data.tool_input | tostring)[0:120]}'
```

What you're looking for is the *trigger* — the user turn, error, or
prior tool result that pushed the agent into the wasteful pattern.
That trigger is what your recommendation will target.

## Carry forward

For each pattern above some threshold, record:

- pattern name
- the count + cohort median for context
- one representative `<sid prefix>:<seq>` for a concrete example
- a one-line hypothesis about *why* (read in the next reference)
