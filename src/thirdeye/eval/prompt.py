from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from thirdeye.eval.definition import EvalDefinition
from thirdeye.paths import meta_path, session_dir, usage_db_path, usage_jsonl_path

OUTPUT_CONTRACT = """\
=== Required output ===
Respond with a fenced ```json block containing the structured findings,
followed by a markdown narrative explaining your reasoning.

```json
{
  "verdict": "pass" | "warn" | "fail",
  "summary": "1-2 sentence takeaway",
  "scores": { "<dimension>": <number 0-10>, ... },
  "findings": [
    {
      "seq": <integer or null>,
      "severity": "info" | "warn" | "error",
      "category": "<optional string, e.g. tokens|tools|errors>",
      "note": "<one-line observation>"
    }
  ]
}
```

After the JSON block, write the full narrative (multiple paragraphs OK).
"""


def build_prompt(
    *,
    thirdeye_home: Path,
    platform: str,
    session_id: str,
    definition: EvalDefinition,
    event_lines: list[str] | None = None,
    max_timeline_lines: int = 200,
) -> str:
    """Assemble the three-block evaluator prompt.

    Block 1: definition.directive (verbatim).
    Block 2: session context (auto-assembled from disk).
    Block 3: OUTPUT_CONTRACT (constant).
    """
    sd = session_dir(thirdeye_home, platform, session_id)

    blocks: list[str] = [definition.directive.rstrip(), ""]

    blocks.append("=== Session being evaluated ===")
    blocks.extend(_render_meta(sd, platform, session_id))
    blocks.append("")

    usage_lines = _render_usage_summary(sd)
    if usage_lines:
        blocks.append("=== Usage summary ===")
        blocks.extend(usage_lines)
        blocks.append("")

    if event_lines:
        blocks.append("=== Event timeline (condensed) ===")
        blocks.extend(_truncate_timeline(event_lines, max_timeline_lines))
        blocks.append("")
    else:
        blocks.append("=== Event timeline ===")
        blocks.append(f"(not pre-rendered — use `thirdeye events {session_id}` to inspect)")
        blocks.append("")

    db = usage_db_path(thirdeye_home)
    blocks.append("=== Tool inventory ===")
    blocks.append("You have read-only access to:")
    blocks.append(
        "- `thirdeye` CLI: list, events, show, tail, event, search, tag, tags, stats, usage"
    )
    blocks.append(f"- `sqlite3 {db}`")
    blocks.append("- `jq`, `Read`")
    blocks.append("")

    blocks.append(OUTPUT_CONTRACT)

    return "\n".join(blocks)


def _render_meta(sd: Path, platform: str, session_id: str) -> list[str]:
    """Format key session metadata into prompt lines."""
    lines = [
        f"session_id: {session_id}",
        f"platform:   {platform}",
    ]
    try:
        import yaml

        from thirdeye.meta import read_meta

        m = read_meta(meta_path(sd))
    except (OSError, ImportError, TypeError, ValueError, yaml.YAMLError):
        return lines
    except Exception:
        # Last-resort guard — build_prompt must never raise on bad metadata.
        return lines
    if m is None:
        return lines
    lines.append(f"cwd:        {m.cwd}")
    lines.append(f"started:    {m.started_at}")
    if m.last_ts:
        lines.append(f"last_activity: {m.last_ts}")
    lines.append(f"event_count: {m.event_count}")
    lines.append(f"tag_count:   {m.tag_count}")
    lines.append(f"status:     {m.status}")
    return lines


def _render_usage_summary(sd: Path) -> list[str]:
    """Build the usage summary block by streaming usage.jsonl directly."""
    path = usage_jsonl_path(sd)
    if not path.is_file():
        return []
    total_in = total_out = total = 0
    turns = 0
    models: Counter[str] = Counter()
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            turns += 1
            total_in += int(row.get("input_tokens", 0) or 0)
            total_out += int(row.get("output_tokens", 0) or 0)
            total += int(row.get("total_tokens", 0) or 0)
            model = str(row.get("model", "")).strip()
            if model:
                models[model] += 1
    except OSError:
        return []
    if turns == 0:
        return []
    lines = [f"turns: {turns}"]
    if models:
        lines.append("models: " + ", ".join(f"{m} ({n} turns)" for m, n in models.most_common()))
    lines.append(f"total_input_tokens:  {total_in:,}")
    lines.append(f"total_output_tokens: {total_out:,}")
    lines.append(f"total_tokens:        {total:,}")
    return lines


def _truncate_timeline(lines: list[str], maximum: int) -> list[str]:
    """Keep head + tail with an elided middle if `lines` exceeds `maximum`."""
    if len(lines) <= maximum:
        return lines
    head = maximum // 2
    tail = maximum - head - 1
    elided = len(lines) - head - tail
    return lines[:head] + [f"... ({elided} lines elided) ..."] + lines[-tail:]
