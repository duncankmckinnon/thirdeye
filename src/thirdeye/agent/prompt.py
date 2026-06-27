from __future__ import annotations

import re
from datetime import date
from importlib import resources
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)

VALID_SKILLS: frozenset[str] = frozenset(
    ["use-thirdeye", "thirdeye-review", "thirdeye-evals", "thirdeye-filter"]
)
DEFAULT_SKILLS: list[str] = ["use-thirdeye", "thirdeye-review"]


def _load_skill(name: str) -> str:
    """Return SKILL.md body with YAML frontmatter stripped."""
    res = resources.files("thirdeye").joinpath("skills", name, "SKILL.md")
    text = res.read_text(encoding="utf-8")
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip()


def build_agent_prompt(
    task: str,
    *,
    skills: list[str] | None = None,
    cwd: Path | str | None = None,
    thirdeye_home: Path | str | None = None,
) -> str:
    """Build the full prompt string to pass to the agent subprocess.

    Structure:
        {skill_1_body}

        ---

        {skill_2_body}

        ---

        Context:
          date: YYYY-MM-DD
          thirdeye_home: ...   (if provided)
          cwd: ...             (if provided)

        TASK:
        {task}
    """
    if skills is None:
        skills = DEFAULT_SKILLS

    unknown = [s for s in skills if s not in VALID_SKILLS]
    if unknown:
        raise ValueError(f"unknown skill(s): {unknown!r}. Valid: {sorted(VALID_SKILLS)}")

    parts: list[str] = []
    for name in skills:
        parts.append(_load_skill(name))

    context_lines = [f"date: {date.today().isoformat()}"]
    if thirdeye_home is not None:
        context_lines.append(f"thirdeye_home: {thirdeye_home}")
    if cwd is not None:
        context_lines.append(f"cwd: {cwd}")
    context_block = "Context:\n" + "\n".join(f"  {line}" for line in context_lines)
    parts.append(context_block)
    parts.append(f"TASK:\n{task}")

    return "\n\n---\n\n".join(parts)
