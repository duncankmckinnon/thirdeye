from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from importlib import resources
from urllib.parse import urlencode

from thirdeye.config import Config
from thirdeye.eval.agents import BUILTIN_ADAPTERS, get_adapter, list_agent_names
from thirdeye.eval.agents.exec import invoke_agent
from thirdeye.web.vocabulary import Surface, build_vocabulary_block


@dataclass(frozen=True)
class ProposedFilters:
    q: str | None
    platform: str | None
    cwd: str | None
    tags: list[str]
    since: str | None
    until: str | None
    status: str | None
    order: str | None
    rationale: str | None

    def query_string(self) -> str:
        pairs: list[tuple[str, str]] = []
        if self.q:
            pairs.append(("q", self.q))
        for k, v in (
            ("platform", self.platform),
            ("cwd", self.cwd),
            ("since", self.since),
            ("until", self.until),
            ("status", self.status),
            ("order", self.order),
        ):
            if v:
                pairs.append((k, v))
        for t in self.tags:
            pairs.append(("tag", t))
        return "?" + urlencode(pairs) if pairs else ""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def installed_agent_names(config: Config) -> list[str]:
    names = list_agent_names(config.root)
    installed: list[str] = []
    for name in names:
        adapter_cls = BUILTIN_ADAPTERS.get(name)
        cmd = adapter_cls().config.command if adapter_cls else name
        if shutil.which(cmd) is not None:
            installed.append(name)
    return installed


_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


def _read_skill_body() -> str:
    """Return SKILL.md body with YAML frontmatter stripped.

    Frontmatter is for the skill loader (name/description/type).
    Passing it to the agent CLI causes Claude/Codex to parse the
    leading `---` as a flag and reject the prompt.
    """
    res = resources.files("thirdeye").joinpath("skills", "thirdeye-filter", "SKILL.md")
    text = res.read_text(encoding="utf-8")
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip()


def _build_prompt(config: Config, surface: Surface, nl: str) -> str:
    skill = _read_skill_body()
    vocab = build_vocabulary_block(config, surface)
    return f"{skill}\n\n{vocab}\n\nSURFACE: {surface}\nUSER QUERY:\n{nl}\n"


def _parse_envelope(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s).strip()
    return json.loads(s)


def _to_proposed(env: dict, surface: Surface) -> ProposedFilters:
    tags = [t for t in (env.get("tags") or []) if isinstance(t, str)]
    return ProposedFilters(
        q=(env.get("q") or None) if surface == "search" else None,
        platform=env.get("platform") or None,
        cwd=env.get("cwd") or None,
        tags=tags,
        since=env.get("since") or None,
        until=env.get("until") or None,
        status=(env.get("status") or None) if surface == "sessions" else None,
        order=(env.get("order") or None) if surface == "sessions" else None,
        rationale=env.get("rationale") or None,
    )


def propose_filters(
    config: Config, *, nl: str, agent_name: str, surface: Surface
) -> ProposedFilters:
    """Synchronous; callers should wrap with run_in_threadpool inside route."""
    adapter = get_adapter(agent_name, thirdeye_home=config.root)
    prompt = _build_prompt(config, surface, nl)
    inv = invoke_agent(adapter, prompt, cwd=config.root)
    if inv.returncode != 0:
        raise RuntimeError(f"agent exited {inv.returncode}: {inv.stderr[:200]}")
    agent_text, _cost = adapter.parse_output(inv.stdout)
    env = _parse_envelope(agent_text)
    return _to_proposed(env, surface)
