from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping


_TAG_BODY_RE = re.compile(r"[^a-z0-9_\-.#]+")
_TAG_FULL_RE = re.compile(r"^[a-z0-9_\-.#]{1,64}$")


def capture_env(
    patterns: Iterable[str],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return env vars matching the patterns.

    Each pattern is either an exact name (``"WB_PLAN"``) or a prefix glob
    ending in ``*`` (``"WB_*"``). Whitespace around patterns is stripped and
    empty entries are ignored. A bare ``"*"`` is rejected (returns nothing
    for that pattern) so users cannot accidentally capture the entire env.

    Output keys are returned in alphabetical order so tests/snapshots are
    deterministic. ``environ`` defaults to ``os.environ`` and exists to make
    the function pure for unit tests.
    """
    env = dict(environ) if environ is not None else dict(os.environ)
    exact: set[str] = set()
    prefixes: list[str] = []
    for raw in patterns:
        p = raw.strip()
        if not p or p == "*":
            continue
        if p.endswith("*"):
            prefixes.append(p[:-1])
        else:
            exact.add(p)
    out: dict[str, str] = {}
    for name in sorted(env):
        if name in exact or any(name.startswith(pref) for pref in prefixes):
            out[name] = env[name]
    return out


def env_to_tag(name: str, value: str) -> str | None:
    """Derive a session-start tag from an env-var ``(name, value)`` pair.

    Format: ``"<key>-<value>"`` where ``key`` is ``name`` lowercased with a
    leading ``WB_`` (case-insensitive) stripped, and ``value`` is lowercased
    with any character outside ``[a-z0-9_\\-.#]`` replaced by ``-`` and runs
    of dashes collapsed to a single dash, then trimmed of leading/trailing
    ``-``.

    Returns ``None`` (skip silently) when:
      - the key collapses to empty,
      - the sanitized value collapses to empty,
      - the joined tag is longer than 64 chars,
      - the joined tag otherwise fails the tag charset.
    """
    if not name:
        return None
    key = name.lower()
    if key.startswith("wb_"):
        key = key[3:]
    if not key:
        return None
    sanitized = _TAG_BODY_RE.sub("-", value.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        return None
    tag = f"{key}-{sanitized}"
    if not _TAG_FULL_RE.match(tag):
        return None
    return tag
