from __future__ import annotations

from collections import Counter
from typing import Literal

from thirdeye.config import Config
from thirdeye.paths import session_dir as _session_dir
from thirdeye.store import Store
from thirdeye.tags import TagStore

Surface = Literal["search", "sessions"]


def build_vocabulary_block(config: Config, surface: Surface) -> str:
    """Return a human-readable block listing distinct platforms, recent
    cwds, and frequent tags drawn from the store.

    The block goes into the agent prompt above the user's NL query so the
    agent only proposes filter values that actually exist.
    """
    store = Store(config)
    platforms: set[str] = set()
    cwd_recency: list[str] = []
    tag_counter: Counter[str] = Counter()

    for meta in store.list_sessions():
        platforms.add(meta.platform)
        if meta.cwd:
            cwd_recency.append(meta.cwd)
        sd = _session_dir(config.root, meta.platform, meta.session_id)
        for tag in TagStore(sd).unique_tags():
            tag_counter[tag] += 1

    cwds_top: list[str] = []
    seen: set[str] = set()
    for cwd in reversed(cwd_recency):
        if cwd in seen:
            continue
        seen.add(cwd)
        cwds_top.append(cwd)
        if len(cwds_top) >= 20:
            break
    tags_top = [t for t, _ in tag_counter.most_common(50)]

    lines = ["KNOWN VOCABULARY (only use these values for filter fields):"]
    lines.append(f"platforms: {sorted(platforms)}")
    lines.append(f"cwds (top 20, most-recent first): {cwds_top}")
    lines.append(f"tags (top 50 by frequency): {tags_top}")
    if surface == "sessions":
        lines.append('statuses: ["open", "closed"]')
        lines.append('orders: ["newest", "oldest", "longest", "shortest"]')
    return "\n".join(lines)


def inventory_tags(config: Config) -> list[str]:
    """Return sorted distinct tags currently effective across all sessions."""
    store = Store(config)
    tags: set[str] = set()
    for meta in store.list_sessions():
        sd = _session_dir(config.root, meta.platform, meta.session_id)
        tags |= TagStore(sd).unique_tags()
    return sorted(tags)
