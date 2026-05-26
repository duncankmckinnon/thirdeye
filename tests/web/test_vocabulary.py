from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir as _session_dir
from thirdeye.store import Store
from thirdeye.tags import TagStore
from thirdeye.web.vocabulary import build_vocabulary_block

pytest.importorskip("starlette")


def _open_session(
    store: Store, sid: str, *, platform: str, cwd: str, tags: list[str] | None = None
) -> None:
    with store.open_session(sid, platform=platform, cwd=cwd) as w:
        seq = w.append("user_message", {"prompt": "hi"})
    store.close_session(sid, platform=platform)
    if tags:
        sd = _session_dir(store.config.root, platform, sid)
        ts = TagStore(sd)
        for tag in tags:
            ts.add(seq, tag)


def test_empty_store_renders(tmp_path: Path) -> None:
    config = Config(root=tmp_path)
    (tmp_path / "traces").mkdir()

    block = build_vocabulary_block(config, "search")

    assert "KNOWN VOCABULARY" in block
    assert "platforms: []" in block
    assert "cwds (top 20, most-recent first): []" in block
    assert "tags (top 50 by frequency): []" in block


def test_search_block_lists_distinct_platforms_and_cwds(tmp_path: Path) -> None:
    config = Config(root=tmp_path)
    (tmp_path / "traces").mkdir()
    store = Store(config)

    _open_session(store, "01J9VOC001", platform="claude", cwd="/repo/a")
    _open_session(store, "01J9VOC002", platform="codex", cwd="/repo/b")
    _open_session(store, "01J9VOC003", platform="claude", cwd="/repo/c")

    block = build_vocabulary_block(config, "search")

    assert "platforms: ['claude', 'codex']" in block
    assert "/repo/a" in block
    assert "/repo/b" in block
    assert "/repo/c" in block


def test_cwds_ordered_most_recent_first_and_deduped(tmp_path: Path) -> None:
    config = Config(root=tmp_path)
    (tmp_path / "traces").mkdir()
    store = Store(config)

    _open_session(store, "01J9VOC010", platform="claude", cwd="/old")
    _open_session(store, "01J9VOC011", platform="claude", cwd="/dup")
    _open_session(store, "01J9VOC012", platform="claude", cwd="/dup")
    _open_session(store, "01J9VOC013", platform="claude", cwd="/new")

    block = build_vocabulary_block(config, "search")

    cwd_line = next(line for line in block.splitlines() if line.startswith("cwds (top 20"))
    assert cwd_line.count("/dup") == 1
    new_idx = cwd_line.index("/new")
    dup_idx = cwd_line.index("/dup")
    old_idx = cwd_line.index("/old")
    assert new_idx < dup_idx < old_idx


def test_tags_sorted_by_frequency(tmp_path: Path) -> None:
    config = Config(root=tmp_path)
    (tmp_path / "traces").mkdir()
    store = Store(config)

    _open_session(store, "01J9VOC020", platform="claude", cwd="/p", tags=["frequent"])
    _open_session(store, "01J9VOC021", platform="claude", cwd="/p", tags=["frequent"])
    _open_session(store, "01J9VOC022", platform="claude", cwd="/p", tags=["frequent"])
    _open_session(store, "01J9VOC023", platform="claude", cwd="/p", tags=["rare"])

    block = build_vocabulary_block(config, "search")

    tag_line = next(line for line in block.splitlines() if line.startswith("tags (top 50"))
    assert tag_line.index("frequent") < tag_line.index("rare")


def test_sessions_surface_adds_statuses_and_orders(tmp_path: Path) -> None:
    config = Config(root=tmp_path)
    (tmp_path / "traces").mkdir()

    search_block = build_vocabulary_block(config, "search")
    sessions_block = build_vocabulary_block(config, "sessions")

    assert "statuses:" not in search_block
    assert "orders:" not in search_block
    assert 'statuses: ["open", "closed"]' in sessions_block
    assert 'orders: ["newest", "oldest", "longest", "shortest"]' in sessions_block
