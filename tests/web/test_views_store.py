from __future__ import annotations

import json

import pytest

from thirdeye.web.views_store import SavedView, ViewStore, now_iso


def _view(name: str = "recent", path: str = "?since=7d") -> SavedView:
    return SavedView(name=name, path=path, created_at=now_iso())


def test_list_missing_file_returns_empty(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    assert store.list() == []


def test_save_creates_file_and_roundtrips(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    view = _view()
    store.save(view)

    assert (tmp_path / "views" / "sessions.json").exists()
    loaded = store.list()
    assert len(loaded) == 1
    assert loaded[0].name == "recent"
    assert loaded[0].path == "?since=7d"
    assert loaded[0].created_at == view.created_at


def test_save_upserts_by_name(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    store.save(_view(name="x", path="?since=7d"))
    store.save(_view(name="x", path="?since=30d"))

    loaded = store.list()
    assert len(loaded) == 1
    assert loaded[0].path == "?since=30d"


def test_save_preserves_other_views(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    store.save(_view(name="a", path="?a=1"))
    store.save(_view(name="b", path="?b=2"))

    names = sorted(v.name for v in store.list())
    assert names == ["a", "b"]


def test_delete_removes_by_name(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    store.save(_view(name="a", path="?a=1"))
    store.save(_view(name="b", path="?b=2"))
    store.delete("a")

    names = [v.name for v in store.list()]
    assert names == ["b"]


def test_delete_unknown_is_noop(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    store.save(_view(name="a", path="?a=1"))
    store.delete("does-not-exist")

    assert [v.name for v in store.list()] == ["a"]


@pytest.mark.parametrize("bad", ["a/b", "", "x" * 65, "name?bad", "tab\there"])
def test_save_rejects_invalid_name(tmp_path, bad):
    store = ViewStore(tmp_path, "sessions")
    with pytest.raises(ValueError):
        store.save(SavedView(name=bad, path="?ok=1", created_at=now_iso()))


@pytest.mark.parametrize("bad", ["since=7d", "?<script>", "?has space", "?a=1#frag", ""])
def test_save_rejects_invalid_path(tmp_path, bad):
    store = ViewStore(tmp_path, "sessions")
    with pytest.raises(ValueError):
        store.save(SavedView(name="ok", path=bad, created_at=now_iso()))


def test_unknown_page_raises(tmp_path):
    with pytest.raises(ValueError):
        ViewStore(tmp_path, "evals")


def test_atomic_write_leaves_no_tmp(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    store.save(_view())

    leftovers = list((tmp_path / "views").glob("*.tmp"))
    assert leftovers == []


def test_corrupt_file_returns_empty(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    store.file.parent.mkdir(parents=True, exist_ok=True)
    store.file.write_text("not json", encoding="utf-8")
    assert store.list() == []


def test_on_disk_schema(tmp_path):
    store = ViewStore(tmp_path, "sessions")
    view = _view(name="recent", path="?since=7d")
    store.save(view)

    data = json.loads(store.file.read_text(encoding="utf-8"))
    assert data == {
        "views": [{"name": "recent", "path": "?since=7d", "created_at": view.created_at}]
    }


def test_now_iso_format():
    s = now_iso()
    assert s.endswith("Z")
    assert "+00:00" not in s
