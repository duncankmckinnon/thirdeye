from __future__ import annotations

from pathlib import Path

import pytest

import thirdeye.index as index_module
from thirdeye.codec import encode_event
from thirdeye.index import IndexReader, IndexWriter, rebuild_index


class TestIndexWriter:
    def test_appends_offsets(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        w = IndexWriter(p)
        w.append(0)
        w.append(123)
        w.append(456)
        w.close()
        assert p.stat().st_size == 24  # 3 * 8 bytes

    def test_creates_parent_dirs(self, tmp_path: Path):
        p = tmp_path / "a" / "b" / "events.idx"
        w = IndexWriter(p)
        w.append(0)
        w.close()
        assert p.exists()

    def test_context_manager(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p) as w:
            w.append(0)
            w.append(100)
        assert p.stat().st_size == 16

    def test_append_large_offset(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        large = 2**48
        with IndexWriter(p) as w:
            w.append(large)
        r = IndexReader(p)
        assert r.get(0) == large

    def test_empty_file_when_no_appends(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p):
            pass
        assert p.stat().st_size == 0

    def test_appends_are_persistent(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p) as w:
            w.append(10)
        # Open a second writer that appends to the same file
        with IndexWriter(p) as w:
            w.append(20)
        assert p.stat().st_size == 16
        assert IndexReader(p).all_offsets() == [10, 20]

    def test_two_writers_interleave_appends(self, tmp_path: Path):
        """Writers created before a rebuild append to its replacement index."""
        events_log = tmp_path / "events.alog"
        index = tmp_path / "events.idx"
        offsets = TestRebuildIndex()._write_events_log(events_log, 2)
        first = IndexWriter(index)
        second = IndexWriter(index)
        try:
            rebuild_index(events_log, index)
            first.append(100)
            second.append(200)
        finally:
            first.close()
            second.close()

        assert IndexReader(index).all_offsets() == [*offsets, 100, 200]


class TestIndexReader:
    def test_get_offset(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p) as w:
            for off in [0, 100, 250, 700]:
                w.append(off)
        r = IndexReader(p)
        assert r.get(0) == 0
        assert r.get(1) == 100
        assert r.get(2) == 250
        assert r.get(3) == 700

    def test_count(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p) as w:
            for off in [0, 100, 250, 700]:
                w.append(off)
        assert IndexReader(p).count() == 4

    def test_count_missing_file(self, tmp_path: Path):
        p = tmp_path / "nonexistent.idx"
        assert IndexReader(p).count() == 0

    def test_count_empty_file(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        p.touch()
        assert IndexReader(p).count() == 0

    def test_get_out_of_range(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        IndexWriter(p).close()
        with pytest.raises(IndexError):
            IndexReader(p).get(0)

    def test_get_negative_index(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p) as w:
            w.append(42)
        with pytest.raises(IndexError):
            IndexReader(p).get(-1)

    def test_get_past_end(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p) as w:
            w.append(0)
            w.append(100)
        with pytest.raises(IndexError):
            IndexReader(p).get(2)

    def test_all_offsets(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        with IndexWriter(p) as w:
            for off in [10, 20, 30]:
                w.append(off)
        assert IndexReader(p).all_offsets() == [10, 20, 30]

    def test_all_offsets_missing_file(self, tmp_path: Path):
        p = tmp_path / "nonexistent.idx"
        assert IndexReader(p).all_offsets() == []

    def test_all_offsets_empty_file(self, tmp_path: Path):
        p = tmp_path / "events.idx"
        p.touch()
        assert IndexReader(p).all_offsets() == []


class TestRebuildIndex:
    def _write_events_log(self, path: Path, count: int) -> list[int]:
        offsets = []
        with open(path, "wb") as f:
            for seq in range(count):
                offsets.append(f.tell())
                f.write(encode_event({"t": "x", "ts": "now", "seq": seq}))
        return offsets

    def test_rebuild_from_events_log(self, tmp_path: Path):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        offsets = self._write_events_log(events_log, 3)
        count = rebuild_index(events_log, idx)
        assert count == 3
        assert IndexReader(idx).all_offsets() == offsets

    def test_rebuild_returns_event_count(self, tmp_path: Path):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        self._write_events_log(events_log, 5)
        assert rebuild_index(events_log, idx) == 5

    def test_rebuild_empty_log(self, tmp_path: Path):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        events_log.touch()
        count = rebuild_index(events_log, idx)
        assert count == 0
        assert IndexReader(idx).all_offsets() == []

    def test_rebuild_missing_log(self, tmp_path: Path):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        count = rebuild_index(events_log, idx)
        assert count == 0
        assert idx.exists()

    def test_rebuild_overwrites_existing_index(self, tmp_path: Path):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        # Write a stale index with garbage
        with IndexWriter(idx) as w:
            w.append(999)
            w.append(888)
        offsets = self._write_events_log(events_log, 2)
        rebuild_index(events_log, idx)
        assert IndexReader(idx).all_offsets() == offsets
        assert IndexReader(idx).count() == 2

    def test_rebuild_leaves_old_index_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        self._write_events_log(events_log, 2)
        with IndexWriter(idx) as writer:
            writer.append(999)

        def replace_fails(src: Path, dst: Path) -> None:
            raise OSError("simulated replacement failure")

        monkeypatch.setattr(index_module.fsops, "replace", replace_fails)

        with pytest.raises(OSError, match="simulated replacement failure"):
            rebuild_index(events_log, idx)

        assert IndexReader(idx).all_offsets() == [999]

    def test_rebuild_single_event(self, tmp_path: Path):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        offsets = self._write_events_log(events_log, 1)
        rebuild_index(events_log, idx)
        assert IndexReader(idx).all_offsets() == [0]

    def test_rebuild_offsets_increase(self, tmp_path: Path):
        events_log = tmp_path / "events.alog"
        idx = tmp_path / "events.idx"
        self._write_events_log(events_log, 10)
        rebuild_index(events_log, idx)
        offsets = IndexReader(idx).all_offsets()
        assert offsets == sorted(offsets)
        assert offsets[0] == 0
        # Each subsequent offset must be strictly greater
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1]
