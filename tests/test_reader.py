from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from thirdeye.codec import encode_event
from thirdeye.index import IndexWriter
from thirdeye.paths import events_path, index_path
from thirdeye.reader import SessionReader
from thirdeye.writer import SessionWriter

# -- helpers -------------------------------------------------------------------


def _make_session(tmp_path: Path, events: list[tuple[str, object]]) -> Path:
    sd = tmp_path / "01J9G7"
    w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
    for t, data in events:
        w.append(t, data)
    w.close()
    return sd


def _make_session_manual(tmp_path: Path, events: list[dict], *, subdir: str = "session") -> Path:
    """Build a session dir by hand using codec + index, no SessionWriter needed."""
    sd = tmp_path / subdir
    sd.mkdir(parents=True, exist_ok=True)
    elog = events_path(sd)
    idxp = index_path(sd)
    with open(elog, "wb") as f, IndexWriter(idxp) as iw:
        for evt in events:
            offset = f.tell()
            f.write(encode_event(evt))
            iw.append(offset)
    return sd


# -- iter_events ---------------------------------------------------------------


class TestIterEvents:
    def test_yields_in_order(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2), ("c", 3)])
        events = list(SessionReader(sd).iter_events())
        assert [e["t"] for e in events] == ["a", "b", "c"]
        assert [e["seq"] for e in events] == [0, 1, 2]
        assert [e["data"] for e in events] == [1, 2, 3]

    def test_empty_session(self, tmp_path: Path):
        sd = _make_session(tmp_path, [])
        events = list(SessionReader(sd).iter_events())
        assert events == []

    def test_single_event(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("only", 42)])
        events = list(SessionReader(sd).iter_events())
        assert len(events) == 1
        assert events[0]["t"] == "only"
        assert events[0]["data"] == 42

    def test_preserves_complex_data(self, tmp_path: Path):
        data = {"nested": {"key": [1, 2, 3]}, "flag": True}
        sd = _make_session(tmp_path, [("complex", data)])
        events = list(SessionReader(sd).iter_events())
        assert events[0]["data"] == data

    def test_many_events(self, tmp_path: Path):
        items = [(f"t{i}", i) for i in range(100)]
        sd = _make_session(tmp_path, items)
        events = list(SessionReader(sd).iter_events())
        assert len(events) == 100
        assert events[0]["data"] == 0
        assert events[99]["data"] == 99

    def test_reader_snapshot_during_concurrent_append(self, tmp_path: Path):
        session_dir = _make_session(tmp_path, [("initial", 0)])
        worker = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "from pathlib import Path",
                        "from thirdeye.writer import SessionWriter",
                        f"session_dir = Path({str(session_dir)!r})",
                        "writer = SessionWriter.open("
                        "session_dir, session_id='01J9G7', platform='claude', cwd='/p')",
                        "for value in range(100):",
                        "    writer.append('concurrent', value)",
                        "writer.close()",
                    ]
                ),
            ]
        )
        snapshots: list[list[dict]] = []
        try:
            while worker.poll() is None:
                snapshots.append(list(SessionReader(session_dir).iter_events()))
                time.sleep(0.001)
        finally:
            assert worker.wait(timeout=10) == 0

        snapshots.append(list(SessionReader(session_dir).iter_events()))
        assert snapshots
        for events in snapshots:
            assert [event["seq"] for event in events] == list(range(len(events)))
        assert [event["data"] for event in snapshots[-1]] == [0, *range(100)]


# -- filter by type ------------------------------------------------------------


class TestIterEventsFilterTypes:
    def test_filter_single_type(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2), ("a", 3)])
        events = list(SessionReader(sd).iter_events(types={"a"}))
        assert [e["data"] for e in events] == [1, 3]

    def test_filter_multiple_types(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2), ("c", 3), ("b", 4)])
        events = list(SessionReader(sd).iter_events(types={"a", "b"}))
        assert [e["data"] for e in events] == [1, 2, 4]

    def test_filter_no_matches(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2)])
        events = list(SessionReader(sd).iter_events(types={"z"}))
        assert events == []

    def test_filter_all_match(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("a", 2)])
        events = list(SessionReader(sd).iter_events(types={"a"}))
        assert len(events) == 2

    def test_types_none_returns_all(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2)])
        events = list(SessionReader(sd).iter_events(types=None))
        assert len(events) == 2


# -- seq_range -----------------------------------------------------------------


class TestIterEventsSeqRange:
    def test_seq_range_middle_slice(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 0), ("b", 1), ("c", 2), ("d", 3)])
        events = list(SessionReader(sd).iter_events(seq_range=(1, 3)))
        assert [e["seq"] for e in events] == [1, 2]

    def test_seq_range_from_start(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 0), ("b", 1), ("c", 2)])
        events = list(SessionReader(sd).iter_events(seq_range=(0, 2)))
        assert [e["seq"] for e in events] == [0, 1]

    def test_seq_range_to_end(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 0), ("b", 1), ("c", 2)])
        events = list(SessionReader(sd).iter_events(seq_range=(1, 3)))
        assert [e["seq"] for e in events] == [1, 2]

    def test_seq_range_single_event(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 0), ("b", 1), ("c", 2)])
        events = list(SessionReader(sd).iter_events(seq_range=(1, 2)))
        assert [e["seq"] for e in events] == [1]

    def test_seq_range_empty(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 0), ("b", 1)])
        events = list(SessionReader(sd).iter_events(seq_range=(1, 1)))
        assert events == []

    def test_seq_range_full(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 0), ("b", 1)])
        events = list(SessionReader(sd).iter_events(seq_range=(0, 2)))
        assert len(events) == 2


# -- combined filters ----------------------------------------------------------


class TestIterEventsCombinedFilters:
    def test_type_and_seq_range(self, tmp_path: Path):
        sd = _make_session(
            tmp_path,
            [("a", 0), ("b", 1), ("a", 2), ("b", 3), ("a", 4)],
        )
        events = list(SessionReader(sd).iter_events(types={"a"}, seq_range=(1, 4)))
        assert [e["data"] for e in events] == [2]


# -- get_event -----------------------------------------------------------------


class TestGetEvent:
    def test_get_by_seq(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2), ("c", 3)])
        r = SessionReader(sd)
        assert r.get_event(0)["t"] == "a"
        assert r.get_event(1)["t"] == "b"
        assert r.get_event(2)["data"] == 3

    def test_get_first_event(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("first", "hello")])
        assert SessionReader(sd).get_event(0)["t"] == "first"

    def test_get_last_event(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2), ("last", 99)])
        assert SessionReader(sd).get_event(2)["t"] == "last"

    def test_get_event_out_of_range(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1)])
        with pytest.raises(IndexError):
            SessionReader(sd).get_event(5)

    def test_get_event_negative_index(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1)])
        with pytest.raises(IndexError):
            SessionReader(sd).get_event(-1)

    def test_get_event_returns_full_dict(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("msg", {"text": "hi"})])
        evt = SessionReader(sd).get_event(0)
        assert "t" in evt
        assert "ts" in evt
        assert "seq" in evt
        assert evt["data"] == {"text": "hi"}


# -- torn tail -----------------------------------------------------------------


class TestTornTail:
    def test_torn_tail_recovers_valid_events(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2)])
        # Append a partial/corrupt zstd frame
        with open(events_path(sd), "ab") as f:
            f.write(b"\x28\xb5\x2f\xfd\xff\xff")
        # Add a fake index entry pointing to the corrupt frame
        with IndexWriter(index_path(sd)) as iw:
            iw.append(events_path(sd).stat().st_size - 6)
        r = SessionReader(sd)
        events = list(r.iter_events())
        assert [e["seq"] for e in events] == [0, 1]
        assert r.truncated_tail is True

    def test_no_torn_tail_flag_is_false(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1), ("b", 2)])
        r = SessionReader(sd)
        list(r.iter_events())
        assert r.truncated_tail is False

    def test_torn_tail_single_corrupt_event(self, tmp_path: Path):
        """Session with only a corrupt frame yields nothing."""
        sd = tmp_path / "corrupt"
        sd.mkdir(parents=True)
        elog = events_path(sd)
        idxp = index_path(sd)
        with open(elog, "wb") as f:
            f.write(b"\x28\xb5\x2f\xfd\xff\xff")
        with IndexWriter(idxp) as iw:
            iw.append(0)
        r = SessionReader(sd)
        events = list(r.iter_events())
        assert events == []
        assert r.truncated_tail is True


# -- constructor / init --------------------------------------------------------


class TestSessionReaderInit:
    def test_session_dir_stored(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1)])
        r = SessionReader(sd)
        assert r.session_dir == sd

    def test_truncated_tail_initially_false(self, tmp_path: Path):
        sd = _make_session(tmp_path, [("a", 1)])
        r = SessionReader(sd)
        assert r.truncated_tail is False


# -- manual construction (no SessionWriter dependency) -------------------------


class TestIterEventsManual:
    """Tests using _make_session_manual to verify reader works with
    hand-built session dirs (codec + index only)."""

    def test_manual_session_iter(self, tmp_path: Path):
        events = [
            {"t": "x", "ts": "2026-04-30T00:00:00.000Z", "seq": 0, "data": "a"},
            {"t": "y", "ts": "2026-04-30T00:00:01.000Z", "seq": 1, "data": "b"},
        ]
        sd = _make_session_manual(tmp_path, events)
        result = list(SessionReader(sd).iter_events())
        assert [e["t"] for e in result] == ["x", "y"]
        assert [e["data"] for e in result] == ["a", "b"]

    def test_manual_session_get_event(self, tmp_path: Path):
        events = [
            {"t": "first", "ts": "now", "seq": 0, "data": 10},
            {"t": "second", "ts": "now", "seq": 1, "data": 20},
        ]
        sd = _make_session_manual(tmp_path, events)
        r = SessionReader(sd)
        assert r.get_event(0)["data"] == 10
        assert r.get_event(1)["data"] == 20

    def test_manual_session_filter_types(self, tmp_path: Path):
        events = [
            {"t": "a", "ts": "now", "seq": 0},
            {"t": "b", "ts": "now", "seq": 1},
            {"t": "a", "ts": "now", "seq": 2},
        ]
        sd = _make_session_manual(tmp_path, events)
        result = list(SessionReader(sd).iter_events(types={"b"}))
        assert len(result) == 1
        assert result[0]["seq"] == 1


# -- empty session edge cases (reader.py fix) ----------------------------------


class TestEmptySessionEdgeCases:
    def test_iter_events_no_events_file(self, tmp_path: Path):
        """Session dir exists with index but no events.alog file."""
        sd = tmp_path / "no_events"
        sd.mkdir(parents=True)
        idxp = index_path(sd)
        with IndexWriter(idxp) as iw:
            pass  # empty index
        result = list(SessionReader(sd).iter_events())
        assert result == []

    def test_iter_events_no_events_file_with_seq_range(self, tmp_path: Path):
        """Events file missing + explicit seq_range that would normally iterate."""
        sd = tmp_path / "no_events_range"
        sd.mkdir(parents=True)
        idxp = index_path(sd)
        # Write a fake index entry pointing at offset 0
        with IndexWriter(idxp) as iw:
            iw.append(0)
        result = list(SessionReader(sd).iter_events(seq_range=(0, 1)))
        assert result == []

    def test_iter_events_start_equals_end(self, tmp_path: Path):
        """start == end should yield no events even with valid data."""
        sd = _make_session(tmp_path, [("a", 1), ("b", 2)])
        result = list(SessionReader(sd).iter_events(seq_range=(1, 1)))
        assert result == []

    def test_iter_events_start_greater_than_end(self, tmp_path: Path):
        """start > end should yield no events."""
        sd = _make_session(tmp_path, [("a", 1), ("b", 2)])
        result = list(SessionReader(sd).iter_events(seq_range=(2, 1)))
        assert result == []
