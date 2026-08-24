from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

from thirdeye.codec import decode_event
from thirdeye.index import IndexReader
from thirdeye.meta import SessionMeta, read_meta, write_meta
from thirdeye.paths import events_path, index_path, meta_path
from thirdeye.reader import SessionReader
from thirdeye.writer import SessionWriter, _utc_iso_ms

# -- helpers -------------------------------------------------------------------

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _open_writer(tmp_path: Path, sid: str = "01J9G7XK4P", **kw) -> SessionWriter:
    sd = tmp_path / sid
    defaults = dict(session_id=sid, platform="claude", cwd="/proj")
    defaults.update(kw)
    return SessionWriter.open(sd, **defaults)


# -- open / directory creation -------------------------------------------------


class TestOpen:
    def test_creates_session_dir(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        assert sd.is_dir()
        w.close()

    def test_creates_nested_session_dir(self, tmp_path: Path):
        sd = tmp_path / "deep" / "nested" / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        assert sd.is_dir()
        w.close()

    def test_writes_meta_yaml_on_open(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        m = read_meta(meta_path(sd))
        assert m is not None
        assert m.status == "open"
        assert m.session_id == "01J9G7"
        assert m.platform == "claude"
        assert m.cwd == "/p"
        assert m.event_count == 0
        assert m.last_seq == -1
        assert m.last_ts is None
        assert m.ended_at is None
        w.close()

    def test_meta_started_at_is_utc_iso_ms(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        m = read_meta(meta_path(sd))
        assert _TS_RE.match(m.started_at)
        w.close()

    def test_extra_passed_through(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        extra = {"model": "opus", "tags": ["a"]}
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p", extra=extra)
        m = read_meta(meta_path(sd))
        assert m.extra == extra
        w.close()

    def test_extra_defaults_to_empty_dict(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        m = read_meta(meta_path(sd))
        assert m.extra == {}
        w.close()

    def test_idempotent_open_on_existing_dir(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        sd.mkdir(parents=True)
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        assert sd.is_dir()
        w.close()


# -- append --------------------------------------------------------------------


class TestAppend:
    def test_returns_seq_zero_for_first_event(self, tmp_path: Path):
        w = _open_writer(tmp_path)
        assert w.append("user_message", "hi") == 0
        w.close()

    def test_increments_seq(self, tmp_path: Path):
        w = _open_writer(tmp_path)
        assert w.append("a", 1) == 0
        assert w.append("b", 2) == 1
        assert w.append("c", 3) == 2
        w.close()

    def test_writes_decodable_frame(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("user_message", "hello")
        w.close()
        idx = IndexReader(index_path(sd))
        offset = idx.get(0)
        with open(events_path(sd), "rb") as f:
            f.seek(offset)
            frame = f.read()
        event = decode_event(frame)
        assert event["t"] == "user_message"
        assert event["seq"] == 0
        assert event["data"] == "hello"
        assert "ts" in event

    def test_event_ts_is_utc_iso_ms(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("msg", "x")
        w.close()
        idx = IndexReader(index_path(sd))
        with open(events_path(sd), "rb") as f:
            f.seek(idx.get(0))
            frame = f.read()
        event = decode_event(frame)
        assert _TS_RE.match(event["ts"])

    def test_data_none_omits_data_key(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("heartbeat")
        w.close()
        idx = IndexReader(index_path(sd))
        with open(events_path(sd), "rb") as f:
            f.seek(idx.get(0))
            frame = f.read()
        event = decode_event(frame)
        assert event["t"] == "heartbeat"
        assert "data" not in event

    def test_data_complex_value(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        payload = {"key": "val", "nested": [1, 2, 3]}
        w.append("tool_call", payload)
        w.close()
        idx = IndexReader(index_path(sd))
        with open(events_path(sd), "rb") as f:
            f.seek(idx.get(0))
            frame = f.read()
        event = decode_event(frame)
        assert event["data"] == payload

    def test_index_entry_written_per_event(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("a", 1)
        w.append("b", 2)
        w.append("c", 3)
        w.close()
        idx = IndexReader(index_path(sd))
        assert idx.count() == 3

    def test_index_offsets_are_monotonically_increasing(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        for i in range(5):
            w.append("evt", f"data_{i}")
        w.close()
        offsets = IndexReader(index_path(sd)).all_offsets()
        assert offsets[0] == 0
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1]

    def test_events_log_created_on_first_append(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        assert not events_path(sd).exists() or events_path(sd).stat().st_size == 0
        w.append("x", 1)
        w.close()
        assert events_path(sd).exists()
        assert events_path(sd).stat().st_size > 0

    def test_multiple_events_all_decodable(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        for i in range(5):
            w.append(f"type_{i}", {"idx": i})
        w.close()
        idx = IndexReader(index_path(sd))
        offsets = idx.all_offsets()
        data = events_path(sd).read_bytes()
        for i, offset in enumerate(offsets):
            # Determine frame boundary: next offset or end of file
            end = offsets[i + 1] if i + 1 < len(offsets) else len(data)
            frame = data[offset:end]
            event = decode_event(frame)
            assert event["t"] == f"type_{i}"
            assert event["seq"] == i
            assert event["data"] == {"idx": i}


# -- close ---------------------------------------------------------------------


class TestClose:
    def test_marks_session_closed(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("x", 1)
        w.close()
        m = read_meta(meta_path(sd))
        assert m.status == "closed"

    def test_sets_ended_at(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.close()
        m = read_meta(meta_path(sd))
        assert m.ended_at is not None
        assert _TS_RE.match(m.ended_at)

    def test_updates_event_count_in_meta(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("a", 1)
        w.append("b", 2)
        w.append("c", 3)
        w.close()
        m = read_meta(meta_path(sd))
        assert m.event_count == 3
        assert m.last_seq == 2

    def test_updates_last_ts_in_meta(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("x", 1)
        w.close()
        m = read_meta(meta_path(sd))
        assert m.last_ts is not None
        assert _TS_RE.match(m.last_ts)

    def test_custom_status_on_close(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.close(status="stale")
        m = read_meta(meta_path(sd))
        assert m.status == "stale"

    def test_close_no_events_meta_consistent(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.close()
        m = read_meta(meta_path(sd))
        assert m.event_count == 0
        assert m.last_seq == -1
        assert m.last_ts is None
        assert m.status == "closed"


# -- reopen / resume -----------------------------------------------------------


class TestReopen:
    def test_resumes_seq_after_close(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("a", 1)
        w.append("b", 2)
        w.close()

        w2 = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        assert w2.append("c", 3) == 2
        w2.close()

    def test_reopen_preserves_existing_events(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("a", 1)
        w.append("b", 2)
        w.close()

        w2 = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w2.append("c", 3)
        w2.close()

        idx = IndexReader(index_path(sd))
        assert idx.count() == 3
        # First two events should still be decodable at their original offsets
        offsets = idx.all_offsets()
        data = events_path(sd).read_bytes()
        for i, offset in enumerate(offsets):
            end = offsets[i + 1] if i + 1 < len(offsets) else len(data)
            event = decode_event(data[offset:end])
            assert event["seq"] == i

    def test_reopen_sets_status_to_open(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.close()
        m = read_meta(meta_path(sd))
        assert m.status == "closed"

        w2 = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        m2 = read_meta(meta_path(sd))
        assert m2.status == "open"
        assert m2.ended_at is None
        w2.close()

    def test_reopen_preserves_started_at(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.close()
        original_started = read_meta(meta_path(sd)).started_at

        w2 = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w2.close()
        assert read_meta(meta_path(sd)).started_at == original_started

    def test_reopen_updates_event_count_correctly(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("a", 1)
        w.append("b", 2)
        w.close()

        w2 = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w2.append("c", 3)
        w2.append("d", 4)
        w2.close()
        m = read_meta(meta_path(sd))
        assert m.event_count == 4
        assert m.last_seq == 3


# -- context manager -----------------------------------------------------------


class TestContextManager:
    def test_context_manager_closes(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        with SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p") as w:
            w.append("a", 1)
        m = read_meta(meta_path(sd))
        assert m.status == "closed"
        assert m.event_count == 1

    def test_context_manager_returns_self(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        with w as ctx:
            assert ctx is w
        # already closed by __exit__


# -- index rebuild on open with stale index ------------------------------------


class TestIndexRebuildOnOpen:
    def test_rebuilds_index_when_empty_idx_but_nonempty_log(self, tmp_path: Path):
        sd = tmp_path / "01J9G7XK4P"
        sd.mkdir(parents=True)
        # Create an events log with one event but no index entries
        from thirdeye.codec import encode_event

        frame = encode_event({"t": "x", "ts": "2026-04-30T00:00:00.000Z", "seq": 0})
        with open(events_path(sd), "wb") as f:
            f.write(frame)
        # Create empty index
        index_path(sd).touch()
        # Write a meta so open() finds existing session
        meta = SessionMeta(
            session_id="01J9G7XK4P",
            platform="claude",
            cwd="/p",
            started_at="2026-04-30T00:00:00.000Z",
            ended_at="2026-04-30T00:01:00.000Z",
            status="closed",
            event_count=1,
            last_seq=0,
            last_ts="2026-04-30T00:00:00.000Z",
            extra={},
        )
        write_meta(meta_path(sd), meta)
        # Reopen: should rebuild index and resume at seq 1
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        assert w.append("y", 2) == 1
        w.close()
        assert IndexReader(index_path(sd)).count() == 2


# -- _utc_iso_ms helper --------------------------------------------------------


class TestUtcIsoMs:
    def test_format(self):
        ts = _utc_iso_ms()
        assert _TS_RE.match(ts), f"Timestamp {ts!r} does not match expected format"

    def test_ends_with_z(self):
        ts = _utc_iso_ms()
        assert ts.endswith("Z")

    def test_contains_milliseconds(self):
        ts = _utc_iso_ms()
        # ms part is the 3 digits before the Z
        ms_part = ts[-4:-1]
        assert ms_part.isdigit()
        assert len(ms_part) == 3


# -- meta not written on every append ------------------------------------------


class TestMetaWriteFrequency:
    def test_meta_not_updated_on_every_append(self, tmp_path: Path):
        """meta.yaml is only written at open() and close(), not per-event."""
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        m_after_open = read_meta(meta_path(sd))
        assert m_after_open.event_count == 0

        w.append("a", 1)
        w.append("b", 2)
        # meta.yaml should still show event_count=0 (not updated mid-session)
        m_mid = read_meta(meta_path(sd))
        assert m_mid.event_count == 0

        w.close()
        # Only after close should meta reflect the final state
        m_final = read_meta(meta_path(sd))
        assert m_final.event_count == 2


# -- fsync discipline ----------------------------------------------------------


class TestFsyncDiscipline:
    def test_events_log_is_fsynced(self, tmp_path: Path):
        """After append, the data should be on disk (testable by re-reading)."""
        sd = tmp_path / "01J9G7XK4P"
        w = SessionWriter.open(sd, session_id="01J9G7XK4P", platform="claude", cwd="/p")
        w.append("important", "data")
        # Even before close, the event data should be readable on disk
        assert events_path(sd).stat().st_size > 0
        w.close()


# -- various platforms ---------------------------------------------------------


class TestPlatforms:
    @pytest.mark.parametrize("platform", ["claude", "cursor", "codex", "gemini", "copilot"])
    def test_open_with_various_platforms(self, tmp_path: Path, platform: str):
        sd = tmp_path / platform / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform=platform, cwd="/p")
        w.append("evt", 1)
        w.close()
        m = read_meta(meta_path(sd))
        assert m.platform == platform
        assert m.event_count == 1


# -- flush_and_detach ----------------------------------------------------------


class TestFlushAndDetach:
    def test_keeps_status_open(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        w.append("user_message", "hi")
        w.flush_and_detach()
        m = read_meta(sd / "meta.yaml")
        assert m.status == "open"
        assert m.ended_at is None
        assert m.event_count == 1

    def test_releases_index_fd(self, tmp_path: Path):
        """A second writer can append after detach."""
        sd = tmp_path / "01J9G7"
        w1 = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        w1.append("a", 1)
        w1.flush_and_detach()
        w2 = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        seq = w2.append("b", 2)
        w2.close()
        assert seq == 1

    def test_no_events_before_detach(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        w.flush_and_detach()
        m = read_meta(meta_path(sd))
        assert m.status == "open"
        assert m.event_count == 0
        assert m.last_seq == -1

    def test_preserves_started_at(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        m_before = read_meta(meta_path(sd))
        w.append("x", 1)
        w.flush_and_detach()
        m_after = read_meta(meta_path(sd))
        assert m_after.started_at == m_before.started_at

    def test_meta_updated_by_detach(self, tmp_path: Path):
        """flush_and_detach writes meta with current event_count."""
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        w.append("x", 1)
        w.append("y", 2)
        m_mid = read_meta(meta_path(sd))
        assert m_mid.event_count == 0
        w.flush_and_detach()
        m_after = read_meta(meta_path(sd))
        assert m_after.event_count == 2
        assert m_after.last_seq == 1
        assert m_after.status == "open"

    def test_close_after_detach_raises_or_noop(self, tmp_path: Path):
        """Calling close() after flush_and_detach() should not corrupt state."""
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        w.append("x", 1)
        w.flush_and_detach()
        # close() writes meta with status=closed; the fd is already released
        # so index_w.close() may be a no-op or raise - either way meta should be ok
        # This verifies the detach->close sequence doesn't blow up
        try:
            w.close()
        except Exception:
            pass
        # Session dir still exists and is readable
        m = read_meta(meta_path(sd))
        assert m is not None

    def test_multiple_detach_reopen_cycles(self, tmp_path: Path):
        """Simulates multiple hook invocations writing to the same session."""
        sd = tmp_path / "01J9G7"
        for i in range(5):
            w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
            seq = w.append(f"evt_{i}", i)
            w.flush_and_detach()
            assert seq == i
        # Final close
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        w.close()
        m = read_meta(meta_path(sd))
        assert m.status == "closed"
        assert m.event_count == 5

    def test_events_readable_after_detach(self, tmp_path: Path):
        sd = tmp_path / "01J9G7"
        w = SessionWriter.open(sd, session_id="01J9G7", platform="claude", cwd="/p")
        w.append("user_message", "hello")
        w.flush_and_detach()
        idx = IndexReader(index_path(sd))
        assert idx.count() == 1
        with open(events_path(sd), "rb") as f:
            f.seek(idx.get(0))
            frame = f.read()
        event = decode_event(frame)
        assert event["t"] == "user_message"
        assert event["data"] == "hello"


# -- Concurrent writers ----------------------------------------------------------


class TestConcurrentWriters:
    def test_concurrent_writers_produce_unique_contiguous_seqs(self, tmp_path: Path):
        """Every hook invocation is a fresh process, so a fresh SessionWriter
        for the same session. If two are constructed around the same moment
        and each derives its `seq` from a stale pre-lock snapshot, they can
        claim the same seq for different events, and their unsynchronized
        index appends can land out of step with the log -- corrupting the
        index enough that reading the session back can fail outright.
        """
        sid = "01J9G7XK4P"
        sd = tmp_path / sid
        n_threads = 5
        n_per_thread = 40
        total = n_threads * n_per_thread

        errors: list[str] = []
        written_pairs: set[tuple[str, int]] = set()
        results_lock = threading.Lock()

        def worker(tag: str) -> None:
            for i in range(n_per_thread):
                try:
                    w = SessionWriter.open(sd, session_id=sid, platform="claude", cwd="/proj")
                    try:
                        w.append("tool_call", {"worker": tag, "i": i})
                    finally:
                        w.flush_and_detach()
                    with results_lock:
                        written_pairs.add((tag, i))
                except Exception as e:  # noqa: BLE001 -- must record, not swallow
                    with results_lock:
                        errors.append(f"{tag}: {type(e).__name__}: {e}")
                    return

        threads = [threading.Thread(target=worker, args=(tag,)) for tag in "ABCDE"[:n_threads]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(written_pairs) == total

        idx = IndexReader(index_path(sd))
        assert idx.count() == total
        offsets = idx.all_offsets()
        assert offsets == sorted(offsets), "index offsets must stay monotonically increasing"

        events = list(SessionReader(sd).iter_events())
        assert len(events) == total
        seqs = [e["seq"] for e in events]
        assert sorted(seqs) == list(range(total)), "seqs must be unique and contiguous"

        read_pairs = {(e["data"]["worker"], e["data"]["i"]) for e in events}
        assert read_pairs == written_pairs, "every written event must read back with its own data"

    def test_concurrent_construction_with_missing_index_repairs_cleanly(self, tmp_path: Path):
        """SessionWriter.open() rebuilds the index from the log if the index
        is empty but the log has data (e.g. a crash wiped the index after an
        earlier write). If two processes hit that repair path at once, the
        repair itself must not corrupt the index it's trying to fix.
        """
        sid = "01J9G7XK4P"
        sd = tmp_path / sid
        seed = SessionWriter.open(sd, session_id=sid, platform="claude", cwd="/proj")
        seed.append("user_message", {"n": 0})
        seed.append("user_message", {"n": 1})
        seed.flush_and_detach()

        # Simulate a crash that wiped the index but left the log intact.
        index_path(sd).write_bytes(b"")

        errors: list[str] = []
        errors_lock = threading.Lock()
        start = threading.Barrier(5)

        def worker() -> None:
            start.wait(timeout=5)
            try:
                w = SessionWriter.open(sd, session_id=sid, platform="claude", cwd="/proj")
                try:
                    w.append("user_message", {"n": 2})
                finally:
                    w.flush_and_detach()
            except Exception as e:  # noqa: BLE001
                with errors_lock:
                    errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        idx = IndexReader(index_path(sd))
        assert idx.count() == 7  # 2 seeded + 5 concurrent
        offsets = idx.all_offsets()
        assert offsets == sorted(offsets)

        events = list(SessionReader(sd).iter_events())
        assert len(events) == 7
        assert {e["seq"] for e in events} == set(range(7))


# -- utc_iso_ms public alias ---------------------------------------------------


class TestUtcIsoMsPublic:
    def test_format_matches_private(self):
        from thirdeye.writer import utc_iso_ms

        ts = utc_iso_ms()
        assert _TS_RE.match(ts), f"Timestamp {ts!r} does not match expected format"

    def test_returns_string(self):
        from thirdeye.writer import utc_iso_ms

        assert isinstance(utc_iso_ms(), str)

    def test_ends_with_z(self):
        from thirdeye.writer import utc_iso_ms

        assert utc_iso_ms().endswith("Z")
