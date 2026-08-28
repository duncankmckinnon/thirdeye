from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.meta import SessionMeta, read_meta, write_meta
from thirdeye.paths import meta_path, session_dir, sessions_root
from thirdeye.reader import SessionReader
from thirdeye.store import Store
from thirdeye.tags import TagStore
from thirdeye.writer import SessionWriter

# -- open_session --------------------------------------------------------------


class TestOpenSession:
    def test_returns_writer(self, tmp_store: Store):
        with tmp_store.open_session("01J9G7", platform="claude", cwd="/p") as w:
            assert isinstance(w, SessionWriter)

    def test_writer_append_returns_seq_zero(self, tmp_store: Store):
        with tmp_store.open_session("01J9G7", platform="claude", cwd="/p") as w:
            assert w.append("user_message", "hi") == 0

    def test_creates_session_dir(self, tmp_store: Store):
        with tmp_store.open_session("01J9G7", platform="claude", cwd="/p") as w:
            sd = session_dir(tmp_store.config.root, "claude", "01J9G7")
            assert sd.is_dir()

    def test_passes_extra_to_writer(self, tmp_store: Store):
        extra = {"model": "opus"}
        with tmp_store.open_session("01J9G7", platform="claude", cwd="/p", extra=extra) as w:
            w.append("x", 1)
        m = tmp_store.get_meta("01J9G7")
        assert m.extra == extra

    def test_extra_defaults_none(self, tmp_store: Store):
        with tmp_store.open_session("01J9G7", platform="claude", cwd="/p") as w:
            w.append("x", 1)
        m = tmp_store.get_meta("01J9G7")
        assert m.extra == {}


# -- list_sessions -------------------------------------------------------------


class TestListSessions:
    def test_all(self, populated_store: Store):
        metas = list(populated_store.list_sessions())
        assert sorted(m.session_id for m in metas) == ["01J9G7XK4P", "02ABCDEF12"]

    def test_returns_session_meta_objects(self, populated_store: Store):
        metas = list(populated_store.list_sessions())
        for m in metas:
            assert isinstance(m, SessionMeta)

    def test_filter_platform(self, populated_store: Store):
        metas = list(populated_store.list_sessions(platform="claude"))
        assert [m.session_id for m in metas] == ["01J9G7XK4P"]

    def test_filter_platform_cursor(self, populated_store: Store):
        metas = list(populated_store.list_sessions(platform="cursor"))
        assert [m.session_id for m in metas] == ["02ABCDEF12"]

    def test_filter_cwd(self, populated_store: Store):
        metas = list(populated_store.list_sessions(cwd="/proj/b"))
        assert [m.session_id for m in metas] == ["02ABCDEF12"]

    def test_filter_status(self, tmp_store: Store):
        with tmp_store.open_session("AAAA", platform="claude", cwd="/p") as w:
            w.append("x", 1)
        with tmp_store.open_session("BBBB", platform="claude", cwd="/p") as w:
            w.append("x", 1)
        # Reopen BBBB so it stays open
        w2 = tmp_store.open_session("BBBB", platform="claude", cwd="/p")
        metas = list(tmp_store.list_sessions(status="open"))
        assert [m.session_id for m in metas] == ["BBBB"]
        w2.close()

    def test_filter_platform_no_matches(self, populated_store: Store):
        metas = list(populated_store.list_sessions(platform="codex"))
        assert metas == []

    def test_filter_cwd_no_matches(self, populated_store: Store):
        metas = list(populated_store.list_sessions(cwd="/nonexistent"))
        assert metas == []

    def test_empty_store(self, tmp_store: Store):
        metas = list(tmp_store.list_sessions())
        assert metas == []

    def test_sessions_root_doesnt_exist(self, tmp_path: Path):
        store = Store(Config(root=tmp_path / "nonexistent"))
        metas = list(store.list_sessions())
        assert metas == []

    def test_ignores_non_directory_entries(self, tmp_store: Store):
        with tmp_store.open_session("01J9G7", platform="claude", cwd="/p") as w:
            w.append("x", 1)
        # Create a stray file inside platform dir
        stray = sessions_root(tmp_store.config.root) / "claude" / "not-a-dir.txt"
        stray.write_text("junk")
        metas = list(tmp_store.list_sessions())
        assert len(metas) == 1

    def test_ignores_session_dir_without_meta(self, tmp_store: Store):
        # Create a session dir with no meta.yaml
        sd = session_dir(tmp_store.config.root, "claude", "ORPHAN")
        sd.mkdir(parents=True)
        metas = list(tmp_store.list_sessions())
        assert metas == []

    def test_multiple_platforms(self, tmp_store: Store):
        for p in ["claude", "cursor", "codex"]:
            with tmp_store.open_session(f"SID_{p}", platform=p, cwd="/p") as w:
                w.append("x", 1)
        metas = list(tmp_store.list_sessions())
        assert len(metas) == 3
        assert sorted(m.platform for m in metas) == ["claude", "codex", "cursor"]


# -- resolve_session_id --------------------------------------------------------


class TestResolveSessionId:
    def test_unique_prefix(self, populated_store: Store):
        assert populated_store.resolve_session_id("01J9") == ("claude", "01J9G7XK4P")

    def test_full_id(self, populated_store: Store):
        assert populated_store.resolve_session_id("02ABCDEF12") == (
            "cursor",
            "02ABCDEF12",
        )

    def test_no_match_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="no session matching"):
            populated_store.resolve_session_id("ZZZZZ")

    def test_ambiguous_prefix_raises(self, tmp_store: Store):
        with tmp_store.open_session("AB1111", platform="claude", cwd="/p") as w:
            w.append("x", 1)
        with tmp_store.open_session("AB2222", platform="cursor", cwd="/p") as w:
            w.append("x", 1)
        with pytest.raises(ValueError, match="ambiguous"):
            tmp_store.resolve_session_id("AB")

    def test_same_full_id_is_ambiguous_without_platform(self, tmp_store: Store):
        session_id = "SAME_SESSION_ID"
        for platform in ("claude", "cursor"):
            with tmp_store.open_session(session_id, platform=platform, cwd="/p") as w:
                w.append("x", 1)

        with pytest.raises(ValueError, match="ambiguous") as exc_info:
            tmp_store.resolve_session_id(session_id)

        message = str(exc_info.value)
        assert f"claude:{session_id}" in message
        assert f"cursor:{session_id}" in message

    def test_qualified_full_id_resolves_each_platform(self, tmp_store: Store):
        session_id = "SAME_SESSION_ID"
        for platform in ("claude", "cursor"):
            with tmp_store.open_session(session_id, platform=platform, cwd="/p") as w:
                w.append("x", 1)

        assert tmp_store.resolve_session_id(f"claude:{session_id}") == (
            "claude",
            session_id,
        )
        assert tmp_store.resolve_session_id(f"cursor:{session_id}") == (
            "cursor",
            session_id,
        )

    def test_qualified_short_prefix_searches_one_platform(self, tmp_store: Store):
        with tmp_store.open_session("SHARED_CLAUDE", platform="claude", cwd="/p") as w:
            w.append("x", 1)
        with tmp_store.open_session("SHARED_CURSOR", platform="cursor", cwd="/p") as w:
            w.append("x", 1)

        assert tmp_store.resolve_session_id("cursor:SHARED") == (
            "cursor",
            "SHARED_CURSOR",
        )

    def test_qualified_prefix_ambiguous_within_platform(self, tmp_store: Store):
        for session_id in ("AB1111", "AB2222"):
            with tmp_store.open_session(session_id, platform="cursor", cwd="/p") as w:
                w.append("x", 1)
        with tmp_store.open_session("AB3333", platform="claude", cwd="/p") as w:
            w.append("x", 1)

        with pytest.raises(ValueError, match="ambiguous") as exc_info:
            tmp_store.resolve_session_id("cursor:AB")

        message = str(exc_info.value)
        assert "cursor:AB1111" in message
        assert "cursor:AB2222" in message
        assert "claude:AB3333" not in message

    def test_unknown_platform_qualifier_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="unknown platform qualifier.*unknown"):
            populated_store.resolve_session_id("unknown:01J9")

    def test_unknown_platform_codex_qualifier_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match=r"unknown platform.*codex"):
            populated_store.resolve_session_id("codex:01J9")

    def test_known_platform_with_no_matching_session_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="no session matching.*claude:ZZZZZ"):
            populated_store.resolve_session_id("claude:ZZZZZ")

    def test_empty_platform_qualifier_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="platform qualifier.*empty"):
            populated_store.resolve_session_id(":01J9")

    def test_empty_session_qualifier_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="session (prefix|qualifier).*empty"):
            populated_store.resolve_session_id("claude:")

    def test_empty_store_raises(self, tmp_store: Store):
        with pytest.raises(ValueError, match="no session matching"):
            tmp_store.resolve_session_id("anything")

    def test_single_char_prefix(self, populated_store: Store):
        # "0" matches both sessions -> ambiguous
        with pytest.raises(ValueError, match="ambiguous"):
            populated_store.resolve_session_id("0")

    def test_returns_platform_and_sid_tuple(self, populated_store: Store):
        result = populated_store.resolve_session_id("01J9G7XK4P")
        assert isinstance(result, tuple)
        assert len(result) == 2
        platform, sid = result
        assert platform == "claude"
        assert sid == "01J9G7XK4P"

    def test_qualified_reference_supported_by_store_consumers(self, tmp_store: Store):
        session_id = "SAME_SESSION_ID"
        tmp_store.append_event(
            session_id=session_id,
            platform="claude",
            cwd="/claude",
            t="message",
            data="from claude",
        )
        tmp_store.append_event(
            session_id=session_id,
            platform="cursor",
            cwd="/cursor",
            t="message",
            data="from cursor",
        )

        assert (
            list(tmp_store.reader(f"claude:{session_id}").iter_events())[0]["data"] == "from claude"
        )
        assert tmp_store.get_meta(f"cursor:{session_id}").cwd == "/cursor"
        assert tmp_store.stats(session_id=f"cursor:{session_id}")["platform"] == "cursor"


# -- reader --------------------------------------------------------------------


class TestReader:
    def test_returns_session_reader(self, populated_store: Store):
        r = populated_store.reader("01J9")
        assert isinstance(r, SessionReader)

    def test_reader_can_iterate_events(self, populated_store: Store):
        r = populated_store.reader("01J9")
        events = list(r.iter_events())
        assert len(events) == 2
        assert events[0]["t"] == "user_message"
        assert events[1]["t"] == "assistant_message"

    def test_reader_no_match_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="no session matching"):
            populated_store.reader("ZZZZZ")


# -- get_meta ------------------------------------------------------------------


class TestGetMeta:
    def test_returns_meta(self, populated_store: Store):
        m = populated_store.get_meta("01J9G7XK4P")
        assert isinstance(m, SessionMeta)
        assert m.session_id == "01J9G7XK4P"
        assert m.platform == "claude"

    def test_by_prefix(self, populated_store: Store):
        m = populated_store.get_meta("02AB")
        assert m.session_id == "02ABCDEF12"
        assert m.platform == "cursor"

    def test_no_match_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="no session matching"):
            populated_store.get_meta("ZZZZZ")

    def test_event_count(self, populated_store: Store):
        m = populated_store.get_meta("01J9")
        assert m.event_count == 2

    def test_cwd(self, populated_store: Store):
        m = populated_store.get_meta("01J9")
        assert m.cwd == "/proj/a"


# -- stats ---------------------------------------------------------------------


class TestStats:
    def test_session_stats(self, populated_store: Store):
        s = populated_store.stats(session_id="01J9G7XK4P")
        assert s["event_count"] == 2
        assert s["bytes_compressed"] > 0
        assert s["session_id"] == "01J9G7XK4P"
        assert s["platform"] == "claude"

    def test_session_stats_by_prefix(self, populated_store: Store):
        s = populated_store.stats(session_id="02AB")
        assert s["event_count"] == 1
        assert s["platform"] == "cursor"

    def test_global_stats(self, populated_store: Store):
        s = populated_store.stats()
        assert s["session_count"] == 2
        assert s["event_count"] == 3
        assert s["bytes_compressed"] > 0

    def test_global_stats_empty_store(self, tmp_store: Store):
        s = tmp_store.stats()
        assert s["session_count"] == 0
        assert s["event_count"] == 0
        assert s["bytes_compressed"] == 0

    def test_session_stats_no_match_raises(self, populated_store: Store):
        with pytest.raises(ValueError, match="no session matching"):
            populated_store.stats(session_id="ZZZZZ")

    def test_session_stats_bytes_increases_with_events(self, tmp_store: Store):
        with tmp_store.open_session("SID1", platform="claude", cwd="/p") as w:
            w.append("x", "short")
        s1 = tmp_store.stats(session_id="SID1")
        with tmp_store.open_session("SID2", platform="claude", cwd="/p") as w:
            for i in range(20):
                w.append("x", f"data_{i}" * 50)
        s2 = tmp_store.stats(session_id="SID2")
        assert s2["bytes_compressed"] > s1["bytes_compressed"]


# -- constructor ---------------------------------------------------------------


class TestStoreInit:
    def test_config_stored(self, tmp_path: Path):
        cfg = Config(root=tmp_path)
        store = Store(cfg)
        assert store.config is cfg


# -- append_event --------------------------------------------------------------


class TestAppendEvent:
    def test_one_shot(self, tmp_store: Store):
        seq = tmp_store.append_event(
            session_id="01J9G7XK4P",
            platform="claude",
            cwd="/p",
            t="user_message",
            data="hi",
        )
        assert seq == 0
        metas = list(tmp_store.list_sessions())
        assert len(metas) == 1
        assert metas[0].session_id == "01J9G7XK4P"
        assert metas[0].status == "open"
        assert metas[0].event_count == 1

    def test_appends_to_existing_session(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="01J9G7XK4P",
            platform="claude",
            cwd="/p",
            t="user_message",
            data="hi",
        )
        seq = tmp_store.append_event(
            session_id="01J9G7XK4P",
            platform="claude",
            cwd="/p",
            t="assistant_message",
            data="hello",
        )
        assert seq == 1

    def test_returns_int(self, tmp_store: Store):
        result = tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        assert isinstance(result, int)

    def test_data_none_accepted(self, tmp_store: Store):
        seq = tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="session_start",
        )
        assert seq == 0

    def test_complex_data(self, tmp_store: Store):
        payload = {"tool_name": "Read", "args": {"path": "/foo"}, "nested": [1, 2]}
        seq = tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="tool_call",
            data=payload,
        )
        assert seq == 0
        r = tmp_store.reader("SID1")
        events = list(r.iter_events())
        assert events[0]["data"] == payload

    def test_event_readable_after_append(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="user_message",
            data="hello",
        )
        r = tmp_store.reader("SID1")
        events = list(r.iter_events())
        assert len(events) == 1
        assert events[0]["t"] == "user_message"
        assert events[0]["data"] == "hello"

    def test_many_sequential_appends(self, tmp_store: Store):
        for i in range(10):
            seq = tmp_store.append_event(
                session_id="SID1",
                platform="claude",
                cwd="/p",
                t=f"evt_{i}",
                data=i,
            )
            assert seq == i
        m = tmp_store.get_meta("SID1")
        assert m.event_count == 10

    def test_different_sessions_independent(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID_A",
            platform="claude",
            cwd="/p",
            t="a",
            data=1,
        )
        tmp_store.append_event(
            session_id="SID_B",
            platform="claude",
            cwd="/q",
            t="b",
            data=2,
        )
        metas = sorted(tmp_store.list_sessions(), key=lambda m: m.session_id)
        assert len(metas) == 2
        assert metas[0].session_id == "SID_A"
        assert metas[1].session_id == "SID_B"

    def test_session_stays_open_after_append(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        m = tmp_store.get_meta("SID1")
        assert m.status == "open"
        assert m.ended_at is None

    def test_different_platforms(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        tmp_store.append_event(
            session_id="SID2",
            platform="cursor",
            cwd="/p",
            t="y",
            data=2,
        )
        claude_sessions = list(tmp_store.list_sessions(platform="claude"))
        cursor_sessions = list(tmp_store.list_sessions(platform="cursor"))
        assert len(claude_sessions) == 1
        assert len(cursor_sessions) == 1


# -- close_session -------------------------------------------------------------


class TestCloseSession:
    def test_marks_closed(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="01J9G7XK4P",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        tmp_store.close_session("01J9G7XK4P", platform="claude")
        m = next(tmp_store.list_sessions())
        assert m.status == "closed"
        assert m.ended_at is not None

    def test_no_op_for_missing_session(self, tmp_store: Store):
        tmp_store.close_session("does-not-exist", platform="claude")

    def test_no_op_for_missing_platform(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        tmp_store.close_session("SID1", platform="cursor")
        # Original session should still be open
        m = tmp_store.get_meta("SID1")
        assert m.status == "open"

    def test_sets_ended_at_timestamp(self, tmp_store: Store):
        import re

        ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        tmp_store.close_session("SID1", platform="claude")
        m = tmp_store.get_meta("SID1")
        assert ts_re.match(m.ended_at)

    def test_preserves_event_count(self, tmp_store: Store):
        for i in range(3):
            tmp_store.append_event(
                session_id="SID1",
                platform="claude",
                cwd="/p",
                t=f"evt_{i}",
                data=i,
            )
        tmp_store.close_session("SID1", platform="claude")
        m = tmp_store.get_meta("SID1")
        assert m.event_count == 3

    def test_preserves_started_at(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        started = tmp_store.get_meta("SID1").started_at
        tmp_store.close_session("SID1", platform="claude")
        m = tmp_store.get_meta("SID1")
        assert m.started_at == started

    def test_close_then_append_reopens(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="a",
            data=1,
        )
        tmp_store.close_session("SID1", platform="claude")
        assert tmp_store.get_meta("SID1").status == "closed"
        seq = tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="b",
            data=2,
        )
        assert seq == 1
        assert tmp_store.get_meta("SID1").status == "open"

    def test_idempotent_close(self, tmp_store: Store):
        tmp_store.append_event(
            session_id="SID1",
            platform="claude",
            cwd="/p",
            t="x",
            data=1,
        )
        tmp_store.close_session("SID1", platform="claude")
        tmp_store.close_session("SID1", platform="claude")
        m = tmp_store.get_meta("SID1")
        assert m.status == "closed"

    def test_no_op_when_session_dir_exists_but_no_meta(self, tmp_store: Store):
        sd = session_dir(tmp_store.config.root, "claude", "ORPHAN")
        sd.mkdir(parents=True)
        tmp_store.close_session("ORPHAN", platform="claude")


# -- list_sessions tag filter --------------------------------------------------


class TestListSessionsTagFilter:
    @staticmethod
    def _setup(tmp_store: Store) -> None:
        tmp_store.append_event(
            session_id="TAGGED",
            platform="claude",
            cwd="/p",
            t="user_message",
            data="hi",
        )
        tmp_store.append_event(
            session_id="UNTAGGED",
            platform="claude",
            cwd="/p",
            t="user_message",
            data="hi",
        )
        sd = session_dir(tmp_store.config.root, "claude", "TAGGED")
        ts = TagStore(sd)
        ts.add(0, "alpha")
        ts.add(0, "beta")

    def test_single_tag_matches(self, tmp_store: Store):
        self._setup(tmp_store)
        metas = list(tmp_store.list_sessions(tags={"alpha"}))
        assert [m.session_id for m in metas] == ["TAGGED"]

    def test_all_required_tags_present(self, tmp_store: Store):
        self._setup(tmp_store)
        metas = list(tmp_store.list_sessions(tags={"alpha", "beta"}))
        assert [m.session_id for m in metas] == ["TAGGED"]

    def test_missing_tag_excludes(self, tmp_store: Store):
        self._setup(tmp_store)
        metas = list(tmp_store.list_sessions(tags={"alpha", "gamma"}))
        assert metas == []

    def test_empty_tag_set_no_filter(self, tmp_store: Store):
        self._setup(tmp_store)
        metas = list(tmp_store.list_sessions(tags=set()))
        assert sorted(m.session_id for m in metas) == ["TAGGED", "UNTAGGED"]

    def test_none_tag_no_filter(self, tmp_store: Store):
        self._setup(tmp_store)
        metas = list(tmp_store.list_sessions(tags=None))
        assert sorted(m.session_id for m in metas) == ["TAGGED", "UNTAGGED"]


# -- list_sessions date filter -------------------------------------------------


def _set_window(
    store: Store, sid: str, platform: str, started_at: str, last_ts: str | None
) -> None:
    sd = session_dir(store.config.root, platform, sid)
    mp = meta_path(sd)
    m = read_meta(mp)
    assert m is not None
    m.started_at = started_at
    m.last_ts = last_ts
    write_meta(mp, m)


class TestListSessionsDateFilter:
    def _build(self, tmp_store: Store) -> None:
        tmp_store.append_event(session_id="OLD", platform="claude", cwd="/p", t="x", data=1)
        tmp_store.append_event(session_id="NEW", platform="claude", cwd="/p", t="x", data=1)
        _set_window(
            tmp_store,
            "OLD",
            "claude",
            "2026-01-01T00:00:00.000Z",
            "2026-01-02T00:00:00.000Z",
        )
        _set_window(
            tmp_store,
            "NEW",
            "claude",
            "2026-03-01T00:00:00.000Z",
            "2026-03-02T00:00:00.000Z",
        )

    def test_since_excludes_older(self, tmp_store: Store):
        self._build(tmp_store)
        since = datetime(2026, 2, 1, tzinfo=UTC)
        metas = list(tmp_store.list_sessions(since=since))
        assert [m.session_id for m in metas] == ["NEW"]

    def test_until_excludes_newer(self, tmp_store: Store):
        self._build(tmp_store)
        until = datetime(2026, 2, 1, tzinfo=UTC)
        metas = list(tmp_store.list_sessions(until=until))
        assert [m.session_id for m in metas] == ["OLD"]

    def test_since_and_until_require_overlap(self, tmp_store: Store):
        self._build(tmp_store)
        since = datetime(2026, 1, 15, tzinfo=UTC)
        until = datetime(2026, 2, 15, tzinfo=UTC)
        metas = list(tmp_store.list_sessions(since=since, until=until))
        assert metas == []

    def test_overlap_at_window_start(self, tmp_store: Store):
        self._build(tmp_store)
        since = datetime(2026, 1, 2, tzinfo=UTC)
        until = datetime(2026, 1, 15, tzinfo=UTC)
        metas = list(tmp_store.list_sessions(since=since, until=until))
        assert [m.session_id for m in metas] == ["OLD"]

    def test_last_ts_none_zero_width_window(self, tmp_store: Store):
        tmp_store.append_event(session_id="EMPTY", platform="claude", cwd="/p", t="x", data=1)
        _set_window(tmp_store, "EMPTY", "claude", "2026-02-15T00:00:00.000Z", None)

        # Within window
        since = datetime(2026, 2, 1, tzinfo=UTC)
        until = datetime(2026, 3, 1, tzinfo=UTC)
        metas = list(tmp_store.list_sessions(since=since, until=until))
        assert [m.session_id for m in metas] == ["EMPTY"]

        # Strictly before since
        since2 = datetime(2026, 3, 1, tzinfo=UTC)
        metas = list(tmp_store.list_sessions(since=since2))
        assert metas == []

        # Strictly after until
        until2 = datetime(2026, 2, 1, tzinfo=UTC)
        metas = list(tmp_store.list_sessions(until=until2))
        assert metas == []


# -- list_sessions combined filters --------------------------------------------


class TestListSessionsCombined:
    def test_platform_cwd_tags_since(self, tmp_store: Store):
        # MATCH: claude, /proj/a, tagged, within window
        tmp_store.append_event(session_id="MATCH", platform="claude", cwd="/proj/a", t="x", data=1)
        _set_window(
            tmp_store,
            "MATCH",
            "claude",
            "2026-03-01T00:00:00.000Z",
            "2026-03-02T00:00:00.000Z",
        )
        TagStore(session_dir(tmp_store.config.root, "claude", "MATCH")).add(0, "wanted")

        # WRONG_PLATFORM
        tmp_store.append_event(
            session_id="WRONGPLAT", platform="cursor", cwd="/proj/a", t="x", data=1
        )
        _set_window(
            tmp_store,
            "WRONGPLAT",
            "cursor",
            "2026-03-01T00:00:00.000Z",
            "2026-03-02T00:00:00.000Z",
        )
        TagStore(session_dir(tmp_store.config.root, "cursor", "WRONGPLAT")).add(0, "wanted")

        # WRONG_CWD
        tmp_store.append_event(
            session_id="WRONGCWD", platform="claude", cwd="/proj/b", t="x", data=1
        )
        _set_window(
            tmp_store,
            "WRONGCWD",
            "claude",
            "2026-03-01T00:00:00.000Z",
            "2026-03-02T00:00:00.000Z",
        )
        TagStore(session_dir(tmp_store.config.root, "claude", "WRONGCWD")).add(0, "wanted")

        # UNTAGGED
        tmp_store.append_event(
            session_id="UNTAGGED", platform="claude", cwd="/proj/a", t="x", data=1
        )
        _set_window(
            tmp_store,
            "UNTAGGED",
            "claude",
            "2026-03-01T00:00:00.000Z",
            "2026-03-02T00:00:00.000Z",
        )

        # TOO_OLD
        tmp_store.append_event(session_id="TOOOLD", platform="claude", cwd="/proj/a", t="x", data=1)
        _set_window(
            tmp_store,
            "TOOOLD",
            "claude",
            "2026-01-01T00:00:00.000Z",
            "2026-01-02T00:00:00.000Z",
        )
        TagStore(session_dir(tmp_store.config.root, "claude", "TOOOLD")).add(0, "wanted")

        since = datetime(2026, 2, 1, tzinfo=UTC)
        metas = list(
            tmp_store.list_sessions(
                platform="claude",
                cwd="/proj/a",
                tags={"wanted"},
                since=since,
            )
        )
        assert [m.session_id for m in metas] == ["MATCH"]
