from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.meta import SessionMeta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.usage.aggregate import aggregate_by_day
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


def _make_session(
    root: Path,
    *,
    sid: str,
    platform: str,
    started_at: str,
    last_ts: str | None = None,
) -> Path:
    sd = session_dir(root, platform, sid)
    sd.mkdir(parents=True, exist_ok=True)
    meta = SessionMeta(
        session_id=sid,
        platform=platform,
        cwd="/tmp/proj",
        started_at=started_at,
        ended_at=None,
        status="open",
        event_count=0,
        last_seq=-1,
        last_ts=last_ts or started_at,
    )
    write_meta(meta_path(sd), meta)
    return sd


def _row(
    *,
    sid: str,
    platform: str,
    seq: int,
    ts: str,
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> UsageRow:
    return UsageRow(
        session_id=sid,
        seq=seq,
        ts=ts,
        platform=platform,
        model="claude-opus-4-7",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(root=tmp_path)


def test_empty_store_returns_empty_report(config: Config) -> None:
    report = aggregate_by_day(config)
    assert report.buckets == []
    assert report.totals.day == "TOTAL"
    assert report.totals.sessions == 0
    assert report.totals.events == 0
    assert report.totals.input_tokens == 0
    assert report.totals.output_tokens == 0


def test_two_sessions_two_days_aggregates(config: Config) -> None:
    sd1 = _make_session(
        config.root,
        sid="s1",
        platform="claude",
        started_at="2026-05-20T10:00:00.000Z",
    )
    UsageStore(sd1).append(
        [
            _row(sid="s1", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z"),
            _row(sid="s1", platform="claude", seq=1, ts="2026-05-20T11:00:00.000Z"),
        ]
    )
    sd2 = _make_session(
        config.root,
        sid="s2",
        platform="claude",
        started_at="2026-05-21T10:00:00.000Z",
    )
    UsageStore(sd2).append(
        [
            _row(
                sid="s2",
                platform="claude",
                seq=0,
                ts="2026-05-21T10:00:00.000Z",
                input_tokens=50,
                output_tokens=5,
            ),
        ]
    )

    report = aggregate_by_day(config)
    assert [b.day for b in report.buckets] == ["2026-05-20", "2026-05-21"]

    by_day = {b.day: b for b in report.buckets}
    assert by_day["2026-05-20"].events == 2
    assert by_day["2026-05-20"].sessions == 1
    assert by_day["2026-05-20"].input_tokens == 200
    assert by_day["2026-05-20"].output_tokens == 20
    assert by_day["2026-05-21"].events == 1
    assert by_day["2026-05-21"].sessions == 1
    assert by_day["2026-05-21"].input_tokens == 50
    assert by_day["2026-05-21"].output_tokens == 5

    assert report.totals.events == 3
    assert report.totals.sessions == 2
    assert report.totals.input_tokens == 250
    assert report.totals.output_tokens == 25
    assert report.totals.total_tokens == 275


def test_platform_filter_excludes_others(config: Config) -> None:
    sd_claude = _make_session(
        config.root, sid="c1", platform="claude", started_at="2026-05-20T10:00:00.000Z"
    )
    UsageStore(sd_claude).append(
        [_row(sid="c1", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z")]
    )
    sd_codex = _make_session(
        config.root, sid="x1", platform="codex", started_at="2026-05-20T10:00:00.000Z"
    )
    UsageStore(sd_codex).append(
        [
            _row(
                sid="x1",
                platform="codex",
                seq=0,
                ts="2026-05-20T10:00:00.000Z",
                input_tokens=999,
                output_tokens=0,
            )
        ]
    )

    report = aggregate_by_day(config, platform="claude")
    assert report.totals.sessions == 1
    assert report.totals.input_tokens == 100
    assert report.totals.events == 1


def test_since_until_filter_at_session_level(config: Config) -> None:
    sd_old = _make_session(
        config.root,
        sid="old",
        platform="claude",
        started_at="2026-05-01T10:00:00.000Z",
        last_ts="2026-05-01T11:00:00.000Z",
    )
    UsageStore(sd_old).append(
        [_row(sid="old", platform="claude", seq=0, ts="2026-05-01T10:00:00.000Z")]
    )
    sd_new = _make_session(
        config.root,
        sid="new",
        platform="claude",
        started_at="2026-05-20T10:00:00.000Z",
        last_ts="2026-05-20T11:00:00.000Z",
    )
    UsageStore(sd_new).append(
        [_row(sid="new", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z")]
    )

    since = datetime(2026, 5, 15, tzinfo=UTC)
    until = datetime(2026, 5, 25, tzinfo=UTC)
    report = aggregate_by_day(config, since=since, until=until)
    assert report.totals.sessions == 1
    assert report.totals.events == 1
    assert "2026-05-20" in {b.day for b in report.buckets}
    assert "2026-05-01" not in {b.day for b in report.buckets}


def test_session_counted_distinctly_per_day(config: Config) -> None:
    sd = _make_session(
        config.root,
        sid="s1",
        platform="claude",
        started_at="2026-05-20T10:00:00.000Z",
        last_ts="2026-05-21T10:00:00.000Z",
    )
    UsageStore(sd).append(
        [
            _row(sid="s1", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z"),
            _row(sid="s1", platform="claude", seq=1, ts="2026-05-21T10:00:00.000Z"),
        ]
    )

    report = aggregate_by_day(config)
    by_day = {b.day: b for b in report.buckets}
    assert by_day["2026-05-20"].sessions == 1
    assert by_day["2026-05-21"].sessions == 1
    assert report.totals.sessions == 1


def test_zero_day_buckets_between_seen_days(config: Config) -> None:
    sd1 = _make_session(
        config.root, sid="s1", platform="claude", started_at="2026-05-20T10:00:00.000Z"
    )
    UsageStore(sd1).append(
        [_row(sid="s1", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z")]
    )
    sd2 = _make_session(
        config.root, sid="s2", platform="claude", started_at="2026-05-23T10:00:00.000Z"
    )
    UsageStore(sd2).append(
        [_row(sid="s2", platform="claude", seq=0, ts="2026-05-23T10:00:00.000Z")]
    )

    report = aggregate_by_day(config)
    assert [b.day for b in report.buckets] == [
        "2026-05-20",
        "2026-05-21",
        "2026-05-22",
        "2026-05-23",
    ]
    by_day = {b.day: b for b in report.buckets}
    assert by_day["2026-05-21"].events == 0
    assert by_day["2026-05-21"].sessions == 0
    assert by_day["2026-05-21"].input_tokens == 0
    assert by_day["2026-05-22"].events == 0


def test_bounds_extend_range_when_wider_than_data(config: Config) -> None:
    sd = _make_session(
        config.root, sid="s1", platform="claude", started_at="2026-05-20T10:00:00.000Z"
    )
    UsageStore(sd).append([_row(sid="s1", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z")])

    since = datetime(2026, 5, 18, tzinfo=UTC)
    until = datetime(2026, 5, 22, tzinfo=UTC)
    report = aggregate_by_day(config, since=since, until=until)
    days = [b.day for b in report.buckets]
    assert days[0] == "2026-05-18"
    assert days[-1] == "2026-05-22"
    assert len(days) == 5


def test_row_day_handles_offset_timestamp(config: Config) -> None:
    sd = _make_session(
        config.root,
        sid="s1",
        platform="claude",
        started_at="2026-05-20T23:00:00.000Z",
        last_ts="2026-05-20T23:30:00.000Z",
    )
    UsageStore(sd).append(
        [
            UsageRow(
                session_id="s1",
                seq=0,
                ts="2026-05-20T23:30:00.000+00:00",
                platform="claude",
                model="m",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            ),
            UsageRow(
                session_id="s1",
                seq=1,
                ts="2026-05-21T01:30:00.000+02:00",
                platform="claude",
                model="m",
                input_tokens=2,
                output_tokens=2,
                total_tokens=4,
            ),
        ]
    )

    report = aggregate_by_day(config)
    by_day = {b.day: b.events for b in report.buckets}
    assert by_day.get("2026-05-20") == 2


def test_empty_with_bounds_produces_zero_buckets(config: Config) -> None:
    since = datetime(2026, 5, 18, tzinfo=UTC)
    until = datetime(2026, 5, 20, tzinfo=UTC)
    report = aggregate_by_day(config, since=since, until=until)
    days = [b.day for b in report.buckets]
    assert days == ["2026-05-18", "2026-05-19", "2026-05-20"]
    assert all(b.events == 0 and b.sessions == 0 for b in report.buckets)
    assert report.totals.events == 0


def test_aggregate_returns_immutable_dataclasses(config: Config) -> None:
    report = aggregate_by_day(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.totals.events = 99  # type: ignore[misc]


def test_row_day_helper_handles_z_and_offset() -> None:
    from thirdeye.usage.aggregate import _row_day

    assert _row_day("2026-05-20T10:00:00.000Z") == "2026-05-20"
    assert _row_day("2026-05-21T01:30:00.000+02:00") == "2026-05-20"
